"""Experimental mixed-precision QPSK CC/CAP inference helpers.

This module is intentionally separate from the production online inference
path.  It reuses ``FullyOnlinePipelineGPU`` for raw-IQ loading, BOI/shift,
FFT3 feature extraction, LR9 inference, and blind Rs/CFO estimation, while
replacing only the QPSK-like CC and CAP stages for precision experiments.

PyTorch has no complex-bfloat16 dtype.  BF16 complex arithmetic is therefore
represented by separate real and imaginary bfloat16 tensors.  The NCO phase,
phase increment, recursive oscillator state, and rotation factor use the
independently selected phase dtype (float64/complex128 by default).  The
rotated product is rounded back to the selected work dtype before the
streaming-style reduction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import pre_compact_fft3_vector_generator as compact
from . import pre_qpsklike_rs_cfo_estimator as rc
from .fully_online_pipeline_inference import (
    COARSE_CLASS_NAMES,
    FINE_CLASS_NAMES,
    _cc_power as _cc_power_cpu,
)
from .fully_online_pipeline_inference_gpu import (
    _cc_power,
    _estimate_rs_cfo_from_torch_vectors,
)


K_LIST = tuple(range(-5, 6))
K_TO_INDEX = {k: k + 5 for k in K_LIST}
FIRST15_NM = (
    (2, 0), (2, 1), (2, 2),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
)
NM_TO_COL = {nm: col for col, nm in enumerate(FIRST15_NM)}
ACTIVE_K = {
    (2, 1): (-1, 0, 1),
    (4, 0): (-2, -1, 0, 1, 2),
    (4, 2): (-2, -1, 0, 1, 2),
    (4, 4): (-2, -1, 0, 1, 2),
    (6, 1): (-2, -1, 0, 1, 2),
    (6, 3): (-2, -1, 0, 1, 2),
    (6, 5): (-2, -1, 0, 1, 2),
}


def resolve_real_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key == "float32":
        return torch.float32
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError("dtype must be 'float32' or 'bfloat16'")


def resolve_phase_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key == "float32":
        return torch.float32
    if key in {"float64", "double"}:
        return torch.float64
    raise ValueError("phase dtype must be 'float32' or 'float64'")


def select_nonnegative_snr_reference_rows(
    reference_frame,
    years: Sequence[int],
    batch_ids: Sequence[int] | None = None,
    n_per_batch: int = 4000,
    snr_column: str = "snr_db",
):
    """Return finite, non-negative-SNR rows for the requested evaluation set.

    The returned frame is the single source of truth for both inference index
    construction and result comparison in the mixed-precision notebooks.  A
    negative-SNR row therefore cannot enter either population accidentally.
    When ``batch_ids`` is ``None``, every row listed for the requested years is
    retained; otherwise the historical contiguous batch window is applied.
    """
    required = {"dataset", "index", snr_column}
    missing = required - set(reference_frame.columns)
    if missing:
        raise KeyError(f"Reference frame is missing columns: {sorted(missing)}")
    years = tuple(int(year) for year in years)
    mask = reference_frame["dataset"].astype(int).isin(years)
    if batch_ids is not None:
        batch_ids = tuple(int(batch_id) for batch_id in batch_ids)
        if not batch_ids:
            raise ValueError("batch_ids must not be empty")
        first_index = (min(batch_ids) - 1) * int(n_per_batch) + 1
        last_index = max(batch_ids) * int(n_per_batch)
        mask &= reference_frame["index"].astype(int).between(first_index, last_index)
    frame = reference_frame.loc[mask].copy()
    frame["dataset"] = frame["dataset"].astype(int)
    frame["index"] = frame["index"].astype(int)
    frame[snr_column] = frame[snr_column].astype(float)
    eligible = np.isfinite(frame[snr_column].to_numpy()) & frame[snr_column].ge(0.0).to_numpy()
    frame = frame.loc[eligible].copy()
    if frame[snr_column].lt(0.0).any() or not np.isfinite(frame[snr_column]).all():
        raise RuntimeError("Negative or non-finite SNR escaped the eligibility filter")
    if frame.duplicated(["dataset", "index"]).any():
        raise ValueError("Duplicate dataset/index rows in eligible reference frame")
    return frame.sort_values(["dataset", "index"]).reset_index(drop=True)


def _validate_nonnegative_snr_batch(snr_db, expected_count: int) -> np.ndarray:
    values = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if int(values.size) != int(expected_count):
        raise ValueError(
            f"Expected {expected_count} SNR values for the inference chunk, got {values.size}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Non-finite SNR is forbidden in mixed-precision inference")
    negative = np.flatnonzero(values < 0.0)
    if negative.size:
        raise ValueError(
            f"Negative-SNR samples are forbidden in mixed-precision inference; "
            f"chunk positions={negative[:10].tolist()}"
        )
    return values


def _zeros(length: int, device: torch.device, dtype: torch.dtype):
    return (
        torch.zeros(length, device=device, dtype=dtype),
        torch.zeros(length, device=device, dtype=dtype),
    )


def _complex_mul(a, b, dtype: torch.dtype):
    ar, ai = a
    br, bi = b
    real = (ar * br - ai * bi).to(dtype)
    imag = (ar * bi + ai * br).to(dtype)
    return real, imag


def _complex_conj(value):
    real, imag = value
    return real, (-imag).to(imag.dtype)


def _assign(sequence, k: int, value) -> None:
    sequence[0][K_TO_INDEX[int(k)]] = value[0]
    sequence[1][K_TO_INDEX[int(k)]] = value[1]


def _at(sequence, k: int):
    idx = K_TO_INDEX[int(k)]
    return sequence[0][idx], sequence[1][idx]


def _project_many_with_nco(
    value,
    t0,
    f0,
    n: int,
    m: int,
    k_values: Sequence[int],
    work_dtype: torch.dtype,
    phase_dtype: torch.dtype,
):
    """Project moments with the selected recursive NCO and work-dtype MAC."""
    real, imag = value
    device = real.device
    n_samples = int(real.numel())
    k_values = tuple(int(k) for k in k_values)
    if phase_dtype not in {torch.float32, torch.float64}:
        raise ValueError("phase_dtype must be torch.float32 or torch.float64")
    complex_phase_dtype = (
        torch.complex64 if phase_dtype == torch.float32 else torch.complex128
    )
    k = torch.tensor(k_values, device=device, dtype=phase_dtype)
    t0 = torch.as_tensor(t0, device=device, dtype=phase_dtype)
    f0 = torch.as_tensor(f0, device=device, dtype=phase_dtype)
    alpha = float(n - 2 * m) * f0 + k / t0
    phase_step = -2.0 * torch.pi * alpha
    step = torch.polar(torch.ones_like(phase_step), phase_step).to(complex_phase_dtype)

    oscillator = torch.ones(
        (len(k_values), n_samples),
        device=device,
        dtype=complex_phase_dtype,
    )
    if n_samples > 1:
        repeated_step = step[:, None].expand(-1, n_samples - 1)
        oscillator[:, 1:] = torch.cumprod(repeated_step, dim=1)

    # Rotation is evaluated in the selected phase precision, then the
    # multiplier output is rounded to the requested non-phase work dtype.
    product_real = (
        real.to(phase_dtype)[None, :] * oscillator.real
        - imag.to(phase_dtype)[None, :] * oscillator.imag
    ).to(work_dtype)
    product_imag = (
        real.to(phase_dtype)[None, :] * oscillator.imag
        + imag.to(phase_dtype)[None, :] * oscillator.real
    ).to(work_dtype)

    # The result dtype is explicitly low precision.  cumsum exposes the
    # streaming-reduction rounding at each stored prefix on supported CUDA
    # backends instead of silently requesting a float64 accumulator.
    sum_real = torch.cumsum(product_real, dim=1, dtype=work_dtype)[:, -1]
    sum_imag = torch.cumsum(product_imag, dim=1, dtype=work_dtype)[:, -1]
    divisor = torch.tensor(float(n_samples), device=device, dtype=work_dtype)
    return (sum_real / divisor).to(work_dtype), (sum_imag / divisor).to(work_dtype)


def _build_independent_moments(
    x: torch.Tensor,
    t0,
    f0,
    work_dtype: torch.dtype,
    phase_dtype: torch.dtype,
):
    if x.ndim != 1:
        raise ValueError(f"x must have shape (N,), got {tuple(x.shape)}")
    x = x.to(torch.complex64)
    xr = x.real.to(work_dtype)
    xi = x.imag.to(work_dtype)
    z = torch.zeros_like(xr)

    x_pair = (xr, xi)
    x_conj = (xr, (-xi).to(work_dtype))
    x2 = _complex_mul(x_pair, x_pair, work_dtype)
    x4 = _complex_mul(x2, x2, work_dtype)
    x5 = _complex_mul(x4, x_pair, work_dtype)
    abs2 = (xr * xr + xi * xi).to(work_dtype)
    abs4 = (abs2 * abs2).to(work_dtype)
    abs6 = (abs4 * abs2).to(work_dtype)

    sources = {
        (2, 1): (abs2, z),
        (4, 0): x4,
        (4, 2): (abs4, z),
        (6, 1): _complex_mul(x5, x_conj, work_dtype),
        (6, 3): (abs6, z),
    }
    independent_k = {
        (2, 1): (0, 1),
        (4, 0): (-2, -1, 0, 1, 2),
        (4, 2): (0, 1, 2),
        (6, 1): (-2, -1, 0, 1, 2),
        (6, 3): (0, 1, 2),
    }

    moments = {}
    for nm, source in sources.items():
        sequence = _zeros(len(K_LIST), x.device, work_dtype)
        values = _project_many_with_nco(
            source,
            t0,
            f0,
            nm[0],
            nm[1],
            independent_k[nm],
            work_dtype,
            phase_dtype,
        )
        for pos, k_value in enumerate(independent_k[nm]):
            _assign(sequence, k_value, (values[0][pos], values[1][pos]))
        moments[nm] = sequence

    for nm in ((2, 1), (4, 2), (6, 3)):
        positive_limit = 1 if nm == (2, 1) else 2
        for k_value in range(1, positive_limit + 1):
            value = _at(moments[nm], k_value)
            _assign(moments[nm], -k_value, _complex_conj(value))

    m44 = _zeros(len(K_LIST), x.device, work_dtype)
    m65 = _zeros(len(K_LIST), x.device, work_dtype)
    for k_value in (-2, -1, 0, 1, 2):
        _assign(m44, k_value, _complex_conj(_at(moments[(4, 0)], -k_value)))
        _assign(m65, k_value, _complex_conj(_at(moments[(6, 1)], -k_value)))
    moments[(4, 4)] = m44
    moments[(6, 5)] = m65
    return moments


def _conv_same(a, b, work_dtype: torch.dtype):
    out = _zeros(len(K_LIST), a[0].device, work_dtype)
    for ka in K_LIST:
        av = _at(a, ka)
        for kb in K_LIST:
            kout = ka + kb
            if kout not in K_TO_INDEX:
                continue
            product = _complex_mul(av, _at(b, kb), work_dtype)
            idx = K_TO_INDEX[kout]
            out[0][idx] = (out[0][idx] + product[0]).to(work_dtype)
            out[1][idx] = (out[1][idx] + product[1]).to(work_dtype)
    return out


def _add_scaled(base, terms, work_dtype: torch.dtype):
    real = base[0].clone()
    imag = base[1].clone()
    for coefficient, value in terms:
        scale = torch.tensor(float(coefficient), device=real.device, dtype=work_dtype)
        real = (real + scale * value[0]).to(work_dtype)
        imag = (imag + scale * value[1]).to(work_dtype)
    return real, imag


def _mask(sequence, nm):
    real = torch.zeros_like(sequence[0])
    imag = torch.zeros_like(sequence[1])
    for k_value in ACTIVE_K[nm]:
        idx = K_TO_INDEX[k_value]
        real[idx] = sequence[0][idx]
        imag[idx] = sequence[1][idx]
    return real, imag


def build_qpsk_cap_input_mixed_precision(
    x: torch.Tensor,
    t0,
    f0,
    power,
    compute_dtype: str | torch.dtype,
    phase_dtype: str | torch.dtype = "float64",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return one experimental 11x15 CAP input in float32 or bfloat16."""
    work_dtype = (
        resolve_real_dtype(compute_dtype)
        if isinstance(compute_dtype, str)
        else compute_dtype
    )
    if work_dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("compute_dtype must be torch.float32 or torch.bfloat16")
    nco_dtype = (
        resolve_phase_dtype(phase_dtype)
        if isinstance(phase_dtype, str)
        else phase_dtype
    )
    if nco_dtype not in {torch.float32, torch.float64}:
        raise ValueError("phase_dtype must be torch.float32 or torch.float64")
    moments = _build_independent_moments(x, t0, f0, work_dtype, nco_dtype)
    m21 = moments[(2, 1)]
    m40 = moments[(4, 0)]
    m42 = moments[(4, 2)]
    m44 = moments[(4, 4)]
    m61 = moments[(6, 1)]
    m63 = moments[(6, 3)]
    m65 = moments[(6, 5)]

    c21 = m21
    c40 = m40
    c42 = _add_scaled(m42, ((-2.0, _conv_same(m21, m21, work_dtype)),), work_dtype)
    c44 = m44
    c61 = _add_scaled(m61, ((-5.0, _conv_same(m40, m21, work_dtype)),), work_dtype)
    c63 = _add_scaled(
        m63,
        (
            (-9.0, _conv_same(m42, m21, work_dtype)),
            (12.0, _conv_same(_conv_same(m21, m21, work_dtype), m21, work_dtype)),
        ),
        work_dtype,
    )
    c65 = _add_scaled(m65, ((-5.0, _conv_same(m44, m21, work_dtype)),), work_dtype)

    matrix_real = torch.zeros((11, 15), device=x.device, dtype=work_dtype)
    matrix_imag = torch.zeros_like(matrix_real)
    for nm, sequence in (
        ((2, 1), c21), ((4, 0), c40), ((4, 2), c42), ((4, 4), c44),
        ((6, 1), c61), ((6, 3), c63), ((6, 5), c65),
    ):
        sequence = _mask(sequence, nm)
        col = NM_TO_COL[nm]
        matrix_real[:, col] = sequence[0]
        matrix_imag[:, col] = sequence[1]

    magnitude = torch.sqrt(
        (matrix_real * matrix_real + matrix_imag * matrix_imag).to(work_dtype)
    ).to(work_dtype)
    for col, (n, _m) in enumerate(FIRST15_NM):
        exponent = torch.tensor(2.0 / float(n), device=x.device, dtype=work_dtype)
        magnitude[:, col] = torch.pow(magnitude[:, col], exponent).to(work_dtype)

    power = torch.as_tensor(power, device=x.device, dtype=work_dtype)
    if not bool(torch.isfinite(power).detach().cpu().item()) or not bool(
        (power > torch.tensor(float(eps), device=x.device, dtype=work_dtype)).detach().cpu().item()
    ):
        raise ValueError("Invalid CC normalization power")
    return (magnitude / power).to(work_dtype)


def configure_cap_model_test_dtype(pipeline, nn_dtype: str | torch.dtype) -> torch.dtype:
    dtype = resolve_real_dtype(nn_dtype) if isinstance(nn_dtype, str) else nn_dtype
    if dtype not in {torch.float32, torch.bfloat16}:
        raise ValueError("nn_dtype must be torch.float32 or torch.bfloat16")
    pipeline.cap_model.to(device=pipeline.device, dtype=dtype).eval()
    return dtype


# HLS-oriented fixed-point CAPNetLite accuracy experiments.  PyTorch has no
# native int24/int40 convolution, so integer tensors and binary-point metadata
# are stored in the artifacts while the GPU model emulates every externally
# visible rounding/saturation point in float32.  This is intended for accuracy
# screening; FPGA resource/latency must still come from HLS synthesis.
FIXED_CAP_SCHEMES = {
    "A": {
        "description": "ZCU104: INT16 weights/activations, INT32 products, INT48 accumulators",
        "default_weight_bits": 16,
        "default_activation_bits": 16,
        "default_activation_frac_bits": 10,
        "default_accumulator_bits": 48,
        "overrides": {},
    },
    "B": {
        "description": "ZCU104: INT24 weights, INT16 activations, INT40 products, INT48 accumulators",
        "default_weight_bits": 24,
        "default_activation_bits": 16,
        "default_activation_frac_bits": 10,
        "default_accumulator_bits": 48,
        "overrides": {},
    },
    "C": {
        "description": "ZCU104: INT16 weights, INT24 activations, INT40 products, INT48 accumulators",
        "default_weight_bits": 16,
        "default_activation_bits": 24,
        "default_activation_frac_bits": 16,
        "default_accumulator_bits": 48,
        "overrides": {},
    },
    "D": {
        "description": "ZCU104: INT24 stem/FC weights, INT16 middle path, INT48 accumulators, INT24 residual add",
        "default_weight_bits": 16,
        "default_activation_bits": 16,
        "default_activation_frac_bits": 10,
        "default_accumulator_bits": 48,
        "residual_add_format": {"bits": 24, "frac_bits": 16},
        "overrides": {
            "stem": {
                "weight_bits": 24,
                "input_bits": 16,
                "input_frac_bits": 10,
                "output_bits": 16,
                "output_frac_bits": 10,
                "accumulator_bits": 48,
            },
            "fc": {
                "weight_bits": 24,
                "input_bits": 16,
                "input_frac_bits": 10,
                "output_matches_accumulator": True,
                "accumulator_bits": 48,
            },
        },
    },
}

_FIXED_CAP_LAYER_DEFS = (
    ("stem", "stem.weight", "stem_bn"),
    ("block1_conv1", "block1.conv1.weight", "block1.bn1"),
    ("block1_conv2", "block1.conv2.weight", "block1.bn2"),
    ("down", "down.weight", "down_bn"),
    ("block2_conv1", "block2.conv1.weight", "block2.bn1"),
    ("block2_conv2", "block2.conv2.weight", "block2.bn2"),
    ("fc", "fc.weight", None),
)


def _fixed_integer_bounds(bits: int) -> tuple[int, int]:
    bits = int(bits)
    if bits < 2 or bits > 64:
        raise ValueError(f"Signed fixed-point width must be in [2, 64], got {bits}")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1


def _fixed_layer_spec(scheme: str, layer_name: str) -> dict:
    scheme_key = str(scheme).strip().upper()
    if scheme_key not in FIXED_CAP_SCHEMES:
        raise ValueError(f"Unknown fixed-point scheme {scheme!r}; expected A, B, C, or D")
    cfg = FIXED_CAP_SCHEMES[scheme_key]
    default_bits = int(cfg["default_activation_bits"])
    default_frac = int(cfg["default_activation_frac_bits"])
    spec = {
        "weight_bits": int(cfg["default_weight_bits"]),
        "input_bits": default_bits,
        "input_frac_bits": default_frac,
        "output_bits": default_bits,
        "output_frac_bits": default_frac,
        "accumulator_bits": int(cfg["default_accumulator_bits"]),
    }
    spec.update(cfg.get("overrides", {}).get(layer_name, {}))
    spec["product_bits"] = int(spec["input_bits"]) + int(spec["weight_bits"])
    return spec


def _fold_conv_batch_norm(
    state_dict: dict,
    weight_key: str,
    bn_prefix: str,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = state_dict[weight_key].detach().to(torch.float64)
    gamma = state_dict[f"{bn_prefix}.weight"].detach().to(torch.float64)
    beta = state_dict[f"{bn_prefix}.bias"].detach().to(torch.float64)
    mean = state_dict[f"{bn_prefix}.running_mean"].detach().to(torch.float64)
    variance = state_dict[f"{bn_prefix}.running_var"].detach().to(torch.float64)
    multiplier = gamma / torch.sqrt(variance + float(eps))
    folded_weight = weight * multiplier.reshape(-1, 1, 1, 1)
    folded_bias = beta - mean * multiplier
    return folded_weight, folded_bias


def _choose_weight_frac_bits(
    weight: torch.Tensor,
    bias: torch.Tensor,
    spec: dict,
) -> int:
    weight_bits = int(spec["weight_bits"])
    input_bits = int(spec["input_bits"])
    input_frac = int(spec["input_frac_bits"])
    accumulator_bits = int(spec["accumulator_bits"])
    _qmin, qmax = _fixed_integer_bounds(weight_bits)
    max_abs_weight = float(weight.abs().max().item())
    if not math.isfinite(max_abs_weight) or max_abs_weight <= 0.0:
        precision_frac = weight_bits - 2
    else:
        precision_frac = math.floor(math.log2(float(qmax) / max_abs_weight))

    # Constrain the binary point so the accumulator can cover a conservative
    # full-range input bound after BN folding.  This avoids silently choosing
    # excellent weight resolution with no remaining accumulator integer bits.
    _input_min, input_qmax = _fixed_integer_bounds(input_bits)
    input_real_max = float(input_qmax) / math.ldexp(1.0, input_frac)
    reduce_dims = tuple(range(1, weight.ndim))
    l1_per_output = weight.abs().sum(dim=reduce_dims)
    worst_real = float((l1_per_output * input_real_max + bias.abs()).max().item())
    worst_real = max(worst_real * 1.05, 1.0)
    required_acc_integer_bits = max(2, math.ceil(math.log2(worst_real)) + 1)
    max_frac_from_accumulator = accumulator_bits - required_acc_integer_bits - input_frac
    frac_bits = min(precision_frac, max_frac_from_accumulator)
    if frac_bits < 0:
        raise ValueError(
            f"Accumulator width {accumulator_bits} cannot cover layer range with "
            f"input Q{input_bits - input_frac}.{input_frac}"
        )
    return int(frac_bits)


def _quantize_tensor_to_signed_integer(
    value: torch.Tensor,
    bits: int,
    frac_bits: int,
) -> torch.Tensor:
    qmin, qmax = _fixed_integer_bounds(bits)
    scale = math.ldexp(1.0, int(frac_bits))
    quantized = torch.round(value.to(torch.float64) * scale).clamp(qmin, qmax)
    storage_dtype = torch.int16 if int(bits) <= 16 else torch.int32
    return quantized.to(storage_dtype).contiguous()


def build_fixed_capnet_lite_artifact(
    checkpoint_path: str | Path,
    output_path: str | Path,
    scheme: str,
    split_mode: str,
    batch_norm_eps: float = 1e-5,
) -> dict:
    """Fold BN and save integer-only CAPNetLite weights for scheme A-D."""
    scheme_key = str(scheme).strip().upper()
    if scheme_key not in FIXED_CAP_SCHEMES:
        raise ValueError("scheme must be A, B, C, or D")
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("CAPNetLite checkpoint does not contain a state dict")

    layers = {}
    for layer_name, weight_key, bn_prefix in _FIXED_CAP_LAYER_DEFS:
        spec = _fixed_layer_spec(scheme_key, layer_name)
        if bn_prefix is None:
            weight = state_dict[weight_key].detach().to(torch.float64)
            bias = state_dict["fc.bias"].detach().to(torch.float64)
        else:
            weight, bias = _fold_conv_batch_norm(
                state_dict, weight_key, bn_prefix, float(batch_norm_eps)
            )
        weight_frac_bits = _choose_weight_frac_bits(weight, bias, spec)
        accumulator_frac_bits = int(spec["input_frac_bits"]) + weight_frac_bits
        if bool(spec.pop("output_matches_accumulator", False)):
            spec["output_bits"] = int(spec["accumulator_bits"])
            spec["output_frac_bits"] = int(accumulator_frac_bits)
        weight_q = _quantize_tensor_to_signed_integer(
            weight, int(spec["weight_bits"]), weight_frac_bits
        )
        acc_qmin, acc_qmax = _fixed_integer_bounds(int(spec["accumulator_bits"]))
        bias_scale = math.ldexp(1.0, accumulator_frac_bits)
        bias_q = torch.round(bias * bias_scale).clamp(acc_qmin, acc_qmax).to(torch.int64)
        layers[layer_name] = {
            **spec,
            "weight_frac_bits": int(weight_frac_bits),
            "accumulator_frac_bits": int(accumulator_frac_bits),
            "weight_q": weight_q,
            "bias_q": bias_q.contiguous(),
            "weight_shape": tuple(int(value) for value in weight_q.shape),
        }

    artifact = {
        "format_version": 2,
        "artifact_type": "capnet_lite_hls_fixed_point",
        "hardware_profile": "Xilinx ZCU104 / XCZU7EV / DSP48E2 27x18 + 48-bit accumulator",
        "scheme": scheme_key,
        "scheme_description": FIXED_CAP_SCHEMES[scheme_key]["description"],
        "split_mode": str(split_mode),
        "source_checkpoint": str(checkpoint_path),
        "rounding_mode": "round_to_nearest_even",
        "overflow_mode": "saturate",
        "scale_mode": "power_of_two_per_layer",
        "batch_norm_folded": True,
        "batch_norm_eps": float(batch_norm_eps),
        "residual_add_format": FIXED_CAP_SCHEMES[scheme_key].get("residual_add_format"),
        "layers": layers,
        "fine_class_names": list(FINE_CLASS_NAMES),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return artifact


def _fixed_fake_quantize(value: torch.Tensor, bits: int, frac_bits: int) -> torch.Tensor:
    qmin, qmax = _fixed_integer_bounds(int(bits))
    scale = math.ldexp(1.0, int(frac_bits))
    return torch.round(value * scale).clamp(float(qmin), float(qmax)) / scale


class _FixedAffineEmulator(nn.Module):
    def __init__(self, layer_artifact: dict):
        super().__init__()
        self.spec = {
            key: int(layer_artifact[key])
            for key in (
                "weight_bits", "weight_frac_bits", "input_bits",
                "input_frac_bits", "output_bits", "output_frac_bits",
                "product_bits", "accumulator_bits", "accumulator_frac_bits",
            )
        }
        weight_q = layer_artifact["weight_q"].detach().clone()
        bias_q = layer_artifact["bias_q"].detach().clone()
        self.register_buffer("weight_q", weight_q)
        self.register_buffer("bias_q", bias_q)
        self.register_buffer(
            "weight_real_cache",
            weight_q.to(torch.float32)
            / math.ldexp(1.0, self.spec["weight_frac_bits"]),
        )
        self.register_buffer(
            "bias_real_cache",
            bias_q.to(torch.float32)
            / math.ldexp(1.0, self.spec["accumulator_frac_bits"]),
        )

    def quantize_input(self, value: torch.Tensor) -> torch.Tensor:
        return _fixed_fake_quantize(
            value, self.spec["input_bits"], self.spec["input_frac_bits"]
        )

    def quantize_output(self, value: torch.Tensor) -> torch.Tensor:
        return _fixed_fake_quantize(
            value, self.spec["output_bits"], self.spec["output_frac_bits"]
        )

    def _weight_real(self, dtype: torch.dtype) -> torch.Tensor:
        return self.weight_real_cache.to(dtype=dtype)

    def _bias_real(self, dtype: torch.dtype) -> torch.Tensor:
        return self.bias_real_cache.to(dtype=dtype)

    def _quantize_accumulator(self, value: torch.Tensor) -> torch.Tensor:
        return _fixed_fake_quantize(
            value,
            self.spec["accumulator_bits"],
            self.spec["accumulator_frac_bits"],
        )

    def conv2d(
        self,
        value: torch.Tensor,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> torch.Tensor:
        value = self.quantize_input(value)
        output = F.conv2d(
            value,
            self._weight_real(value.dtype),
            bias=None,
            stride=stride,
            padding=padding,
        )
        output = output + self._bias_real(value.dtype).reshape(1, -1, 1, 1)
        return self._quantize_accumulator(output)

    def linear(self, value: torch.Tensor) -> torch.Tensor:
        value = self.quantize_input(value)
        output = F.linear(value, self._weight_real(value.dtype), bias=None)
        output = output + self._bias_real(value.dtype).reshape(1, -1)
        return self._quantize_accumulator(output)


class FixedPointCAPNetLiteEmulator(nn.Module):
    """GPU accuracy emulator backed only by integer A-D weight artifacts."""

    def __init__(self, artifact: dict):
        super().__init__()
        if artifact.get("artifact_type") != "capnet_lite_hls_fixed_point":
            raise ValueError("Not a CAPNetLite fixed-point artifact")
        if int(artifact.get("format_version", -1)) != 2:
            raise ValueError("Unsupported fixed-point artifact version")
        self.scheme = str(artifact["scheme"])
        self.split_mode = str(artifact["split_mode"])
        self.rounding_mode = str(artifact["rounding_mode"])
        self.overflow_mode = str(artifact["overflow_mode"])
        self.residual_add_format = artifact.get("residual_add_format")
        layers = artifact["layers"]
        self.stem = _FixedAffineEmulator(layers["stem"])
        self.block1_conv1 = _FixedAffineEmulator(layers["block1_conv1"])
        self.block1_conv2 = _FixedAffineEmulator(layers["block1_conv2"])
        self.down = _FixedAffineEmulator(layers["down"])
        self.block2_conv1 = _FixedAffineEmulator(layers["block2_conv1"])
        self.block2_conv2 = _FixedAffineEmulator(layers["block2_conv2"])
        self.fc = _FixedAffineEmulator(layers["fc"])

    def _residual_block(
        self,
        value: torch.Tensor,
        conv1: _FixedAffineEmulator,
        conv2: _FixedAffineEmulator,
    ) -> torch.Tensor:
        identity = value
        output = conv1.conv2d(value, stride=1, padding=1)
        output = conv1.quantize_output(F.relu(output))
        output = conv2.conv2d(output, stride=1, padding=1)
        output = output + identity
        if self.residual_add_format is not None:
            output = _fixed_fake_quantize(
                output,
                int(self.residual_add_format["bits"]),
                int(self.residual_add_format["frac_bits"]),
            )
        return conv2.quantize_output(F.relu(output))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[-2:]) != (11, 15):
            raise ValueError(f"Expected CAP input (batch,1,11,15), got {tuple(value.shape)}")
        value = value.to(torch.float32)
        value = self.stem.conv2d(value, stride=1, padding=1)
        value = self.stem.quantize_output(F.relu(value))
        value = self._residual_block(value, self.block1_conv1, self.block1_conv2)
        value = self.down.conv2d(value, stride=(1, 2), padding=1)
        value = self.down.quantize_output(F.relu(value))
        value = self._residual_block(value, self.block2_conv1, self.block2_conv2)
        # The FC is only 64x4.  Use float64 as the emulation container so its
        # 24x16 products and 48-bit logits are not rounded by float32 before
        # the explicit fixed-point quantization points are applied.
        value = F.adaptive_avg_pool2d(value, (1, 1)).flatten(1).to(torch.float64)
        value = self.fc.quantize_input(value)
        value = self.fc.linear(value)
        return self.fc.quantize_output(value)


def load_fixed_capnet_lite_for_pipeline(
    pipeline,
    artifact_path: str | Path,
    expected_scheme: str | None = None,
    expected_split_mode: str | None = None,
) -> dict:
    """Replace the pipeline's FP32 CAP model with an integer-artifact emulator."""
    artifact_path = Path(artifact_path).resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    scheme = str(artifact.get("scheme", "")).upper()
    split_mode = str(artifact.get("split_mode", ""))
    if expected_scheme is not None and scheme != str(expected_scheme).upper():
        raise ValueError(f"Expected scheme {expected_scheme}, artifact contains {scheme}")
    if expected_split_mode is not None and split_mode != str(expected_split_mode):
        raise ValueError(
            f"Expected split {expected_split_mode}, artifact contains {split_mode}"
        )
    artifact_classes = list(artifact.get("fine_class_names", []))
    if artifact_classes != list(pipeline.fine_class_names):
        raise ValueError(
            f"Artifact class order {artifact_classes} does not match "
            f"pipeline class order {pipeline.fine_class_names}"
        )
    model = FixedPointCAPNetLiteEmulator(artifact).to(pipeline.device).eval()
    pipeline.cap_model = model
    pipeline.fixed_cap_artifact_path = artifact_path
    pipeline.fixed_cap_scheme = scheme
    return artifact


@dataclass(frozen=True)
class MixedPrecisionResult:
    pred_label: str
    coarse_pred: str
    route: str
    rs_est: float = float("nan")
    cfo_est: float = float("nan")


def process_signal_batch_mixed_precision(
    pipeline,
    x_batch,
    cc_compute_dtype: str | torch.dtype,
    nn_dtype: str | torch.dtype,
    snr_db,
    cc_phase_dtype: str | torch.dtype = "float64",
) -> list[MixedPrecisionResult]:
    """Reuse the unchanged GPU BOI/FFT/LR9 front end, then run test CC/CAP."""
    cc_dtype = (
        resolve_real_dtype(cc_compute_dtype)
        if isinstance(cc_compute_dtype, str)
        else cc_compute_dtype
    )
    cap_dtype = resolve_real_dtype(nn_dtype) if isinstance(nn_dtype, str) else nn_dtype
    phase_dtype = (
        resolve_phase_dtype(cc_phase_dtype)
        if isinstance(cc_phase_dtype, str)
        else cc_phase_dtype
    )
    if torch.is_tensor(x_batch):
        x_batch = x_batch.to(device=pipeline.device, dtype=torch.complex128)
    else:
        x_batch = torch.as_tensor(x_batch, device=pipeline.device, dtype=torch.complex128)
    if x_batch.ndim != 2:
        raise ValueError(f"x_batch must have shape (batch,samples), got {tuple(x_batch.shape)}")
    _validate_nonnegative_snr_batch(snr_db, int(x_batch.shape[0]))

    x_processed_rows = []
    rs20_rows = []
    cfo1_rows = []
    power_rows = []
    lr_rows = []
    for row_idx in range(int(x_batch.shape[0])):
        x_processed, lr9, rs20, cfo1, prepared = pipeline._process_loaded_to_vectors(
            x_batch[row_idx]
        )
        x_processed_rows.append(x_processed)
        rs20_rows.append(rs20)
        cfo1_rows.append(cfo1)
        power_rows.append(_cc_power(prepared, x_processed, float(pipeline.cfg["EPS"])))
        lr_rows.append(lr9)

    x_processed_batch = torch.stack(x_processed_rows)
    rs20_batch = torch.stack(rs20_rows)
    cfo1_batch = torch.stack(cfo1_rows)
    power_batch = torch.stack(power_rows)
    coarse_indices = pipeline.lr_model.predict_indices(torch.stack(lr_rows))
    qpsk_like_index = pipeline.lr_model.classes_.index("QPSK-like")
    qpsk_rows = torch.nonzero(
        coarse_indices == int(qpsk_like_index), as_tuple=False
    ).reshape(-1)

    cap_inputs = []
    rs_est_all = torch.full(
        (int(x_batch.shape[0]),), torch.nan, device=pipeline.device, dtype=torch.float64
    )
    cfo_est_all = torch.full_like(rs_est_all, torch.nan)
    fine_index_all = torch.full(
        (int(x_batch.shape[0]),), -1, device=pipeline.device, dtype=torch.long
    )
    for row_idx_t in qpsk_rows:
        rs_est, t0_est, cfo_est = _estimate_rs_cfo_from_torch_vectors(
            rs20_batch[row_idx_t], cfo1_batch[row_idx_t]
        )
        cap_input = build_qpsk_cap_input_mixed_precision(
            x_processed_batch[row_idx_t].to(torch.complex64),
            t0_est,
            cfo_est,
            power_batch[row_idx_t],
            compute_dtype=cc_dtype,
            phase_dtype=phase_dtype,
            eps=float(pipeline.cfg["EPS"]),
        )
        cap_inputs.append(cap_input)
        rs_est_all[row_idx_t] = rs_est
        cfo_est_all[row_idx_t] = cfo_est

    if cap_inputs:
        cap_batch = torch.stack(cap_inputs)[:, None, :, :].to(cap_dtype)
        with torch.inference_mode():
            fine_indices = pipeline.cap_model(cap_batch).argmax(dim=1)
        fine_index_all[qpsk_rows] = fine_indices

    coarse_cpu = coarse_indices.detach().cpu().tolist()
    fine_cpu = fine_index_all.detach().cpu().tolist()
    rs_cpu = rs_est_all.detach().cpu().tolist()
    cfo_cpu = cfo_est_all.detach().cpu().tolist()
    results = []
    for row_idx, class_index in enumerate(coarse_cpu):
        coarse_pred = str(pipeline.lr_model.classes_[int(class_index)])
        if coarse_pred not in COARSE_CLASS_NAMES:
            raise ValueError(f"Unexpected LR prediction: {coarse_pred!r}")
        if coarse_pred != "QPSK-like":
            results.append(MixedPrecisionResult(coarse_pred, coarse_pred, "LR-final"))
            continue
        results.append(
            MixedPrecisionResult(
                pipeline.fine_class_names[int(fine_cpu[row_idx])],
                coarse_pred,
                f"QPSK-like->CC-{str(cc_dtype).split('.')[-1]}->CAP-{str(cap_dtype).split('.')[-1]}",
                float(rs_cpu[row_idx]),
                float(cfo_cpu[row_idx]),
            )
        )
    return results


def process_signal_batch_cpu_frontend_mixed_precision(
    pipeline,
    x_batch,
    cc_compute_dtype: str | torch.dtype,
    nn_dtype: str | torch.dtype,
    snr_db,
    cc_phase_dtype: str | torch.dtype = "float64",
) -> list[MixedPrecisionResult]:
    """Keep the production NumPy/sklearn BOI+FFT+LR9 front end unchanged.

    Only QPSK-like rows cross the CPU/GPU boundary: their processed complex64
    signal is sent to ``pipeline.device`` for the experimental CC builder and
    CAP model.  This is the route used by the generated precision notebooks.
    """
    cc_dtype = (
        resolve_real_dtype(cc_compute_dtype)
        if isinstance(cc_compute_dtype, str)
        else cc_compute_dtype
    )
    cap_dtype = resolve_real_dtype(nn_dtype) if isinstance(nn_dtype, str) else nn_dtype
    phase_dtype = (
        resolve_phase_dtype(cc_phase_dtype)
        if isinstance(cc_phase_dtype, str)
        else cc_phase_dtype
    )
    x_batch = np.asarray(x_batch)
    if x_batch.ndim != 2:
        raise ValueError(f"x_batch must have shape (batch,samples), got {x_batch.shape}")
    _validate_nonnegative_snr_batch(snr_db, int(x_batch.shape[0]))

    staged = []
    lr_rows = []
    for x in x_batch:
        x_processed, lr9, rs20, cfo1, prepared = pipeline._process_loaded_to_vectors(x)
        staged.append(
            {
                "x_processed": x_processed,
                "rs20": rs20,
                "cfo1": cfo1,
                "prepared": prepared,
            }
        )
        lr_rows.append(lr9)

    lr_matrix = np.stack(lr_rows).astype(np.float32, copy=False)
    coarse_preds = [str(value) for value in pipeline.lr_model.predict(lr_matrix)]
    results: list[MixedPrecisionResult | None] = [None] * len(coarse_preds)
    cap_inputs = []
    cap_meta = []

    for row_idx, coarse_pred in enumerate(coarse_preds):
        if coarse_pred not in COARSE_CLASS_NAMES:
            raise ValueError(f"Unexpected LR prediction: {coarse_pred!r}")
        if coarse_pred != "QPSK-like":
            results[row_idx] = MixedPrecisionResult(
                coarse_pred, coarse_pred, "LR-final"
            )
            continue

        item = staged[row_idx]
        x_processed_cc = np.asarray(item["x_processed"], dtype=np.complex64)
        sample = {
            "x_processed": x_processed_cc,
            "vectors": {"rs20": item["rs20"], "cfo1": item["cfo1"]},
        }
        estimate = rc.estimate_rs_cfo_from_sample(
            sample,
            compact.RS20_FEATURES,
            compact.CFO_FEATURES,
            set(),
            rs_vector_key="rs20",
        )
        rs_est = float(estimate["rs_est"])
        t0_est = float(estimate["T0_est"])
        cfo_est = float(estimate["cfo_est"])
        power = _cc_power_cpu(
            item["prepared"], item["x_processed"], float(pipeline.cfg["EPS"])
        )
        x_device = torch.from_numpy(x_processed_cc).to(pipeline.device)
        cap_input = build_qpsk_cap_input_mixed_precision(
            x_device,
            t0_est,
            cfo_est,
            power,
            compute_dtype=cc_dtype,
            phase_dtype=phase_dtype,
            eps=float(pipeline.cfg["EPS"]),
        )
        cap_inputs.append(cap_input)
        cap_meta.append((row_idx, coarse_pred, rs_est, cfo_est))

    if cap_inputs:
        cap_batch = torch.stack(cap_inputs)[:, None, :, :].to(cap_dtype)
        with torch.inference_mode():
            fine_indices = pipeline.cap_model(cap_batch).argmax(dim=1).detach().cpu().tolist()
        for fine_index, (row_idx, coarse_pred, rs_est, cfo_est) in zip(
            fine_indices, cap_meta
        ):
            results[row_idx] = MixedPrecisionResult(
                pipeline.fine_class_names[int(fine_index)],
                coarse_pred,
                f"QPSK-like->CC-{str(cc_dtype).split('.')[-1]}->CAP-{str(cap_dtype).split('.')[-1]}",
                rs_est,
                cfo_est,
            )

    missing = [row_idx for row_idx, result in enumerate(results) if result is None]
    if missing:
        raise RuntimeError(f"Missing mixed-precision results at rows {missing[:10]}")
    return [result for result in results if result is not None]


__all__ = [
    "FIXED_CAP_SCHEMES",
    "FixedPointCAPNetLiteEmulator",
    "MixedPrecisionResult",
    "build_fixed_capnet_lite_artifact",
    "build_qpsk_cap_input_mixed_precision",
    "configure_cap_model_test_dtype",
    "load_fixed_capnet_lite_for_pipeline",
    "process_signal_batch_cpu_frontend_mixed_precision",
    "process_signal_batch_mixed_precision",
    "resolve_phase_dtype",
    "resolve_real_dtype",
    "select_nonnegative_snr_reference_rows",
]
