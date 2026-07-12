"""Four-way CPU/GPU profiler for fixed-QPSK 11x15 CC inference.

The four schemes share the same BOI, FFT/LR, Rs/CFO, cumulant, postprocessing,
and CAP stages.  They differ only by device and by how cyclic moments are
constructed:

* original: project every required k independently and recompute the signal
  power product for every projection;
* simplified: reuse signal powers and recover dependent moments with conjugate
  symmetry.

Both CC implementations are single-signal (not batched), use complex64, write
the seven fixed QPSK cumulant columns directly into an 11x15 matrix, and return
the identical warped/normalised float32 CAP input.
"""

from __future__ import annotations

import gc
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from . import pre_compact_fft3_vector_generator as compact
from . import pre_qpsklike_rs_cfo_estimator as rc
from .fully_online_pipeline_inference import (
    _boi_preprocess as _boi_preprocess_cpu,
    _cc_power as _cc_power_cpu,
    _truth_column,
    load_truth_table,
)
from .fully_online_pipeline_inference_gpu import (
    _cc_power as _cc_power_gpu,
    _estimate_rs_cfo_from_torch_vectors,
    boi_preprocess_torch,
)


K_LIST = tuple(range(-5, 6))
K_TO_ROW = {k: k + 5 for k in K_LIST}
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
STAGE_COLUMNS = (
    "load_s",
    "boi_shift_s",
    "fft3_lr_feature_s",
    "lr_s",
    "rs_cfo_s",
    "cc_s",
    "cap_s",
)
RESULT_COLUMNS = (
    "device",
    "n_samples",
    *STAGE_COLUMNS,
    "total_s",
    "per_sample_s",
)


def eligible_qpsklike_indices(
    dataset_root,
    cfg: Mapping,
    year: int,
    batch_ids: Sequence[int],
    limit: int,
    drop_2018_negative_snr: bool = True,
) -> list[int]:
    """Return true QPSK/QAM indices used only to select the timing workload."""
    truth = load_truth_table(dataset_root, int(year))
    label_col = _truth_column(
        truth,
        ["SignalType", "signal_type", "signal type", "modulation", "modulation_type", "label"],
    )
    snr_col = _truth_column(
        truth,
        ["InbandSNRdB", "inband_snr_db_label", "inband snr db label", "snr", "snr_db"],
    )
    qpsklike = {"QPSK", "16QAM", "64QAM", "256QAM"}
    indices: list[int] = []
    for batch_id in batch_ids:
        start = (int(batch_id) - 1) * int(cfg["N_PER_BATCH"]) + 1
        stop = min(int(batch_id) * int(cfg["N_PER_BATCH"]), int(cfg["N_TOTAL"]))
        for signal_index in range(start, stop + 1):
            if str(truth.loc[signal_index, label_col]).strip().upper() not in qpsklike:
                continue
            if int(year) == 2018 and drop_2018_negative_snr:
                if float(truth.loc[signal_index, snr_col]) < 0.0:
                    continue
            indices.append(int(signal_index))
            if len(indices) == int(limit):
                return indices
    raise ValueError(f"Only found {len(indices)} eligible QPSK-like signals; need {limit}.")


def _sync_cuda(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def round_timing_seconds(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = frame.copy()
    numeric = out.select_dtypes(include=[np.number]).columns
    out[numeric] = out[numeric].round(int(digits))
    return out


# ---------------------------------------------------------------------------
# CPU moments: original direct projection and symmetry-based simplification
# ---------------------------------------------------------------------------

def _project_one_cpu_recompute(
    x: np.ndarray,
    t0: np.float32,
    f0: np.float32,
    n: int,
    m: int,
    k: int,
) -> np.complex64:
    """Project one moment while deliberately recomputing x powers."""
    x = np.asarray(x, dtype=np.complex64)
    u = (x ** int(n - m)) * (np.conj(x) ** int(m))
    alpha = np.float32((n - 2 * m) * f0 + np.float32(k) / t0)
    t = np.arange(x.size, dtype=np.float32)
    phase = np.asarray(-np.float32(2.0 * np.pi) * alpha * t, dtype=np.float32)
    expo = np.exp(np.asarray(1j * phase, dtype=np.complex64))
    return np.asarray(np.sum(expo * u, dtype=np.complex64) / np.float32(x.size), dtype=np.complex64)[()]


def _project_many_cpu(
    u: np.ndarray,
    t0: np.float32,
    f0: np.float32,
    n: int,
    m: int,
    k_values: Sequence[int],
) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex64)
    k = np.asarray(k_values, dtype=np.float32)
    alpha = np.asarray((n - 2 * m) * f0 + k / t0, dtype=np.float32)
    t = np.arange(u.size, dtype=np.float32)
    phase = np.asarray(-np.float32(2.0 * np.pi) * alpha[:, None] * t[None, :], dtype=np.float32)
    expo = np.exp(np.asarray(1j * phase, dtype=np.complex64))
    return np.asarray((expo @ u) / np.float32(u.size), dtype=np.complex64)


def _empty_cpu_sequence() -> np.ndarray:
    return np.zeros(len(K_LIST), dtype=np.complex64)


def _assign_cpu(seq: np.ndarray, k_values: Sequence[int], values) -> None:
    for k, value in zip(k_values, np.atleast_1d(values)):
        seq[K_TO_ROW[int(k)]] = np.complex64(value)


def build_qpsk_moments_original_cpu(x, t0, f0) -> dict[tuple[int, int], np.ndarray]:
    """Compute all 33 required moments independently on the CPU."""
    x = np.asarray(x, dtype=np.complex64)
    t0 = np.float32(t0)
    f0 = np.float32(f0)
    moments = {}
    for nm, k_values in ACTIVE_K.items():
        seq = _empty_cpu_sequence()
        for k in k_values:
            seq[K_TO_ROW[k]] = _project_one_cpu_recompute(x, t0, f0, nm[0], nm[1], k)
        moments[nm] = seq
    return moments


def build_qpsk_moments_simplified_cpu(x, t0, f0) -> dict[tuple[int, int], np.ndarray]:
    """Compute 18 independent moments and recover the rest by symmetry."""
    x = np.asarray(x, dtype=np.complex64)
    t0 = np.float32(t0)
    f0 = np.float32(f0)
    x_conj = np.conj(x)
    abs2 = np.asarray((x * x_conj).real, dtype=np.float32)
    x2 = x * x
    x4 = x2 * x2
    x5 = x4 * x
    u21 = np.asarray(abs2, dtype=np.complex64)
    u40 = np.asarray(x4, dtype=np.complex64)
    u42 = np.asarray(abs2 * abs2, dtype=np.complex64)
    u61 = np.asarray(x5 * x_conj, dtype=np.complex64)
    u63 = np.asarray(abs2 * abs2 * abs2, dtype=np.complex64)

    m21 = _empty_cpu_sequence()
    _assign_cpu(m21, (0, 1), _project_many_cpu(u21, t0, f0, 2, 1, (0, 1)))
    m21[K_TO_ROW[-1]] = np.conj(m21[K_TO_ROW[1]])

    m40 = _empty_cpu_sequence()
    _assign_cpu(m40, (-2, -1, 0, 1, 2), _project_many_cpu(u40, t0, f0, 4, 0, (-2, -1, 0, 1, 2)))

    m42 = _empty_cpu_sequence()
    _assign_cpu(m42, (0, 1, 2), _project_many_cpu(u42, t0, f0, 4, 2, (0, 1, 2)))
    m42[K_TO_ROW[-1]] = np.conj(m42[K_TO_ROW[1]])
    m42[K_TO_ROW[-2]] = np.conj(m42[K_TO_ROW[2]])

    m44 = _empty_cpu_sequence()
    for k in ACTIVE_K[(4, 4)]:
        m44[K_TO_ROW[k]] = np.conj(m40[K_TO_ROW[-k]])

    m61 = _empty_cpu_sequence()
    _assign_cpu(m61, (-2, -1, 0, 1, 2), _project_many_cpu(u61, t0, f0, 6, 1, (-2, -1, 0, 1, 2)))

    m63 = _empty_cpu_sequence()
    _assign_cpu(m63, (0, 1, 2), _project_many_cpu(u63, t0, f0, 6, 3, (0, 1, 2)))
    m63[K_TO_ROW[-1]] = np.conj(m63[K_TO_ROW[1]])
    m63[K_TO_ROW[-2]] = np.conj(m63[K_TO_ROW[2]])

    m65 = _empty_cpu_sequence()
    for k in ACTIVE_K[(6, 5)]:
        m65[K_TO_ROW[k]] = np.conj(m61[K_TO_ROW[-k]])

    return {
        (2, 1): m21, (4, 0): m40, (4, 2): m42, (4, 4): m44,
        (6, 1): m61, (6, 3): m63, (6, 5): m65,
    }


def _conv2_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    full = np.convolve(a, b).astype(np.complex64, copy=False)
    return full[5:16]


def _conv3_cpu(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    full = np.convolve(np.convolve(a, b), c).astype(np.complex64, copy=False)
    return full[10:21]


def _mask_cpu(seq: np.ndarray, nm: tuple[int, int]) -> np.ndarray:
    out = _empty_cpu_sequence()
    for k in ACTIVE_K[nm]:
        out[K_TO_ROW[k]] = seq[K_TO_ROW[k]]
    return out


def build_qpsk_cc_complex_cpu(moments: Mapping) -> np.ndarray:
    m21, m40 = moments[(2, 1)], moments[(4, 0)]
    m42, m44 = moments[(4, 2)], moments[(4, 4)]
    m61, m63, m65 = moments[(6, 1)], moments[(6, 3)], moments[(6, 5)]
    cumulants = {
        (2, 1): m21,
        (4, 0): m40,
        (4, 2): m42 - np.complex64(2.0) * _conv2_cpu(m21, m21),
        (4, 4): m44,
        (6, 1): m61 - np.complex64(5.0) * _conv2_cpu(m40, m21),
        (6, 3): m63 - np.complex64(9.0) * _conv2_cpu(m42, m21)
        + np.complex64(12.0) * _conv3_cpu(m21, m21, m21),
        (6, 5): m65 - np.complex64(5.0) * _conv2_cpu(m44, m21),
    }
    matrix = np.zeros((11, 15), dtype=np.complex64)
    for nm, seq in cumulants.items():
        matrix[:, NM_TO_COL[nm]] = _mask_cpu(seq, nm)
    return matrix


def postprocess_qpsk_cc_cpu(matrix, power, eps: float = 1e-12) -> np.ndarray:
    power = np.float32(power)
    if not np.isfinite(power) or power <= np.float32(eps):
        raise ValueError(f"Invalid CC normalization power: {power}")
    out = np.abs(np.asarray(matrix, dtype=np.complex64)).astype(np.float32)
    for col, (n, _m) in enumerate(FIRST15_NM):
        out[:, col] = out[:, col] ** np.float32(2.0 / n)
    return np.asarray(out / power, dtype=np.float32)


def build_qpsk_cap_input_original_cpu(x, t0, f0, power, eps: float = 1e-12) -> np.ndarray:
    moments = build_qpsk_moments_original_cpu(x, t0, f0)
    return postprocess_qpsk_cc_cpu(build_qpsk_cc_complex_cpu(moments), power, eps)


def build_qpsk_cap_input_simplified_cpu(x, t0, f0, power, eps: float = 1e-12) -> np.ndarray:
    moments = build_qpsk_moments_simplified_cpu(x, t0, f0)
    return postprocess_qpsk_cc_cpu(build_qpsk_cc_complex_cpu(moments), power, eps)


# ---------------------------------------------------------------------------
# GPU moments: the same two algorithms, with every intermediate on CUDA
# ---------------------------------------------------------------------------

def _project_one_gpu_recompute(x, t0, f0, n: int, m: int, k: int) -> torch.Tensor:
    x = x.to(dtype=torch.complex64)
    u = (x ** int(n - m)) * (torch.conj(x) ** int(m))
    t0 = torch.as_tensor(t0, device=x.device, dtype=torch.float32)
    f0 = torch.as_tensor(f0, device=x.device, dtype=torch.float32)
    alpha = float(n - 2 * m) * f0 + float(k) / t0
    t = torch.arange(int(x.numel()), device=x.device, dtype=torch.float32)
    expo = torch.exp((1j * (-2.0 * torch.pi * alpha * t)).to(torch.complex64))
    return torch.sum(expo * u, dtype=torch.complex64) / float(x.numel())


def _project_many_gpu(u, t0, f0, n: int, m: int, k_values: Sequence[int]) -> torch.Tensor:
    u = u.to(dtype=torch.complex64)
    t0 = torch.as_tensor(t0, device=u.device, dtype=torch.float32)
    f0 = torch.as_tensor(f0, device=u.device, dtype=torch.float32)
    k = torch.tensor(k_values, device=u.device, dtype=torch.float32)
    alpha = float(n - 2 * m) * f0 + k / t0
    t = torch.arange(int(u.numel()), device=u.device, dtype=torch.float32)
    phase = -2.0 * torch.pi * alpha[:, None] * t[None, :]
    expo = torch.exp((1j * phase).to(torch.complex64))
    return torch.sum(expo * u[None, :], dim=1, dtype=torch.complex64) / float(u.numel())


def _empty_gpu_sequence(device) -> torch.Tensor:
    return torch.zeros(len(K_LIST), device=device, dtype=torch.complex64)


def build_qpsk_moments_original_gpu(x, t0, f0) -> dict[tuple[int, int], torch.Tensor]:
    """Compute all 33 required moments independently on the selected GPU."""
    if x.device.type != "cuda":
        raise ValueError("GPU CC input must already be a CUDA tensor")
    x = x.to(dtype=torch.complex64)
    moments = {}
    for nm, k_values in ACTIVE_K.items():
        seq = _empty_gpu_sequence(x.device)
        for k in k_values:
            seq[K_TO_ROW[k]] = _project_one_gpu_recompute(x, t0, f0, nm[0], nm[1], k)
        moments[nm] = seq
    return moments


def build_qpsk_moments_simplified_gpu(x, t0, f0) -> dict[tuple[int, int], torch.Tensor]:
    """Compute 18 independent moments and recover the rest on the GPU."""
    if x.device.type != "cuda":
        raise ValueError("GPU CC input must already be a CUDA tensor")
    x = x.to(dtype=torch.complex64)
    x_conj = torch.conj(x)
    abs2 = (x * x_conj).real
    x2 = x * x
    x4 = x2 * x2
    x5 = x4 * x
    u21 = abs2.to(torch.complex64)
    u40 = x4
    u42 = (abs2 * abs2).to(torch.complex64)
    u61 = x5 * x_conj
    u63 = (abs2 * abs2 * abs2).to(torch.complex64)

    m21 = _empty_gpu_sequence(x.device)
    vals = _project_many_gpu(u21, t0, f0, 2, 1, (0, 1))
    m21[K_TO_ROW[0]], m21[K_TO_ROW[1]] = vals[0], vals[1]
    m21[K_TO_ROW[-1]] = torch.conj(m21[K_TO_ROW[1]])

    m40 = _empty_gpu_sequence(x.device)
    m40[3:8] = _project_many_gpu(u40, t0, f0, 4, 0, (-2, -1, 0, 1, 2))

    m42 = _empty_gpu_sequence(x.device)
    vals = _project_many_gpu(u42, t0, f0, 4, 2, (0, 1, 2))
    m42[5:8] = vals
    m42[K_TO_ROW[-1]] = torch.conj(m42[K_TO_ROW[1]])
    m42[K_TO_ROW[-2]] = torch.conj(m42[K_TO_ROW[2]])

    m44 = _empty_gpu_sequence(x.device)
    for k in ACTIVE_K[(4, 4)]:
        m44[K_TO_ROW[k]] = torch.conj(m40[K_TO_ROW[-k]])

    m61 = _empty_gpu_sequence(x.device)
    m61[3:8] = _project_many_gpu(u61, t0, f0, 6, 1, (-2, -1, 0, 1, 2))

    m63 = _empty_gpu_sequence(x.device)
    vals = _project_many_gpu(u63, t0, f0, 6, 3, (0, 1, 2))
    m63[5:8] = vals
    m63[K_TO_ROW[-1]] = torch.conj(m63[K_TO_ROW[1]])
    m63[K_TO_ROW[-2]] = torch.conj(m63[K_TO_ROW[2]])

    m65 = _empty_gpu_sequence(x.device)
    for k in ACTIVE_K[(6, 5)]:
        m65[K_TO_ROW[k]] = torch.conj(m61[K_TO_ROW[-k]])

    return {
        (2, 1): m21, (4, 0): m40, (4, 2): m42, (4, 4): m44,
        (6, 1): m61, (6, 3): m63, (6, 5): m65,
    }


def _conv_same_gpu(sequences: Sequence[torch.Tensor]) -> torch.Tensor:
    out = torch.ones(1, device=sequences[0].device, dtype=torch.complex64)
    for seq in sequences:
        full = torch.zeros(out.numel() + seq.numel() - 1, device=out.device, dtype=torch.complex64)
        for idx in range(int(out.numel())):
            full[idx:idx + seq.numel()] += out[idx] * seq
        out = full
    half_width = 5 * len(sequences)
    center = half_width
    return out[center - 5:center + 6]


def _mask_gpu(seq: torch.Tensor, nm: tuple[int, int]) -> torch.Tensor:
    out = _empty_gpu_sequence(seq.device)
    for k in ACTIVE_K[nm]:
        out[K_TO_ROW[k]] = seq[K_TO_ROW[k]]
    return out


def build_qpsk_cc_complex_gpu(moments: Mapping) -> torch.Tensor:
    m21, m40 = moments[(2, 1)], moments[(4, 0)]
    m42, m44 = moments[(4, 2)], moments[(4, 4)]
    m61, m63, m65 = moments[(6, 1)], moments[(6, 3)], moments[(6, 5)]
    cumulants = {
        (2, 1): m21,
        (4, 0): m40,
        (4, 2): m42 - 2.0 * _conv_same_gpu((m21, m21)),
        (4, 4): m44,
        (6, 1): m61 - 5.0 * _conv_same_gpu((m40, m21)),
        (6, 3): m63 - 9.0 * _conv_same_gpu((m42, m21))
        + 12.0 * _conv_same_gpu((m21, m21, m21)),
        (6, 5): m65 - 5.0 * _conv_same_gpu((m44, m21)),
    }
    matrix = torch.zeros((11, 15), device=m21.device, dtype=torch.complex64)
    for nm, seq in cumulants.items():
        matrix[:, NM_TO_COL[nm]] = _mask_gpu(seq, nm)
    return matrix


def postprocess_qpsk_cc_gpu(matrix, power, eps: float = 1e-12) -> torch.Tensor:
    if matrix.device.type != "cuda":
        raise ValueError("GPU CC matrix must be a CUDA tensor")
    power = torch.as_tensor(power, device=matrix.device, dtype=torch.float32)
    # Avoid a device-to-host validity check; invalid values propagate to the output.
    safe_power = torch.where(
        torch.isfinite(power) & (power > float(eps)),
        power,
        torch.full_like(power, torch.nan),
    )
    out = torch.abs(matrix).to(torch.float32)
    for col, (n, _m) in enumerate(FIRST15_NM):
        out[:, col] = out[:, col] ** (2.0 / float(n))
    return (out / safe_power).to(torch.float32)


def build_qpsk_cap_input_original_gpu(x, t0, f0, power, eps: float = 1e-12) -> torch.Tensor:
    moments = build_qpsk_moments_original_gpu(x, t0, f0)
    return postprocess_qpsk_cc_gpu(build_qpsk_cc_complex_gpu(moments), power, eps)


def build_qpsk_cap_input_simplified_gpu(x, t0, f0, power, eps: float = 1e-12) -> torch.Tensor:
    moments = build_qpsk_moments_simplified_gpu(x, t0, f0)
    return postprocess_qpsk_cc_gpu(build_qpsk_cc_complex_gpu(moments), power, eps)


# ---------------------------------------------------------------------------
# End-to-end stage profilers
# ---------------------------------------------------------------------------

def profile_cpu_scheme(pipeline, year: int, signal_indices: Sequence[int], cc_mode: str) -> dict:
    if cc_mode not in {"original", "simplified"}:
        raise ValueError("cc_mode must be 'original' or 'simplified'")
    gc.collect()
    timings = {}
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    x_batch = pipeline.load_signal_batch(year, signal_indices)
    timings["load_s"] = time.perf_counter() - t0

    staged = []
    t0 = time.perf_counter()
    for x in x_batch:
        x_processed, prepared, center_shift_hz = _boi_preprocess_cpu(x, pipeline.cfg)
        staged.append({
            "x_processed": x_processed,
            "prepared": prepared,
            "center_shift_hz": center_shift_hz,
        })
    timings["boi_shift_s"] = time.perf_counter() - t0

    lr_rows = []
    t0 = time.perf_counter()
    for item in staged:
        lr55, rs20, cfo1 = pipeline._blind_vectors(item["x_processed"], item["center_shift_hz"])
        item["rs20"], item["cfo1"] = rs20, cfo1
        lr_rows.append(lr55)
    lr_matrix = np.stack(lr_rows).astype(np.float32, copy=False)
    timings["fft3_lr_feature_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pipeline.lr_model.predict(lr_matrix)
    timings["lr_s"] = time.perf_counter() - t0

    estimates = []
    t0 = time.perf_counter()
    for item in staged:
        sample = {
            "x_processed": np.asarray(item["x_processed"], dtype=np.complex64),
            "vectors": {"rs20": item["rs20"], "cfo1": item["cfo1"]},
        }
        estimates.append(rc.estimate_rs_cfo_from_sample(
            sample,
            compact.RS20_FEATURES,
            compact.CFO_FEATURES,
            set(),
            rs_vector_key="rs20",
        ))
    timings["rs_cfo_s"] = time.perf_counter() - t0

    cc_builder = (
        build_qpsk_cap_input_original_cpu
        if cc_mode == "original"
        else build_qpsk_cap_input_simplified_cpu
    )
    cap_inputs = []
    t0 = time.perf_counter()
    for item, estimate in zip(staged, estimates):
        power = _cc_power_cpu(item["prepared"], item["x_processed"], float(pipeline.cfg["EPS"]))
        cap_inputs.append(cc_builder(
            np.asarray(item["x_processed"], dtype=np.complex64),
            np.float32(estimate["T0_est"]),
            np.float32(estimate["cfo_est"]),
            np.float32(power),
            eps=float(pipeline.cfg["EPS"]),
        ))
    timings["cc_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cap_array = np.stack(cap_inputs).astype(np.float32, copy=False)
    cap_tensor = torch.from_numpy(cap_array[:, None, :, :]).to(pipeline.device)
    with torch.inference_mode():
        _ = pipeline.cap_model(cap_tensor).argmax(dim=1)
    timings["cap_s"] = time.perf_counter() - t0

    total_s = time.perf_counter() - total_t0
    return {
        "device": f"cpu_{cc_mode}_cc",
        "n_samples": len(signal_indices),
        **timings,
        "total_s": total_s,
        "per_sample_s": total_s / max(len(signal_indices), 1),
    }


def profile_gpu_scheme(pipeline, year: int, signal_indices: Sequence[int], cc_mode: str) -> dict:
    if cc_mode not in {"original", "simplified"}:
        raise ValueError("cc_mode must be 'original' or 'simplified'")
    device = pipeline.device
    if torch.device(device).type != "cuda":
        raise ValueError("GPU profiling requires a CUDA pipeline")
    gc.collect()
    torch.cuda.empty_cache()
    timings = {}
    _sync_cuda(device)
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    x_batch = pipeline.load_signal_batch(year, signal_indices).to(dtype=torch.complex64)
    _sync_cuda(device)
    timings["load_s"] = time.perf_counter() - t0

    staged = []
    t0 = time.perf_counter()
    for row in range(int(x_batch.shape[0])):
        x_processed, prepared, center_shift_hz = boi_preprocess_torch(x_batch[row], pipeline.cfg)
        staged.append({
            "x_processed": x_processed,
            "prepared": prepared,
            "center_shift_hz": center_shift_hz,
        })
    _sync_cuda(device)
    timings["boi_shift_s"] = time.perf_counter() - t0

    lr_rows, rs20_rows, cfo1_rows = [], [], []
    t0 = time.perf_counter()
    for item in staged:
        lr55, rs20, cfo1 = pipeline._blind_vectors(item["x_processed"], item["center_shift_hz"])
        lr_rows.append(lr55)
        rs20_rows.append(rs20)
        cfo1_rows.append(cfo1)
    lr_matrix = torch.stack(lr_rows)
    _sync_cuda(device)
    timings["fft3_lr_feature_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = pipeline.lr_model.predict_indices(lr_matrix)
    _sync_cuda(device)
    timings["lr_s"] = time.perf_counter() - t0

    estimates = []
    t0 = time.perf_counter()
    for rs20, cfo1 in zip(rs20_rows, cfo1_rows):
        estimates.append(_estimate_rs_cfo_from_torch_vectors(rs20, cfo1))
    _sync_cuda(device)
    timings["rs_cfo_s"] = time.perf_counter() - t0

    cc_builder = (
        build_qpsk_cap_input_original_gpu
        if cc_mode == "original"
        else build_qpsk_cap_input_simplified_gpu
    )
    cap_inputs = []
    t0 = time.perf_counter()
    for item, estimate in zip(staged, estimates):
        power = _cc_power_gpu(item["prepared"], item["x_processed"], float(pipeline.cfg["EPS"]))
        cap_inputs.append(cc_builder(
            item["x_processed"].to(dtype=torch.complex64),
            estimate[1].to(device=device, dtype=torch.float32),
            estimate[2].to(device=device, dtype=torch.float32),
            power.to(device=device, dtype=torch.float32),
            eps=float(pipeline.cfg["EPS"]),
        ))
    _sync_cuda(device)
    timings["cc_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cap_tensor = torch.stack(cap_inputs)[:, None, :, :]
    with torch.inference_mode():
        _ = pipeline.cap_model(cap_tensor).argmax(dim=1)
    _sync_cuda(device)
    timings["cap_s"] = time.perf_counter() - t0

    total_s = time.perf_counter() - total_t0
    return {
        "device": f"{device}_{cc_mode}_cc",
        "n_samples": len(signal_indices),
        **timings,
        "total_s": total_s,
        "per_sample_s": total_s / max(len(signal_indices), 1),
    }


def profile_four_schemes(cpu_pipeline, gpu_pipeline, year: int, signal_indices: Sequence[int]) -> pd.DataFrame:
    """Run the complete path four times and return only the requested columns."""
    rows = [
        profile_cpu_scheme(cpu_pipeline, year, signal_indices, "original"),
        profile_gpu_scheme(gpu_pipeline, year, signal_indices, "original"),
        profile_cpu_scheme(cpu_pipeline, year, signal_indices, "simplified"),
        profile_gpu_scheme(gpu_pipeline, year, signal_indices, "simplified"),
    ]
    return pd.DataFrame(rows).loc[:, RESULT_COLUMNS]


__all__ = [
    "RESULT_COLUMNS",
    "STAGE_COLUMNS",
    "build_qpsk_cap_input_original_cpu",
    "build_qpsk_cap_input_original_gpu",
    "build_qpsk_cap_input_simplified_cpu",
    "build_qpsk_cap_input_simplified_gpu",
    "eligible_qpsklike_indices",
    "profile_four_schemes",
    "round_timing_seconds",
]
