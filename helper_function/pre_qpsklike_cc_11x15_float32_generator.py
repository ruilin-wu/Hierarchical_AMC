#!/usr/bin/env python3
"""Generate float32 QPSK-like 1x11x15 CC tensors from Two-FFT FE PT files.

Nothing is generated on import.  The CC implementation is the simplified
CAPNet path: it evaluates only 18 independent cyclic moments, reconstructs
dependent moments with conjugate symmetry, writes the seven active QPSK
cumulant columns directly into an 11x15 matrix.  It saves that raw complex CC:
no absolute value, order warping, scaling, or power normalization is applied.
Saved BOI in-band power remains a separate field so the neural-network dataset
can perform all postprocessing later. Every signal-domain intermediate is
complex64 or float32 on both CPU and CUDA.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


K_LIST = tuple(range(-5, 6))
K_TO_ROW = {k: k + 5 for k in K_LIST}
FIRST15_NM = (
    (2, 0), (2, 1), (2, 2),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6),
)
COL_LABELS = tuple(f"C({n},{m})" for n, m in FIRST15_NM)
NM_TO_COL = {nm: index for index, nm in enumerate(FIRST15_NM)}
ACTIVE_K = {
    (2, 1): (-1, 0, 1),
    (4, 0): (-2, -1, 0, 1, 2),
    (4, 2): (-2, -1, 0, 1, 2),
    (4, 4): (-2, -1, 0, 1, 2),
    (6, 1): (-2, -1, 0, 1, 2),
    (6, 3): (-2, -1, 0, 1, 2),
    (6, 5): (-2, -1, 0, 1, 2),
}
TARGET_SIGNAL_TYPES = frozenset({"qpsk", "16qam", "64qam", "256qam"})
REQUIRED_SAMPLE_FIELDS = (
    "x_processed",
    "cfo_estimate",
    "rs_estimate_candidate_only_124",
)


def _empty_sequence(batch: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((batch, len(K_LIST)), device=device, dtype=torch.complex64)


def _assign_k(sequence: torch.Tensor, k: int, value: torch.Tensor) -> None:
    sequence[:, K_TO_ROW[int(k)]] = value


def _at_k(sequence: torch.Tensor, k: int) -> torch.Tensor:
    return sequence[:, K_TO_ROW[int(k)]]


def _mask_sequence(sequence: torch.Tensor, nm: tuple[int, int]) -> torch.Tensor:
    result = torch.zeros_like(sequence, dtype=torch.complex64)
    for k in ACTIVE_K[nm]:
        result[:, K_TO_ROW[k]] = sequence[:, K_TO_ROW[k]]
    return result


def _conv2_same_k(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(a, dtype=torch.complex64)
    for ka in K_LIST:
        a_value = _at_k(a, ka)
        for kb in K_LIST:
            output_k = ka + kb
            if output_k in K_TO_ROW:
                result[:, K_TO_ROW[output_k]] += a_value * _at_k(b, kb)
    return result


def _conv3_same_k(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(a, dtype=torch.complex64)
    for ka in K_LIST:
        a_value = _at_k(a, ka)
        for kb in K_LIST:
            ab_value = a_value * _at_k(b, kb)
            for kc in K_LIST:
                output_k = ka + kb + kc
                if output_k in K_TO_ROW:
                    result[:, K_TO_ROW[output_k]] += ab_value * _at_k(c, kc)
    return result


def _project_many_float32(
    u: torch.Tensor,
    t0: torch.Tensor,
    f0: torch.Tensor,
    n: int,
    m: int,
    k_values: Iterable[int],
) -> torch.Tensor:
    """Project one signal power product at several cyclic frequencies."""
    u = u.to(dtype=torch.complex64)
    k_values = tuple(int(k) for k in k_values)
    k = torch.tensor(k_values, device=u.device, dtype=torch.float32)
    order_offset = torch.tensor(n - 2 * m, device=u.device, dtype=torch.float32)
    alpha = order_offset * f0[:, None] + k[None, :] / t0[:, None]
    time_index = torch.arange(int(u.shape[-1]), device=u.device, dtype=torch.float32)
    two_pi = torch.tensor(
        np.float32(2.0) * np.float32(np.pi),
        device=u.device,
        dtype=torch.float32,
    )
    phase = -two_pi * alpha[:, :, None] * time_index[None, None, :]
    exponent = torch.exp(torch.complex(torch.zeros_like(phase), phase))
    sample_count = torch.tensor(int(u.shape[-1]), device=u.device, dtype=torch.float32)
    return torch.sum(
        exponent * u[:, None, :], dim=-1, dtype=torch.complex64
    ) / sample_count


def _build_simplified_moments_float32(
    x: torch.Tensor,
    t0: torch.Tensor,
    f0: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    x = x.to(dtype=torch.complex64)
    batch = int(x.shape[0])
    device = x.device
    x_conj = torch.conj(x)
    abs2 = (x.real * x.real + x.imag * x.imag).to(torch.float32)
    x2 = (x * x).to(torch.complex64)
    x4 = (x2 * x2).to(torch.complex64)
    x5 = (x4 * x).to(torch.complex64)

    u21 = torch.complex(abs2, torch.zeros_like(abs2))
    u40 = x4
    abs4 = (abs2 * abs2).to(torch.float32)
    u42 = torch.complex(abs4, torch.zeros_like(abs4))
    u61 = (x5 * x_conj).to(torch.complex64)
    abs6 = (abs4 * abs2).to(torch.float32)
    u63 = torch.complex(abs6, torch.zeros_like(abs6))

    m21 = _empty_sequence(batch, device)
    values = _project_many_float32(u21, t0, f0, 2, 1, (0, 1))
    _assign_k(m21, 0, values[:, 0])
    _assign_k(m21, 1, values[:, 1])
    _assign_k(m21, -1, torch.conj(values[:, 1]))

    m40 = _empty_sequence(batch, device)
    values = _project_many_float32(u40, t0, f0, 4, 0, (-2, -1, 0, 1, 2))
    for index, k in enumerate((-2, -1, 0, 1, 2)):
        _assign_k(m40, k, values[:, index])

    m42 = _empty_sequence(batch, device)
    values = _project_many_float32(u42, t0, f0, 4, 2, (0, 1, 2))
    for index, k in enumerate((0, 1, 2)):
        _assign_k(m42, k, values[:, index])
    _assign_k(m42, -1, torch.conj(values[:, 1]))
    _assign_k(m42, -2, torch.conj(values[:, 2]))

    m44 = _empty_sequence(batch, device)
    for k in ACTIVE_K[(4, 4)]:
        _assign_k(m44, k, torch.conj(_at_k(m40, -k)))

    m61 = _empty_sequence(batch, device)
    values = _project_many_float32(u61, t0, f0, 6, 1, (-2, -1, 0, 1, 2))
    for index, k in enumerate((-2, -1, 0, 1, 2)):
        _assign_k(m61, k, values[:, index])

    m63 = _empty_sequence(batch, device)
    values = _project_many_float32(u63, t0, f0, 6, 3, (0, 1, 2))
    for index, k in enumerate((0, 1, 2)):
        _assign_k(m63, k, values[:, index])
    _assign_k(m63, -1, torch.conj(values[:, 1]))
    _assign_k(m63, -2, torch.conj(values[:, 2]))

    m65 = _empty_sequence(batch, device)
    for k in ACTIVE_K[(6, 5)]:
        _assign_k(m65, k, torch.conj(_at_k(m61, -k)))

    return {
        (2, 1): m21,
        (4, 0): m40,
        (4, 2): m42,
        (4, 4): m44,
        (6, 1): m61,
        (6, 3): m63,
        (6, 5): m65,
    }


def _build_cc_complex_float32(
    moments: dict[tuple[int, int], torch.Tensor],
) -> torch.Tensor:
    m21, m40 = moments[(2, 1)], moments[(4, 0)]
    m42, m44 = moments[(4, 2)], moments[(4, 4)]
    m61, m63, m65 = moments[(6, 1)], moments[(6, 3)], moments[(6, 5)]
    device = m21.device
    c2 = torch.tensor(np.float32(2.0), device=device, dtype=torch.float32)
    c5 = torch.tensor(np.float32(5.0), device=device, dtype=torch.float32)
    c9 = torch.tensor(np.float32(9.0), device=device, dtype=torch.float32)
    c12 = torch.tensor(np.float32(12.0), device=device, dtype=torch.float32)
    cumulants = {
        (2, 1): m21,
        (4, 0): m40,
        (4, 2): m42 - c2 * _conv2_same_k(m21, m21),
        (4, 4): m44,
        (6, 1): m61 - c5 * _conv2_same_k(m40, m21),
        (6, 3): (
            m63
            - c9 * _conv2_same_k(m42, m21)
            + c12 * _conv3_same_k(m21, m21, m21)
        ),
        (6, 5): m65 - c5 * _conv2_same_k(m44, m21),
    }
    matrix = torch.zeros(
        (int(m21.shape[0]), len(K_LIST), len(FIRST15_NM)),
        device=device,
        dtype=torch.complex64,
    )
    for nm, sequence in cumulants.items():
        matrix[:, :, NM_TO_COL[nm]] = _mask_sequence(sequence, nm)
    return matrix


def build_qpsklike_cc_11x15_float32(
    x: torch.Tensor,
    rs_estimate: torch.Tensor,
    cfo_estimate: torch.Tensor,
) -> torch.Tensor:
    """Return raw complex64 CC with shape `(B,11,15)` and no postprocessing."""
    x = torch.as_tensor(x)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if x.ndim != 2:
        raise ValueError(f"x must have shape (N,) or (B,N), got {tuple(x.shape)}")
    x = x.to(dtype=torch.complex64)
    batch = int(x.shape[0])
    rs = torch.as_tensor(rs_estimate, device=x.device, dtype=torch.float32).reshape(-1)
    cfo = torch.as_tensor(cfo_estimate, device=x.device, dtype=torch.float32).reshape(-1)
    if rs.numel() == 1 and batch > 1:
        rs = rs.expand(batch)
    if cfo.numel() == 1 and batch > 1:
        cfo = cfo.expand(batch)
    if rs.numel() != batch or cfo.numel() != batch:
        raise ValueError("Rs/CFO batch length does not match x")
    valid_parameters = torch.isfinite(rs) & (
        rs > torch.tensor(np.float32(0.0), device=x.device, dtype=torch.float32)
    )
    valid_parameters &= torch.isfinite(cfo)
    if not bool(torch.all(valid_parameters).detach().cpu().item()):
        raise ValueError("Rs must be positive finite and CFO must be finite")

    t0 = torch.reciprocal(rs).to(torch.float32)
    moments = _build_simplified_moments_float32(x, t0, cfo)
    return _build_cc_complex_float32(moments).to(torch.complex64)


def _signal_type(record: dict) -> str:
    truth = record.get("truth", {})
    meta = record.get("meta", {})
    return str(
        truth.get("SignalType", meta.get("signal_type", ""))
    ).strip().lower()


def _is_qpsklike(record: dict) -> bool:
    return _signal_type(record) in TARGET_SIGNAL_TYPES


def _input_path(input_root: Path, year: str, batch_id: int) -> Path:
    return input_root / f"{year}_batch_{batch_id:02d}_2fft_fe_float32.pt"


def _output_path(output_root: Path, year: str, batch_id: int) -> Path:
    return output_root / f"{year}_batch_{batch_id:02d}_qpsklike_cc_1x11x15_float32.pt"


def _validate_source_sample(sample: dict) -> None:
    missing = [name for name in REQUIRED_SAMPLE_FIELDS if name not in sample]
    if missing:
        raise KeyError(f"Two-FFT sample misses fields: {missing}")
    x = np.asarray(sample["x_processed"])
    if x.dtype != np.complex64 or x.ndim != 1:
        raise TypeError(f"x_processed must be 1-D complex64, got {x.shape}/{x.dtype}")


def _float32_or_nan(record: dict, key: str) -> np.float32:
    try:
        return np.float32(record[key])
    except (KeyError, TypeError, ValueError):
        return np.float32(np.nan)


def _saved_boi_inband_power(sample: dict) -> np.float32:
    try:
        return np.float32(sample["power"]["p_boi_inband_signal"])
    except (KeyError, TypeError, ValueError):
        return np.float32(np.nan)


def _compute_positions(
    selected: list[dict],
    positions: list[int],
    device: torch.device,
) -> torch.Tensor:
    x_numpy = np.stack([
        np.asarray(selected[position]["x_processed"], dtype=np.complex64)
        for position in positions
    ])
    rs_numpy = np.asarray([
        selected[position]["rs_estimate_candidate_only_124"]
        for position in positions
    ], dtype=np.float32)
    cfo_numpy = np.asarray([
        selected[position]["cfo_estimate"]
        for position in positions
    ], dtype=np.float32)
    x = torch.from_numpy(x_numpy).to(device=device, dtype=torch.complex64)
    rs = torch.from_numpy(rs_numpy).to(device=device, dtype=torch.float32)
    cfo = torch.from_numpy(cfo_numpy).to(device=device, dtype=torch.float32)
    matrix = build_qpsklike_cc_11x15_float32(x, rs, cfo)
    return matrix.detach().cpu()


def generate_one_cc_batch(
    year: str,
    batch_id: int,
    cfg: dict,
    device: torch.device,
) -> Path:
    input_root = Path(cfg["INPUT_ROOT"])
    output_root = Path(cfg["OUT_ROOT"])
    source_path = _input_path(input_root, year, batch_id)
    destination_path = _output_path(output_root, year, batch_id)
    if bool(cfg.get("SKIP_EXISTING", False)) and destination_path.exists():
        print(f"Skip existing: {destination_path}")
        return destination_path
    if not source_path.exists():
        raise FileNotFoundError(f"Two-FFT input PT not found: {source_path}")

    source_pack = torch.load(source_path, map_location="cpu", weights_only=False)
    source_samples = list(source_pack.get("samples", []))
    num_input_samples = len(source_samples)
    selected = []
    for sample in source_samples:
        if _is_qpsklike(sample):
            # CC consumes only x/Rs/CFO/truth. Drop the full x4 spectrum and LR
            # vector immediately to keep a large source PT memory-bounded.
            sample.pop("fft_x4_mag_db", None)
            sample.pop("lr8_20", None)
            selected.append(sample)
    source_pack["samples"] = []
    del source_samples
    count = len(selected)
    missing_boi_power = [
        int(sample.get("meta", {}).get("signal_index", -1))
        for sample in selected
        if (
            not np.isfinite(_saved_boi_inband_power(sample))
            or _saved_boi_inband_power(sample) <= np.float32(0.0)
        )
    ]
    if missing_boi_power:
        raise ValueError(
            f"{source_path.name} has {len(missing_boi_power)} QPSK-like samples without "
            "positive power['p_boi_inband_signal']; regenerate the Two-FFT PT with "
            f"BOI payload v2 first. First signal_index={missing_boi_power[0]}"
        )
    cc_matrix = torch.full(
        (count, 1, 11, 15), complex(np.nan, np.nan), dtype=torch.complex64
    )
    status = torch.zeros((count,), dtype=torch.bool)
    errors = [""] * count

    valid_positions: list[int] = []
    for position, sample in enumerate(selected):
        try:
            _validate_source_sample(sample)
            rs = np.float32(sample["rs_estimate_candidate_only_124"])
            cfo = np.float32(sample["cfo_estimate"])
            if not np.isfinite(rs) or rs <= np.float32(0.0) or not np.isfinite(cfo):
                raise ValueError(f"Invalid Rs/CFO: rs={rs}, cfo={cfo}")
            valid_positions.append(position)
        except Exception as exc:
            errors[position] = repr(exc)
            sample.pop("x_processed", None)

    batch_size = max(1, int(cfg.get("COMPUTE_BATCH_SIZE", 8)))
    chunks = [valid_positions[start:start + batch_size]
              for start in range(0, len(valid_positions), batch_size)]
    iterator = chunks
    if tqdm is not None:
        iterator = tqdm(chunks, desc=f"{year} batch {batch_id:02d} CC", leave=False)

    for positions in iterator:
        try:
            matrix_chunk = _compute_positions(selected, positions, device)
            finite = torch.isfinite(matrix_chunk).flatten(1).all(dim=1)
            for local, position in enumerate(positions):
                if bool(finite[local].item()):
                    cc_matrix[position, 0] = matrix_chunk[local]
                    status[position] = True
                else:
                    errors[position] = "NONFINITE_CC_OUTPUT"
        except Exception:
            batch_traceback = traceback.format_exc()
            # Isolate a bad sample without losing the rest of its compute chunk.
            for position in positions:
                try:
                    matrix_one = _compute_positions(selected, [position], device)
                    if not bool(torch.isfinite(matrix_one).all().item()):
                        raise ValueError("NONFINITE_CC_OUTPUT")
                    cc_matrix[position, 0] = matrix_one[0]
                    status[position] = True
                except Exception as exc:
                    errors[position] = repr(exc) + "\nchunk_error:\n" + batch_traceback
        finally:
            for position in positions:
                selected[position].pop("x_processed", None)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    signal_index = torch.tensor([
        int(sample.get("meta", {}).get("signal_index",
            sample.get("meta", {}).get("global_index", -1)))
        for sample in selected
    ], dtype=torch.int64)
    local_index = torch.tensor([
        int(sample.get("meta", {}).get("batch_local_index", -1))
        for sample in selected
    ], dtype=torch.int64)
    rs_estimate = torch.tensor([
        _float32_or_nan(sample, "rs_estimate_candidate_only_124")
        for sample in selected
    ], dtype=torch.float32)
    cfo_estimate = torch.tensor([
        _float32_or_nan(sample, "cfo_estimate")
        for sample in selected
    ], dtype=torch.float32)
    p_boi_inband_signal = torch.tensor([
        _saved_boi_inband_power(sample) for sample in selected
    ], dtype=torch.float32)
    valid_rs = torch.isfinite(rs_estimate) & (rs_estimate > torch.tensor(0, dtype=torch.float32))
    t0_estimate = torch.where(
        valid_rs,
        torch.reciprocal(rs_estimate),
        torch.full_like(rs_estimate, torch.nan),
    ).to(torch.float32)

    source_skipped = [
        row for row in source_pack.get("skipped", []) if _is_qpsklike(row)
    ]
    source_errors = [
        row for row in source_pack.get("errors", []) if _is_qpsklike(row)
    ]
    output_pack = {
        "version": "qpsklike_raw_cc_1x11x15_complex64_batch_v3",
        "dataset_name": str(year),
        "batch_id": int(batch_id),
        "source_pt": str(source_path),
        "num_input_samples": int(num_input_samples),
        "num_selected_qpsklike": int(count),
        "num_ok": int(status.sum().item()),
        "num_error": int((~status).sum().item()),
        "num_source_qpsklike_skipped": int(len(source_skipped)),
        "target_signal_types": sorted(TARGET_SIGNAL_TYPES),
        "matrix_shape_per_sample": [1, 11, 15],
        "matrix_semantics": (
            "raw complex cyclic cumulant; no abs, warping, scaling, or power normalization"
        ),
        "algorithm": "simplified_18_independent_moments_conjugate_symmetry_float32",
        "precision": {
            "x_processed": "complex64",
            "rs_cfo_t0_power": "float32",
            "cyclic_frequency_phase": "float32",
            "cyclic_moment_and_cumulant": "complex64",
            "saved_cc_matrix": "complex64 (float32 real/imag components)",
            "saved_p_boi_inband_signal": "float32",
        },
        "k_list": list(K_LIST),
        "nm_list": list(FIRST15_NM),
        "column_labels": list(COL_LABELS),
        "signal_index": signal_index,
        "local_index": local_index,
        "signal_type": [_signal_type(sample) for sample in selected],
        "truth": [dict(sample.get("truth", {})) for sample in selected],
        "truth_line": [str(sample.get("truth_line", "")) for sample in selected],
        "rs_estimate_candidate_only_124": rs_estimate,
        "cfo_estimate": cfo_estimate,
        "t0_estimate": t0_estimate,
        "p_boi_inband_signal": p_boi_inband_signal,
        "cc_matrix": cc_matrix,
        "status": status,
        "error": errors,
        "source_qpsklike_skipped": source_skipped,
        "source_qpsklike_errors": source_errors,
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_pack, destination_path)
    print(
        f"Saved {destination_path.name}: selected={count}, "
        f"ok={output_pack['num_ok']}, errors={output_pack['num_error']}, "
        f"source_skipped={len(source_skipped)}"
    )
    del source_pack, selected, output_pack, cc_matrix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return destination_path


def generate_qpsklike_cc_pt_files(cfg: dict) -> list[Path]:
    selection = str(cfg["DATASET_SELECT"]).lower()
    if selection == "both":
        years = ("2018", "2022")
    elif selection in {"2018", "2022"}:
        years = (selection,)
    else:
        raise ValueError("DATASET_SELECT must be '2018', '2022', or 'both'")

    requested_device = str(cfg.get("DEVICE", "auto"))
    if requested_device == "auto":
        requested_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    input_root = Path(cfg["INPUT_ROOT"])
    output_root = Path(cfg["OUT_ROOT"])
    start_batch = max(1, int(cfg.get("START_BATCH", 1)))
    end_batch = min(int(cfg.get("END_BATCH", 28)), int(cfg.get("N_BATCHES", 28)))
    if start_batch > end_batch:
        raise ValueError(f"Invalid batch range: {start_batch}..{end_batch}")

    print(f"Input root: {input_root.resolve()}")
    print(f"Shared flat output root: {output_root.resolve()}")
    print(f"Device: {device}; compute batch size: {int(cfg.get('COMPUTE_BATCH_SIZE', 8))}")
    paths: list[Path] = []
    for year in years:
        for batch_id in range(start_batch, end_batch + 1):
            paths.append(generate_one_cc_batch(year, batch_id, cfg, device))
    print(f"Done. Generated {len(paths)} PT files for selection={selection}")
    return paths


def inspect_qpsklike_cc_pt(path: str | Path) -> dict:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    matrix = pack.get("cc_matrix", torch.empty(0))
    return {
        "path": str(path),
        "version": pack.get("version"),
        "dataset_name": pack.get("dataset_name"),
        "batch_id": pack.get("batch_id"),
        "num_selected_qpsklike": pack.get("num_selected_qpsklike"),
        "num_ok": pack.get("num_ok"),
        "num_error": pack.get("num_error"),
        "cc_matrix_shape": tuple(matrix.shape),
        "cc_matrix_dtype": str(matrix.dtype),
        "per_sample_shape": tuple(matrix.shape[1:]) if matrix.ndim == 4 else None,
        "rs_dtype": str(pack.get("rs_estimate_candidate_only_124", torch.empty(0)).dtype),
        "cfo_dtype": str(pack.get("cfo_estimate", torch.empty(0)).dtype),
        "p_boi_inband_signal_dtype": str(
            pack.get("p_boi_inband_signal", torch.empty(0)).dtype
        ),
    }


__all__ = [
    "ACTIVE_K",
    "COL_LABELS",
    "FIRST15_NM",
    "K_LIST",
    "TARGET_SIGNAL_TYPES",
    "build_qpsklike_cc_11x15_float32",
    "generate_qpsklike_cc_pt_files",
    "inspect_qpsklike_cc_pt",
]
