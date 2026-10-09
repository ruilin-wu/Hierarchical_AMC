#!/usr/bin/env python3
"""QPSK-like 11x15 raw CC matrix generation from `generated_boi_p96_fft2_pt/`.

Reuses the validated pure-math CC engine (`build_qpsklike_cc_11x15_float32` and
its K_LIST/FIRST15_NM/COL_LABELS/TARGET_SIGNAL_TYPES constants) from
`pre_qpsklike_cc_11x15_float32_generator.py` unchanged, but replaces that
module's I/O layer: the original reads the OLD, superseded (no-clip,
candidate-lookup Rs) `{year}_batch_NN_2fft_fe_float32.pt` files and their
`rs_estimate_candidate_only_124` / `cfo_estimate` fields. This module instead
reads the consolidated, validated `generated_boi_p96_fft2_pt/{year}_batch_NN_
boi_p96_fft2_full.pt` files and, by default, the winning combination developed
in `boi_p96_fft3`: `rs_estimate_prominence_corrected` (Method 2's blind
circular-prominence full-grid search, 98.24% on the full 2018/2022 QPSK-like
corpus, corrected by the `FFT{|z|^2}` cross-check arbitration -- 0 regressions
on any group-split partition, 936/167,816 samples recovered, see
`rs_prominence_fftz2_correction`'s docstring in `pre_boi_p96_fft2_generator.py`)
and `cfo_estimate_band_interp` (band-limited + parabolic interpolation, closes
the classification gap to ground-truth CFO to within +/-0.4pp). Either field
name can be overridden via `cfg["RS_FIELD"]` / `cfg["CFO_FIELD"]` (e.g. back to
the UNCORRECTED `rs_estimate_prominence`, or to `rs_estimate_abs_x2_rule` /
`cfo_estimate_argmax`, for ablation).

`x_processed` (unclipped BOI output) is used for the CC computation itself,
NOT `z` -- verified throughout this session's ablation work to give better
downstream classification accuracy than using the P96-clipped `z`.
`power.p_boi_inband_signal` (unclipped in-band power) is used for the later
abs/warp/scale normalization step (done by the consumer, not here -- this
module saves the RAW complex CC, matching the established convention).

The upstream files contain ALL signal types; this is where the QPSK-like
filter is applied. Excluded rows are counted separately in the output metadata,
not silently dropped:
  - `num_upstream_not_qpsklike`: `status=True` rows whose truth signal type is
    bpsk/8psk/dqpsk/msk (they have x_processed/Rs/CFO, but no QPSK-like CC).
  - `num_upstream_negative_snr_skipped`: 2018 negative-SNR rows
    (`status=False`, `SKIP_NEGATIVE_TRUTH_SNR`).
  - `num_upstream_error`: any other upstream `status=False` row (BOI/clip/FFT
    processing error), of any signal type.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .pre_qpsklike_cc_11x15_float32_generator import (
    K_LIST,
    FIRST15_NM,
    COL_LABELS,
    TARGET_SIGNAL_TYPES,
    build_qpsklike_cc_11x15_float32,
)

DEFAULT_RS_FIELD = "rs_estimate_prominence_corrected"
DEFAULT_CFO_FIELD = "cfo_estimate_band_interp"


def _input_path(input_root: Path, year: str, batch_id: int) -> Path:
    return input_root / f"{year}_batch_{batch_id:02d}_boi_p96_fft2_full.pt"


def _output_path(output_root: Path, year: str, batch_id: int) -> Path:
    return output_root / f"{year}_batch_{batch_id:02d}_qpsklike_cc_1x11x15_boi_p96_fft2.pt"


def _signal_type(sample: dict) -> str:
    truth = sample.get("truth") or {}
    return str(truth.get("signal_type", "")).strip().lower()


def _is_usable_qpsklike(sample: dict) -> bool:
    return bool(sample.get("status")) and _signal_type(sample) in TARGET_SIGNAL_TYPES


def _float32_or_nan(sample: dict, key: str) -> np.float32:
    try:
        return np.float32(sample[key])
    except (KeyError, TypeError, ValueError):
        return np.float32(np.nan)


def _saved_boi_inband_power(sample: dict) -> np.float32:
    try:
        return np.float32(sample["power"]["p_boi_inband_signal"])
    except (KeyError, TypeError, ValueError):
        return np.float32(np.nan)


def _compute_positions(selected: list[dict], positions: list[int], rs_field: str,
                        cfo_field: str, device: torch.device) -> torch.Tensor:
    x_numpy = np.stack([
        np.asarray(selected[p]["x_processed"], dtype=np.complex64) for p in positions
    ])
    rs_numpy = np.asarray([np.float32(selected[p][rs_field]) for p in positions], dtype=np.float32)
    cfo_numpy = np.asarray([np.float32(selected[p][cfo_field]) for p in positions], dtype=np.float32)
    x = torch.from_numpy(x_numpy).to(device=device, dtype=torch.complex64)
    rs = torch.from_numpy(rs_numpy).to(device=device, dtype=torch.float32)
    cfo = torch.from_numpy(cfo_numpy).to(device=device, dtype=torch.float32)
    matrix = build_qpsklike_cc_11x15_float32(x, rs, cfo)
    return matrix.detach().cpu()


def generate_one_cc_batch(year: str, batch_id: int, cfg: dict, device: torch.device) -> Path:
    input_root = Path(cfg["INPUT_ROOT"])
    output_root = Path(cfg["OUT_ROOT"])
    rs_field = str(cfg.get("RS_FIELD", DEFAULT_RS_FIELD))
    cfo_field = str(cfg.get("CFO_FIELD", DEFAULT_CFO_FIELD))
    source_path = _input_path(input_root, year, batch_id)
    destination_path = _output_path(output_root, year, batch_id)
    if bool(cfg.get("SKIP_EXISTING", False)) and destination_path.exists():
        print(f"Skip existing: {destination_path}")
        return destination_path
    if not source_path.exists():
        raise FileNotFoundError(f"boi_p96_fft2 input PT not found: {source_path}")

    source_pack = torch.load(source_path, map_location="cpu", weights_only=False)
    source_samples = list(source_pack.get("samples", []))
    num_input_samples = len(source_samples)
    num_upstream_not_qpsklike = sum(
        1 for s in source_samples if s.get("status") and _signal_type(s) not in TARGET_SIGNAL_TYPES
    )
    num_upstream_negative_snr = sum(
        1 for s in source_samples if not s.get("status") and s.get("error") == "SKIP_NEGATIVE_TRUTH_SNR"
    )
    num_upstream_error = sum(
        1 for s in source_samples if not s.get("status") and s.get("error") != "SKIP_NEGATIVE_TRUTH_SNR"
    )

    selected = [s for s in source_samples if _is_usable_qpsklike(s)]
    source_pack["samples"] = []
    del source_samples
    count = len(selected)

    missing_boi_power = [
        int(s.get("meta", {}).get("signal_index", -1)) for s in selected
        if not np.isfinite(_saved_boi_inband_power(s)) or _saved_boi_inband_power(s) <= np.float32(0.0)
    ]
    if missing_boi_power:
        raise ValueError(
            f"{source_path.name} has {len(missing_boi_power)} QPSK-like samples without "
            f"positive power['p_boi_inband_signal']. First signal_index={missing_boi_power[0]}"
        )

    cc_matrix = torch.full((count, 1, 11, 15), complex(np.nan, np.nan), dtype=torch.complex64)
    status = torch.zeros((count,), dtype=torch.bool)
    errors = [""] * count

    valid_positions: list[int] = []
    for position, sample in enumerate(selected):
        try:
            rs = np.float32(sample[rs_field])
            cfo = np.float32(sample[cfo_field])
            if not np.isfinite(rs) or rs <= np.float32(0.0) or not np.isfinite(cfo):
                raise ValueError(f"Invalid Rs/CFO: rs={rs}, cfo={cfo}")
            valid_positions.append(position)
        except Exception as exc:
            errors[position] = repr(exc)

    batch_size = max(1, int(cfg.get("COMPUTE_BATCH_SIZE", 16)))
    chunks = [valid_positions[start:start + batch_size] for start in range(0, len(valid_positions), batch_size)]
    iterator = tqdm(chunks, desc=f"{year} batch {batch_id:02d} CC", leave=False) if tqdm is not None else chunks

    for positions in iterator:
        try:
            matrix_chunk = _compute_positions(selected, positions, rs_field, cfo_field, device)
            finite = torch.isfinite(matrix_chunk).flatten(1).all(dim=1)
            for local, position in enumerate(positions):
                if bool(finite[local].item()):
                    cc_matrix[position, 0] = matrix_chunk[local]
                    status[position] = True
                else:
                    errors[position] = "NONFINITE_CC_OUTPUT"
        except Exception:
            batch_traceback = traceback.format_exc()
            for position in positions:
                try:
                    matrix_one = _compute_positions(selected, [position], rs_field, cfo_field, device)
                    if not bool(torch.isfinite(matrix_one).all().item()):
                        raise ValueError("NONFINITE_CC_OUTPUT")
                    cc_matrix[position, 0] = matrix_one[0]
                    status[position] = True
                except Exception as exc:
                    errors[position] = repr(exc) + "\nchunk_error:\n" + batch_traceback
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    signal_index = torch.tensor(
        [int(s.get("meta", {}).get("signal_index", -1)) for s in selected], dtype=torch.int64
    )
    rs_estimate = torch.tensor([_float32_or_nan(s, rs_field) for s in selected], dtype=torch.float32)
    cfo_estimate = torch.tensor([_float32_or_nan(s, cfo_field) for s in selected], dtype=torch.float32)
    p_boi_inband_signal = torch.tensor([_saved_boi_inband_power(s) for s in selected], dtype=torch.float32)
    valid_rs = torch.isfinite(rs_estimate) & (rs_estimate > torch.tensor(0, dtype=torch.float32))
    t0_estimate = torch.where(valid_rs, torch.reciprocal(rs_estimate), torch.full_like(rs_estimate, torch.nan))

    output_pack = {
        "version": "qpsklike_raw_cc_1x11x15_boi_p96_fft2_v1",
        "dataset_name": str(year),
        "batch_id": int(batch_id),
        "source_pt": str(source_path),
        "num_input_samples": int(num_input_samples),
        "num_upstream_not_qpsklike": int(num_upstream_not_qpsklike),
        "num_upstream_negative_snr_skipped": int(num_upstream_negative_snr),
        "num_upstream_error": int(num_upstream_error),
        "num_selected_qpsklike": int(count),
        "num_ok": int(status.sum().item()),
        "num_error": int((~status).sum().item()),
        "target_signal_types": sorted(TARGET_SIGNAL_TYPES),
        "matrix_shape_per_sample": [1, 11, 15],
        "matrix_semantics": "raw complex cyclic cumulant; no abs, warping, scaling, or power normalization",
        "algorithm": "simplified_18_independent_moments_conjugate_symmetry_float32",
        "rs_field": rs_field,
        "cfo_field": cfo_field,
        "power_field": "p_boi_inband_signal",
        "x_field": "x_processed (unclipped BOI output)",
        "k_list": list(K_LIST),
        "nm_list": list(FIRST15_NM),
        "column_labels": list(COL_LABELS),
        "signal_index": signal_index,
        "signal_type": [_signal_type(s) for s in selected],
        "truth": [dict(s.get("truth") or {}) for s in selected],
        "rs_estimate": rs_estimate,
        "cfo_estimate": cfo_estimate,
        "t0_estimate": t0_estimate,
        "p_boi_inband_signal": p_boi_inband_signal,
        "cc_matrix": cc_matrix,
        "status": status,
        "error": errors,
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_pack, destination_path)
    print(
        f"Saved {destination_path.name}: selected={count}, ok={output_pack['num_ok']}, "
        f"errors={output_pack['num_error']}, upstream_not_qpsklike={num_upstream_not_qpsklike}, "
        f"upstream_neg_snr={num_upstream_negative_snr}, upstream_error={num_upstream_error}"
    )
    del source_pack, selected, output_pack, cc_matrix
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return destination_path


def generate_qpsklike_cc_pt_files(cfg: dict) -> list[Path]:
    selection = str(cfg["DATASET_SELECT"])
    years = ("2018", "2022", "2018R2") if selection.lower() == "all" else (selection,)

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

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Device: {device}; RS_FIELD={cfg.get('RS_FIELD', DEFAULT_RS_FIELD)}; "
          f"CFO_FIELD={cfg.get('CFO_FIELD', DEFAULT_CFO_FIELD)}")
    paths: list[Path] = []
    for year in years:
        for batch_id in range(start_batch, end_batch + 1):
            paths.append(generate_one_cc_batch(year, batch_id, cfg, device))
    print(f"Done. Generated {len(paths)} PT files for selection={selection}")
    return paths


def inspect_qpsklike_cc_pt(path) -> dict:
    pack = torch.load(str(path), map_location="cpu", weights_only=False)
    matrix = pack.get("cc_matrix", torch.empty(0))
    return {
        "path": str(path),
        "version": pack.get("version"),
        "dataset_name": pack.get("dataset_name"),
        "batch_id": pack.get("batch_id"),
        "rs_field": pack.get("rs_field"),
        "cfo_field": pack.get("cfo_field"),
        "num_input_samples": pack.get("num_input_samples"),
        "num_upstream_not_qpsklike": pack.get("num_upstream_not_qpsklike"),
        "num_upstream_negative_snr_skipped": pack.get("num_upstream_negative_snr_skipped"),
        "num_upstream_error": pack.get("num_upstream_error"),
        "num_selected_qpsklike": pack.get("num_selected_qpsklike"),
        "num_ok": pack.get("num_ok"),
        "num_error": pack.get("num_error"),
        "cc_matrix_shape": tuple(matrix.shape),
        "cc_matrix_dtype": str(matrix.dtype),
    }
