#!/usr/bin/env python3
"""Generate BOI-shifted two-FFT FE payloads without running on import.

The fixed BOI detector/filter/frequency shift deliberately reuses the existing
float64/complex128 implementation.  The result is then quantized once to
complex64; x^2, x^4, both FFTs, LR8_20, CFO and fixed-124 Rs estimation are
computed and saved at float32/complex64 precision.
"""

from __future__ import annotations

import gc
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft

from . import pre_boi_fft3_generation_base_from_notebook as base


NFFT = 32768
TOP_K = 8
MIN_SEP_BINS = 8
DC_EXCLUSION = np.float32(5.0 / NFFT)
_WINDOW_INDEX_FLOAT32 = np.arange(NFFT, dtype=np.float32)
HANN_WINDOW_FLOAT32 = (
    np.float32(0.5)
    - np.float32(0.5) * np.cos(
        np.float32(2.0) * np.float32(np.pi) * _WINDOW_INDEX_FLOAT32
        / np.float32(NFFT - 1)
    )
).astype(np.float32, copy=False)

TRUTH_COLUMNS = [
    "SignalIndex",
    "SignalType",
    "BaseSymbolPeriod",
    "CarrierOffset",
    "ExcessBandwidth",
    "UpsampleFactor",
    "DownsampleFactor",
    "InbandSNRdB",
    "NoiseSpectralDensitydB",
]

M8_FEATURES = [
    "x2_top4_f", "x2_top5_f", "x2_top8_f", "x2_top5_minus_top6_db",
    "x2_energy_ratio_top4_of_8", "x2_sym8_score_min", "x2_sym8_score_min2",
    "x4_top1_af", "x4_top2_af", "x4_top3_db", "x4_top6_af", "x4_top7_f",
    "x4_top8_f", "x4_top8_af", "x4_top1_minus_top2_db", "x4_top2_minus_top3_db",
    "x4_top6_minus_top7_db", "x4_count_gt_m20_d8", "x4_sym8_score_min",
    "x4_sym8_score_min2",
]

FIXED_RS_CANDIDATES = np.asarray([
    0.035087719, 0.037037037, 0.039215686, 0.039473684, 0.041666667,
    0.042105263, 0.043859649, 0.044117647, 0.044444444, 0.045112782,
    0.046052632, 0.046296296, 0.046783626, 0.046875000, 0.047058824,
    0.047368421, 0.047619048, 0.048611111, 0.049019608, 0.049382716,
    0.050000000, 0.050420168, 0.051282051, 0.051470588, 0.052083333,
    0.052287582, 0.052631579, 0.052941176, 0.053333333, 0.053571429,
    0.054687500, 0.055555556, 0.056250000, 0.057142857, 0.057692308,
    0.058333333, 0.058823529, 0.059259259, 0.059523810, 0.060000000,
    0.060606061, 0.061224490, 0.061538462, 0.062500000, 0.063492063,
    0.064102564, 0.064285714, 0.065934066, 0.066666667, 0.067307692,
    0.068181818, 0.068376068, 0.069230769, 0.069444444, 0.071428571,
    0.072727273, 0.072916667, 0.074074074, 0.075000000, 0.075757576,
    0.076923077, 0.077922078, 0.079545455, 0.080000000, 0.080808081,
    0.081818182, 0.083333333, 0.085714286, 0.087500000, 0.088888889,
    0.090000000, 0.090909091, 0.092592593, 0.093750000, 0.095238095,
    0.097222222, 0.098765432, 0.100000000, 0.104166667, 0.107142857,
    0.109375000, 0.111111111, 0.112500000, 0.114285714, 0.119047619,
    0.122448980, 0.125000000, 0.126984127, 0.128571429, 0.133333333,
    0.138888889, 0.142857143, 0.145833333, 0.148148148, 0.150000000,
    0.160000000, 0.166666667, 0.171428571, 0.175000000, 0.177777778,
    0.180000000, 0.187500000, 0.200000000, 0.208333333, 0.214285714,
    0.218750000, 0.222222222, 0.225000000, 0.250000000, 0.266666667,
    0.277777778, 0.285714286, 0.291666667, 0.296296296, 0.300000000,
    0.333333333, 0.375000000, 0.400000000, 0.416666667, 0.428571429,
    0.437500000, 0.444444444, 0.450000000, 0.500000000,
], dtype=np.float32)

if len(M8_FEATURES) != 20 or len(set(M8_FEATURES)) != 20:
    raise AssertionError("LR8_20 must contain 20 unique features")
if FIXED_RS_CANDIDATES.shape != (124,) or not np.all(np.diff(FIXED_RS_CANDIDATES) > 0):
    raise AssertionError("The fixed Rs table must contain 124 increasing rates")


def _read_raw_truth(label_path: str | Path) -> tuple[pd.DataFrame, dict[int, str]]:
    label_path = Path(label_path)
    truth_df = pd.read_csv(label_path, sep=r"\s+")
    if list(truth_df.columns) != TRUTH_COLUMNS:
        raise ValueError(
            f"Unexpected truth columns in {label_path}: {list(truth_df.columns)}"
        )
    truth_df = truth_df.set_index("SignalIndex", drop=False)

    lines = label_path.read_text(encoding="utf-8").splitlines()
    raw_lines: dict[int, str] = {}
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        raw_lines[int(stripped.split()[0])] = stripped
    return truth_df, raw_lines


def _truth_dict(raw_truth_df: pd.DataFrame, signal_index: int) -> dict:
    row = raw_truth_df.loc[int(signal_index)]
    result = {
        "SignalIndex": int(row["SignalIndex"]),
        "SignalType": str(row["SignalType"]),
    }
    for name in TRUTH_COLUMNS[2:]:
        result[name] = np.float32(row[name])
    return result


def _fft_mag_db_float32(z: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    """Hann-windowed relative magnitude-dB FFT, with no artificial dB floor."""
    z = np.asarray(z, dtype=np.complex64).reshape(-1)
    nfft = int(nfft)
    work = np.zeros(nfft, dtype=np.complex64)
    count = min(z.size, nfft)
    work[:count] = z[:count]
    if nfft != NFFT:
        raise ValueError(f"This FE definition requires nfft={NFFT}")
    windowed = (work * HANN_WINDOW_FLOAT32).astype(np.complex64, copy=False)
    spectrum = scipy_fft.fftshift(scipy_fft.fft(windowed, n=nfft))
    spectrum = np.asarray(spectrum, dtype=np.complex64)
    magnitude = np.abs(spectrum).astype(np.float32, copy=False)
    peak = np.max(magnitude)
    if not np.isfinite(peak) or peak <= np.float32(0.0):
        raise ValueError("FFT magnitude has no positive finite maximum")
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude_db = np.float32(20.0) * np.log10(magnitude / peak)
    return np.asarray(magnitude_db, dtype=np.float32)


def _top_peaks_float32(magnitude_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the comparison notebook's descending sort + bin-exclusion rule."""
    magnitude_db = np.asarray(magnitude_db, dtype=np.float32).reshape(-1)
    valid_indices = np.flatnonzero(np.isfinite(magnitude_db))
    order = valid_indices[np.argsort(magnitude_db[valid_indices])[::-1]]
    selected: list[int] = []
    for index in order:
        index_int = int(index)
        if all(abs(index_int - old) >= MIN_SEP_BINS for old in selected):
            selected.append(index_int)
            if len(selected) == TOP_K:
                break
    if len(selected) != TOP_K:
        raise ValueError(f"Only {len(selected)} finite separated FFT peaks were found")
    bins = np.asarray(selected, dtype=np.int32)
    freq = ((bins - np.int32(NFFT // 2)).astype(np.float32) /
            np.float32(NFFT)).astype(np.float32)
    return freq, magnitude_db[bins].astype(np.float32, copy=False)


def _symmetry_min2_float32(freq: np.ndarray, db: np.ndarray) -> tuple[np.float32, np.float32]:
    freq = np.asarray(freq, dtype=np.float32)[:TOP_K]
    db = np.asarray(db, dtype=np.float32)[:TOP_K]
    valid = np.isfinite(freq)
    scores: list[np.float32] = []
    for index in range(TOP_K):
        if not valid[index] or np.abs(freq[index]) <= DC_EXCLUSION:
            continue
        differences = np.abs(freq + freq[index]).astype(np.float32)
        differences = np.where(valid, differences, np.float32(np.inf))
        mirror = int(np.argmin(differences))
        score = (
            np.abs(db[index] - db[mirror])
            + np.float32(40.0) * np.abs(freq[mirror] + freq[index])
        )
        scores.append(np.float32(score))
    if not scores:
        return np.float32(99.0), np.float32(99.0)
    ordered = np.sort(np.asarray(scores, dtype=np.float32))
    first = np.float32(ordered[0])
    second = np.float32(ordered[1] if ordered.size >= 2 else ordered[0])
    return first, second


def _lr8_20_float32(
    x2_freq: np.ndarray,
    x2_db: np.ndarray,
    x4_freq: np.ndarray,
    x4_db: np.ndarray,
) -> np.ndarray:
    x4_af = np.abs(x4_freq).astype(np.float32)
    x2_sym_min, x2_sym_min2 = _symmetry_min2_float32(x2_freq, x2_db)
    x4_sym_min, x4_sym_min2 = _symmetry_min2_float32(x4_freq, x4_db)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x2_power = np.power(np.float32(10.0), x2_db / np.float32(10.0)).astype(np.float32)
        energy_ratio = np.sum(x2_power[:4], dtype=np.float32) / np.sum(
            x2_power[:8], dtype=np.float32
        )

    x4_nonzero = np.isfinite(x4_af) & (x4_af > DC_EXCLUSION)
    x4_count = np.sum(x4_nonzero & (x4_db > np.float32(-20.0)), dtype=np.int32)

    columns = {
        "x2_top4_f": x2_freq[3],
        "x2_top5_f": x2_freq[4],
        "x2_top8_f": x2_freq[7],
        "x2_top5_minus_top6_db": x2_db[4] - x2_db[5],
        "x2_energy_ratio_top4_of_8": energy_ratio,
        "x2_sym8_score_min": x2_sym_min,
        "x2_sym8_score_min2": x2_sym_min2,
        "x4_top1_af": x4_af[0],
        "x4_top2_af": x4_af[1],
        "x4_top3_db": x4_db[2],
        "x4_top6_af": x4_af[5],
        "x4_top7_f": x4_freq[6],
        "x4_top8_f": x4_freq[7],
        "x4_top8_af": x4_af[7],
        "x4_top1_minus_top2_db": x4_db[0] - x4_db[1],
        "x4_top2_minus_top3_db": x4_db[1] - x4_db[2],
        "x4_top6_minus_top7_db": x4_db[5] - x4_db[6],
        "x4_count_gt_m20_d8": np.float32(x4_count),
        "x4_sym8_score_min": x4_sym_min,
        "x4_sym8_score_min2": x4_sym_min2,
    }
    return np.asarray([columns[name] for name in M8_FEATURES], dtype=np.float32)


def _wrap_float32(value):
    value = np.asarray(value, dtype=np.float32)
    return ((value + np.float32(0.5)) % np.float32(1.0)) - np.float32(0.5)


def _candidate_only_rs_float32(
    x4_magnitude_db: np.ndarray,
    x4_top1_f: np.float32,
) -> np.float32:
    center = np.float32(x4_top1_f)
    pos = _wrap_float32(center + FIXED_RS_CANDIDATES)
    neg = _wrap_float32(center - FIXED_RS_CANDIDATES)
    idx_pos = np.rint((pos + np.float32(0.5)) * np.float32(NFFT)).astype(np.int32) % NFFT
    idx_neg = np.rint((neg + np.float32(0.5)) * np.float32(NFFT)).astype(np.int32) % NFFT
    scores = np.float32(0.5) * (
        x4_magnitude_db[idx_pos] + x4_magnitude_db[idx_neg]
    )
    return np.float32(FIXED_RS_CANDIDATES[int(np.argmax(scores))])


def _boi_scalar_payload(run: dict) -> tuple[dict, dict]:
    """Preserve the legacy compact PT BOI/preprocess and power contract."""
    prepared = run["prepared"]
    step_b = run["stepB"]
    refined = prepared["selected_refined"]
    requested_filter = prepared["selected_filter"]
    used_filter = step_b["boi_used"]
    power_report = prepared["power_report"]

    preprocess = {
        "boi_algorithm": "fixed",
        "preprocess_mode": "boi_filter_shift",
        "estimated_boi_center_hz": float(run["estimated_boi_center_hz"]),
        "center_shift_hz": float(run["center_shift_hz"]),
        "truth_guide_cfo": float(run["truth_guide_cfo"]),
        "boi_refined_left": float(refined["f_left"]),
        "boi_refined_right": float(refined["f_right"]),
        "boi_refined_center": float(refined["f_center"]),
        "boi_refined_bw": float(refined["bandwidth"]),
        "boi_refined_left_bin": int(refined["left_bin"]),
        "boi_refined_right_bin": int(refined["right_bin"]),
        "boi_refined_width_bins": int(refined["width_bins"]),
        "boi_filter_left": float(requested_filter["f_left"]),
        "boi_filter_right": float(requested_filter["f_right"]),
        "boi_filter_center": float(requested_filter["f_center"]),
        "boi_filter_bw": float(requested_filter["bandwidth"]),
        "boi_filter_expand_ratio": float(
            requested_filter.get("filter_expand_ratio", np.nan)
        ),
        "boi_filter_source": str(step_b.get("boi_filter_source", "")),
        "boi_filter_used_left": float(used_filter["f_left"]),
        "boi_filter_used_right": float(used_filter["f_right"]),
        "boi_filter_used_center": float(used_filter["f_center"]),
        "boi_filter_used_bw": float(used_filter["bandwidth"]),
        "expanded_bw_frac": float(step_b.get("expanded_bw_frac", np.nan)),
        "max_expanded_filter_bw_frac": float(
            step_b.get("max_expanded_filter_bw_frac", np.nan)
        ),
        "stepB_mode": str(step_b["mode"]),
        "stepB_decim": int(step_b["decim"]),
        "stepB_fs_out": float(step_b["fs_out"]),
        "stepB_applied_shift_hz": float(step_b["applied_shift_hz"]),
    }

    # Keep the established names used by the old compact PT and expose the
    # complete scalar BOI power report as well. These are float64 BOI results.
    power = {
        "P_total": float(base.signal_rms_power(run["x_original"])),
        "P_x_bpf": float(base.signal_rms_power(step_b["x_bpf"])),
        "P_x_processed": float(base.signal_rms_power(step_b["x_processed"])),
        "p_boi_inband_signal": float(power_report["p_signal_boi_local"]),
        "noise_floor_raw_per_bin": float(power_report["noise_floor_raw_per_bin"]),
        "p_noise_fullband_hat": float(power_report["p_noise_fullband_hat"]),
        "p_noise_inband_hat": float(power_report["p_noise_inband_hat"]),
        "p_noise_outband_hat": float(power_report["p_noise_outband_hat"]),
        "p_signal_fullband_excess": float(power_report["p_signal_fullband_excess"]),
        "p_signal_recommended": float(power_report["p_signal_recommended"]),
        "snr_boi_local_db": float(power_report["snr_boi_local_db"]),
        "snr_fullband_excess_db": float(power_report["snr_fullband_excess_db"]),
        "snr_recommended_db": float(power_report["snr_recommended_db"]),
    }
    return preprocess, power


def build_twofft_fe_payload(
    signal_index: int,
    cfg: dict,
    boi_truth_df: pd.DataFrame,
    raw_truth_df: pd.DataFrame,
    raw_truth_lines: dict[int, str],
    root_data: str,
    dataset_name: str,
    batch_id: int,
    batch_start: int,
) -> dict:
    run = base.run_one_fixed_boi_shift(
        signal_index=signal_index,
        cfg=cfg,
        truth_df=boi_truth_df,
        root_data=root_data,
    )

    # Single precision boundary: BOI and frequency shift above remain complex128.
    x_processed = np.asarray(run["x_processed"], dtype=np.complex64)
    x2 = (x_processed * x_processed).astype(np.complex64, copy=False)
    x4 = (x2 * x2).astype(np.complex64, copy=False)
    fft_x2_mag_db = _fft_mag_db_float32(x2, nfft=int(cfg["FFT_NFFT"]))
    fft_x4_mag_db = _fft_mag_db_float32(x4, nfft=int(cfg["FFT_NFFT"]))
    x2_freq, x2_peak_db = _top_peaks_float32(fft_x2_mag_db)
    x4_freq, x4_peak_db = _top_peaks_float32(fft_x4_mag_db)
    lr8_20 = _lr8_20_float32(x2_freq, x2_peak_db, x4_freq, x4_peak_db)
    x4_top1_f = np.float32(x4_freq[0])
    cfo_estimate = np.float32(_wrap_float32(x4_top1_f / np.float32(4.0)))
    rs_estimate = _candidate_only_rs_float32(fft_x4_mag_db, x4_top1_f)
    preprocess, power = _boi_scalar_payload(run)

    return {
        "version": "boi_shift_2fft_fe_float32_v2",
        "meta": {
            "dataset_name": str(dataset_name),
            "signal_index": int(signal_index),
            "batch_id": int(batch_id),
            "batch_local_index": int(signal_index - batch_start),
            "global_index": int(signal_index),
        },
        "truth": _truth_dict(raw_truth_df, signal_index),
        "truth_line": raw_truth_lines[int(signal_index)],
        "preprocess": preprocess,
        "power": power,
        "x_processed": x_processed,
        "fft_x4_mag_db": fft_x4_mag_db,
        "lr8_20": lr8_20,
        "x4_top1_f": x4_top1_f,
        "cfo_estimate": cfo_estimate,
        "rs_estimate_candidate_only_124": rs_estimate,
    }


def _negative_snr_skip(
    dataset_name: str,
    batch_id: int,
    signal_index: int,
    truth: dict,
    truth_line: str,
) -> dict:
    return {
        "dataset_name": str(dataset_name),
        "batch_id": int(batch_id),
        "signal_index": int(signal_index),
        "truth": truth,
        "truth_line": truth_line,
        "truth_snr": np.float32(truth["InbandSNRdB"]),
        "skip_reason": "negative_truth_snr_2018",
        "error": "SKIP_NEGATIVE_TRUTH_SNR",
        "traceback": "",
    }


def generate_twofft_fe_float32_pt_files(cfg: dict) -> None:
    """Generate flat, year-prefixed batch PT files for one or both datasets."""
    if base.torch is None:
        raise RuntimeError("PyTorch is required for torch.save().")
    if int(cfg["FFT_NFFT"]) != NFFT:
        raise ValueError(f"This FE definition requires FFT_NFFT={NFFT}")
    if str(cfg["FFT_WINDOW"]).lower() != "hann":
        raise ValueError("This FE definition requires FFT_WINDOW='hann'")

    selection = str(cfg["DATASET_SELECT"]).lower()
    if selection == "both":
        dataset_list = ["2018", "2022"]
    elif selection in {"2018", "2022"}:
        dataset_list = [selection]
    else:
        raise ValueError("DATASET_SELECT must be '2018', '2022', or 'both'")

    out_root = Path(base.ensure_dir(cfg["OUT_ROOT"]))
    for dataset_name in dataset_list:
        local_cfg = dict(cfg)
        local_cfg["DATASET_NAME"] = dataset_name
        root_data, label_path, boi_truth_df = base.set_dataset(
            dataset_name=dataset_name,
            root_map=local_cfg["ROOT_MAP"],
            label_map=local_cfg["LABEL_MAP"],
        )
        raw_truth_df, raw_truth_lines = _read_raw_truth(label_path)

        n_per_batch = int(local_cfg["N_PER_BATCH"])
        n_total = int(local_cfg["N_TOTAL"])
        start_pt = max(1, int(local_cfg["START_PT"]))
        end_pt = min(int(local_cfg["END_PT"]), int(local_cfg["N_BATCHES"]))
        if start_pt > end_pt:
            raise ValueError(f"Invalid PT range {start_pt}..{end_pt}")

        print("=" * 80)
        print(f"Two-FFT float32 FE generation | dataset={dataset_name}")
        print(f"truth={label_path}")
        print(f"data={root_data}")
        print(f"flat output={out_root}")
        print("=" * 80)

        for batch_id in range(start_pt, end_pt + 1):
            batch_start = (batch_id - 1) * n_per_batch + 1
            batch_end = min(batch_id * n_per_batch, n_total)
            save_path = out_root / f"{dataset_name}_batch_{batch_id:02d}_2fft_fe_float32.pt"
            if bool(local_cfg.get("SKIP_EXISTING", False)) and save_path.exists():
                print(f"Skip existing: {save_path}")
                continue

            samples: list[dict] = []
            skipped: list[dict] = []
            errors: list[dict] = []
            iterator = range(batch_start, batch_end + 1)
            if base.tqdm is not None:
                iterator = base.tqdm(
                    iterator,
                    desc=f"{dataset_name} 2FFT batch {batch_id:02d}",
                    total=batch_end - batch_start + 1,
                    dynamic_ncols=True,
                )

            for signal_index in iterator:
                truth = _truth_dict(raw_truth_df, signal_index)
                truth_line = raw_truth_lines[int(signal_index)]
                # This branch is deliberately before .tim loading and all BOI/FFT work.
                if dataset_name == "2018" and truth["InbandSNRdB"] < np.float32(0.0):
                    skipped.append(_negative_snr_skip(
                        dataset_name, batch_id, signal_index, truth, truth_line
                    ))
                    continue
                try:
                    sample = build_twofft_fe_payload(
                        signal_index=signal_index,
                        cfg=local_cfg,
                        boi_truth_df=boi_truth_df,
                        raw_truth_df=raw_truth_df,
                        raw_truth_lines=raw_truth_lines,
                        root_data=root_data,
                        dataset_name=dataset_name,
                        batch_id=batch_id,
                        batch_start=batch_start,
                    )
                    samples.append(sample)
                except Exception as exc:
                    errors.append({
                        "dataset_name": dataset_name,
                        "batch_id": int(batch_id),
                        "signal_index": int(signal_index),
                        "truth": truth,
                        "truth_line": truth_line,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
                finally:
                    gc.collect()

            batch_pack = {
                "version": "boi_shift_2fft_fe_float32_batch_v2",
                "dataset_name": dataset_name,
                "batch_id": int(batch_id),
                "start_signal_index": int(batch_start),
                "end_signal_index": int(batch_end),
                "num_samples_expected": int(batch_end - batch_start + 1),
                "num_samples_saved": int(len(samples)),
                "num_samples_skipped": int(len(skipped)),
                "num_errors": int(len(errors)),
                "truth_columns": list(TRUTH_COLUMNS),
                "lr8_20_feature_names": list(M8_FEATURES),
                "fixed_rs_candidates": FIXED_RS_CANDIDATES.copy(),
                "precision": {
                    "boi_detection_and_shift": "float64/complex128",
                    "boi_preprocess_and_power_metadata": "float64 scalars",
                    "x_processed": "complex64",
                    "post_boi_computation_and_real_arrays": "float32",
                    "fft_implementation": "scipy.fft (complex64 preserving)",
                    "fft_x4_db_floor": None,
                },
                "fft": {
                    "nfft": NFFT,
                    "window": "hann_float32",
                    "stored_spectrum": "complete relative magnitude dB",
                },
                "samples": samples,
                "skipped": skipped,
                "errors": errors,
            }
            base.torch.save(batch_pack, save_path)
            print(
                f"Saved {save_path.name}: samples={len(samples)}, "
                f"skipped={len(skipped)}, errors={len(errors)}"
            )
            del samples, skipped, errors, batch_pack
            gc.collect()

    print(f"Done. All selected datasets share flat folder: {out_root.resolve()}")


def inspect_twofft_fe_pt(pt_path: str | Path) -> dict:
    if base.torch is None:
        raise RuntimeError("PyTorch is required for inspection")
    pack = base.torch.load(pt_path, map_location="cpu", weights_only=False)
    sample = pack["samples"][0] if pack.get("samples") else {}
    return {
        "path": str(pt_path),
        "version": pack.get("version"),
        "num_samples_saved": pack.get("num_samples_saved"),
        "num_samples_skipped": pack.get("num_samples_skipped"),
        "num_errors": pack.get("num_errors"),
        "sample_keys": list(sample.keys()),
        "truth_keys": list(sample.get("truth", {}).keys()),
        "preprocess_keys": list(sample.get("preprocess", {}).keys()),
        "power_keys": list(sample.get("power", {}).keys()),
        "p_boi_inband_signal": sample.get("power", {}).get("p_boi_inband_signal"),
        "x_processed_shape": tuple(sample["x_processed"].shape) if sample else None,
        "x_processed_dtype": str(sample["x_processed"].dtype) if sample else None,
        "fft_x4_shape": tuple(sample["fft_x4_mag_db"].shape) if sample else None,
        "fft_x4_dtype": str(sample["fft_x4_mag_db"].dtype) if sample else None,
        "lr8_20_shape": tuple(sample["lr8_20"].shape) if sample else None,
        "lr8_20_dtype": str(sample["lr8_20"].dtype) if sample else None,
        "scalar_types": {
            key: type(sample[key]).__name__
            for key in ("x4_top1_f", "cfo_estimate", "rs_estimate_candidate_only_124")
            if sample
        },
    }
