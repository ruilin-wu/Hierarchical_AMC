#!/usr/bin/env python3
"""Generate compact BOI+CFO-shift payloads with only selected FFT feature vectors.

This module is additive and does not modify the existing large FFT payload
generator.  It reuses the original BOI+CFO-shift and FFT code, extracts the
same blind feature columns used by the notebooks, then drops the FFT spectra
and top-peak structures before saving.
"""

from __future__ import annotations

import gc
import os
import traceback
from pathlib import Path

import numpy as np

from . import pre_boi_fft3_generation_base_from_notebook as base


META_COLS = {
    "year",
    "pt_name",
    "batch_id",
    "signal_index",
    "signal_type",
    "label5",
    "snr",
}

LR7_FEATURES = [
    "x4_top2_db",
    "x4_top1_af",
    "x4_top1_db",
    "x4_top1_minus_top5_db",
    "x2_top4_db",
    "x2_top1_db",
    "abs_x2_top1_minus_top5_db",
]

LR55_FEATURES = [
    "x4_top1_minus_top2_db",
    "x4_top2_db",
    "x4_top3_db",
    "x4_top1_af",
    "x4_top1_db",
    "x4_int_grid_3rd_db",
    "x4_top1_minus_top5_db",
    "x4_top4_db",
    "x2_nonzero_count_gt_m10",
    "x2_nonzero_count_gt_m20",
    "x4_int_grid_2nd_db",
    "x2_top4_db",
    "x2_top1_minus_top5_db",
    "x2_top3_db",
    "x2_half_grid_best_db",
    "x2_half_grid_2nd_db",
    "x4_nonzero_count_gt_m10",
    "x2_top1_db",
    "abs_x2_top4_db",
    "abs_x2_top1_minus_top5_db",
    "x4_int_grid_count_gt_m10",
    "x2_top1_minus_top2_db",
    "x2_top2_db",
    "x4_top2_af",
    "x4_half_grid_count_gt_m10",
    "x2_half_grid_count_gt_m20",
    "x2_half_grid_count_gt_m30",
    "x2_top1_af",
    "abs_x2_nonzero_count_gt_m30",
    "x4_top1_af_over_rs",
    "x2_top3_af",
    "x2_int_grid_count_gt_m20",
    "x2_int_grid_2nd_db",
    "x2_top2_af",
    "abs_x2_nonzero_count_gt_m20",
    "abs_x2_top1_minus_top2_db",
    "abs_x2_nonzero_best_db",
    "abs_x2_top2_db",
    "x2_int_grid_count_gt_m30",
    "x2_half_grid_count_gt_m10",
    "x4_top3_af",
    "abs_x2_top3_db",
    "x4_half_grid_3rd_db",
    "x2_nonzero_count_gt_m30",
    "x4_nonzero_count_gt_m20",
    "x2_nonzero_best_db",
    "abs_x2_top4_af",
    "x4_half_grid_count_gt_m20",
    "x2_half_grid_3rd_db",
    "x4_top4_af",
    "x4_half_minus_int_top3_db",
    "x4_half_grid_best_db",
    "abs_x2_half_grid_mean_db",
    "abs_x2_int_grid_mean_db",
    "x4_half_grid_mean_top3_db",
]


def scheme_features(x4_k: int, abs_ranks: list[int]) -> list[str]:
    features: list[str] = []
    features += [f"x4_top{i}_f" for i in range(1, int(x4_k) + 1)]
    features += [f"x4_top{i}_db" for i in range(1, int(x4_k) + 1)]
    features += [f"abs_x2_top{i}_af" for i in abs_ranks]
    features += ["rs_est"]
    return features


RS18_FEATURES = scheme_features(
    x4_k=3,
    abs_ranks=[2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16],
)
RS28_FEATURES = scheme_features(
    x4_k=4,
    abs_ranks=list(range(2, 21)),
)
RS20_FEATURES = (
    ["abs_x2_top1_minus_top2_db"]
    + [f"abs_x2_top{i}_af" for i in range(2, 21, 2)]
    + ["rs_est"]
    + [f"x4_top{i}_f" for i in range(1, 5)]
    + [f"x4_top{i}_db" for i in range(1, 5)]
)
CFO_FEATURES = ["x4_top1_f"]
NFFT = 32768


def wrap_freq_scalar(f: float) -> float:
    return ((float(f) + 0.5) % 1.0) - 0.5


def wrap_freq(f: float) -> float:
    return ((float(f) + 0.5) % 1.0) - 0.5


def idx_for_freq(f: float, nfft: int = NFFT) -> int:
    f = wrap_freq(f)
    idx = int(round(f * nfft + nfft / 2))
    return max(0, min(nfft - 1, idx))


def sample_mag_db_near(mag_db, target_freq: float, bins: int = 3) -> float:
    idx = idx_for_freq(target_freq)
    lo = max(0, idx - bins)
    hi = min(len(mag_db), idx + bins + 1)
    vals = np.asarray(mag_db[lo:hi], dtype=np.float32)
    if vals.size == 0:
        return -120.0
    return float(np.nanmax(vals))


def label5(signal_type: str) -> str:
    sig = str(signal_type).lower()
    if sig == "bpsk":
        return "BPSK"
    if sig == "msk":
        return "MSK"
    if sig == "dqpsk":
        return "DQPSK"
    if sig == "8psk":
        return "8PSK"
    if sig in {"qpsk", "16qam", "64qam", "256qam"}:
        return "QPSK-like"
    raise ValueError(f"Unsupported signal_type={signal_type!r}")


def get_top_peaks(sample: dict, branch: str):
    return sample["fft"][branch]["top_peaks"]


def estimate_rs_from_abs_x2(sample: dict) -> float:
    candidates = []
    for peak in get_top_peaks(sample, "abs_x2"):
        f = abs(float(peak["freq"]))
        db = float(peak["mag_db"])
        if f > 5 / NFFT and f < 0.49:
            candidates.append((db, f))
    if not candidates:
        return float("nan")
    candidates.sort(reverse=True)
    return float(candidates[0][1])


def get_truth_shifted_cfo_for_debug(sample: dict) -> float:
    preprocess = sample.get("preprocess", {})
    if "truth_guide_cfo" in preprocess:
        return float(preprocess["truth_guide_cfo"])

    truth = sample["truth"]
    if "carrier_offset_label" in truth and "center_shift_hz" in preprocess:
        return float(truth["carrier_offset_label"]) - float(preprocess["center_shift_hz"])

    return float("nan")


def extract_sample_features(sample: dict, year: str, pt_name: str) -> dict:
    """Extract the same 445 blind FFT pattern features from the current sample."""
    meta = sample["meta"]
    truth = sample["truth"]
    preprocess = sample.get("preprocess", {})
    sig = str(meta["signal_type"]).lower()
    snr = float(truth["inband_snr_db_label"])

    row = {
        "year": str(year),
        "pt_name": pt_name,
        "batch_id": int(meta.get("batch_id", -1)),
        "signal_index": int(meta.get("signal_index", meta.get("global_index", -1))),
        "signal_type": sig.upper(),
        "label5": label5(sig),
        "snr": snr,
    }

    rs = estimate_rs_from_abs_x2(sample)
    row["rs_est"] = rs

    row["rs_truth_for_debug"] = float(truth["symbol_rate"])
    row["rs_abs_err_for_debug"] = (
        abs(rs - float(truth["symbol_rate"])) if np.isfinite(rs) else np.nan
    )
    row["cfo_truth_shifted"] = get_truth_shifted_cfo_for_debug(sample)
    row["carrier_offset_truth_for_debug"] = float(
        truth.get("carrier_offset_label", np.nan)
    )
    row["boi_center_shift_for_debug"] = float(
        preprocess.get("center_shift_hz", np.nan)
    )

    for branch in ["abs_x2", "x2", "x4"]:
        peaks = get_top_peaks(sample, branch)

        for i in range(20):
            if i < len(peaks):
                f = float(peaks[i]["freq"])
                db = float(peaks[i]["mag_db"])
            else:
                f = np.nan
                db = -120.0

            row[f"{branch}_top{i + 1}_f"] = f
            row[f"{branch}_top{i + 1}_af"] = abs(f) if np.isfinite(f) else np.nan
            row[f"{branch}_top{i + 1}_db"] = db
            row[f"{branch}_top{i + 1}_f_over_rs"] = (
                f / rs if np.isfinite(rs) and rs > 0 else np.nan
            )
            row[f"{branch}_top{i + 1}_af_over_rs"] = (
                abs(f) / rs if np.isfinite(rs) and rs > 0 else np.nan
            )

        dbs = np.array([float(p["mag_db"]) for p in peaks], dtype=float)
        freqs = np.array([float(p["freq"]) for p in peaks], dtype=float)
        abs_freqs = np.abs(freqs)
        nonzero_dbs = dbs[abs_freqs > 5 / NFFT]

        row[f"{branch}_nonzero_best_db"] = (
            float(np.max(nonzero_dbs)) if nonzero_dbs.size else -120.0
        )
        row[f"{branch}_nonzero_count_gt_m10"] = (
            int(np.sum(nonzero_dbs > -10)) if nonzero_dbs.size else 0
        )
        row[f"{branch}_nonzero_count_gt_m20"] = (
            int(np.sum(nonzero_dbs > -20)) if nonzero_dbs.size else 0
        )
        row[f"{branch}_nonzero_count_gt_m30"] = (
            int(np.sum(nonzero_dbs > -30)) if nonzero_dbs.size else 0
        )

        if len(dbs):
            row[f"{branch}_top1_minus_top2_db"] = (
                float(dbs[0] - dbs[1]) if len(dbs) > 1 else 120.0
            )
            row[f"{branch}_top1_minus_top5_db"] = float(
                dbs[0] - dbs[min(4, len(dbs) - 1)]
            )

        sym = []
        for f, db in zip(freqs, dbs):
            if abs(f) <= 5 / NFFT:
                continue
            j = int(np.argmin(np.abs(freqs + f))) if len(freqs) else 0
            sym.append(abs(db - dbs[j]) + 40 * abs(freqs[j] + f))
        row[f"{branch}_sym_score_mean"] = float(np.mean(sym)) if sym else 99.0
        row[f"{branch}_sym_score_min"] = float(np.min(sym)) if sym else 99.0

    if np.isfinite(rs) and rs > 0:
        for branch in ["abs_x2", "x2", "x4"]:
            mag_db = sample["fft"][branch]["mag_db"]
            peaks = get_top_peaks(sample, branch)
            center = 0.0 if branch == "abs_x2" else float(peaks[0]["freq"])
            row[f"{branch}_center_f"] = center

            for mode, offset in [("int", 0.0), ("half", 0.5 * rs)]:
                vals = []
                for k in range(-5, 6):
                    target = k * rs
                    if branch != "abs_x2":
                        target += center + offset
                    value = sample_mag_db_near(mag_db, target, bins=3)
                    row[f"{branch}_{mode}_grid_k{k:+d}_db"] = value
                    vals.append(value)

                vals = np.array(vals, dtype=float)
                vals_sorted = np.sort(vals)[::-1]
                row[f"{branch}_{mode}_grid_best_db"] = float(vals_sorted[0])
                row[f"{branch}_{mode}_grid_2nd_db"] = float(vals_sorted[1])
                row[f"{branch}_{mode}_grid_3rd_db"] = float(vals_sorted[2])
                row[f"{branch}_{mode}_grid_mean_top3_db"] = float(
                    np.mean(vals_sorted[:3])
                )
                row[f"{branch}_{mode}_grid_mean_db"] = float(np.mean(vals))
                row[f"{branch}_{mode}_grid_count_gt_m10"] = int(np.sum(vals > -10))
                row[f"{branch}_{mode}_grid_count_gt_m20"] = int(np.sum(vals > -20))
                row[f"{branch}_{mode}_grid_count_gt_m30"] = int(np.sum(vals > -30))

            row[f"{branch}_half_minus_int_top3_db"] = (
                row[f"{branch}_half_grid_mean_top3_db"]
                - row[f"{branch}_int_grid_mean_top3_db"]
            )

    return row


def get_blind445_feature_columns_from_row(row: dict) -> list[str]:
    forbidden = [
        c for c in row.keys()
        if "truth" in str(c).lower() or "debug" in str(c).lower()
    ]
    return [c for c in row.keys() if c not in META_COLS and c not in forbidden]


def vector_from_row(row: dict, features: list[str]) -> np.ndarray:
    missing = [name for name in features if name not in row]
    if missing:
        raise KeyError(f"Missing feature columns: {missing[:10]}")
    return np.asarray([row[name] for name in features], dtype=np.float32)


def build_vectors_from_full_payload(
    full_payload: dict,
    year: str,
    pt_name_for_feature_order: str,
    blind445_features: list[str] | None = None,
) -> tuple[dict, dict, dict]:
    """Return vector arrays, feature-name lists, and the full feature row."""
    row = extract_sample_features(
        sample=full_payload,
        year=str(year),
        pt_name=str(pt_name_for_feature_order),
    )

    if blind445_features is None:
        blind445_features = get_blind445_feature_columns_from_row(row)
        if len(blind445_features) != 445:
            raise ValueError(f"Expected 445 blind features, got {len(blind445_features)}")

    names = {
        "lr445": list(blind445_features),
        "lr55": list(LR55_FEATURES),
        "lr7": list(LR7_FEATURES),
        "rs18": list(RS18_FEATURES),
        "rs28": list(RS28_FEATURES),
        "rs20": list(RS20_FEATURES),
        "cfo1": list(CFO_FEATURES),
    }
    vectors = {key: vector_from_row(row, value) for key, value in names.items()}

    if names["rs18"][0] != "x4_top1_f" or names["rs28"][0] != "x4_top1_f":
        raise AssertionError("x4_top1_f must be the first Rs/CFO estimator feature")

    return vectors, names, row


def make_compact_payload(full_payload: dict, vectors: dict, feature_row: dict) -> dict:
    meta = dict(full_payload["meta"])

    truth = dict(full_payload["truth"])
    cfo_truth_shifted = float(full_payload["preprocess"]["truth_guide_cfo"])
    truth["rs_truth"] = float(truth["symbol_rate"])
    truth["cfo_truth"] = float(truth["carrier_offset_label"])
    truth["snr_truth_db"] = float(truth["inband_snr_db_label"])
    truth["cfo_truth_shifted"] = cfo_truth_shifted

    preprocess = dict(full_payload["preprocess"])
    preprocess["cfo_truth_shifted"] = cfo_truth_shifted

    power = dict(full_payload["power"])
    if "p_signal_boi_local" in power:
        power["p_boi_inband_signal"] = float(power.pop("p_signal_boi_local"))

    x4_top1_f = float(feature_row["x4_top1_f"])
    rs_est = float(feature_row["rs_est"])

    return {
        "version": "boi_shift_fft3_compact_vectors_v1",
        "meta": meta,
        "truth": truth,
        "preprocess": preprocess,
        "power": power,
        "fft_config": {
            "nfft": int(full_payload["fft_config"]["nfft"]),
            "window": str(full_payload["fft_config"]["window"]),
            "fftshift": True,
            "computed_branches": ["abs_x2", "x2", "x4"],
            "saved_fft_mag_db": False,
            "saved_fft_top_peaks": False,
            "saved_fft_payload": False,
            "mag_format_used_for_features": "db_relative_to_max",
        },
        "x_processed": np.asarray(full_payload["x_processed"], dtype=np.complex64),
        "vectors": {
            "lr445": np.asarray(vectors["lr445"], dtype=np.float32),
            "lr55": np.asarray(vectors["lr55"], dtype=np.float32),
            "lr7": np.asarray(vectors["lr7"], dtype=np.float32),
            "rs18": np.asarray(vectors["rs18"], dtype=np.float32),
            "rs28": np.asarray(vectors["rs28"], dtype=np.float32),
            "rs20": np.asarray(vectors["rs20"], dtype=np.float32),
            "cfo1": np.asarray(vectors["cfo1"], dtype=np.float32),
        },
        "summary": {
            "rs_est_from_features": rs_est,
            "cfo_est_from_x4_top1_f": wrap_freq_scalar(x4_top1_f / 4.0),
            "x4_top1_f": x4_top1_f,
            "candidate_pattern_hint": str(
                full_payload.get("summary", {}).get("candidate_pattern_hint", "")
            ),
        },
    }


def run_one_and_build_compact_payload(
    signal_index: int,
    cfg: dict,
    truth_df,
    root_data: str,
    year: str,
    batch_id: int,
    batch_start: int,
    pt_name_for_feature_order: str,
    blind445_features: list[str] | None = None,
) -> tuple[dict, dict]:
    full_payload = None
    try:
        full_payload = base.run_one_and_build_payload(
            signal_index=int(signal_index),
            cfg=cfg,
            truth_df=truth_df,
            root_data=root_data,
        )
        full_payload["meta"]["batch_id"] = int(batch_id)
        full_payload["meta"]["batch_local_index"] = int(signal_index - batch_start)
        full_payload["meta"]["global_index"] = int(signal_index)

        vectors, names, row = build_vectors_from_full_payload(
            full_payload=full_payload,
            year=year,
            pt_name_for_feature_order=pt_name_for_feature_order,
            blind445_features=blind445_features,
        )
        compact_payload = make_compact_payload(
            full_payload=full_payload,
            vectors=vectors,
            feature_row=row,
        )
        return compact_payload, names
    finally:
        if isinstance(full_payload, dict):
            full_payload.pop("fft", None)
            full_payload.pop("x_processed", None)
        del full_payload
        gc.collect()
        if base.torch is not None and base.torch.cuda.is_available():
            base.torch.cuda.empty_cache()


def save_compact_batch_payloads(
    dataset_name: str,
    batch_id: int,
    batch_start: int,
    batch_end: int,
    payloads: list[dict],
    errors: list[dict],
    save_path: str | Path,
    cfg: dict,
    vector_feature_names: dict,
) -> Path:
    if base.torch is None:
        raise RuntimeError("PyTorch is required for torch.save().")

    payloads = sorted(payloads, key=lambda x: int(x["meta"]["signal_index"]))
    batch_pack = {
        "version": "boi_shift_fft3_compact_vectors_batch_v1",
        "dataset_name": str(dataset_name),
        "batch_id": int(batch_id),
        "batch_start_index": int(batch_start),
        "batch_end_index": int(batch_end),
        "n_per_batch": int(cfg["N_PER_BATCH"]),
        "num_samples_expected": int(batch_end - batch_start + 1),
        "num_samples_saved": int(len(payloads)),
        "num_errors": int(len(errors)),
        "vector_feature_names": vector_feature_names,
        "samples": payloads,
        "errors": errors,
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    base.torch.save(batch_pack, save_path)
    return save_path


def generate_compact_vector_pt_files(cfg: dict) -> None:
    if base.torch is None:
        raise RuntimeError("PyTorch is required for torch.save().")

    dataset_select = str(cfg["DATASET_SELECT"]).lower()
    if dataset_select == "both":
        dataset_list = ["2018", "2022"]
    elif dataset_select in ["2018", "2022"]:
        dataset_list = [dataset_select]
    else:
        raise ValueError("DATASET_SELECT must be '2018', '2022', or 'both'.")

    out_root = base.ensure_dir(cfg["OUT_ROOT"])
    blind445_features = None

    global_vector_feature_names = None
    all_error_rows = []

    for dataset_name in dataset_list:
        local_cfg = dict(cfg)
        local_cfg["DATASET_NAME"] = str(dataset_name)

        root_data, label_path, truth_df = base.set_dataset(
            dataset_name=dataset_name,
            root_map=local_cfg["ROOT_MAP"],
            label_map=local_cfg["LABEL_MAP"],
        )

        n_per_batch = int(local_cfg["N_PER_BATCH"])
        n_total = int(local_cfg["N_TOTAL"])
        n_batches = int(local_cfg["N_BATCHES"])
        start_pt = max(1, int(local_cfg["START_PT"]))
        end_pt = min(int(local_cfg["END_PT"]), n_batches)
        if start_pt > end_pt:
            raise ValueError(f"Invalid PT batch range: {start_pt}..{end_pt}")

        dataset_out_dir = base.ensure_dir(os.path.join(out_root, dataset_name))
        print("=" * 80)
        print("Compact BOI + Shift + FFT3 vector PT generation")
        print("=" * 80)
        print(f"dataset_name     : {dataset_name}")
        print(f"label_path       : {label_path}")
        print(f"root_data        : {root_data}")
        print(f"out_dir          : {dataset_out_dir}")
        print(f"nfft             : {local_cfg['FFT_NFFT']}")
        print(f"window           : {local_cfg['FFT_WINDOW']}")
        print("=" * 80)

        for batch_id in range(start_pt, end_pt + 1):
            batch_start = (batch_id - 1) * n_per_batch + 1
            batch_end = min(batch_id * n_per_batch, n_total)
            compact_pt_name = (
                f"{dataset_name}_batch_{batch_id:02d}_boi_shift_fft3_compact_vectors.pt"
            )
            reference_pt_name = (
                f"{dataset_name}_batch_{batch_id:02d}_boi_shift_fft3_cfpattern.pt"
            )
            batch_save_path = os.path.join(dataset_out_dir, compact_pt_name)

            if local_cfg.get("SKIP_EXISTING", False) and os.path.exists(batch_save_path):
                print(f"Skip existing: {batch_save_path}")
                continue

            print("=" * 80)
            print(f"Start {dataset_name} batch {batch_id:02d}")
            print(f"Signal index range: {batch_start} to {batch_end}")
            print(f"Output: {batch_save_path}")
            print("=" * 80)

            payloads = []
            errors = []
            iterator = range(batch_start, batch_end + 1)
            if base.tqdm is not None:
                iterator = base.tqdm(
                    iterator,
                    desc=f"{dataset_name} compact batch {batch_id:02d}",
                    total=batch_end - batch_start + 1,
                    dynamic_ncols=True,
                )

            for signal_index in iterator:
                try:
                    skip_sample, truth_for_skip = base.should_skip_negative_truth_snr(
                        dataset_name=dataset_name,
                        signal_index=signal_index,
                        truth_df=truth_df,
                    )
                    if skip_sample:
                        err_row = base.make_negative_snr_skip_row(
                            dataset_name=dataset_name,
                            signal_index=signal_index,
                            truth=truth_for_skip,
                            batch_id=batch_id,
                        )
                        errors.append(err_row)
                        all_error_rows.append(err_row)
                        if base.tqdm is not None:
                            iterator.set_postfix_str(
                                f"SKIP neg SNR #{signal_index}: "
                                f"{err_row['truth_snr']:.3f} dB | saved={len(payloads)} err={len(errors)}"
                            )
                        continue

                    payload, names = run_one_and_build_compact_payload(
                        signal_index=signal_index,
                        cfg=local_cfg,
                        truth_df=truth_df,
                        root_data=root_data,
                        year=dataset_name,
                        batch_id=batch_id,
                        batch_start=batch_start,
                        pt_name_for_feature_order=reference_pt_name,
                        blind445_features=blind445_features,
                    )
                    if global_vector_feature_names is None:
                        global_vector_feature_names = names
                    blind445_features = names["lr445"]
                    payloads.append(payload)

                    if base.tqdm is not None:
                        iterator.set_postfix_str(
                            f"ok #{signal_index} {payload['meta']['signal_type']} "
                            f"| saved={len(payloads)} err={len(errors)}"
                        )
                except Exception as exc:
                    err_row = {
                        "dataset_name": str(dataset_name),
                        "batch_id": int(batch_id),
                        "signal_index": int(signal_index),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(err_row)
                    all_error_rows.append(err_row)
                    if base.tqdm is not None:
                        iterator.set_postfix_str(
                            f"ERROR #{signal_index}: {type(exc).__name__} "
                            f"| saved={len(payloads)} err={len(errors)}"
                        )
                    else:
                        print(f"ERROR | {dataset_name} #{signal_index}: {repr(exc)}")

                gc.collect()
                if base.torch.cuda.is_available():
                    base.torch.cuda.empty_cache()

            if global_vector_feature_names is None:
                global_vector_feature_names = {
                    "lr445": list(blind445_features) if blind445_features is not None else [],
                    "lr55": list(LR55_FEATURES),
                    "lr7": list(LR7_FEATURES),
                    "rs18": list(RS18_FEATURES),
                    "rs28": list(RS28_FEATURES),
                    "rs20": list(RS20_FEATURES),
                    "cfo1": list(CFO_FEATURES),
                }

            save_compact_batch_payloads(
                dataset_name=dataset_name,
                batch_id=batch_id,
                batch_start=batch_start,
                batch_end=batch_end,
                payloads=payloads,
                errors=errors,
                save_path=batch_save_path,
                cfg=local_cfg,
                vector_feature_names=global_vector_feature_names,
            )
            print(
                f"Saved {dataset_name} compact batch {batch_id:02d}: "
                f"samples={len(payloads)} | errors={len(errors)}"
            )

            del payloads
            del errors
            gc.collect()
            if base.torch.cuda.is_available():
                base.torch.cuda.empty_cache()

    print(f"Done. Compact PT files saved under: {os.path.abspath(out_root)}")
    print(f"Errors are stored inside each .pt batch only; no CSV was written. total_errors={len(all_error_rows)}")


def inspect_compact_pt(compact_pt_path: str | Path) -> dict:
    if base.torch is None:
        raise RuntimeError("PyTorch is required for inspection.")
    pack = base.torch.load(compact_pt_path, map_location="cpu", weights_only=False)
    sample = pack["samples"][0] if pack["samples"] else {}
    vectors = sample.get("vectors", {})
    return {
        "path": str(compact_pt_path),
        "version": pack.get("version"),
        "num_samples_saved": pack.get("num_samples_saved"),
        "num_errors": pack.get("num_errors"),
        "sample_keys": list(sample.keys()),
        "has_fft_key": "fft" in sample,
        "x_processed_shape": tuple(sample.get("x_processed", np.empty(0)).shape)
        if sample else None,
        "x_processed_dtype": str(sample.get("x_processed", np.empty(0)).dtype)
        if sample else None,
        "vector_shapes": {key: tuple(value.shape) for key, value in vectors.items()},
        "power_keys": list(sample.get("power", {}).keys()) if sample else [],
    }
