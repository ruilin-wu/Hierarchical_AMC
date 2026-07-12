#!/usr/bin/env python3
"""Blind Rs/CFO estimation helpers for QPSK-like CC matrix generation.

The legacy Rs logic matches the `rs28_x4top4_absx2top20` path used in
Rs-CFO-estimation-Rs18-28-60-CFO-ptvectors.ipynb:
  - confident x4 triplets are used first
  - otherwise abs_x2 candidate bins from ranks 2..20 are used
  - final fallback is the `rs_est` value in the rs28 vector

The rs20 logic is the strict single-signal hand-written rule explored in
pre_Rs-CFO-estimation-QPSKlike-pkl-strict-hand-unsupervised-more-features.ipynb.
It uses only the current sample's rs20 vector and does not use candidate bins
or any dataset-level statistics.

CFO is estimated as wrap_freq(cfo1[0] / 4).
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

NFFT = 32768
BIN = 1.0 / NFFT
RS_ABS_RANKS = list(range(2, 21))
RS20_ABS_RANKS = list(range(2, 21, 2))
RS20_FEATURES = (
    ["abs_x2_top1_minus_top2_db"]
    + [f"abs_x2_top{i}_af" for i in RS20_ABS_RANKS]
    + ["rs_est"]
    + [f"x4_top{i}_f" for i in range(1, 5)]
    + [f"x4_top{i}_db" for i in range(1, 5)]
)
RS20_TRUST_ABS2_DB_GAP_MAX = 23.941201956505633
RS20_LATE_EVEN_MIN_BIN = 1800
TARGET_SIGNAL_TYPES = {"qpsk", "16qam", "64qam", "256qam"}
NEGATIVE_SNR_ERROR = "SKIP_NEGATIVE_SNR"


def wrap_freq(x):
    return ((np.asarray(x, dtype=float) + 0.5) % 1.0) - 0.5


def wrap_freq_scalar(f):
    return ((float(f) + 0.5) % 1.0) - 0.5


def sample_signal_type(sample):
    return str(sample.get("meta", {}).get("signal_type", "")).lower()


def sample_snr(sample):
    truth = sample.get("truth", {})
    return float(truth.get("inband_snr_db_label", truth.get("snr_truth_db", np.nan)))


def is_qpsklike_sample(sample, target_signal_types=TARGET_SIGNAL_TYPES):
    return sample_signal_type(sample) in set(target_signal_types)


def is_negative_snr_skip(sample, year, drop_2018_negative_snr=True):
    return (
        str(year) == "2018"
        and bool(drop_2018_negative_snr)
        and sample_snr(sample) < 0.0
    )


def load_pack(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def validate_vector_names(pack, rs_vector_key="rs28", cfo_vector_key="cfo1"):
    names = pack["vector_feature_names"]
    if rs_vector_key not in names:
        raise KeyError(f"Missing vector_feature_names[{rs_vector_key!r}]")
    if cfo_vector_key not in names:
        raise KeyError(f"Missing vector_feature_names[{cfo_vector_key!r}]")

    rs_names = list(names[rs_vector_key])
    cfo_names = list(names[cfo_vector_key])
    if rs_vector_key == "rs28" and len(rs_names) != 28:
        raise ValueError(f"Expected rs28 to have 28 feature names, got {len(rs_names)}")
    if rs_vector_key == "rs20":
        if len(rs_names) != 20:
            raise ValueError(f"Expected rs20 to have 20 feature names, got {len(rs_names)}")
        if rs_names != RS20_FEATURES:
            raise ValueError(f"Unexpected rs20 feature names: {rs_names}")
    if cfo_names != ["x4_top1_f"]:
        raise ValueError(f"Unexpected cfo1 names: {cfo_names}")
    return rs_names, cfo_names


def validate_compact_pack_schema(
    pack,
    rs_vector_key="rs28",
    cfo_vector_key="cfo1",
    lr_vector_key="lr445",
    require_lr_vector=True,
):
    """Validate compact-vector PT fields consumed by the QPSK-like CC flow.

    Required data contract:
      - sample["x_processed"] is the BOI+CFO-shifted signal used by CC matrix
      - sample["vectors"][rs_vector_key] is the Rs vector, commonly rs28 or rs20
      - sample["vectors"]["cfo1"] is a 1-value vector with feature name x4_top1_f
      - sample["vectors"]["lr445"] exists when require_lr_vector=True

    The lr445 vector is validated because it is part of the compact payload used
    by the surrounding QPSK-like flow.  The Rs/CFO estimator itself uses the
    configured Rs vector and cfo1 only.
    """
    rs_names, cfo_names = validate_vector_names(pack, rs_vector_key, cfo_vector_key)
    vector_names = pack.get("vector_feature_names", {})
    if require_lr_vector and lr_vector_key not in vector_names:
        raise KeyError(f"Missing vector_feature_names[{lr_vector_key!r}]")

    samples = pack.get("samples", [])
    if not samples:
        raise ValueError("Compact PT contains no samples.")

    for row_idx, sample in enumerate(samples):
        if "x_processed" not in sample:
            raise KeyError(f"sample[{row_idx}] missing x_processed")
        if "vectors" not in sample:
            raise KeyError(f"sample[{row_idx}] missing vectors")
        vectors = sample["vectors"]
        for key in ([lr_vector_key] if require_lr_vector else []) + [rs_vector_key, cfo_vector_key]:
            if key not in vectors:
                raise KeyError(f"sample[{row_idx}] missing vectors[{key!r}]")

        x = np.asarray(sample["x_processed"])
        if x.ndim != 1 or x.size == 0:
            raise ValueError(f"sample[{row_idx}] x_processed has invalid shape {x.shape}")

        rs_vec = np.asarray(vectors[rs_vector_key])
        cfo_vec = np.asarray(vectors[cfo_vector_key])
        if rs_vector_key == "rs28" and rs_vec.shape != (28,):
            raise ValueError(f"sample[{row_idx}] vectors['rs28'] shape {rs_vec.shape}, expected (28,)")
        if rs_vector_key == "rs20" and rs_vec.shape != (20,):
            raise ValueError(f"sample[{row_idx}] vectors['rs20'] shape {rs_vec.shape}, expected (20,)")
        if cfo_vec.shape != (1,):
            raise ValueError(f"sample[{row_idx}] vectors['cfo1'] shape {cfo_vec.shape}, expected (1,)")

    return {
        "rs_names": rs_names,
        "cfo_names": cfo_names,
        "lr_names": list(vector_names.get(lr_vector_key, [])),
        "validated_sample_count": int(len(samples)),
    }


def vector_column_value(vec, names, feature_name):
    return float(np.asarray(vec, dtype=np.float64)[names.index(feature_name)])


def collect_absx2_candidate_bins(
    input_root,
    years=("2018", "2022"),
    n_batches=28,
    min_count=800,
    rs_vector_key="rs28",
    cfo_vector_key="cfo1",
    target_signal_types=TARGET_SIGNAL_TYPES,
    drop_2018_negative_snr=True,
):
    """Scan QPSK-like rs28 rows and collect global abs_x2 candidate bins.

    This mirrors build_absx2_candidate_estimates_for_scheme() in
    Rs-CFO-estimation-Rs18-28-60-CFO-ptvectors.ipynb.  The
    drop_2018_negative_snr argument is kept only so callers can report how many
    rows will be error-marked during CC generation; it does not remove those
    rows from candidate-bin statistics.
    """
    input_root = Path(input_root)
    bins = []
    rows = []
    t0 = time.time()

    for year in years:
        for batch_id in range(1, int(n_batches) + 1):
            path = input_root / str(year) / f"{year}_batch_{batch_id:02d}_boi_shift_fft3_compact_vectors.pt"
            if not path.exists():
                raise FileNotFoundError(path)

            pack = load_pack(path)
            rs_names, _ = validate_vector_names(pack, rs_vector_key, cfo_vector_key)

            n_rows = 0
            n_qpsklike = 0
            n_negative_snr_marked = 0
            n_used_for_candidates = 0

            for sample in pack["samples"]:
                n_rows += 1
                if not is_qpsklike_sample(sample, target_signal_types):
                    continue
                n_qpsklike += 1
                if is_negative_snr_skip(sample, year, drop_2018_negative_snr):
                    n_negative_snr_marked += 1

                n_used_for_candidates += 1
                vec = np.asarray(sample["vectors"][rs_vector_key], dtype=np.float64)
                for rank in RS_ABS_RANKS:
                    f = vector_column_value(vec, rs_names, f"abs_x2_top{rank}_af")
                    if np.isfinite(f) and 0.03 <= f <= 0.5001:
                        bins.append(int(round(f * NFFT)))

            rows.append({
                "year": str(year),
                "batch_id": int(batch_id),
                "rows": int(n_rows),
                "qpsklike": int(n_qpsklike),
                "negative_snr_error_marked": int(n_negative_snr_marked),
                "used_for_candidate_bins": int(n_used_for_candidates),
            })
            print(
                f"candidate pass {year} batch {batch_id:02d}: "
                f"qpsklike={n_qpsklike} negative_snr_marked={n_negative_snr_marked} "
                f"used={n_used_for_candidates} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
            del pack
            gc.collect()

    counts = pd.Series(bins).value_counts() if bins else pd.Series(dtype=int)
    candidate_bins = set(counts[counts >= int(min_count)].index.astype(int).tolist())
    return {
        "candidate_bins": sorted(candidate_bins),
        "candidate_bin_count": len(candidate_bins),
        "candidate_min_count": int(min_count),
        "scan_summary": pd.DataFrame(rows),
        "top_bin_counts": counts.head(30).rename_axis("bin").reset_index(name="count"),
    }


def estimate_x4_triplet_from_rs28(vec, names):
    x4_freq = np.array([vector_column_value(vec, names, f"x4_top{i}_f") for i in range(1, 5)], dtype=float)
    x4_db = np.array([vector_column_value(vec, names, f"x4_top{i}_db") for i in range(1, 5)], dtype=float)

    best = None
    min_sep = 5.0 / NFFT
    for ci in range(4):
        cf = x4_freq[ci]
        cdb = x4_db[ci]
        if not (np.isfinite(cf) and np.isfinite(cdb)):
            continue
        for ai in range(4):
            if ai == ci:
                continue
            af = x4_freq[ai]
            adb = x4_db[ai]
            if not (np.isfinite(af) and np.isfinite(adb)):
                continue
            da = wrap_freq_scalar(af - cf)
            if abs(da) <= min_sep:
                continue
            for bi in range(ai + 1, 4):
                if bi == ci:
                    continue
                bf = x4_freq[bi]
                bdb = x4_db[bi]
                if not (np.isfinite(bf) and np.isfinite(bdb)):
                    continue
                db = wrap_freq_scalar(bf - cf)
                if abs(db) <= min_sep or da * db >= 0:
                    continue

                if da < 0:
                    left_i, right_i = ai, bi
                    left_f, right_f = af, bf
                    left_db, right_db = adb, bdb
                else:
                    left_i, right_i = bi, ai
                    left_f, right_f = bf, af
                    left_db, right_db = bdb, adb

                dl = wrap_freq_scalar(left_f - cf)
                dr = wrap_freq_scalar(right_f - cf)
                rs_est = 0.5 * (abs(dl) + abs(dr))
                if not (min_sep < rs_est < 0.49):
                    continue

                sym_err = abs(abs(dl) - abs(dr))
                center_from_sides = wrap_freq_scalar(cf + 0.5 * (dl + dr))
                center_offset = abs(wrap_freq_scalar(center_from_sides - cf))
                side_mean = 0.5 * (left_db + right_db)
                side_min = min(left_db, right_db)
                score = (
                    1.0 * cdb
                    + 0.9 * side_mean
                    + 0.25 * side_min
                    - 900.0 * sym_err
                    - 450.0 * center_offset
                    - 0.03 * ((ci + 1) + (left_i + 1) + (right_i + 1))
                )
                cand = (score, rs_est, left_i + 1, right_i + 1, sym_err)
                if best is None or score > best[0]:
                    best = cand

    if best is None:
        return np.nan, np.nan, np.nan, np.nan
    return float(best[1]), float(best[2]), float(best[3]), float(best[4])


def estimate_rs_absx2_from_rs28(vec, names, candidate_bins):
    for rank in RS_ABS_RANKS:
        f = vector_column_value(vec, names, f"abs_x2_top{rank}_af")
        b = int(round(f * NFFT)) if np.isfinite(f) else -1
        if np.isfinite(f) and 0.03 <= f <= 0.5001 and b in candidate_bins:
            return float(b / NFFT), f"absx2_top{rank}_candidate_bin"
    fallback = vector_column_value(vec, names, "rs_est")
    return float(fallback), "rs_est_fallback"


def quantize_rs(f):
    if not np.isfinite(f):
        return float("nan")
    return float(round(float(f) * NFFT) / NFFT)


def valid_rs_value(f):
    return np.isfinite(f) and 0.03 <= float(f) <= 0.5001


def estimate_rs_from_rs20(vec, names):
    """Strict local hand-written rs20 estimator.

    Rule from pre_Rs-CFO-estimation-QPSKlike-pkl-strict-hand-unsupervised-more-features.ipynb:
      1. If abs_x2_top1_minus_top2_db is not too large, trust abs_x2_top2_af.
      2. Otherwise use a confident x4 symmetric triplet.
      3. Otherwise use the first valid even abs_x2 candidate with bin >= 1800.
      4. Fallback to rs_est.
    """
    abs2 = quantize_rs(vector_column_value(vec, names, "abs_x2_top2_af"))
    db_gap = vector_column_value(vec, names, "abs_x2_top1_minus_top2_db")

    if np.isfinite(db_gap) and db_gap <= RS20_TRUST_ABS2_DB_GAP_MAX:
        return float(abs2), "rs20_abs_x2_top2_trusted_db_gap", np.nan, np.nan, np.nan

    rs_x4, left_rank, right_rank, sym_err = estimate_x4_triplet_from_rs28(vec, names)
    x4_gate = (
        np.isfinite(sym_err)
        and sym_err <= 2.0 * BIN
        and left_rank <= 4
        and right_rank <= 4
    )
    if x4_gate:
        return float(rs_x4), "rs20_x4_triplet_conf", left_rank, right_rank, sym_err

    for rank in RS20_ABS_RANKS:
        f = vector_column_value(vec, names, f"abs_x2_top{rank}_af")
        if not valid_rs_value(f):
            continue
        b = int(round(float(f) * NFFT))
        if b >= RS20_LATE_EVEN_MIN_BIN:
            return float(b / NFFT), f"rs20_abs_x2_top{rank}_first_even_minbin", left_rank, right_rank, sym_err

    fallback = vector_column_value(vec, names, "rs_est")
    return float(fallback), "rs20_rs_est_fallback", left_rank, right_rank, sym_err


def estimate_rs_cfo_from_sample(
    sample,
    rs_names,
    cfo_names,
    candidate_bins,
    rs_vector_key="rs28",
    cfo_vector_key="cfo1",
):
    rs_vec = np.asarray(sample["vectors"][rs_vector_key], dtype=np.float64)
    cfo_vec = np.asarray(sample["vectors"][cfo_vector_key], dtype=np.float64)

    if rs_vector_key == "rs20":
        rs_est, rs_source, left_rank, right_rank, sym_err = estimate_rs_from_rs20(rs_vec, rs_names)
    else:
        rs_x4, left_rank, right_rank, sym_err = estimate_x4_triplet_from_rs28(rs_vec, rs_names)
        x4_gate = np.isfinite(sym_err) and (sym_err < 2.0 * BIN) and (left_rank <= 4) and (right_rank <= 4)

        if x4_gate:
            rs_est = float(rs_x4)
            rs_source = "x4_triplet_conf"
        else:
            rs_est, rs_source = estimate_rs_absx2_from_rs28(rs_vec, rs_names, candidate_bins)

    if not np.isfinite(rs_est) or rs_est <= 0.0:
        raise ValueError(f"Invalid blind Rs estimate: {rs_est}")

    cfo_est = wrap_freq_scalar(float(cfo_vec[0]) / 4.0)
    return {
        "rs_est": float(rs_est),
        "T0_est": float(1.0 / rs_est),
        "cfo_est": float(cfo_est),
        "rs_source": rs_source,
        "x4_left_rank": float(left_rank),
        "x4_right_rank": float(right_rank),
        "x4_sym_err": float(sym_err),
    }
