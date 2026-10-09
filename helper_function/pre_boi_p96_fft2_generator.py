#!/usr/bin/env python3
"""Consolidated BOI + P96-clip + 3FFT (z^2, z^4, |z|^2) generator.

`boi_p96_fft3` fork of `boi_p96_fft2`: adds ONE extra FFT (`FFT{|z|^2}`, the envelope
power of the already-clipped `z`, computed once in `build_one_sample` and shared), used
for (1) a cross-check/arbitration branch for `rs_estimate_prominence` -- see
`rs_prominence_fftz2_correction` -- and (2) 8 additional features for the LR router,
which is now `lr8_32` (32-dim; supersedes and replaces the old 24-dim `lr8_24`) -- see
`az2_lr_features` / `LR8_32_FEATURES`. Everything else (BOI/clip/z^4/CFO) is unchanged
from `boi_p96_fft2`.

Single source of truth for the validated pipeline used throughout the SPL
CFO/Rs investigation:
  - BOI detection/filter/frequency-shift (float64/complex128, unchanged from
    the established `run_one_fixed_boi_shift`).
  - P96 magnitude clipping (phase preserved) to build `z` from `x_processed`,
    with the clip threshold itself (`clip_thr`) saved explicitly.
  - `z^4` FFT (Hann window, float32 log-magnitude spectrum).
  - Rs: blind circular-prominence symmetric full-grid search on the z^4
    spectrum (verified 98.24% on the full 2018/2022 QPSK-like corpus), PLUS
    the independent abs_x2 rule-based estimator (trust gate -> x4-triplet ->
    ranked fallback, ~95% on 2018R2), computed directly from `x_processed`
    without any intermediate 445/55/28/20-feature vectors, PLUS
    `rs_estimate_prominence_corrected`: an `FFT{|z|^2}` cross-check arbitration
    on top of the prominence estimate (3rd FFT). Validated on the full 167,816-
    sample QPSK-like corpus with an honest group-split train/val/test +
    fully-external (2018R2) breakdown: 0 regressions on any split, 936 samples
    recovered, raising accuracy 97.175%->97.965% (2018), 99.309%->99.429%
    (2022), 97.355%->98.120% (2018R2). See `rs_prominence_fftz2_correction`'s
    docstring for the full validation detail, including why a more aggressive
    search-tuned threshold was rejected (1 test-set regression) in favor of
    reusing the already-validated `RS20_TRUST_ABS2_DB_GAP_MAX` verbatim.
  - CFO: the original unrestricted-argmax estimate (kept for comparison) AND
    the band-limited (|cfo|<=0.03) + 3-point parabolic interpolation estimate
    developed this session (closes the classification gap to ground-truth CFO
    to within +/-0.4pp; verified on 2018/2022/2018R2).
  - `lr8_32`: the 32-feature vector for the LR8_32 5-class coarse router
    (BPSK/MSK/DQPSK/8PSK/QPSK-like) -- 24 base features from the top-8 separated
    peaks of the Hann `FFT{z^2}` and `FFT{z^4}` spectra (same functions as
    `FPGA_pytorch_sim_2FFT_z-test/helper_function/pre_twofft_fe_float32_generator.py`)
    PLUS 8 features from the top-16 peaks / full spectrum of `FFT{|z|^2}` (the
    3rd FFT), picked by a 5-fold-CV forward search: 0 regressions on any of the
    4 held-out metrics (2018train20/2022train20 same+external), every metric
    improved +0.026pp to +0.050pp. Only the 32-dim vector is saved; the
    spectra/peaks themselves are not.

ALL signal types are processed (the LR router needs every class). The Rs/CFO
estimators are QPSK-like-specific and give meaningless (but finite, non-
crashing) values for bpsk/8psk/dqpsk/msk; QPSK-like selection happens
downstream, in the CC-matrix generator.

Every raw truth.txt column is preserved verbatim (renamed to snake_case) via
`run_one_fixed_boi_shift`'s own truth dict, plus convenience aliases
(rs_truth, cfo_truth, cfo_truth_shifted, snr_truth_db) used elsewhere in the
project.

2018 negative-SNR samples are detected from the raw truth line ALONE (no BOI
extraction, no clip, no FFT is ever run for them) and recorded with
`status=False`, `error="SKIP_NEGATIVE_TRUTH_SNR"`; all heavy array fields are
left as `None` for these rows to avoid wasting disk space.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from . import pre_boi_fft3_generation_base_from_notebook as base

NFFT = 32768
CLIP_PCT = 96
BASELINE_HALF_WIDTH = 3200
RS_LO, RS_HI = 0.03, 0.5001
RS_OFFSET_LO = int(np.ceil(RS_LO * NFFT))
RS_OFFSET_HI = int(np.floor(RS_HI * NFFT))
RS_OFFSETS = np.arange(RS_OFFSET_LO, RS_OFFSET_HI + 1, dtype=np.int32)
if RS_OFFSETS.shape != (15404,):
    raise AssertionError(f"Expected 15404 blind Rs offsets, got {RS_OFFSETS.shape}")

HANN_WINDOW = np.hanning(NFFT)
FREQ = np.fft.fftshift(np.fft.fftfreq(NFFT))
BIN = 1.0 / NFFT

CFO_BAND_HALFWIDTH = 0.12  # => |cfo| <= 0.03, validated on 2018/2022/2018R2
CFO_BAND_IDX = np.where(np.abs(FREQ) <= CFO_BAND_HALFWIDTH)[0]

# LR8_32 peak-feature constants and feature order. The base-24 names are verbatim from
# the validated FPGA_pytorch_sim_2FFT_z-test generator (order matters: it defines the
# vector); the az2_* names are the 8 |z|^2 additions selected in boi_p96_fft3.
TOP_K = 8
TOP_K16 = 16  # only for the |z|^2 envelope peak search (lr8_32's az2_* features)
MIN_SEP_BINS = 8
DC_EXCLUSION = np.float32(5.0 / NFFT)
_LR_BASE24_NAMES = [
    "x2_top5_f",
    "x2_top8_f",
    "x2_top5_minus_top6_db",
    "x2_energy_ratio_top4_of_8",
    "x2_sym8_score_min",
    "x2_sym8_score_min2",
    "x4_top1_af",
    "x4_top2_af",
    "x4_top3_db",
    "x4_top6_af",
    "x4_top7_f",
    "x4_top8_f",
    "x4_top8_af",
    "x4_top1_minus_top2_db",
    "x4_top2_minus_top3_db",
    "x4_top6_minus_top7_db",
    "x4_count_gt_m20_d8",
    "x4_sym8_score_min",
    "x4_sym8_score_min2",
    "x2_top3_db",
    "x4_top7_minus_top8_db",
    "x2_top1_minus_top2_db",
    "x2_top7_f",
    "x4_top2_f",
]
# 8 additions from FFT{|z|^2} (envelope power of the P96-clipped z -- the 3rd FFT), picked
# by a 5-fold-CV forward search over a 41-candidate pool: 0 regressions on ALL 4 held-out
# metrics (2018train20/2022train20 same-domain + external), every metric improved
# (+0.026pp to +0.050pp). See rs_prominence_fftz2_correction's docstring for the sibling
# Rs-correction use of the same |z|^2 spectrum.
_LR_AZ2_NAMES = [
    "az2_peak16_energy_ratio",
    "az2_p1_db",
    "az2_p6_db",
    "az2_p3_minus_p5_db",
    "az2_p2_minus_p3_db",
    "az2_p3af_over_p1af",
    "az2_p5_minus_p6_db",
    "az2_count_gt_m10_d8",
]
LR8_32_FEATURES = _LR_BASE24_NAMES + _LR_AZ2_NAMES
if len(LR8_32_FEATURES) != 32 or len(set(LR8_32_FEATURES)) != 32:
    raise AssertionError("LR8_32 must contain 32 unique features")

ABS2_TOPK = 20
ABS2_TOPK_EXTENDED = 60
RS20_TRUST_ABS2_DB_GAP_MAX = 23.941201956505633
RS20_LATE_EVEN_MIN_BIN = 983  # = RS_LO * NFFT

GENERATION_METHOD_VERSION = {
    "clip": {"method": "per-sample magnitude percentile, phase preserved", "percentile": CLIP_PCT},
    "rs_prominence": {
        "method": "blind circular-prominence symmetric full-grid",
        "baseline_half_width": BASELINE_HALF_WIDTH,
        "offset_lo": RS_OFFSET_LO, "offset_hi": RS_OFFSET_HI,
        "num_hypotheses": int(RS_OFFSETS.size),
    },
    "rs_abs_x2_rule": {
        "method": "trust gate (top1/top2 dB gap) -> x4-triplet confidence -> ranked fallback",
        "trust_db_gap_max": RS20_TRUST_ABS2_DB_GAP_MAX,
        "topk": ABS2_TOPK, "topk_extended": ABS2_TOPK_EXTENDED,
        "late_even_min_bin": RS20_LATE_EVEN_MIN_BIN,
    },
    "rs_prominence_corrected": {
        "method": "FFT{|z|^2} cross-check arbitration on top of rs_prominence (3rd FFT); "
                   "overrides only when the |z|^2 top1/top2 dB gap is trusted AND disagrees "
                   "with rs_prominence",
        "trust_db_gap_max": RS20_TRUST_ABS2_DB_GAP_MAX,
        "candidate_rank": 2,
        "validated_regressions": 0,
        "validated_recoveries": 936,
        "validated_n": 167816,
    },
    "cfo_argmax": {"method": "unrestricted global argmax of z^4 spectrum, no interpolation"},
    "cfo_band_interp": {
        "method": "band-limited top-1 peak search + 3-point parabolic interpolation",
        "band_halfwidth": CFO_BAND_HALFWIDTH,
    },
    "lr8_32": {
        "method": "24 base features (top-8 separated peaks of Hann FFT{z^2} and FFT{z^4}) "
                   "+ 8 features from the top-16 separated peaks / full spectrum of "
                   "Hann FFT{|z|^2} (3rd FFT, envelope power of the clipped z), picked by "
                   "a 5-fold-CV forward search -- 0 regressions, all 4 held-out metrics "
                   "improved +0.026pp to +0.050pp (2018/2022 train20, same+external)",
        "top_k": TOP_K, "top_k16": TOP_K16, "min_sep_bins": MIN_SEP_BINS,
        "base24_feature_names": list(_LR_BASE24_NAMES),
        "az2_feature_names": list(_LR_AZ2_NAMES),
        "feature_names": list(LR8_32_FEATURES),
    },
}


def wrap_freq_scalar(f) -> float:
    return ((float(f) + 0.5) % 1.0) - 0.5


def prep_clip_p96(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (z as complex64, clip threshold thr as float)."""
    x_work = np.asarray(x, dtype=np.complex128).reshape(-1)
    amplitude = np.abs(x_work)
    thr = float(np.percentile(amplitude, CLIP_PCT))
    if not np.isfinite(thr) or thr < 0.0:
        raise ValueError("P96 clipping threshold is not finite and non-negative")
    scale = np.minimum(1.0, thr / (amplitude + 1e-12))
    z = np.asarray(x_work * scale, dtype=np.complex64)
    return z, thr


def fft_mag_db_float32(z_power: np.ndarray, nfft: int = NFFT) -> np.ndarray:
    """Hann-windowed relative magnitude-dB FFT, fftshift-ordered, float32."""
    z_power = np.asarray(z_power, dtype=np.complex128).reshape(-1)
    work = np.zeros(nfft, dtype=np.complex128)
    count = min(z_power.size, nfft)
    work[:count] = z_power[:count]
    spectrum = np.fft.fftshift(np.fft.fft(work * HANN_WINDOW, n=nfft))
    magnitude = np.abs(spectrum)
    peak = float(np.max(magnitude))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("FFT magnitude has no positive finite maximum")
    magnitude_db = 20.0 * np.log10(magnitude / (peak + 1e-12) + 1e-12)
    return magnitude_db.astype(np.float32, copy=False)


def blind_prominence_rs(fft_z4_mag_db: np.ndarray) -> tuple[float, float]:
    """Blind circular-prominence symmetric full-grid Rs search (98.24% validated).
    Returns (rs_estimate, fused_prominence_score)."""
    magnitude_db = np.asarray(fft_z4_mag_db, dtype=np.float64).reshape(-1)
    if magnitude_db.shape != (NFFT,):
        raise ValueError(f"Expected z4 spectrum shape ({NFFT},), got {magnitude_db.shape}")
    center_index = int(np.argmax(magnitude_db))

    half_width = BASELINE_HALF_WIDTH
    width = 2 * half_width + 1
    extended = np.concatenate([magnitude_db[-half_width:], magnitude_db, magnitude_db[:half_width]])
    cumulative = np.cumsum(np.insert(extended, 0, 0.0), dtype=np.float64)
    baseline = (cumulative[width:] - cumulative[:-width]) / float(width)
    prominence = magnitude_db - baseline

    positive = (center_index + RS_OFFSETS) % NFFT
    negative = (center_index - RS_OFFSETS) % NFFT
    scores = 0.5 * (prominence[positive] + prominence[negative])
    best = int(np.argmax(scores))
    return float(RS_OFFSETS[best] / float(NFFT)), float(scores[best])


def cfo_argmax(fft_z4_mag_db: np.ndarray) -> float:
    """Original unrestricted-argmax CFO estimate, kept for comparison."""
    center_index = int(np.argmax(np.asarray(fft_z4_mag_db)))
    center_freq = float((center_index - NFFT // 2) / NFFT)
    return wrap_freq_scalar(center_freq / 4.0)


def cfo_band_interp(fft_z4_mag_db: np.ndarray) -> float:
    """Band-limited (|cfo|<=0.03) top-1 peak search + 3-point parabolic
    interpolation on the log-magnitude spectrum. Verified (2018/2022,
    N=111,816): outlier rate 0.0376%->0.0018%, median error 1.9e-6->2.4e-7.
    Verified (2018R2, N=3,000): 0 outliers, median error 2.7e-7. Closes the
    downstream classification gap to ground-truth CFO to within +/-0.4pp."""
    mag_db = np.asarray(fft_z4_mag_db)
    local_argmax = np.argmax(mag_db[CFO_BAND_IDX])
    bin_idx = CFO_BAND_IDX[local_argmax]
    left = mag_db[bin_idx - 1]
    center = mag_db[bin_idx]
    right = mag_db[bin_idx + 1]
    denom = left - 2.0 * center + right
    delta = 0.0
    if denom != 0:
        delta = float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))
    top1_f = FREQ[bin_idx] + delta * BIN
    return wrap_freq_scalar(top1_f / 4.0)


# ---------------------------------------------------------------------------
# LR8_32 base-24 features: _top_peaks_float32 / _symmetry_min2_float32 (z^2/z^4 part of
# lr8_32_features) are verbatim copies of _top_peaks_float32 / _symmetry_min2_float32 /
# _lr8_24_float32 in FPGA_pytorch_sim_2FFT_z-test's pre_twofft_fe_float32_generator.py.
# The 8 az2_* additions (from |z|^2, the 3rd FFT) are new to boi_p96_fft3.
# ---------------------------------------------------------------------------
def _top_peaks_float32(magnitude_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descending full-spectrum sort with the established 8-bin exclusion."""
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
    frequency = (
        (bins - np.int32(NFFT // 2)).astype(np.float32) / np.float32(NFFT)
    ).astype(np.float32)
    return frequency, magnitude_db[bins].astype(np.float32, copy=False)


def _top_peaks_generic(magnitude_db: np.ndarray, top_k: int, min_sep_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Same algorithm as `_top_peaks_float32` but with a configurable `top_k` (that one
    hardcodes TOP_K=8 via the module global). Used for the |z|^2 top-16 search that
    lr8_32's az2_* features need -- `_top_peaks_float32` itself is left untouched since
    it's still used at TOP_K=8 for the z^2/z^4 base-24 features and for the Rs correction."""
    magnitude_db = np.asarray(magnitude_db, dtype=np.float32).reshape(-1)
    valid_indices = np.flatnonzero(np.isfinite(magnitude_db))
    order = valid_indices[np.argsort(magnitude_db[valid_indices])[::-1]]
    selected: list[int] = []
    for index in order:
        index_int = int(index)
        if all(abs(index_int - old) >= min_sep_bins for old in selected):
            selected.append(index_int)
            if len(selected) == top_k:
                break
    if len(selected) != top_k:
        raise ValueError(f"Only {len(selected)} finite separated FFT peaks were found")
    bins = np.asarray(selected, dtype=np.int32)
    frequency = (
        (bins - np.int32(NFFT // 2)).astype(np.float32) / np.float32(NFFT)
    ).astype(np.float32)
    return frequency, magnitude_db[bins].astype(np.float32, copy=False)


def az2_lr_features(mag_db_env2: np.ndarray) -> dict:
    """The 8 FFT{|z|^2} features selected for LR8_32 (see `_LR_AZ2_NAMES`), from the
    top-16 separated peaks of the envelope-power spectrum plus 2 global (full-spectrum)
    spectral-shape stats. Picked by a 5-fold-CV forward search over a 41-candidate pool;
    see `LR8_32_FEATURES`'s definition and `GENERATION_METHOD_VERSION["lr8_32"]`."""
    mag_db_env2 = np.asarray(mag_db_env2, dtype=np.float32).reshape(-1)
    freq16, db16 = _top_peaks_generic(mag_db_env2, TOP_K16, MIN_SEP_BINS)
    abs_freq16 = np.abs(freq16).astype(np.float32)
    p_af = [abs_freq16[2 * i + 1] for i in range(8)]
    p_db = [db16[2 * i + 1] for i in range(8)]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        power16 = np.power(np.float32(10.0), db16 / np.float32(10.0)).astype(np.float32)
        full_power = np.power(np.float32(10.0), mag_db_env2 / np.float32(10.0))
        full_power = np.clip(full_power, np.float32(1e-12), None)
        log_mean = np.mean(np.log(full_power), dtype=np.float64)
        arith_mean = np.mean(full_power, dtype=np.float64)
        peak16_energy_ratio = float(np.sum(power16, dtype=np.float64)) / (
            float(np.sum(full_power, dtype=np.float64)) + 1e-30
        )

    nonzero = np.isfinite(abs_freq16) & (abs_freq16 > DC_EXCLUSION)
    count_gt_m10_d8 = np.sum(nonzero & (db16 > np.float32(-10.0)), dtype=np.int32)

    return {
        "az2_peak16_energy_ratio": np.float32(peak16_energy_ratio),
        "az2_p1_db": p_db[0],
        "az2_p6_db": p_db[5],
        "az2_p3_minus_p5_db": p_db[2] - p_db[4],
        "az2_p2_minus_p3_db": p_db[1] - p_db[2],
        "az2_p3af_over_p1af": np.float32(p_af[2] / (p_af[0] + np.float32(1e-9))),
        "az2_p5_minus_p6_db": p_db[4] - p_db[5],
        "az2_count_gt_m10_d8": np.float32(count_gt_m10_d8),
    }


def _symmetry_min2_float32(frequency: np.ndarray, db: np.ndarray) -> tuple[np.float32, np.float32]:
    frequency = np.asarray(frequency, dtype=np.float32)[:TOP_K]
    db = np.asarray(db, dtype=np.float32)[:TOP_K]
    valid = np.isfinite(frequency)
    scores: list[np.float32] = []
    for index in range(TOP_K):
        if not valid[index] or np.abs(frequency[index]) <= DC_EXCLUSION:
            continue
        differences = np.abs(frequency + frequency[index]).astype(np.float32)
        differences = np.where(valid, differences, np.float32(np.inf))
        mirror = int(np.argmin(differences))
        score = (
            np.abs(db[index] - db[mirror])
            + np.float32(40.0) * np.abs(frequency[mirror] + frequency[index])
        )
        scores.append(np.float32(score))
    if not scores:
        return np.float32(99.0), np.float32(99.0)
    ordered = np.sort(np.asarray(scores, dtype=np.float32))
    first = np.float32(ordered[0])
    second = np.float32(ordered[1] if ordered.size >= 2 else ordered[0])
    return first, second


def lr8_32_features(fft_z2_mag_db: np.ndarray, fft_z4_mag_db: np.ndarray, mag_db_env2: np.ndarray) -> np.ndarray:
    """32-dim float32 LR8_32 vector: the 24 base features from the z^2 / z^4 log-magnitude
    spectra, plus 8 features from the |z|^2 (3rd FFT) spectrum -- see `az2_lr_features`."""
    z2_frequency, z2_db = _top_peaks_float32(fft_z2_mag_db)
    z4_frequency, z4_db = _top_peaks_float32(fft_z4_mag_db)
    z4_abs_frequency = np.abs(z4_frequency).astype(np.float32)
    z2_sym_min, z2_sym_min2 = _symmetry_min2_float32(z2_frequency, z2_db)
    z4_sym_min, z4_sym_min2 = _symmetry_min2_float32(z4_frequency, z4_db)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        z2_power = np.power(np.float32(10.0), z2_db / np.float32(10.0)).astype(np.float32)
        energy_ratio = np.sum(z2_power[:4], dtype=np.float32) / np.sum(z2_power[:8], dtype=np.float32)

    z4_nonzero = np.isfinite(z4_abs_frequency) & (z4_abs_frequency > DC_EXCLUSION)
    z4_count = np.sum(z4_nonzero & (z4_db > np.float32(-20.0)), dtype=np.int32)

    columns = {
        "x2_top5_f": z2_frequency[4],
        "x2_top8_f": z2_frequency[7],
        "x2_top5_minus_top6_db": z2_db[4] - z2_db[5],
        "x2_energy_ratio_top4_of_8": energy_ratio,
        "x2_sym8_score_min": z2_sym_min,
        "x2_sym8_score_min2": z2_sym_min2,
        "x4_top1_af": z4_abs_frequency[0],
        "x4_top2_af": z4_abs_frequency[1],
        "x4_top3_db": z4_db[2],
        "x4_top6_af": z4_abs_frequency[5],
        "x4_top7_f": z4_frequency[6],
        "x4_top8_f": z4_frequency[7],
        "x4_top8_af": z4_abs_frequency[7],
        "x4_top1_minus_top2_db": z4_db[0] - z4_db[1],
        "x4_top2_minus_top3_db": z4_db[1] - z4_db[2],
        "x4_top6_minus_top7_db": z4_db[5] - z4_db[6],
        "x4_count_gt_m20_d8": np.float32(z4_count),
        "x4_sym8_score_min": z4_sym_min,
        "x4_sym8_score_min2": z4_sym_min2,
        "x2_top3_db": z2_db[2],
        "x4_top7_minus_top8_db": z4_db[6] - z4_db[7],
        "x2_top1_minus_top2_db": z2_db[0] - z2_db[1],
        "x2_top7_f": z2_frequency[6],
        "x4_top2_f": z4_frequency[1],
    }
    columns.update(az2_lr_features(mag_db_env2))
    return np.asarray([columns[name] for name in LR8_32_FEATURES], dtype=np.float32)


def rs_prominence_fftz2_correction(mag_db_env2: np.ndarray, rs_prom: float) -> float:
    """FFT{|z|^2} cross-check arbitration for Method 2 (`blind_prominence_rs`).

    Takes the ALREADY-COMPUTED |z|^2 log-magnitude spectrum (`mag_db_env2`, the envelope
    power of the P96-clipped `z`) -- `build_one_sample` computes this ONE extra FFT once
    and shares it with both this correction and `lr8_32_features`'s az2_* features, rather
    than each recomputing it independently. This envelope spectrum is expected to show a
    discrete line at the true symbol rate for many QPSK-like signals, via a completely
    different mechanism than the z^4 phase-stripping trick `blind_prominence_rs` relies
    on. When that line is strong (top1/top2 dB gap on the |z|^2 spectrum is small --
    reusing RS20_TRUST_ABS2_DB_GAP_MAX verbatim, the SAME threshold already validated for
    `estimate_rs_abs_x2_rule`, not a value newly fit to this task) AND it disagrees with
    `rs_prom`, this candidate is trusted instead.

    Validated on the full corpus (167,816 QPSK-like samples, 2018/2022/2018R2, honest
    group-split train/val/test + fully-external 2018R2): ZERO samples that `rs_prom` had
    already gotten right were changed (0 regressions on train=78,272 / val=5,592 /
    test=27,952 / external=56,000), while 936 samples (0.56%) that `rs_prom` had gotten
    wrong were recovered, raising accuracy 97.175%->97.965% (2018), 99.309%->99.429%
    (2022), 97.355%->98.120% (2018R2). A more aggressive, search-tuned dB-gap threshold
    (25 vs. the reused 23.94) recovered slightly more on train/val but introduced 1
    regression on held-out test -- rejected in favor of this verbatim-threshold version,
    which is what is implemented here.
    """
    mag_db_env2 = np.asarray(mag_db_env2, dtype=np.float32).reshape(-1)
    frequency, db = _top_peaks_float32(mag_db_env2)
    db_gap = float(db[0] - db[1])
    candidate = float(abs(frequency[1]))
    fires = (
        db_gap <= RS20_TRUST_ABS2_DB_GAP_MAX
        and RS_LO <= candidate <= RS_HI
        and abs(candidate - rs_prom) > BIN
    )
    return candidate if fires else float(rs_prom)


def _mag_db_hann(z: np.ndarray) -> np.ndarray:
    Z = np.fft.fftshift(np.fft.fft(np.asarray(z, dtype=np.complex128) * HANN_WINDOW))
    mag = np.abs(Z)
    return 20.0 * np.log10(mag / (mag.max() + 1e-12) + 1e-12)


def _top_peaks(freq, mag_db, top_k, min_sep_bins):
    cand_idx = np.arange(mag_db.shape[0])
    order = cand_idx[np.argsort(-mag_db[cand_idx])]
    selected_f, selected_db, selected_idx = [], [], []
    for idx in order:
        if len(selected_idx) >= top_k:
            break
        if all(abs(int(idx) - int(j)) >= min_sep_bins for j in selected_idx):
            selected_idx.append(int(idx))
            selected_f.append(float(freq[idx]))
            selected_db.append(float(mag_db[idx]))
    while len(selected_f) < top_k:
        selected_f.append(np.nan)
        selected_db.append(np.nan)
    return np.array(selected_f), np.array(selected_db)


def _x4_triplet(x4_f, x4_db):
    best = None
    min_sep = 5.0 / NFFT
    for ci in range(4):
        cf, cdb = x4_f[ci], x4_db[ci]
        if not (np.isfinite(cf) and np.isfinite(cdb)):
            continue
        for ai in range(4):
            if ai == ci:
                continue
            af, adb = x4_f[ai], x4_db[ai]
            if not (np.isfinite(af) and np.isfinite(adb)):
                continue
            da = wrap_freq_scalar(af - cf)
            if abs(da) <= min_sep:
                continue
            for bi in range(ai + 1, 4):
                if bi == ci:
                    continue
                bf, bdb = x4_f[bi], x4_db[bi]
                if not (np.isfinite(bf) and np.isfinite(bdb)):
                    continue
                db_ = wrap_freq_scalar(bf - cf)
                if abs(db_) <= min_sep or da * db_ >= 0:
                    continue
                if da < 0:
                    left_i, right_i, left_f, right_f, left_db, right_db = ai, bi, af, bf, adb, bdb
                else:
                    left_i, right_i, left_f, right_f, left_db, right_db = bi, ai, bf, af, bdb, adb
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
                score = (1.0 * cdb + 0.9 * side_mean + 0.25 * side_min
                         - 900.0 * sym_err - 450.0 * center_offset
                         - 0.03 * ((ci + 1) + (left_i + 1) + (right_i + 1)))
                cand = (score, rs_est, left_i + 1, right_i + 1, sym_err)
                if best is None or score > best[0]:
                    best = cand
    if best is None:
        return np.nan, np.nan, np.nan, np.nan
    return float(best[1]), float(best[2]), float(best[3]), float(best[4])


def estimate_rs_abs_x2_rule(x_processed: np.ndarray) -> tuple[float, str]:
    """Standalone abs_x2 rule-based Rs estimator (trust gate -> x4-triplet
    confidence -> ranked fallback), computed directly from x_processed with
    no dependency on any precomputed feature-vector scheme. Verified ~95% on
    2018R2, matches the deployed `estimate_rs_from_rs20` reference."""
    x = np.asarray(x_processed, dtype=np.complex128)
    env2 = np.abs(x) ** 2  # no mean removal
    mag_db2 = _mag_db_hann(env2)
    abs2_f, abs2_db = _top_peaks(FREQ, mag_db2, top_k=ABS2_TOPK_EXTENDED, min_sep_bins=8)
    abs2_f = np.abs(abs2_f)

    mag_db4 = _mag_db_hann(x ** 4)
    x4_f, x4_db = _top_peaks(FREQ, mag_db4, top_k=4, min_sep_bins=8)

    db_gap = abs2_db[0] - abs2_db[1] if np.isfinite(abs2_db[0]) and np.isfinite(abs2_db[1]) else np.nan
    if np.isfinite(db_gap) and db_gap <= RS20_TRUST_ABS2_DB_GAP_MAX:
        return float(round(abs2_f[1] * NFFT) / NFFT), "trusted_abs2_top2"

    rs_x4, left_rank, right_rank, sym_err = _x4_triplet(x4_f, x4_db)
    if np.isfinite(sym_err) and sym_err <= 2.0 * BIN and left_rank <= 4 and right_rank <= 4:
        return float(rs_x4), "x4_triplet_conf"

    for rank in range(1, ABS2_TOPK, 2):
        f = abs2_f[rank]
        if not (np.isfinite(f) and RS_LO <= f <= RS_HI):
            continue
        b = int(round(f * NFFT))
        if b >= RS20_LATE_EVEN_MIN_BIN:
            return float(b / NFFT), f"abs_x2_top{rank+1}_first_even_minbin"

    for rank in range(ABS2_TOPK - 1, ABS2_TOPK_EXTENDED, 2):
        f = abs2_f[rank]
        if not (np.isfinite(f) and RS_LO <= f <= RS_HI):
            continue
        b = int(round(f * NFFT))
        if b >= RS20_LATE_EVEN_MIN_BIN:
            return float(b / NFFT), f"abs_x2_top{rank+1}_extended_minbin"

    fallback = abs2_f[1] if np.isfinite(abs2_f[1]) else 0.03
    return float(fallback), "rs_est_fallback_proxy"


def _truth_dict_from_run(run: dict) -> dict:
    truth = dict(run["truth"])  # all 9 raw truth.txt columns, renamed snake_case, + T0_eff/symbol_rate
    cfo_truth_shifted = float(run["truth_guide_cfo"])
    truth["rs_truth"] = float(truth["symbol_rate"])
    truth["cfo_truth"] = float(truth["carrier_offset_label"])
    truth["snr_truth_db"] = float(truth["inband_snr_db_label"])
    truth["cfo_truth_shifted"] = cfo_truth_shifted
    return truth


def _preprocess_dict_from_run(run: dict) -> dict:
    prepared = run["prepared"]
    step_b = run["stepB"]
    refined = prepared["selected_refined"]
    requested_filter = prepared["selected_filter"]
    used_filter = step_b["boi_used"]
    return {
        "boi_algorithm": "fixed",
        "preprocess_mode": "boi_filter_shift",
        "estimated_boi_center_hz": float(run["estimated_boi_center_hz"]),
        "center_shift_hz": float(run["center_shift_hz"]),
        "truth_guide_cfo": float(run["truth_guide_cfo"]),
        "boi_refined_left": float(refined["f_left"]),
        "boi_refined_right": float(refined["f_right"]),
        "boi_refined_center": float(refined["f_center"]),
        "boi_refined_bw": float(refined["bandwidth"]),
        "boi_filter_left": float(requested_filter["f_left"]),
        "boi_filter_right": float(requested_filter["f_right"]),
        "boi_filter_bw": float(requested_filter["bandwidth"]),
        "boi_filter_expand_ratio": float(requested_filter.get("filter_expand_ratio", np.nan)),
        "boi_filter_used_left": float(used_filter["f_left"]),
        "boi_filter_used_right": float(used_filter["f_right"]),
        "boi_filter_used_bw": float(used_filter["bandwidth"]),
        "stepB_mode": str(step_b["mode"]),
        "stepB_decim": int(step_b["decim"]),
        "stepB_fs_out": float(step_b["fs_out"]),
    }


def _power_dict(run: dict, z: np.ndarray) -> dict:
    power_report = run["prepared"]["power_report"]
    # Matches the validated pipeline exactly: power is computed from stepB's own
    # (un-rounded, float64) x_processed, not the complex64-cast copy kept for storage.
    p_x_processed = float(base.signal_rms_power(run["stepB"]["x_processed"]))
    p_z_total = float(base.signal_rms_power(np.asarray(z, dtype=np.complex64)))
    p_boi_inband_signal = float(power_report["p_signal_boi_local"])
    if not np.isfinite(p_x_processed) or p_x_processed <= 0.0:
        raise ValueError("P_x_processed must be positive and finite")
    p_boi_inband_signal_z = float(p_boi_inband_signal * p_z_total / (p_x_processed + 1e-12))
    return {
        "P_x_processed": p_x_processed,
        "P_z_total": p_z_total,
        "p_boi_inband_signal": p_boi_inband_signal,
        "p_boi_inband_signal_z": p_boi_inband_signal_z,
    }


def build_one_sample(signal_index: int, cfg: dict, truth_df, root_data: str,
                      dataset_name: str, batch_id: int, batch_start: int) -> dict:
    meta = {
        "dataset_name": str(dataset_name),
        "signal_index": int(signal_index),
        "batch_id": int(batch_id),
        "batch_local_index": int(signal_index - batch_start),
        "global_index": int(signal_index),
    }

    _empty_fields = {
        "truth": None, "preprocess": None, "power": None,
        "x_processed": None, "z": None, "clip_thr": None,
        "rs_estimate_prominence": None, "rs_score_prominence": None,
        "rs_estimate_prominence_corrected": None,
        "rs_estimate_abs_x2_rule": None, "rs_abs_x2_rule_source": None,
        "t0_estimate": None,
        "cfo_estimate_argmax": None, "cfo_estimate_band_interp": None,
        "lr8_32": None,
    }

    # 2018 negative-SNR samples: detected from the truth ROW alone, no BOI
    # extraction / clip / FFT is ever attempted for them.
    row_snr = float(truth_df.loc[int(signal_index), "InbandSNRdB"])
    if dataset_name == "2018" and row_snr < 0.0:
        return {"meta": meta, **_empty_fields, "status": False, "error": "SKIP_NEGATIVE_TRUTH_SNR"}

    try:
        run = base.run_one_fixed_boi_shift(
            signal_index=signal_index, cfg=cfg, truth_df=truth_df, root_data=root_data,
        )
        # Clip on the RAW (float64/complex128) run["x_processed"], exactly like the
        # validated pipeline -- NOT on a complex64-rounded copy. x_processed is only
        # cast to complex64 afterwards, for storage / downstream use.
        z, clip_thr = prep_clip_p96(run["x_processed"])
        x_processed = np.asarray(run["x_processed"], dtype=np.complex64)
        # z^4 via two explicit complex64-rounded squarings (z2 then z4), matching the
        # validated pipeline's exact rounding path bit-for-bit (not a single z**4 call,
        # which numpy would evaluate without that intermediate complex64 rounding).
        z2 = np.asarray(z * z, dtype=np.complex64)
        z4 = np.asarray(z2 * z2, dtype=np.complex64)
        fft_z2_mag_db = fft_mag_db_float32(z2, nfft=int(cfg["FFT_NFFT"]))
        fft_z4_mag_db = fft_mag_db_float32(z4, nfft=int(cfg["FFT_NFFT"]))
        # |z|^2 (3rd FFT): computed ONCE here and shared by both lr8_32's az2_* features
        # and the Rs-prominence correction below, instead of each recomputing it.
        env2 = np.asarray(np.abs(z) ** 2, dtype=np.complex64)
        mag_db_env2 = fft_mag_db_float32(env2, nfft=int(cfg["FFT_NFFT"]))
        lr8_32 = lr8_32_features(fft_z2_mag_db, fft_z4_mag_db, mag_db_env2)

        rs_prom, rs_score = blind_prominence_rs(fft_z4_mag_db)
        rs_prom_corrected = rs_prominence_fftz2_correction(mag_db_env2, rs_prom)
        rs_abs_x2, rs_abs_x2_src = estimate_rs_abs_x2_rule(x_processed)
        cfo_old = cfo_argmax(fft_z4_mag_db)
        cfo_new = cfo_band_interp(fft_z4_mag_db)
        t0_estimate = float(1.0 / rs_prom) if rs_prom > 0 else float("nan")

        return {
            "meta": meta,
            "truth": _truth_dict_from_run(run),
            "preprocess": _preprocess_dict_from_run(run),
            "power": _power_dict(run, z),
            "x_processed": x_processed,
            "z": z,
            "clip_thr": np.float32(clip_thr),
            "rs_estimate_prominence": np.float32(rs_prom),
            "rs_score_prominence": np.float32(rs_score),
            "rs_estimate_prominence_corrected": np.float32(rs_prom_corrected),
            "rs_estimate_abs_x2_rule": np.float32(rs_abs_x2),
            "rs_abs_x2_rule_source": str(rs_abs_x2_src),
            "t0_estimate": np.float32(t0_estimate),
            "cfo_estimate_argmax": np.float32(cfo_old),
            "cfo_estimate_band_interp": np.float32(cfo_new),
            "lr8_32": lr8_32,
            "status": True, "error": "",
        }
    except Exception as exc:
        return {"meta": meta, **_empty_fields, "status": False,
                "error": repr(exc) + "\n" + traceback.format_exc()}


def generate_boi_p96_fft2_pt_files(cfg: dict) -> None:
    if base.torch is None:
        raise RuntimeError("PyTorch is required for torch.save().")
    if int(cfg["FFT_NFFT"]) != NFFT:
        raise ValueError(f"This generator requires FFT_NFFT={NFFT}")

    selection = str(cfg["DATASET_SELECT"])
    dataset_list = ["2018", "2022", "2018R2"] if selection.lower() == "all" else [selection]

    out_root = Path(base.ensure_dir(cfg["OUT_ROOT"]))
    for dataset_name in dataset_list:
        local_cfg = dict(cfg)
        local_cfg["DATASET_NAME"] = dataset_name
        root_data, label_path, truth_df = base.set_dataset(
            dataset_name=dataset_name, root_map=local_cfg["ROOT_MAP"], label_map=local_cfg["LABEL_MAP"],
        )

        n_per_batch = int(local_cfg["N_PER_BATCH"])
        n_total = int(local_cfg["N_TOTAL"])
        start_pt = max(1, int(local_cfg["START_PT"]))
        end_pt = min(int(local_cfg["END_PT"]), int(local_cfg["N_BATCHES"]))

        print("=" * 80)
        print(f"boi_p96_fft2 full generation | dataset={dataset_name} truth={label_path} data={root_data}")
        print("=" * 80)

        for batch_id in range(start_pt, end_pt + 1):
            batch_start = (batch_id - 1) * n_per_batch + 1
            batch_end = min(batch_id * n_per_batch, n_total)
            save_path = out_root / f"{dataset_name}_batch_{batch_id:02d}_boi_p96_fft2_full.pt"
            if bool(local_cfg.get("SKIP_EXISTING", False)) and save_path.exists():
                print(f"Skip existing: {save_path}")
                continue

            samples = []
            iterator = range(batch_start, batch_end + 1)
            if base.tqdm is not None:
                iterator = base.tqdm(iterator, desc=f"{dataset_name} boi_p96_fft2 batch {batch_id:02d}",
                                      total=batch_end - batch_start + 1, dynamic_ncols=True)

            for signal_index in iterator:
                samples.append(build_one_sample(
                    signal_index=signal_index, cfg=local_cfg, truth_df=truth_df, root_data=root_data,
                    dataset_name=dataset_name, batch_id=batch_id, batch_start=batch_start,
                ))

            n_ok = sum(1 for s in samples if s["status"])
            n_neg_snr = sum(1 for s in samples if not s["status"] and s["error"] == "SKIP_NEGATIVE_TRUTH_SNR")
            n_err = len(samples) - n_ok - n_neg_snr
            batch_pack = {
                "version": "boi_p96_fft2_full_v4",
                "dataset_name": dataset_name,
                "batch_id": int(batch_id),
                "batch_start": int(batch_start),
                "batch_end": int(batch_end),
                "num_samples": len(samples),
                "num_ok": int(n_ok),
                "num_negative_snr_skipped": int(n_neg_snr),
                "num_error": int(n_err),
                "lr8_32_feature_names": list(LR8_32_FEATURES),
                "generation_method_version": GENERATION_METHOD_VERSION,
                "samples": samples,
            }
            base.torch.save(batch_pack, save_path)
            print(f"Saved {save_path} | ok={n_ok} neg_snr_skip={n_neg_snr} error={n_err} / total={len(samples)}")


def inspect_boi_p96_fft2_pt(path) -> dict:
    pack = base.torch.load(str(path), map_location="cpu", weights_only=False)
    samples = pack["samples"]
    ok_sample = next((s for s in samples if s["status"]), None)
    return {
        "version": pack["version"],
        "dataset_name": pack["dataset_name"],
        "batch_id": pack["batch_id"],
        "num_samples": pack["num_samples"],
        "num_ok": pack["num_ok"],
        "num_negative_snr_skipped": pack["num_negative_snr_skipped"],
        "num_error": pack["num_error"],
        "lr8_32_feature_names": pack["lr8_32_feature_names"],
        "example_ok_sample_keys": sorted(ok_sample.keys()) if ok_sample else None,
        "example_truth_keys": sorted(ok_sample["truth"].keys()) if ok_sample else None,
        "example_x_processed_shape": tuple(ok_sample["x_processed"].shape) if ok_sample else None,
        "example_x_processed_dtype": str(ok_sample["x_processed"].dtype) if ok_sample else None,
        "example_z_shape": tuple(ok_sample["z"].shape) if ok_sample else None,
        "example_clip_thr": ok_sample["clip_thr"] if ok_sample else None,
        "example_rs_estimate_prominence": ok_sample["rs_estimate_prominence"] if ok_sample else None,
        "example_rs_estimate_prominence_corrected": ok_sample["rs_estimate_prominence_corrected"] if ok_sample else None,
        "example_rs_estimate_abs_x2_rule": ok_sample["rs_estimate_abs_x2_rule"] if ok_sample else None,
        "example_cfo_estimate_argmax": ok_sample["cfo_estimate_argmax"] if ok_sample else None,
        "example_cfo_estimate_band_interp": ok_sample["cfo_estimate_band_interp"] if ok_sample else None,
        "example_lr8_32": ok_sample["lr8_32"] if ok_sample else None,
    }
