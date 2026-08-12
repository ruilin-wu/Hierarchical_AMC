#!/usr/bin/env python3
"""Helper functions mechanically extracted from BOI+FFT-Generate-2018+2022.ipynb.

This module is intentionally additive: it does not run generation on import and
keeps the original notebook untouched.  The compact-vector generator imports
these functions to preserve the existing BOI+CFO-shift and FFT behavior.
"""

# ============================================================
# Extracted fixed BOI + CFO shift + FFT3 helpers
#
# Pipeline:
#   raw .tim
#     -> Fixed BOI detection
#     -> BOI filtering
#     -> frequency shift to baseband
#     -> FFT(|x|^2), FFT(x^2), FFT(x^4)
#     -> save payload .pt
#
# Important:
#   1. BOI algorithm is fixed and kept consistent with previous code.
#   2. No SSCA, no CC, no figure generation.
#   3. x_processed is saved in each sample payload as complex64.
#   4. 2018 samples with negative truth SNR are skipped before BOI/FFT.
#   5. Each sample payload follows:
#        version = "boi_shift_fft3_cfpattern_v1"
#
# Usage:
#   1) Change USER SETTINGS at the bottom.
#   2) Run this whole cell.
# ============================================================

import os
import gc
import math
import traceback
import numpy as np
import pandas as pd

try:
    import torch
except Exception:
    torch = None

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ============================================================
# 1. Dataset / I/Q loading
# ============================================================

def load_truth_labels(label_path):
    df = pd.read_csv(label_path, sep=r"\s+")

    expected = [
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

    if list(df.columns) != expected:
        raise ValueError(
            f"Unexpected label columns.\n"
            f"Got     : {list(df.columns)}\n"
            f"Expected: {expected}"
        )

    df["SignalType"] = df["SignalType"].astype(str).str.lower()

    D = df["DownsampleFactor"].to_numpy(dtype=np.float64)
    U = df["UpsampleFactor"].to_numpy(dtype=np.float64)
    Tbase = df["BaseSymbolPeriod"].to_numpy(dtype=np.float64)

    D_eff = np.where(D == 0, 1.0, D)
    U_eff = np.where(U == 0, 1.0, U)

    Rs = D_eff / (U_eff * Tbase)

    df["SymbolRate"] = Rs
    df["T0_eff"] = 1.0 / Rs
    df["CFO"] = df["CarrierOffset"].astype(np.float64)

    return df


def set_dataset(dataset_name, root_map, label_map):
    dataset_name = str(dataset_name)

    if dataset_name not in root_map:
        raise ValueError(
            f"Unsupported dataset_name={dataset_name}. "
            f"Use one of {list(root_map.keys())}."
        )

    root_data = root_map[dataset_name]
    label_path = os.path.join(root_data, label_map[dataset_name])

    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Label file not found: {label_path}")

    truth_df = load_truth_labels(label_path).set_index("SignalIndex", drop=False)

    return root_data, label_path, truth_df


def get_truth_params(truth_df, signal_index):
    signal_index = int(signal_index)

    if signal_index not in truth_df.index:
        raise ValueError(f"Signal index {signal_index} not found in truth labels.")

    row = truth_df.loc[signal_index]

    return {
        "signal_index": int(row["SignalIndex"]),
        "signal_type": str(row["SignalType"]),
        "base_symbol_period": float(row["BaseSymbolPeriod"]),
        "T0_eff": float(row["T0_eff"]),
        "symbol_rate": float(row["SymbolRate"]),
        "carrier_offset_label": float(row["CarrierOffset"]),
        "excess_bw": float(row["ExcessBandwidth"]),
        "upsample": float(row["UpsampleFactor"]),
        "downsample": float(row["DownsampleFactor"]),
        "inband_snr_db_label": float(row["InbandSNRdB"]),
        "noise_spectral_density_db_label": float(row["NoiseSpectralDensitydB"]),
    }


def global_to_path(root_data, gidx, n_per_batch, n_total):
    gidx = int(gidx)

    if gidx < 1 or gidx > int(n_total):
        raise ValueError(f"global index out of range: {gidx}, valid range is 1..{n_total}")

    batch_dir = (gidx - 1) // int(n_per_batch) + 1
    tim_path = os.path.join(root_data, f"Batch_Dir_{batch_dir}", f"signal_{gidx}.tim")

    return tim_path, batch_dir


def tim_to_complex_iq(tim_path, samples_per_sig):
    if not os.path.exists(tim_path):
        raise FileNotFoundError(f".tim file not found: {tim_path}")

    raw = np.fromfile(tim_path, dtype=np.float32)
    need = 2 + 2 * int(samples_per_sig)

    if raw.size < need:
        raise ValueError(f"{tim_path}: {raw.size} floats < required {need}")

    iq = raw[2:need]
    I = iq[0::2].astype(np.float64)
    Q = iq[1::2].astype(np.float64)
    x = I + 1j * Q

    if x.size != int(samples_per_sig):
        raise ValueError(f"{tim_path}: expected {samples_per_sig} samples, got {x.size}")

    return x.astype(np.complex128)


# ============================================================
# 2. Basic utilities
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def signal_rms_power(x):
    x = np.asarray(x, dtype=np.complex128)
    return float(np.mean(np.abs(x) ** 2))


def fftshift_freqs(n, fs=1.0):
    return np.fft.fftshift(np.fft.fftfreq(int(n), d=1.0 / float(fs)))


def wrap_freq_to_nyquist(f, fs=1.0):
    fs = float(fs)
    return ((float(f) + fs / 2.0) % fs) - fs / 2.0


def moving_average_same(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = int(max(1, w))

    if w == 1:
        return x.copy()

    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, kernel, mode="same")


def contiguous_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)

    if mask.size == 0:
        return []

    padded = np.r_[False, mask, False]
    changes = np.diff(padded.astype(np.int8))
    starts = np.where(changes == 1)[0]
    stops = np.where(changes == -1)[0] - 1

    return list(zip(starts, stops))


def interval_mask(n, left, right):
    n = int(n)
    left = int(max(0, min(left, n - 1)))
    right = int(max(0, min(right, n - 1)))

    if right < left:
        left, right = right, left

    m = np.zeros(n, dtype=bool)
    m[left:right + 1] = True
    return m


def to_builtin_number(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x
    return x


def to_small_scalar_dict(d):
    """
    Keep only scalar fields from dict.
    This avoids accidentally saving masks, arrays, or complex signals.
    """
    out = {}

    if not isinstance(d, dict):
        return out

    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        elif isinstance(v, dict):
            out[k] = to_small_scalar_dict(v)
        else:
            pass

    return out


# ============================================================
# 3. Fixed BOI detection + refinement
# ============================================================

def fixed_resolution_boi(
    x,
    fs,
    delta_f0,
    threshold_db,
    min_width_bins=2,
    eps=1e-12,
):
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    fs = float(fs)

    freqs = fftshift_freqs(n, fs=fs)
    df = fs / n

    X = np.fft.fftshift(np.fft.fft(x))
    raw_bin_power = (np.abs(X) ** 2) / (n ** 2)

    W = int(np.floor(n * float(delta_f0)))
    W = max(1, W)

    if W % 2 == 0:
        W += 1

    S = moving_average_same(raw_bin_power, W)

    half = W // 2
    valid_left = half
    valid_right = n - half

    if valid_right <= valid_left:
        raise ValueError(f"Invalid smoothing width W={W} for n={n}.")

    S_valid = S[valid_left:valid_right]

    q = 10.0 ** (float(threshold_db) / 10.0)
    S_min = float(np.min(S_valid))
    F = q * S_min

    boi_mask_valid = S_valid >= F
    runs_valid = contiguous_true_runs(boi_mask_valid)

    boi_list = []
    full_boi_mask = np.zeros(n, dtype=bool)

    for local_l, local_r in runs_valid:
        left = valid_left + local_l
        right = valid_left + local_r

        if local_l == 0:
            left = 0
        if local_r == len(boi_mask_valid) - 1:
            right = n - 1

        width_bins = int(right - left + 1)

        if width_bins < int(min_width_bins):
            continue

        m = interval_mask(n, left, right)
        full_boi_mask |= m

        power_raw = float(np.sum(raw_bin_power[m]))
        power_smooth = float(np.sum(S[m]))

        f_left = float(freqs[left])
        f_right = float(freqs[right])
        f_center = wrap_freq_to_nyquist(0.5 * (f_left + f_right), fs=fs)
        bandwidth = float(width_bins * df)

        boi_list.append({
            "left_bin": int(left),
            "right_bin": int(right),
            "width_bins": int(width_bins),
            "f_left": f_left,
            "f_right": f_right,
            "f_center": float(f_center),
            "bandwidth": bandwidth,
            "power_raw": power_raw,
            "power_smooth": power_smooth,
        })

    non_boi_mask = ~full_boi_mask

    valid_mask_full = np.zeros(n, dtype=bool)
    valid_mask_full[valid_left:valid_right] = True

    noise_mask = non_boi_mask & valid_mask_full

    if np.any(noise_mask):
        noise_floor_raw_per_bin = float(np.mean(raw_bin_power[noise_mask]))
        noise_floor_smooth_per_bin = float(np.mean(S[noise_mask]))
    else:
        noise_floor_raw_per_bin = float(np.min(raw_bin_power[valid_left:valid_right]))
        noise_floor_smooth_per_bin = float(S_min)

    for b in boi_list:
        noise_power_raw = noise_floor_raw_per_bin * b["width_bins"]
        signal_power_raw = max(b["power_raw"] - noise_power_raw, 0.0)

        noise_power_smooth = noise_floor_smooth_per_bin * b["width_bins"]
        signal_power_smooth = max(b["power_smooth"] - noise_power_smooth, 0.0)

        b["noise_power_hat_raw"] = float(noise_power_raw)
        b["signal_power_hat_raw"] = float(signal_power_raw)
        b["noise_power_hat_smooth"] = float(noise_power_smooth)
        b["signal_power_hat_smooth"] = float(signal_power_smooth)
        b["excess_power_score"] = float(signal_power_raw)

    boi_list = sorted(boi_list, key=lambda d: d["f_left"])

    return {
        "fs": float(fs),
        "n": int(n),
        "df": float(df),
        "W": int(W),
        "delta_f0": float(delta_f0),
        "threshold_db": float(threshold_db),
        "q_linear": float(q),
        "S_min": float(S_min),
        "threshold_F": float(F),
        "freqs": freqs,
        "raw_bin_power": raw_bin_power,
        "S_smooth": S,
        "boi_mask": full_boi_mask,
        "valid_left": int(valid_left),
        "valid_right": int(valid_right),
        "noise_floor_raw_per_bin": float(noise_floor_raw_per_bin),
        "noise_floor_smooth_per_bin": float(noise_floor_smooth_per_bin),
        "bois": boi_list,
    }


def select_boi(boi_result, mode="strongest"):
    bois = boi_result["bois"]

    if len(bois) == 0:
        raise RuntimeError("No BOI detected.")

    mode = str(mode).lower()

    if mode == "strongest":
        return max(bois, key=lambda b: b["excess_power_score"])

    if mode == "widest":
        return max(bois, key=lambda b: b["width_bins"])

    if mode == "first":
        return bois[0]

    raise ValueError(f"Unknown BOI select mode: {mode}")


def recompute_boi_metrics(boi_result, left, right, tag=None):
    n = int(boi_result["n"])
    freqs = boi_result["freqs"]
    raw = boi_result["raw_bin_power"]
    S = boi_result["S_smooth"]
    df = float(boi_result["df"])
    fs = float(boi_result["fs"])

    left = int(max(0, min(left, n - 1)))
    right = int(max(0, min(right, n - 1)))

    if right < left:
        left, right = right, left

    m = interval_mask(n, left, right)
    width_bins = int(right - left + 1)

    power_raw = float(np.sum(raw[m]))
    power_smooth = float(np.sum(S[m]))

    noise_power_raw = float(boi_result["noise_floor_raw_per_bin"] * width_bins)
    signal_power_raw = float(max(power_raw - noise_power_raw, 0.0))

    noise_power_smooth = float(boi_result["noise_floor_smooth_per_bin"] * width_bins)
    signal_power_smooth = float(max(power_smooth - noise_power_smooth, 0.0))

    out = {
        "left_bin": int(left),
        "right_bin": int(right),
        "width_bins": int(width_bins),
        "f_left": float(freqs[left]),
        "f_right": float(freqs[right]),
        "f_center": float(wrap_freq_to_nyquist(0.5 * (freqs[left] + freqs[right]), fs=fs)),
        "bandwidth": float(width_bins * df),
        "power_raw": float(power_raw),
        "power_smooth": float(power_smooth),
        "noise_power_hat_raw": float(noise_power_raw),
        "signal_power_hat_raw": float(signal_power_raw),
        "noise_power_hat_smooth": float(noise_power_smooth),
        "signal_power_hat_smooth": float(signal_power_smooth),
        "excess_power_score": float(signal_power_raw),
    }

    if tag is not None:
        out["tag"] = str(tag)

    return out


def merge_neighbors_around_selected(
    boi_result,
    selected,
    merge_gap_w_factor,
    min_neighbor_power_frac=0.001,
    eps=1e-12,
):
    max_gap_bins = int(float(merge_gap_w_factor) * boi_result["W"])
    bois = boi_result["bois"]

    left = int(selected["left_bin"])
    right = int(selected["right_bin"])
    main_power = max(float(selected["signal_power_hat_raw"]), eps)

    changed = True

    while changed:
        changed = False

        for b in bois:
            bl = int(b["left_bin"])
            br = int(b["right_bin"])

            if bl >= left and br <= right:
                continue

            if br < left:
                gap = left - br - 1
            elif bl > right:
                gap = bl - right - 1
            else:
                gap = 0

            neighbor_power = float(b["signal_power_hat_raw"])
            power_ok = neighbor_power >= min_neighbor_power_frac * main_power

            if gap <= max_gap_bins and power_ok:
                left = min(left, bl)
                right = max(right, br)
                changed = True

    merged = recompute_boi_metrics(boi_result, left, right, tag="merged_neighbors")
    merged["merge_max_gap_bins"] = int(max_gap_bins)

    return merged


def refine_boi_edges_hysteresis(
    boi_result,
    boi,
    boundary_threshold_db,
    max_expand_w_factor,
):
    S = boi_result["S_smooth"]
    n = int(boi_result["n"])

    max_expand_bins = int(float(max_expand_w_factor) * boi_result["W"])

    q2 = 10.0 ** (float(boundary_threshold_db) / 10.0)
    F2 = q2 * boi_result["S_min"]

    left = int(boi["left_bin"])
    right = int(boi["right_bin"])

    steps = 0
    while left > 0 and S[left - 1] >= F2 and steps < max_expand_bins:
        left -= 1
        steps += 1

    steps = 0
    while right < n - 1 and S[right + 1] >= F2 and steps < max_expand_bins:
        right += 1
        steps += 1

    refined = recompute_boi_metrics(boi_result, left, right, tag="hysteresis_refined")
    refined["boundary_threshold_db"] = float(boundary_threshold_db)
    refined["boundary_threshold_F"] = float(F2)
    refined["max_expand_bins"] = int(max_expand_bins)

    return refined


def estimate_boi_center_by_excess_power(boi_result, boi, eps=1e-12):
    freqs = boi_result["freqs"]
    raw = boi_result["raw_bin_power"]
    nf = float(boi_result["noise_floor_raw_per_bin"])

    left = int(boi["left_bin"])
    right = int(boi["right_bin"])

    idx = np.arange(left, right + 1)
    f = freqs[idx]

    w = raw[idx] - nf
    w = np.maximum(w, 0.0)

    if np.sum(w) <= eps:
        return float(boi["f_center"])

    return float(np.sum(f * w) / np.sum(w))


def compute_power_report(x, boi_result, boi_for_power, eps=1e-12):
    p_total = signal_rms_power(x)
    n = int(boi_result["n"])

    noise_floor = float(boi_result["noise_floor_raw_per_bin"])
    width_bins = int(boi_for_power["width_bins"])

    p_noise_fullband_hat = noise_floor * n
    p_noise_inband_hat = noise_floor * width_bins
    p_noise_outband_hat = noise_floor * (n - width_bins)

    p_signal_boi_local = float(boi_for_power["signal_power_hat_raw"])
    p_signal_fullband_excess = max(p_total - p_noise_fullband_hat, 0.0)

    snr_boi_local_db = 10.0 * np.log10((p_signal_boi_local + eps) / (p_noise_inband_hat + eps))
    snr_fullband_excess_db = 10.0 * np.log10((p_signal_fullband_excess + eps) / (p_noise_inband_hat + eps))

    return {
        "p_total": float(p_total),
        "noise_floor_raw_per_bin": float(noise_floor),
        "p_noise_fullband_hat": float(p_noise_fullband_hat),
        "p_noise_inband_hat": float(p_noise_inband_hat),
        "p_noise_outband_hat": float(p_noise_outband_hat),
        "p_signal_boi_local": float(p_signal_boi_local),
        "p_signal_fullband_excess": float(p_signal_fullband_excess),
        "p_signal_recommended": float(p_signal_fullband_excess),
        "snr_boi_local_db": float(snr_boi_local_db),
        "snr_fullband_excess_db": float(snr_fullband_excess_db),
        "snr_recommended_db": float(snr_fullband_excess_db),
    }


def choose_filter_expand_ratio(snr_est_db, cfg):
    snr_est_db = float(snr_est_db)

    if snr_est_db < cfg["SNR_LOW_TH_DB"]:
        return cfg["FILTER_EXPAND_RATIO_LOW_SNR"]

    if snr_est_db < cfg["SNR_HIGH_TH_DB"]:
        return cfg["FILTER_EXPAND_RATIO_MID_SNR"]

    return cfg["FILTER_EXPAND_RATIO_HIGH_SNR"]


def expand_boi_for_filtering(boi, fs, expand_ratio):
    b = dict(boi)

    bw = float(boi["bandwidth"])
    guard = float(expand_ratio) * bw

    f_left = max(float(boi["f_left"]) - guard, -float(fs) / 2.0)
    f_right = min(float(boi["f_right"]) + guard, float(fs) / 2.0)

    b["f_left"] = float(f_left)
    b["f_right"] = float(f_right)
    b["f_center"] = float(boi["f_center"])
    b["bandwidth"] = float(f_right - f_left)
    b["expanded_for_filtering"] = True
    b["filter_expand_ratio"] = float(expand_ratio)

    return b


def prepare_boi_for_filter_shift(x, boi_result, cfg):
    selected_initial = select_boi(
        boi_result,
        mode=cfg["BOI_SELECT_MODE"],
    )

    selected_merged = merge_neighbors_around_selected(
        boi_result=boi_result,
        selected=selected_initial,
        merge_gap_w_factor=cfg["MERGE_GAP_W_FACTOR"],
        min_neighbor_power_frac=0.001,
        eps=cfg["EPS"],
    )

    selected_refined = refine_boi_edges_hysteresis(
        boi_result=boi_result,
        boi=selected_merged,
        boundary_threshold_db=cfg["BOUNDARY_THRESHOLD_DB"],
        max_expand_w_factor=cfg["MAX_EXPAND_W_FACTOR"],
    )

    selected_for_power = dict(selected_refined)

    fc_midpoint = float(selected_for_power["f_center"])
    fc_centroid = estimate_boi_center_by_excess_power(
        boi_result=boi_result,
        boi=selected_for_power,
        eps=cfg["EPS"],
    )

    selected_for_power["f_center_midpoint"] = fc_midpoint
    selected_for_power["f_center"] = fc_centroid
    selected_for_power["f_center_method"] = "excess_power_centroid"

    power_report = compute_power_report(
        x=x,
        boi_result=boi_result,
        boi_for_power=selected_for_power,
        eps=cfg["EPS"],
    )

    if cfg["ADAPTIVE_FILTER_EXPAND"]:
        expand_ratio = choose_filter_expand_ratio(
            power_report["snr_boi_local_db"],
            cfg,
        )
    else:
        expand_ratio = cfg["FILTER_EXPAND_RATIO"]

    selected_for_power["filter_expand_ratio_selected_from_snr_db"] = float(
        power_report["snr_boi_local_db"]
    )
    selected_for_power["filter_expand_ratio_used"] = float(expand_ratio)

    selected_for_filter = expand_boi_for_filtering(
        boi=selected_for_power,
        fs=cfg["FS_IN"],
        expand_ratio=expand_ratio,
    )

    return {
        "selected_initial": selected_initial,
        "selected_merged": selected_merged,
        "selected_refined": selected_for_power,
        "selected_filter": selected_for_filter,
        "power_report": power_report,
    }


def run_fixed_boi_detector(x, cfg):
    boi_result = fixed_resolution_boi(
        x=x,
        fs=cfg["FS_IN"],
        delta_f0=cfg["DELTA_F0"],
        threshold_db=cfg["THRESHOLD_DB"],
        min_width_bins=2,
        eps=cfg["EPS"],
    )

    if len(boi_result["bois"]) == 0:
        raise RuntimeError("No BOI detected by fixed-resolution algorithm.")

    prepared = prepare_boi_for_filter_shift(
        x=x,
        boi_result=boi_result,
        cfg=cfg,
    )

    return {
        "algorithm": "fixed",
        "root_boi_result": boi_result,
        "boi_result_for_plot": boi_result,
        "prepared": prepared,
        "history": [],
    }


# ============================================================
# 4. BOI filter + CFO shift
# ============================================================

def fft_bandpass_extract(x, fs, f_left, f_right, guard_bins):
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    fs = float(fs)

    freqs = fftshift_freqs(n, fs=fs)
    X = np.fft.fftshift(np.fft.fft(x))

    df = fs / n
    guard_hz = int(guard_bins) * df

    lo = float(f_left) - guard_hz
    hi = float(f_right) + guard_hz

    if lo >= -fs / 2.0 and hi < fs / 2.0:
        mask = (freqs >= lo) & (freqs <= hi)
    else:
        lo_w = wrap_freq_to_nyquist(lo, fs=fs)
        hi_w = wrap_freq_to_nyquist(hi, fs=fs)

        if lo_w <= hi_w:
            mask = (freqs >= lo_w) & (freqs <= hi_w)
        else:
            mask = (freqs >= lo_w) | (freqs <= hi_w)

    X_filt = np.zeros_like(X)
    X_filt[mask] = X[mask]

    x_filt = np.fft.ifft(np.fft.ifftshift(X_filt))

    return x_filt, mask


def extract_filter_shift_noresample(x, boi, fs, guard_bins):
    """
    BOI filtering + frequency shift.
    No decimation. Output length remains 32768 and fs_out remains 1.0.
    """
    x = np.asarray(x, dtype=np.complex128)
    n = x.size
    fs = float(fs)

    x_bpf, bp_mask = fft_bandpass_extract(
        x=x,
        fs=fs,
        f_left=boi["f_left"],
        f_right=boi["f_right"],
        guard_bins=guard_bins,
    )

    t = np.arange(n, dtype=np.float64) / fs

    center = float(boi["f_center"])

    x_processed = x_bpf * np.exp(-1j * 2.0 * np.pi * center * t)

    return {
        "x_bpf": x_bpf,
        "x_processed": x_processed,
        "fs_in": float(fs),
        "fs_out": float(fs),
        "decim": 1,
        "frac_bw_out": float(boi["bandwidth"] / fs),
        "bandpass_mask": bp_mask,
        "boi_used": dict(boi),
        "mode": "boi_filter_shift_noresample",
        "do_shift": True,
        "applied_shift_hz": float(center),
    }


def run_one_fixed_boi_shift(signal_index, cfg, truth_df, root_data):
    dataset_name = cfg["DATASET_NAME"]

    truth = get_truth_params(truth_df, signal_index)

    tim_path, batch_dir = global_to_path(
        root_data=root_data,
        gidx=signal_index,
        n_per_batch=cfg["N_PER_BATCH"],
        n_total=cfg["N_TOTAL"],
    )

    x = tim_to_complex_iq(
        tim_path=tim_path,
        samples_per_sig=cfg["SAMPLES_PER_SIG"],
    )

    boi_pack = run_fixed_boi_detector(x, cfg)
    boi_result = boi_pack["boi_result_for_plot"]
    prepared = boi_pack["prepared"]

    boi_for_filter = prepared["selected_filter"]
    boi_filter_source = "selected_filter_expanded"

    max_filter_bw_frac = float(cfg.get("MAX_EXPANDED_FILTER_BW_FRAC", 0.95))
    expanded_bw_frac = float(boi_for_filter["bandwidth"] / cfg["FS_IN"])

    if expanded_bw_frac >= max_filter_bw_frac:
        boi_for_filter = prepared["selected_refined"]
        boi_filter_source = "selected_refined_fallback"

    stepB = extract_filter_shift_noresample(
        x=x,
        boi=boi_for_filter,
        fs=cfg["FS_IN"],
        guard_bins=cfg["BOI_GUARD_BINS"],
    )

    stepB["boi_filter_source"] = boi_filter_source
    stepB["expanded_bw_frac"] = float(expanded_bw_frac)
    stepB["max_expanded_filter_bw_frac"] = float(max_filter_bw_frac)

    estimated_center = float(prepared["selected_refined"]["f_center"])
    center_shift_hz = estimated_center
    truth_guide_cfo = float(truth["carrier_offset_label"]) - estimated_center

    return {
        "dataset_name": dataset_name,
        "signal_index": int(signal_index),
        "truth": truth,
        "x_original": x,
        "x_processed": np.asarray(stepB["x_processed"], dtype=np.complex128),
        "fs_processed": float(stepB["fs_out"]),
        "estimated_boi_center_hz": float(estimated_center),
        "center_shift_hz": float(center_shift_hz),
        "truth_guide_cfo": float(truth_guide_cfo),
        "tim_path": tim_path,
        "batch_dir": int(batch_dir),
        "boi_result": boi_result,
        "prepared": prepared,
        "stepB": stepB,
    }


# ============================================================
# 5. FFT helpers
# ============================================================

def fft3_freq_axis(nfft=32768, fs=1.0):
    return np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / float(fs)))


def compute_fft_mag_db_relative(z, nfft=32768, fs=1.0, window="hann", eps=1e-12):
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    nfft = int(nfft)

    if z.size < nfft:
        zz = np.zeros(nfft, dtype=np.complex128)
        zz[:z.size] = z
    else:
        zz = z[:nfft].copy()

    window = str(window).lower()

    if window == "hann":
        w = np.hanning(nfft)
    elif window in ["none", "rect", "rectangular"]:
        w = np.ones(nfft, dtype=np.float64)
    else:
        raise ValueError("window must be 'hann' or 'none'.")

    Z = np.fft.fftshift(np.fft.fft(zz * w, n=nfft))
    freq = fft3_freq_axis(nfft=nfft, fs=fs)

    mag = np.abs(Z)
    mag_db = 20.0 * np.log10(mag / (np.max(mag) + eps) + eps)

    return freq, mag_db


def find_top_peaks_1d(
    freq,
    mag_db,
    top_k=20,
    min_sep_bins=8,
    xlim=(-0.5, 0.5),
):
    freq = np.asarray(freq, dtype=np.float64).reshape(-1)
    mag_db = np.asarray(mag_db, dtype=np.float64).reshape(-1)

    mask = (
        np.isfinite(freq)
        & np.isfinite(mag_db)
        & (freq >= float(xlim[0]))
        & (freq <= float(xlim[1]))
    )

    idx_all = np.where(mask)[0]

    if idx_all.size == 0:
        return []

    order = idx_all[np.argsort(mag_db[idx_all])[::-1]]

    selected = []
    selected_idx = []

    for idx in order:
        if len(selected) >= int(top_k):
            break

        if all(abs(int(idx) - int(j)) >= int(min_sep_bins) for j in selected_idx):
            selected_idx.append(int(idx))
            selected.append({
                "freq": float(freq[idx]),
                "mag_db": float(mag_db[idx]),
                "bin": int(idx),
            })

    return selected


def sample_mag_db_near_freq(
    freq,
    mag_db,
    target_freq,
    fs=1.0,
    search_bins=3,
):
    freq = np.asarray(freq, dtype=np.float64).reshape(-1)
    mag_db = np.asarray(mag_db, dtype=np.float64).reshape(-1)

    f0 = wrap_freq_to_nyquist(float(target_freq), fs=fs)

    idx0 = int(np.argmin(np.abs(freq - f0)))

    l = max(0, idx0 - int(search_bins))
    r = min(freq.size - 1, idx0 + int(search_bins))

    local = np.arange(l, r + 1)
    best = local[np.argmax(mag_db[local])]

    return {
        "target_freq": float(f0),
        "nearest_bin": int(idx0),
        "best_bin": int(best),
        "best_freq": float(freq[best]),
        "mag_db": float(mag_db[best]),
        "freq_error": float(freq[best] - f0),
    }


def make_integer_rs_guides(rs, cfo_term=0.0, kmax=5, fs=1.0):
    """
    Generic guide:
        alpha = cfo_term + k * Rs
    """
    out = []

    for k in range(-int(kmax), int(kmax) + 1):
        f = wrap_freq_to_nyquist(float(cfo_term) + k * float(rs), fs=fs)
        out.append({
            "k": int(k),
            "freq": float(f),
        })

    return out


def make_half_rs_guides(rs, cfo_term=0.0, kmax=5, fs=1.0):
    """
    Generic half-symbol guide:
        alpha = cfo_term + (k + 0.5) * Rs
    """
    out = []

    for k in range(-int(kmax), int(kmax) + 1):
        f = wrap_freq_to_nyquist(float(cfo_term) + (k + 0.5) * float(rs), fs=fs)
        out.append({
            "k": int(k),
            "freq": float(f),
        })

    return out


def score_guides_on_fft(freq, mag_db, guides, fs=1.0, search_bins=3):
    rows = []

    for g in guides:
        row = dict(g)
        row.update(
            sample_mag_db_near_freq(
                freq=freq,
                mag_db=mag_db,
                target_freq=g["freq"],
                fs=fs,
                search_bins=search_bins,
            )
        )
        rows.append(row)

    rows = sorted(rows, key=lambda d: d["mag_db"], reverse=True)
    return rows


def build_fft_branch(
    z,
    description,
    nfft,
    fs,
    window,
    top_k,
    peak_min_sep_bins,
    guide_scores=None,
    guide_scores_integer_Rs=None,
    guide_scores_half_Rs=None,
    mag_dtype=np.float16,
):
    freq, mag_db = compute_fft_mag_db_relative(
        z=z,
        nfft=nfft,
        fs=fs,
        window=window,
    )

    branch = {
        "description": str(description),
        "mag_db": np.asarray(mag_db, dtype=mag_dtype),
        "top_peaks": find_top_peaks_1d(
            freq=freq,
            mag_db=mag_db,
            top_k=top_k,
            min_sep_bins=peak_min_sep_bins,
            xlim=(-fs / 2.0, fs / 2.0),
        ),
    }

    if guide_scores is not None:
        branch["guide_scores"] = guide_scores

    if guide_scores_integer_Rs is not None:
        branch["guide_scores_integer_Rs"] = guide_scores_integer_Rs

    if guide_scores_half_Rs is not None:
        branch["guide_scores_half_Rs"] = guide_scores_half_Rs

    return branch


def infer_candidate_pattern_hint_from_fft(payload):
    """
    Lightweight hint only, not final modulation classification.
    It uses relative FFT peak evidence after BOI shift.
    """
    try:
        abs_x2_top = payload["fft"]["abs_x2"]["top_peaks"]
        x2_top = payload["fft"]["x2"]["top_peaks"]
        x4_top = payload["fft"]["x4"]["top_peaks"]

        abs_top_db = abs_x2_top[0]["mag_db"] if len(abs_x2_top) else -999.0
        x2_top_db = x2_top[0]["mag_db"] if len(x2_top) else -999.0
        x4_top_db = x4_top[0]["mag_db"] if len(x4_top) else -999.0

        # Since mag_db is relative to each branch max, top1 is usually near 0 dB.
        # So this hint mainly checks whether guide-aligned peaks exist.
        x2_int_best = payload["fft"]["x2"]["guide_scores_integer_Rs"][0]["mag_db"]
        x4_int_best = payload["fft"]["x4"]["guide_scores_integer_Rs"][0]["mag_db"]

        if x2_int_best > -10.0:
            return "second_order_conj_present_bpsk_like_or_real_structure"

        if x4_int_best > -10.0:
            return "fourth_power_conj_present_qpsk_8psk_dqpsk_like"

        if abs_top_db > -3.0:
            return "nonconj_symbol_rate_structure_present"

        return "weak_or_ambiguous_fft_pattern"

    except Exception:
        return "hint_unavailable"


# ============================================================
# 6. Payload builder
# ============================================================

def build_boi_shift_fft3_payload(run, cfg):
    truth = run["truth"]
    prepared = run["prepared"]
    stepB = run["stepB"]

    x_processed = np.asarray(run["x_processed"], dtype=np.complex128)
    fs = float(run["fs_processed"])
    n = int(x_processed.size)

    nfft = int(cfg["FFT_NFFT"])
    window = str(cfg["FFT_WINDOW"])

    rs = float(truth["symbol_rate"])
    residual_cfo = float(run["truth_guide_cfo"])

    kmax_int = int(cfg["GUIDE_KMAX_INTEGER_RS"])
    kmax_half = int(cfg["GUIDE_KMAX_HALF_RS"])
    search_bins = int(cfg["GUIDE_SEARCH_BINS"])

    freq_axis = fft3_freq_axis(nfft=nfft, fs=fs)

    # ------------------------------------------------------------
    # Branch 1:
    #   FFT(|x|^2)
    #   Non-conjugate CF guide:
    #       alpha = k * Rs
    # ------------------------------------------------------------
    z_abs_x2 = np.abs(x_processed) ** 2

    freq_abs_x2, mag_abs_x2 = compute_fft_mag_db_relative(
        z=z_abs_x2,
        nfft=nfft,
        fs=fs,
        window=window,
    )

    abs_x2_integer_guides = make_integer_rs_guides(
        rs=rs,
        cfo_term=0.0,
        kmax=kmax_int,
        fs=fs,
    )

    abs_x2_guide_scores = score_guides_on_fft(
        freq=freq_abs_x2,
        mag_db=mag_abs_x2,
        guides=abs_x2_integer_guides,
        fs=fs,
        search_bins=search_bins,
    )

    # ------------------------------------------------------------
    # Branch 2:
    #   FFT(x^2)
    #   Conjugate CF guide for x(t):
    #       alpha = 2 * residual_CFO + k * Rs
    #   Half-Rs diagnostic:
    #       alpha = 2 * residual_CFO + (k+0.5) * Rs
    # ------------------------------------------------------------
    z_x2 = x_processed ** 2

    freq_x2, mag_x2 = compute_fft_mag_db_relative(
        z=z_x2,
        nfft=nfft,
        fs=fs,
        window=window,
    )

    x2_integer_guides = make_integer_rs_guides(
        rs=rs,
        cfo_term=2.0 * residual_cfo,
        kmax=kmax_int,
        fs=fs,
    )

    x2_half_guides = make_half_rs_guides(
        rs=rs,
        cfo_term=2.0 * residual_cfo,
        kmax=kmax_half,
        fs=fs,
    )

    x2_guide_scores_integer = score_guides_on_fft(
        freq=freq_x2,
        mag_db=mag_x2,
        guides=x2_integer_guides,
        fs=fs,
        search_bins=search_bins,
    )

    x2_guide_scores_half = score_guides_on_fft(
        freq=freq_x2,
        mag_db=mag_x2,
        guides=x2_half_guides,
        fs=fs,
        search_bins=search_bins,
    )

    # ------------------------------------------------------------
    # Branch 3:
    #   FFT(x^4)
    #   Conjugate CF guide for x^2(t):
    #       alpha = 4 * residual_CFO + k * Rs
    #   Half-Rs diagnostic:
    #       alpha = 4 * residual_CFO + (k+0.5) * Rs
    # ------------------------------------------------------------
    z_x4 = x_processed ** 4

    freq_x4, mag_x4 = compute_fft_mag_db_relative(
        z=z_x4,
        nfft=nfft,
        fs=fs,
        window=window,
    )

    x4_integer_guides = make_integer_rs_guides(
        rs=rs,
        cfo_term=4.0 * residual_cfo,
        kmax=kmax_int,
        fs=fs,
    )

    x4_half_guides = make_half_rs_guides(
        rs=rs,
        cfo_term=4.0 * residual_cfo,
        kmax=kmax_half,
        fs=fs,
    )

    x4_guide_scores_integer = score_guides_on_fft(
        freq=freq_x4,
        mag_db=mag_x4,
        guides=x4_integer_guides,
        fs=fs,
        search_bins=search_bins,
    )

    x4_guide_scores_half = score_guides_on_fft(
        freq=freq_x4,
        mag_db=mag_x4,
        guides=x4_half_guides,
        fs=fs,
        search_bins=search_bins,
    )

    # ------------------------------------------------------------
    # Power fields
    # ------------------------------------------------------------
    P_total = signal_rms_power(run["x_original"])
    P_x_bpf = signal_rms_power(stepB["x_bpf"])
    P_x_processed = signal_rms_power(stepB["x_processed"])

    power_report = prepared["power_report"]

    # ------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------
    payload = {
        "version": "boi_shift_fft3_cfpattern_v1",

        "meta": {
            "dataset_name": str(run["dataset_name"]),
            "signal_index": int(run["signal_index"]),
            "signal_type": str(truth["signal_type"]).upper(),
            "n": int(n),
            "fs": float(fs),
        },

        "truth": {
            "T0_eff": float(truth["T0_eff"]),
            "symbol_rate": float(truth["symbol_rate"]),
            "carrier_offset_label": float(truth["carrier_offset_label"]),
            "inband_snr_db_label": float(truth["inband_snr_db_label"]),
            "excess_bw": float(truth["excess_bw"]),
        },

        "preprocess": {
            "boi_algorithm": "fixed",
            "preprocess_mode": "boi_filter_shift",
            "estimated_boi_center_hz": float(run["estimated_boi_center_hz"]),
            "center_shift_hz": float(run["center_shift_hz"]),
            "truth_guide_cfo": float(run["truth_guide_cfo"]),

            "boi_refined_left": float(prepared["selected_refined"]["f_left"]),
            "boi_refined_right": float(prepared["selected_refined"]["f_right"]),
            "boi_refined_bw": float(prepared["selected_refined"]["bandwidth"]),

            "boi_filter_left": float(prepared["selected_filter"]["f_left"]),
            "boi_filter_right": float(prepared["selected_filter"]["f_right"]),
            "boi_filter_bw": float(prepared["selected_filter"]["bandwidth"]),
            "boi_filter_expand_ratio": float(prepared["selected_filter"].get("filter_expand_ratio", np.nan)),

            "stepB_mode": str(stepB["mode"]),
            "stepB_decim": int(stepB["decim"]),
            "stepB_fs_out": float(stepB["fs_out"]),
        },

        "power": {
            "P_total": float(P_total),
            "P_x_bpf": float(P_x_bpf),
            "P_x_processed": float(P_x_processed),
            "p_signal_boi_local": float(power_report["p_signal_boi_local"]),
            "p_noise_inband_hat": float(power_report["p_noise_inband_hat"]),
            "snr_recommended_db": float(power_report["snr_recommended_db"]),
        },

        "fft_config": {
            "nfft": int(nfft),
            "window": str(window),
            "fftshift": True,
            "freq_axis_reconstructable": True,
            "mag_format": "db_relative_to_max",
        },

        "x_processed": np.asarray(x_processed, dtype=np.complex64),

        "fft": {
            "abs_x2": {
                "description": "FFT(|x|^2), corresponds to x(t) non-conj CF",
                "mag_db": np.asarray(mag_abs_x2, dtype=np.float16),
                "top_peaks": find_top_peaks_1d(
                    freq=freq_abs_x2,
                    mag_db=mag_abs_x2,
                    top_k=cfg["FFT_TOP_K"],
                    min_sep_bins=cfg["FFT_PEAK_MIN_SEP_BINS"],
                    xlim=(-fs / 2.0, fs / 2.0),
                ),
                "guide_scores": abs_x2_guide_scores,
            },

            "x2": {
                "description": "FFT(x^2), corresponds to x(t) conj CF",
                "mag_db": np.asarray(mag_x2, dtype=np.float16),
                "top_peaks": find_top_peaks_1d(
                    freq=freq_x2,
                    mag_db=mag_x2,
                    top_k=cfg["FFT_TOP_K"],
                    min_sep_bins=cfg["FFT_PEAK_MIN_SEP_BINS"],
                    xlim=(-fs / 2.0, fs / 2.0),
                ),
                "guide_scores_integer_Rs": x2_guide_scores_integer,
                "guide_scores_half_Rs": x2_guide_scores_half,
            },

            "x4": {
                "description": "FFT(x^4), corresponds to x^2(t) conj CF",
                "mag_db": np.asarray(mag_x4, dtype=np.float16),
                "top_peaks": find_top_peaks_1d(
                    freq=freq_x4,
                    mag_db=mag_x4,
                    top_k=cfg["FFT_TOP_K"],
                    min_sep_bins=cfg["FFT_PEAK_MIN_SEP_BINS"],
                    xlim=(-fs / 2.0, fs / 2.0),
                ),
                "guide_scores_integer_Rs": x4_guide_scores_integer,
                "guide_scores_half_Rs": x4_guide_scores_half,
            },
        },

        "summary": {
            "fft_abs_x2_top1_freq": float(
                find_top_peaks_1d(freq_abs_x2, mag_abs_x2, top_k=1)[0]["freq"]
            ),
            "fft_x2_top1_freq": float(
                find_top_peaks_1d(freq_x2, mag_x2, top_k=1)[0]["freq"]
            ),
            "fft_x4_top1_freq": float(
                find_top_peaks_1d(freq_x4, mag_x4, top_k=1)[0]["freq"]
            ),
            "candidate_pattern_hint": "",
        },
    }

    payload["summary"]["candidate_pattern_hint"] = infer_candidate_pattern_hint_from_fft(payload)

    # x_processed is saved as complex64; all FFT computations above still use complex128.
    return payload


# ============================================================
# 7. Single sample and batch saving
# ============================================================

def run_one_and_build_payload(signal_index, cfg, truth_df, root_data):
    run = None

    try:
        run = run_one_fixed_boi_shift(
            signal_index=int(signal_index),
            cfg=cfg,
            truth_df=truth_df,
            root_data=root_data,
        )

        payload = build_boi_shift_fft3_payload(run, cfg)

        return payload

    finally:
        if isinstance(run, dict):
            run.pop("x_original", None)
            run.pop("x_processed", None)
            run.pop("boi_result", None)
            run.pop("prepared", None)
            run.pop("stepB", None)

        del run
        gc.collect()

        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def should_skip_negative_truth_snr(dataset_name, signal_index, truth_df):
    """Skip 2018 negative-truth-SNR samples before BOI/FFT computation."""
    if str(dataset_name) != "2018":
        return False, None

    truth = get_truth_params(truth_df, int(signal_index))
    truth_snr = float(truth["inband_snr_db_label"])
    return truth_snr < 0.0, truth


def make_negative_snr_skip_row(dataset_name, signal_index, truth, batch_id=None):
    row = {
        "dataset_name": str(dataset_name),
        "signal_index": int(signal_index),
        "signal_type": str(truth.get("signal_type", "")).upper(),
        "truth_snr": float(truth["inband_snr_db_label"]),
        "skip_reason": "negative_truth_snr_2018",
        "error": "SKIP_NEGATIVE_TRUTH_SNR",
        "traceback": "",
    }
    if batch_id is not None:
        row["batch_id"] = int(batch_id)
    return row

# The original notebook save/generate entrypoints are intentionally omitted here.
# compact_fft3_vector_generator.py owns all saving and writes no CSV files.
