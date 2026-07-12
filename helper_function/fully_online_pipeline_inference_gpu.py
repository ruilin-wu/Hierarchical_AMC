"""GPU-oriented fully online inference for the BOI/FFT -> LR -> CC/CAP flow.

This module keeps the same signal-processing decisions and feature definitions
as ``fully_online_pipeline_inference`` while moving the numerically heavy online
path to PyTorch tensors on the selected device. Full-length IQ vectors stay on
the selected device through BOI, FFT, LR, and Rs/CFO estimation. By default,
QPSK-like CC matrix construction is offloaded to the CPU and only the cropped
11x15 CAP input is sent back to the selected device.
"""

from __future__ import annotations

import gc
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from . import pre_cc_matrix_core_from_old as cc
from . import pre_compact_fft3_vector_generator as compact
from .fully_online_pipeline_inference import (
    CAPNet,
    COARSE_CLASS_NAMES,
    FINE_CLASS_NAMES,
    InferenceResult,
    _normalize_label,
    _signal_path,
    _truth_column,
    load_truth_table,
    make_default_config,
    print_and_plot_cm,
)


def _wrap_freq_scalar(f: float, fs: float = 1.0) -> float:
    fs = float(fs)
    return ((float(f) + fs / 2.0) % fs) - fs / 2.0


def _wrap_freq_torch(f: torch.Tensor, fs: float = 1.0):
    fs = float(fs)
    return torch.remainder(f + fs / 2.0, fs) - fs / 2.0


def _freq_axis_torch(n: int, fs: float, device: torch.device, dtype: torch.dtype):
    return (torch.arange(int(n), device=device, dtype=dtype) - int(n) // 2) * (
        float(fs) / float(n)
    )


def _hann_window_torch(n: int, device: torch.device, dtype: torch.dtype):
    if int(n) <= 1:
        return torch.ones(int(n), device=device, dtype=dtype)
    idx = torch.arange(int(n), device=device, dtype=dtype)
    return 0.5 - 0.5 * torch.cos(2.0 * torch.pi * idx / float(int(n) - 1))


def tim_to_complex_iq_torch(tim_path: Path, samples_per_sig: int, device: torch.device):
    """Load one .tim file and move the IQ vector to the target device."""
    tim_path = Path(tim_path)
    if not tim_path.exists():
        raise FileNotFoundError(f".tim file not found: {tim_path}")
    raw = np.fromfile(tim_path, dtype=np.float32)
    need = 2 + 2 * int(samples_per_sig)
    if raw.size < need:
        raise ValueError(f"{tim_path}: {raw.size} floats < required {need}")
    iq = torch.from_numpy(raw[2:need].copy()).to(device=device, dtype=torch.float64)
    x = torch.complex(iq[0::2], iq[1::2])
    if int(x.numel()) != int(samples_per_sig):
        raise ValueError(f"{tim_path}: expected {samples_per_sig} samples, got {x.numel()}")
    return x


def tim_to_complex_iq_batch_torch(
    tim_paths: Sequence[Path],
    samples_per_sig: int,
    device: torch.device,
):
    """Load a chunk of .tim files and move the whole chunk to the target device once."""
    tim_paths = [Path(path) for path in tim_paths]
    samples_per_sig = int(samples_per_sig)
    need = 2 + 2 * samples_per_sig
    batch = np.empty((len(tim_paths), samples_per_sig), dtype=np.complex128)
    for row, path in enumerate(tim_paths):
        if not path.exists():
            raise FileNotFoundError(f".tim file not found: {path}")
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size < need:
            raise ValueError(f"{path}: {raw.size} floats < required {need}")
        iq = raw[2:need]
        batch[row] = iq[0::2].astype(np.float64) + 1j * iq[1::2].astype(np.float64)
    return torch.from_numpy(batch).to(device=device)


def _moving_average_same_torch(x: torch.Tensor, width: int):
    width = int(max(1, width))
    if width == 1:
        return x.clone()
    kernel = torch.ones(1, 1, width, device=x.device, dtype=x.dtype) / float(width)
    y = torch.nn.functional.conv1d(x.reshape(1, 1, -1), kernel, padding=width // 2)
    return y.reshape(-1)


def _contiguous_true_runs_torch(mask: torch.Tensor, valid_left: int, n: int):
    mask_i = mask.to(torch.int8)
    padded = torch.cat(
        [
            torch.zeros(1, device=mask.device, dtype=torch.int8),
            mask_i,
            torch.zeros(1, device=mask.device, dtype=torch.int8),
        ]
    )
    diff = padded[1:] - padded[:-1]
    starts_local = torch.nonzero(diff == 1, as_tuple=False).reshape(-1)
    stops_local = torch.nonzero(diff == -1, as_tuple=False).reshape(-1) - 1
    starts = starts_local + int(valid_left)
    stops = stops_local + int(valid_left)
    starts = torch.where(starts_local == 0, torch.zeros_like(starts), starts)
    stops = torch.where(
        stops_local == int(mask.numel()) - 1,
        torch.full_like(stops, int(n) - 1),
        stops,
    )
    widths = stops - starts + 1
    keep = widths >= 2
    return starts[keep].to(torch.long), stops[keep].to(torch.long)


def _scalar_tensor(value, device: torch.device, dtype=torch.float64):
    return torch.as_tensor(value, device=device, dtype=dtype)


def _empty_bois(device: torch.device):
    return {
        "left_bin": torch.empty(0, device=device, dtype=torch.long),
        "right_bin": torch.empty(0, device=device, dtype=torch.long),
        "width_bins": torch.empty(0, device=device, dtype=torch.float64),
        "f_left": torch.empty(0, device=device, dtype=torch.float64),
        "f_right": torch.empty(0, device=device, dtype=torch.float64),
        "f_center": torch.empty(0, device=device, dtype=torch.float64),
        "bandwidth": torch.empty(0, device=device, dtype=torch.float64),
        "power_raw": torch.empty(0, device=device, dtype=torch.float64),
        "power_smooth": torch.empty(0, device=device, dtype=torch.float64),
        "noise_power_hat_raw": torch.empty(0, device=device, dtype=torch.float64),
        "signal_power_hat_raw": torch.empty(0, device=device, dtype=torch.float64),
        "noise_power_hat_smooth": torch.empty(0, device=device, dtype=torch.float64),
        "signal_power_hat_smooth": torch.empty(0, device=device, dtype=torch.float64),
        "excess_power_score": torch.empty(0, device=device, dtype=torch.float64),
    }


def _boi_masks_from_edges(lefts: torch.Tensor, rights: torch.Tensor, n: int):
    positions = torch.arange(int(n), device=lefts.device, dtype=torch.long)
    return (positions[None, :] >= lefts[:, None]) & (positions[None, :] <= rights[:, None])


def _boi_metrics_from_edges(boi_result: Mapping, left, right):
    n = int(boi_result["n"])
    device = boi_result["raw_bin_power"].device
    left = torch.as_tensor(left, device=device, dtype=torch.long).clamp(0, n - 1)
    right = torch.as_tensor(right, device=device, dtype=torch.long).clamp(0, n - 1)
    lo = torch.minimum(left, right)
    hi = torch.maximum(left, right)
    width_bins_i = hi - lo + 1
    width_bins = width_bins_i.to(torch.float64)
    positions = torch.arange(n, device=device, dtype=torch.long)
    mask = (positions >= lo) & (positions <= hi)
    raw = boi_result["raw_bin_power"]
    smooth = boi_result["S_smooth"]
    freqs = boi_result["freqs"]
    fs = float(boi_result["fs"])
    df = float(boi_result["df"])
    power_raw = torch.sum(torch.where(mask, raw, torch.zeros_like(raw)))
    power_smooth = torch.sum(torch.where(mask, smooth, torch.zeros_like(smooth)))
    noise_power_raw = boi_result["noise_floor_raw_per_bin"] * width_bins
    noise_power_smooth = boi_result["noise_floor_smooth_per_bin"] * width_bins
    signal_power_raw = torch.clamp(power_raw - noise_power_raw, min=0.0)
    signal_power_smooth = torch.clamp(power_smooth - noise_power_smooth, min=0.0)
    f_left = freqs[lo]
    f_right = freqs[hi]
    return {
        "left_bin": lo,
        "right_bin": hi,
        "width_bins": width_bins,
        "f_left": f_left,
        "f_right": f_right,
        "f_center": _wrap_freq_torch(0.5 * (f_left + f_right), fs),
        "bandwidth": width_bins * df,
        "power_raw": power_raw,
        "power_smooth": power_smooth,
        "noise_power_hat_raw": noise_power_raw,
        "signal_power_hat_raw": signal_power_raw,
        "noise_power_hat_smooth": noise_power_smooth,
        "signal_power_hat_smooth": signal_power_smooth,
        "excess_power_score": signal_power_raw,
    }


def fixed_resolution_boi_torch(x: torch.Tensor, cfg: Mapping):
    x = x.to(dtype=torch.complex128)
    n = int(x.numel())
    fs = float(cfg["FS_IN"])
    df = fs / float(n)
    rdtype = torch.float64
    freqs = _freq_axis_torch(n, fs, x.device, rdtype)

    X = torch.fft.fftshift(torch.fft.fft(x))
    raw_bin_power = (torch.abs(X) ** 2) / float(n**2)

    width = int(math.floor(n * float(cfg["DELTA_F0"])))
    width = max(1, width)
    if width % 2 == 0:
        width += 1

    smooth = _moving_average_same_torch(raw_bin_power, width)
    half = width // 2
    valid_left = half
    valid_right = n - half
    if valid_right <= valid_left:
        raise ValueError(f"Invalid smoothing width W={width} for n={n}.")

    smooth_valid = smooth[valid_left:valid_right]
    q = 10.0 ** (float(cfg["THRESHOLD_DB"]) / 10.0)
    smooth_min = torch.min(smooth_valid)
    threshold = smooth_min * q
    boi_mask_valid = smooth_valid >= threshold
    starts, stops = _contiguous_true_runs_torch(boi_mask_valid, valid_left, n)
    if starts.numel() == 0:
        full_boi_mask = torch.zeros(n, device=x.device, dtype=torch.bool)
        bois = _empty_bois(x.device)
    else:
        boi_masks = _boi_masks_from_edges(starts, stops, n)
        full_boi_mask = torch.any(boi_masks, dim=0)
        width_bins_i = stops - starts + 1
        width_bins = width_bins_i.to(rdtype)
        power_raw = torch.sum(boi_masks.to(rdtype) * raw_bin_power[None, :], dim=1)
        power_smooth = torch.sum(boi_masks.to(rdtype) * smooth[None, :], dim=1)
        f_left = freqs[starts]
        f_right = freqs[stops]
        bois = {
            "left_bin": starts,
            "right_bin": stops,
            "width_bins": width_bins,
            "f_left": f_left,
            "f_right": f_right,
            "f_center": _wrap_freq_torch(0.5 * (f_left + f_right), fs),
            "bandwidth": width_bins * df,
            "power_raw": power_raw,
            "power_smooth": power_smooth,
        }

    non_boi_mask = ~full_boi_mask
    valid_mask = torch.zeros(n, device=x.device, dtype=torch.bool)
    valid_mask[valid_left:valid_right] = True
    noise_mask = non_boi_mask & valid_mask
    noise_count = torch.sum(noise_mask).to(rdtype)
    noise_floor_raw = torch.where(
        noise_count > 0,
        torch.sum(torch.where(noise_mask, raw_bin_power, torch.zeros_like(raw_bin_power)))
        / torch.clamp(noise_count, min=1.0),
        torch.min(raw_bin_power[valid_left:valid_right]),
    )
    noise_floor_smooth = torch.where(
        noise_count > 0,
        torch.sum(torch.where(noise_mask, smooth, torch.zeros_like(smooth)))
        / torch.clamp(noise_count, min=1.0),
        smooth_min,
    )

    if starts.numel() != 0:
        noise_power_raw = noise_floor_raw * bois["width_bins"]
        noise_power_smooth = noise_floor_smooth * bois["width_bins"]
        bois["noise_power_hat_raw"] = noise_power_raw
        bois["signal_power_hat_raw"] = torch.clamp(bois["power_raw"] - noise_power_raw, min=0.0)
        bois["noise_power_hat_smooth"] = noise_power_smooth
        bois["signal_power_hat_smooth"] = torch.clamp(
            bois["power_smooth"] - noise_power_smooth,
            min=0.0,
        )
        bois["excess_power_score"] = bois["signal_power_hat_raw"]

    return {
        "fs": fs,
        "n": n,
        "df": df,
        "W": int(width),
        "S_min": smooth_min,
        "freqs": freqs,
        "raw_bin_power": raw_bin_power,
        "S_smooth": smooth,
        "boi_mask": full_boi_mask,
        "valid_left": int(valid_left),
        "valid_right": int(valid_right),
        "noise_floor_raw_per_bin": noise_floor_raw,
        "noise_floor_smooth_per_bin": noise_floor_smooth,
        "bois": bois,
    }


def _select_boi(boi_result: Mapping, mode: str):
    bois = boi_result["bois"]
    if bois["left_bin"].numel() == 0:
        raise RuntimeError("No BOI detected.")
    mode = str(mode).lower()
    if mode == "strongest":
        idx = torch.argmax(bois["excess_power_score"])
    if mode == "widest":
        idx = torch.argmax(bois["width_bins"])
    elif mode == "first":
        idx = torch.zeros((), device=bois["left_bin"].device, dtype=torch.long)
    elif mode != "strongest":
        raise ValueError(f"Unknown BOI select mode: {mode}")
    return {key: value[idx] for key, value in bois.items()}


def _recompute_boi_metrics(boi_result: Mapping, left, right, tag: str | None = None):
    return _boi_metrics_from_edges(boi_result, left, right)


def _merge_neighbors_around_selected(boi_result: Mapping, selected: Mapping, cfg: Mapping):
    max_gap_bins = int(float(cfg["MERGE_GAP_W_FACTOR"]) * int(boi_result["W"]))
    bois = boi_result["bois"]
    left = selected["left_bin"]
    right = selected["right_bin"]
    main_power = torch.clamp(selected["signal_power_hat_raw"], min=float(cfg["EPS"]))
    bl = bois["left_bin"]
    br = bois["right_bin"]
    signal = bois["signal_power_hat_raw"]
    for _ in range(max(1, int(bl.numel()))):
        inside = (bl >= left) & (br <= right)
        gap = torch.where(
            br < left,
            left - br - 1,
            torch.where(bl > right, bl - right - 1, torch.zeros_like(bl)),
        )
        eligible = (~inside) & (gap <= max_gap_bins) & (signal >= 0.001 * main_power)
        left = torch.minimum(left, torch.min(torch.where(eligible, bl, left.expand_as(bl))))
        right = torch.maximum(right, torch.max(torch.where(eligible, br, right.expand_as(br))))
    return _recompute_boi_metrics(boi_result, left, right, tag="merged_neighbors")


def _refine_boi_edges_hysteresis(boi_result: Mapping, boi: Mapping, cfg: Mapping):
    smooth = boi_result["S_smooth"]
    n = int(boi_result["n"])
    max_expand_bins = int(float(cfg["MAX_EXPAND_W_FACTOR"]) * int(boi_result["W"]))
    threshold = (10.0 ** (float(cfg["BOUNDARY_THRESHOLD_DB"]) / 10.0)) * boi_result["S_min"]
    left = boi["left_bin"]
    right = boi["right_bin"]
    positions = torch.arange(n, device=smooth.device, dtype=torch.long)
    min_left = torch.clamp(left - max_expand_bins, min=0)
    max_right = torch.clamp(right + max_expand_bins, max=n - 1)
    left_range = (positions >= min_left) & (positions < left)
    right_range = (positions > right) & (positions <= max_right)
    bad_left = left_range & (smooth < threshold)
    bad_right = right_range & (smooth < threshold)
    nearest_bad_left = torch.max(
        torch.where(bad_left, positions, torch.full_like(positions, -1))
    )
    nearest_bad_right = torch.min(
        torch.where(bad_right, positions, torch.full_like(positions, n))
    )
    new_left = torch.where(nearest_bad_left >= 0, nearest_bad_left + 1, min_left)
    new_right = torch.where(nearest_bad_right < n, nearest_bad_right - 1, max_right)
    return _recompute_boi_metrics(boi_result, new_left, new_right, tag="hysteresis_refined")


def _estimate_boi_center_by_excess_power(boi_result: Mapping, boi: Mapping, eps: float):
    n = int(boi_result["n"])
    positions = torch.arange(n, device=boi_result["freqs"].device, dtype=torch.long)
    mask = (positions >= boi["left_bin"]) & (positions <= boi["right_bin"])
    freqs = boi_result["freqs"]
    raw = boi_result["raw_bin_power"]
    weights = torch.where(
        mask,
        torch.clamp(raw - boi_result["noise_floor_raw_per_bin"], min=0.0),
        torch.zeros_like(raw),
    )
    denom = torch.sum(weights)
    return torch.where(
        denom > float(eps),
        torch.sum(freqs * weights) / torch.clamp(denom, min=float(eps)),
        boi["f_center"],
    )


def _power_report_torch(x: torch.Tensor, boi_result: Mapping, boi_for_power: Mapping, eps: float):
    p_total = torch.mean(torch.abs(x.to(torch.complex128)) ** 2).to(torch.float64)
    n = int(boi_result["n"])
    noise_floor = boi_result["noise_floor_raw_per_bin"]
    width_bins = boi_for_power["width_bins"]
    p_noise_fullband = noise_floor * n
    p_noise_inband = noise_floor * width_bins
    p_noise_outband = noise_floor * (n - width_bins)
    p_signal_boi = boi_for_power["signal_power_hat_raw"]
    p_signal_fullband = torch.clamp(p_total - p_noise_fullband, min=0.0)
    return {
        "p_total": p_total,
        "noise_floor_raw_per_bin": noise_floor,
        "p_noise_fullband_hat": p_noise_fullband,
        "p_noise_inband_hat": p_noise_inband,
        "p_noise_outband_hat": p_noise_outband,
        "p_signal_boi_local": p_signal_boi,
        "p_signal_fullband_excess": p_signal_fullband,
        "p_signal_recommended": p_signal_fullband,
        "snr_boi_local_db": 10.0 * torch.log10((p_signal_boi + eps) / (p_noise_inband + eps)),
        "snr_fullband_excess_db": 10.0
        * torch.log10((p_signal_fullband + eps) / (p_noise_inband + eps)),
        "snr_recommended_db": 10.0
        * torch.log10((p_signal_fullband + eps) / (p_noise_inband + eps)),
    }


def _prepare_boi_for_filter_shift_torch(x: torch.Tensor, boi_result: Mapping, cfg: Mapping):
    selected_initial = _select_boi(boi_result, cfg["BOI_SELECT_MODE"])
    selected_merged = _merge_neighbors_around_selected(boi_result, selected_initial, cfg)
    selected_refined = _refine_boi_edges_hysteresis(boi_result, selected_merged, cfg)

    selected_for_power = dict(selected_refined)
    selected_for_power["f_center_midpoint"] = selected_for_power["f_center"]
    selected_for_power["f_center"] = _estimate_boi_center_by_excess_power(
        boi_result, selected_for_power, float(cfg["EPS"])
    )

    power_report = _power_report_torch(x, boi_result, selected_for_power, float(cfg["EPS"]))
    if bool(cfg["ADAPTIVE_FILTER_EXPAND"]):
        snr = power_report["snr_boi_local_db"]
        low = _scalar_tensor(cfg["FILTER_EXPAND_RATIO_LOW_SNR"], x.device)
        mid = _scalar_tensor(cfg["FILTER_EXPAND_RATIO_MID_SNR"], x.device)
        high = _scalar_tensor(cfg["FILTER_EXPAND_RATIO_HIGH_SNR"], x.device)
        expand_ratio = torch.where(
            snr < float(cfg["SNR_LOW_TH_DB"]),
            low,
            torch.where(snr < float(cfg["SNR_HIGH_TH_DB"]), mid, high),
        )
    else:
        expand_ratio = _scalar_tensor(cfg["FILTER_EXPAND_RATIO"], x.device)

    selected_filter = dict(selected_for_power)
    guard = expand_ratio * selected_for_power["bandwidth"]
    fs = float(cfg["FS_IN"])
    selected_filter["f_left"] = torch.maximum(
        selected_for_power["f_left"] - guard,
        _scalar_tensor(-fs / 2.0, x.device),
    )
    selected_filter["f_right"] = torch.minimum(
        selected_for_power["f_right"] + guard,
        _scalar_tensor(fs / 2.0, x.device),
    )
    selected_filter["bandwidth"] = selected_filter["f_right"] - selected_filter["f_left"]
    selected_filter["filter_expand_ratio"] = expand_ratio

    return {
        "selected_initial": selected_initial,
        "selected_merged": selected_merged,
        "selected_refined": selected_for_power,
        "selected_filter": selected_filter,
        "power_report": power_report,
    }


def _extract_filter_shift_noresample_torch(x: torch.Tensor, boi: Mapping, cfg: Mapping):
    x = x.to(torch.complex128)
    n = int(x.numel())
    fs = float(cfg["FS_IN"])
    freqs = _freq_axis_torch(n, fs, x.device, torch.float64)
    X = torch.fft.fftshift(torch.fft.fft(x))
    df = fs / float(n)
    guard_hz = int(cfg["BOI_GUARD_BINS"]) * df
    lo = boi["f_left"] - guard_hz
    hi = boi["f_right"] + guard_hz
    lo_w = _wrap_freq_torch(lo, fs)
    hi_w = _wrap_freq_torch(hi, fs)
    no_wrap = (lo >= -fs / 2.0) & (hi < fs / 2.0)
    wrap_ordered = lo_w <= hi_w
    mask_no_wrap = (freqs >= lo) & (freqs <= hi)
    mask_wrap_ordered = (freqs >= lo_w) & (freqs <= hi_w)
    mask_wrap_split = (freqs >= lo_w) | (freqs <= hi_w)
    mask_wrap = torch.where(wrap_ordered, mask_wrap_ordered, mask_wrap_split)
    mask = torch.where(no_wrap, mask_no_wrap, mask_wrap)
    X_filt = torch.zeros_like(X)
    X_filt[mask] = X[mask]
    x_bpf = torch.fft.ifft(torch.fft.ifftshift(X_filt))
    t = torch.arange(n, device=x.device, dtype=torch.float64) / fs
    center = boi["f_center"].to(torch.float64)
    osc = torch.exp((-1j * 2.0 * torch.pi * center * t).to(torch.complex128))
    x_processed = x_bpf * osc
    return {"x_bpf": x_bpf, "x_processed": x_processed, "bandpass_mask": mask}


def boi_preprocess_torch(x: torch.Tensor, cfg: Mapping):
    boi_result = fixed_resolution_boi_torch(x, cfg)
    if boi_result["bois"]["left_bin"].numel() == 0:
        raise RuntimeError("No BOI detected by fixed-resolution algorithm.")
    prepared = _prepare_boi_for_filter_shift_torch(x, boi_result, cfg)
    boi_filter = prepared["selected_filter"]
    max_bw = float(cfg["MAX_EXPANDED_FILTER_BW_FRAC"]) * float(cfg["FS_IN"])
    use_refined = boi_filter["bandwidth"] >= max_bw
    refined = prepared["selected_refined"]
    boi_filter = dict(boi_filter)
    for key in ("f_left", "f_right", "f_center", "bandwidth"):
        boi_filter[key] = torch.where(use_refined, refined[key], boi_filter[key])
    shifted = _extract_filter_shift_noresample_torch(x, boi_filter, cfg)
    center_shift_hz = prepared["selected_refined"]["f_center"]
    return shifted["x_processed"], prepared, center_shift_hz


def _fft_mag_db_relative_torch(z: torch.Tensor, cfg: Mapping):
    nfft = int(cfg["FFT_NFFT"])
    z = z.reshape(-1).to(torch.complex128)
    if int(z.numel()) < nfft:
        zz = torch.zeros(nfft, device=z.device, dtype=torch.complex128)
        zz[: int(z.numel())] = z
    else:
        zz = z[:nfft].clone()
    window_name = str(cfg["FFT_WINDOW"]).lower()
    if window_name == "hann":
        window = _hann_window_torch(nfft, z.device, torch.float64)
    elif window_name in {"none", "rect", "rectangular"}:
        window = torch.ones(nfft, device=z.device, dtype=torch.float64)
    else:
        raise ValueError("window must be 'hann' or 'none'.")
    Z = torch.fft.fftshift(torch.fft.fft(zz * window))
    mag = torch.abs(Z)
    mag_db = 20.0 * torch.log10(mag / (torch.max(mag) + float(cfg["EPS"])) + float(cfg["EPS"]))
    freq = _freq_axis_torch(nfft, float(cfg["FS_IN"]), z.device, torch.float64)
    return freq, mag_db


def _find_top_peaks_torch(freq: torch.Tensor, mag_db: torch.Tensor, cfg: Mapping):
    mask = (
        torch.isfinite(freq)
        & torch.isfinite(mag_db)
        & (freq >= -float(cfg["FS_IN"]) / 2.0)
        & (freq <= float(cfg["FS_IN"]) / 2.0)
    )
    scores = torch.full_like(mag_db, -torch.inf)
    scores[mask] = mag_db[mask]
    min_sep = max(1, int(cfg["FFT_PEAK_MIN_SEP_BINS"]))
    top_k = int(cfg["FFT_TOP_K"])

    kernel = max(1, 2 * min_sep - 1)
    pooled = torch.nn.functional.max_pool1d(
        scores.reshape(1, 1, -1),
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    ).reshape(-1)
    local_max = torch.isfinite(scores) & (scores == pooled)
    candidate_scores = torch.where(local_max, scores, torch.full_like(scores, -torch.inf))
    values, indices = torch.topk(candidate_scores, k=top_k, largest=True, sorted=True)
    valid = torch.isfinite(values)
    nan = torch.full_like(values, torch.nan)
    return {
        "freq": torch.where(valid, freq[indices], nan),
        "mag_db": torch.where(valid, values, torch.full_like(values, -120.0)),
        "bin": indices,
        "valid": valid,
    }


def _idx_for_freq_torch(f: torch.Tensor, nfft: int = compact.NFFT):
    f = _wrap_freq_torch(f)
    idx = torch.round(f * int(nfft) + int(nfft) / 2).to(torch.long)
    return torch.clamp(idx, 0, int(nfft) - 1)


def _sample_mag_db_near_many_torch(mag_db: torch.Tensor, target_freq: torch.Tensor, bins: int = 3):
    target_freq = target_freq.to(device=mag_db.device, dtype=torch.float64).reshape(-1)
    idx = _idx_for_freq_torch(target_freq, int(mag_db.numel()))
    offsets = torch.arange(-int(bins), int(bins) + 1, device=mag_db.device, dtype=torch.long)
    idxs = torch.clamp(idx[:, None] + offsets[None, :], 0, int(mag_db.numel()) - 1)
    vals = mag_db[idxs]
    vals = torch.where(torch.isfinite(vals), vals, torch.full_like(vals, -120.0))
    return torch.max(vals, dim=1).values


def _extract_vectors_torch(x_processed: torch.Tensor, center_shift_hz: float, cfg: Mapping, lr_names):
    abs_x2 = torch.abs(x_processed) ** 2
    branches = {
        "abs_x2": abs_x2.to(torch.complex128),
        "x2": x_processed**2,
        "x4": x_processed**4,
    }
    fft = {}
    for name, z in branches.items():
        freq, mag_db = _fft_mag_db_relative_torch(z, cfg)
        fft[name] = {
            "mag_db": mag_db,
            "top_peaks": _find_top_peaks_torch(freq, mag_db, cfg),
        }

    device = x_processed.device
    rdtype = torch.float64
    required = set(lr_names) | set(compact.RS20_FEATURES) | set(compact.CFO_FEATURES)
    row = {}

    def need(name: str) -> bool:
        return name in required

    def has_prefix(prefix: str) -> bool:
        return any(name.startswith(prefix) for name in required)

    abs_peaks = fft["abs_x2"]["top_peaks"]
    abs_freqs = torch.abs(abs_peaks["freq"])
    abs_dbs = abs_peaks["mag_db"]
    abs_valid = abs_peaks["valid"]
    rs_candidates = abs_valid & (abs_freqs > 5 / compact.NFFT) & (abs_freqs < 0.49)
    rs_scores = torch.where(rs_candidates, abs_dbs, torch.full_like(abs_dbs, -torch.inf))
    rs_idx = torch.argmax(rs_scores)
    rs_has = torch.any(rs_candidates)
    rs = torch.where(
        rs_has,
        abs_freqs[rs_idx],
        torch.tensor(torch.nan, device=device, dtype=rdtype),
    )
    row["rs_est"] = rs

    for branch in ["abs_x2", "x2", "x4"]:
        peaks = fft[branch]["top_peaks"]
        top_f = peaks["freq"]
        top_db = peaks["mag_db"]
        top_valid = peaks["valid"]
        top_af = torch.abs(top_f)
        safe_rs = torch.where(
            torch.isfinite(rs) & (rs > 0),
            rs,
            torch.tensor(1.0, device=device, dtype=rdtype),
        )
        for i in range(20):
            key = f"{branch}_top{i + 1}_f"
            if need(key):
                row[key] = top_f[i]
            key = f"{branch}_top{i + 1}_af"
            if need(key):
                row[key] = top_af[i]
            key = f"{branch}_top{i + 1}_db"
            if need(key):
                row[key] = top_db[i]
            key = f"{branch}_top{i + 1}_f_over_rs"
            if need(key):
                row[key] = torch.where(
                    torch.isfinite(rs) & (rs > 0),
                    top_f[i] / safe_rs,
                    torch.tensor(torch.nan, device=device, dtype=rdtype),
                )
            key = f"{branch}_top{i + 1}_af_over_rs"
            if need(key):
                row[key] = torch.where(
                    torch.isfinite(rs) & (rs > 0),
                    top_af[i] / safe_rs,
                    torch.tensor(torch.nan, device=device, dtype=rdtype),
                )

        nonzero_mask = top_valid & (top_af > 5 / compact.NFFT)
        nonzero_for_max = torch.where(nonzero_mask, top_db, torch.full_like(top_db, -torch.inf))
        has_nonzero = torch.any(nonzero_mask)
        key = f"{branch}_nonzero_best_db"
        if need(key):
            row[key] = torch.where(
                has_nonzero,
                torch.max(nonzero_for_max),
                torch.tensor(-120.0, device=device, dtype=rdtype),
            )
        key = f"{branch}_nonzero_count_gt_m10"
        if need(key):
            row[key] = torch.sum(nonzero_mask & (top_db > -10)).to(rdtype)
        key = f"{branch}_nonzero_count_gt_m20"
        if need(key):
            row[key] = torch.sum(nonzero_mask & (top_db > -20)).to(rdtype)
        key = f"{branch}_nonzero_count_gt_m30"
        if need(key):
            row[key] = torch.sum(nonzero_mask & (top_db > -30)).to(rdtype)
        key = f"{branch}_top1_minus_top2_db"
        if need(key):
            row[key] = torch.where(
                top_valid[0] & top_valid[1],
                top_db[0] - top_db[1],
                torch.tensor(120.0, device=device, dtype=rdtype),
            )
        key = f"{branch}_top1_minus_top5_db"
        if need(key):
            row[key] = torch.where(
                top_valid[0] & top_valid[4],
                top_db[0] - top_db[4],
                torch.tensor(120.0, device=device, dtype=rdtype),
            )

        need_sym = need(f"{branch}_sym_score_mean") or need(f"{branch}_sym_score_min")
        if need_sym:
            pair_dist = torch.abs(top_f[:, None] + top_f[None, :])
            pair_dist = torch.where(
                top_valid[None, :] & top_valid[:, None],
                pair_dist,
                torch.full_like(pair_dist, torch.inf),
            )
            sym_j = torch.argmin(pair_dist, dim=1)
            sym_freq = top_f[sym_j]
            sym_db = top_db[sym_j]
            sym_vals = torch.abs(top_db - sym_db) + 40.0 * torch.abs(sym_freq + top_f)
            sym_mask = nonzero_mask & top_valid[sym_j]
            sym_count = torch.sum(sym_mask).to(rdtype)
            sym_vals_masked = torch.where(sym_mask, sym_vals, torch.zeros_like(sym_vals))
            key = f"{branch}_sym_score_mean"
            if need(key):
                row[key] = torch.where(
                    sym_count > 0,
                    torch.sum(sym_vals_masked) / torch.clamp(sym_count, min=1.0),
                    torch.tensor(99.0, device=device, dtype=rdtype),
                )
            key = f"{branch}_sym_score_min"
            if need(key):
                row[key] = torch.where(
                    sym_count > 0,
                    torch.min(torch.where(sym_mask, sym_vals, torch.full_like(sym_vals, torch.inf))),
                    torch.tensor(99.0, device=device, dtype=rdtype),
                )

        need_grid = (
            has_prefix(f"{branch}_int_grid_")
            or has_prefix(f"{branch}_half_grid_")
            or need(f"{branch}_half_minus_int_top3_db")
            or need(f"{branch}_center_f")
        )
        if need_grid:
            mag_db = fft[branch]["mag_db"]
            center = torch.tensor(0.0, device=device, dtype=rdtype) if branch == "abs_x2" else top_f[0]
            center = torch.where(
                torch.isfinite(center),
                center,
                torch.tensor(0.0, device=device, dtype=rdtype),
            )
            key = f"{branch}_center_f"
            if need(key):
                row[key] = center
            grid_values_by_mode = {}
            k_vals = torch.arange(-5, 6, device=device, dtype=rdtype)
            rs_valid = torch.isfinite(rs) & (rs > 0)
            for mode, offset in [("int", 0.0), ("half", 0.5)]:
                target = k_vals * safe_rs
                if branch != "abs_x2":
                    target = target + center + float(offset) * safe_rs
                vals = _sample_mag_db_near_many_torch(mag_db, target, bins=3)
                vals = torch.where(rs_valid, vals, torch.full_like(vals, -120.0))
                grid_values_by_mode[mode] = vals
                for idx, k in enumerate(range(-5, 6)):
                    key = f"{branch}_{mode}_grid_k{k:+d}_db"
                    if need(key):
                        row[key] = vals[idx]
                vals_sorted = torch.sort(vals, descending=True).values
                stats = {
                    f"{branch}_{mode}_grid_best_db": vals_sorted[0],
                    f"{branch}_{mode}_grid_2nd_db": vals_sorted[1],
                    f"{branch}_{mode}_grid_3rd_db": vals_sorted[2],
                    f"{branch}_{mode}_grid_mean_top3_db": torch.mean(vals_sorted[:3]),
                    f"{branch}_{mode}_grid_mean_db": torch.mean(vals),
                    f"{branch}_{mode}_grid_count_gt_m10": torch.sum(vals > -10).to(rdtype),
                    f"{branch}_{mode}_grid_count_gt_m20": torch.sum(vals > -20).to(rdtype),
                    f"{branch}_{mode}_grid_count_gt_m30": torch.sum(vals > -30).to(rdtype),
                }
                for key, value in stats.items():
                    if need(key):
                        row[key] = value
            key = f"{branch}_half_minus_int_top3_db"
            if need(key):
                row[key] = (
                    torch.mean(torch.sort(grid_values_by_mode["half"], descending=True).values[:3])
                    - torch.mean(torch.sort(grid_values_by_mode["int"], descending=True).values[:3])
                )

    def tensor_from_row(names):
        missing = [name for name in names if name not in row]
        if missing:
            raise KeyError(f"Missing feature columns: {missing[:10]}")
        return torch.stack([row[name].to(device=device, dtype=rdtype) for name in names])

    return (
        tensor_from_row(lr_names),
        tensor_from_row(compact.RS20_FEATURES),
        tensor_from_row(compact.CFO_FEATURES),
    )


def _vector_value_torch(vec: torch.Tensor, names, feature_name: str):
    return vec[list(names).index(feature_name)]


def _quantize_rs_torch(f: torch.Tensor):
    return torch.round(f * compact.NFFT) / float(compact.NFFT)


def _valid_rs_torch(f: torch.Tensor):
    return torch.isfinite(f) & (f >= 0.03) & (f <= 0.5001)


def _estimate_x4_triplet_from_rs20_torch(vec: torch.Tensor, names):
    device = vec.device
    rdtype = torch.float64
    x4_freq = torch.stack([_vector_value_torch(vec, names, f"x4_top{i}_f") for i in range(1, 5)])
    x4_db = torch.stack([_vector_value_torch(vec, names, f"x4_top{i}_db") for i in range(1, 5)])

    combos = []
    for ci in range(4):
        for ai in range(4):
            if ai == ci:
                continue
            for bi in range(ai + 1, 4):
                if bi == ci:
                    continue
                combos.append((ci, ai, bi))
    idx = torch.tensor(combos, device=device, dtype=torch.long)
    ci = idx[:, 0]
    ai = idx[:, 1]
    bi = idx[:, 2]

    cf = x4_freq[ci]
    cdb = x4_db[ci]
    af = x4_freq[ai]
    adb = x4_db[ai]
    bf = x4_freq[bi]
    bdb = x4_db[bi]
    da = _wrap_freq_torch(af - cf)
    db = _wrap_freq_torch(bf - cf)
    min_sep = 5.0 / compact.NFFT
    valid = (
        torch.isfinite(cf)
        & torch.isfinite(cdb)
        & torch.isfinite(af)
        & torch.isfinite(adb)
        & torch.isfinite(bf)
        & torch.isfinite(bdb)
        & (torch.abs(da) > min_sep)
        & (torch.abs(db) > min_sep)
        & (da * db < 0)
    )

    left_is_a = da < 0
    left_f = torch.where(left_is_a, af, bf)
    right_f = torch.where(left_is_a, bf, af)
    left_db = torch.where(left_is_a, adb, bdb)
    right_db = torch.where(left_is_a, bdb, adb)
    left_rank = torch.where(left_is_a, ai + 1, bi + 1).to(rdtype)
    right_rank = torch.where(left_is_a, bi + 1, ai + 1).to(rdtype)

    dl = _wrap_freq_torch(left_f - cf)
    dr = _wrap_freq_torch(right_f - cf)
    rs_est = 0.5 * (torch.abs(dl) + torch.abs(dr))
    valid = valid & (rs_est > min_sep) & (rs_est < 0.49)

    sym_err = torch.abs(torch.abs(dl) - torch.abs(dr))
    center_from_sides = _wrap_freq_torch(cf + 0.5 * (dl + dr))
    center_offset = torch.abs(_wrap_freq_torch(center_from_sides - cf))
    side_mean = 0.5 * (left_db + right_db)
    side_min = torch.minimum(left_db, right_db)
    rank_penalty = (ci + 1).to(rdtype) + left_rank + right_rank
    score = (
        1.0 * cdb
        + 0.9 * side_mean
        + 0.25 * side_min
        - 900.0 * sym_err
        - 450.0 * center_offset
        - 0.03 * rank_penalty
    )
    score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    best = torch.argmax(score)
    has = torch.any(valid)
    nan = torch.tensor(torch.nan, device=device, dtype=rdtype)
    return (
        torch.where(has, rs_est[best], nan),
        torch.where(has, left_rank[best], nan),
        torch.where(has, right_rank[best], nan),
        torch.where(has, sym_err[best], nan),
    )


def _estimate_rs_cfo_from_torch_vectors(rs20: torch.Tensor, cfo1: torch.Tensor):
    device = rs20.device
    rdtype = torch.float64
    names = compact.RS20_FEATURES
    abs2 = _quantize_rs_torch(_vector_value_torch(rs20, names, "abs_x2_top2_af"))
    db_gap = _vector_value_torch(rs20, names, "abs_x2_top1_minus_top2_db")
    trust_abs2 = torch.isfinite(db_gap) & (db_gap <= rc20_trust_abs2_db_gap_max())

    rs_x4, left_rank, right_rank, sym_err = _estimate_x4_triplet_from_rs20_torch(rs20, names)
    x4_gate = (
        torch.isfinite(sym_err)
        & (sym_err <= 2.0 / compact.NFFT)
        & (left_rank <= 4)
        & (right_rank <= 4)
    )

    abs_rank_features = [f"abs_x2_top{i}_af" for i in range(2, 21, 2)]
    abs_candidates = torch.stack([_vector_value_torch(rs20, names, name) for name in abs_rank_features])
    valid = _valid_rs_torch(abs_candidates)
    bins = torch.round(abs_candidates * compact.NFFT)
    valid = valid & (bins >= 1800)
    positions = torch.arange(valid.numel(), device=device, dtype=torch.long)
    first_pos = torch.argmin(torch.where(valid, positions, torch.full_like(positions, valid.numel())))
    has_abs = torch.any(valid)
    abs_rs = bins[first_pos] / float(compact.NFFT)
    fallback = _vector_value_torch(rs20, names, "rs_est")
    rs_est = torch.where(
        trust_abs2,
        abs2,
        torch.where(x4_gate, rs_x4, torch.where(has_abs, abs_rs, fallback)),
    ).to(rdtype)
    cfo_est = _wrap_freq_torch(cfo1[0].to(rdtype) / 4.0)
    t0_est = 1.0 / rs_est
    return rs_est, t0_est, cfo_est


def rc20_trust_abs2_db_gap_max():
    return 23.941201956505633


class TorchLogisticPipeline(nn.Module):
    """Torch equivalent of SimpleImputer -> StandardScaler -> LogisticRegression."""

    def __init__(self, sklearn_pipeline, device: torch.device):
        super().__init__()
        imputer = sklearn_pipeline.named_steps["simpleimputer"]
        scaler = sklearn_pipeline.named_steps["standardscaler"]
        lr = sklearn_pipeline.named_steps["logisticregression"]
        self.classes_ = list(lr.classes_)
        self.register_buffer(
            "impute_statistics",
            torch.as_tensor(imputer.statistics_, device=device, dtype=torch.float64),
        )
        self.register_buffer("mean", torch.as_tensor(scaler.mean_, device=device, dtype=torch.float64))
        self.register_buffer("scale", torch.as_tensor(scaler.scale_, device=device, dtype=torch.float64))
        self.register_buffer("coef", torch.as_tensor(lr.coef_, device=device, dtype=torch.float64))
        self.register_buffer(
            "intercept", torch.as_tensor(lr.intercept_, device=device, dtype=torch.float64)
        )

    def forward(self, x: torch.Tensor):
        x = x.to(device=self.mean.device, dtype=torch.float64)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        stats = self.impute_statistics.unsqueeze(0).expand_as(x)
        x = torch.where(torch.isnan(x), stats, x)
        x = (x - self.mean) / self.scale
        return x @ self.coef.T + self.intercept

    def predict_indices(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return torch.argmax(self.forward(x), dim=1)


def _cyclic_moment_sequence_torch(
    x: torch.Tensor,
    T0,
    f0,
    n: int,
    m: int,
    K: int,
    allowed_k,
    remove_mean: bool = False,
):
    x = x.to(torch.complex128)
    if remove_mean:
        x = x - torch.mean(x)
    out = torch.zeros(2 * int(K) + 1, device=x.device, dtype=torch.complex128)
    allowed_k = sorted([int(k) for k in allowed_k if -int(K) <= int(k) <= int(K)])
    if not allowed_k:
        return out
    u = (x ** int(n - m)) * (torch.conj(x) ** int(m))
    k_t = torch.tensor(allowed_k, device=x.device, dtype=torch.float64)
    T0_t = torch.as_tensor(T0, device=x.device, dtype=torch.float64)
    f0_t = torch.as_tensor(f0, device=x.device, dtype=torch.float64)
    beta = float(n - 2 * m) * f0_t + k_t * (1.0 / T0_t)
    t = torch.arange(int(x.numel()), device=x.device, dtype=torch.float64)
    expo = torch.exp((-1j * 2.0 * torch.pi * beta[:, None] * t[None, :]).to(torch.complex128))
    vals = (expo @ u) / float(x.numel())
    for i, kk in enumerate(allowed_k):
        out[int(kk) + int(K)] = vals[i]
    return out


def _conv_k_sequences_torch(seqs):
    """Convolve short k-axis sequences on CPU while keeping full IQ work on GPU.

    These sequences are tiny compared with the 32768-point signal. Running this
    convolution as many small CUDA slice-update kernels is much slower than
    moving the short sequences to CPU, using NumPy's optimized 1-D convolution,
    and returning the short result to the original device.
    """
    if not seqs:
        raise ValueError("seqs must contain at least one sequence")

    device = seqs[0].device
    dtype = seqs[0].dtype
    out = np.ones(1, dtype=np.complex128)
    for seq in seqs:
        seq_cpu = seq.detach().to("cpu").numpy()
        out = np.convolve(out, seq_cpu)
    return torch.as_tensor(out, device=device, dtype=dtype)


def _cyclic_cumulant_sequence_torch(x, T0, f0, n, m, K, pattern_dict, remove_mean=False):
    if (int(n), int(m)) not in pattern_dict:
        return torch.zeros(2 * int(K) + 1, device=x.device, dtype=torch.complex128)
    is_conj = [0] * (int(n) - int(m)) + [1] * int(m)
    parts = cc._set_partitions_cc(int(n))
    needed = set()
    for pi in parts:
        for block in pi:
            nb = len(block)
            mb = sum(is_conj[int(idx)] for idx in block)
            needed.add((nb, mb))
    moment_cache = {}
    for nb, mb in needed:
        if (nb, mb) not in pattern_dict:
            moment_cache[(nb, mb)] = torch.zeros(2 * int(K) + 1, device=x.device, dtype=torch.complex128)
        else:
            moment_cache[(nb, mb)] = _cyclic_moment_sequence_torch(
                x, T0, f0, nb, mb, K, cc.get_allowed_k_list(pattern_dict, nb, mb), remove_mean
            )
    full_half = int(n) * int(K)
    center = full_half
    c_full = torch.zeros(2 * full_half + 1, device=x.device, dtype=torch.complex128)
    for pi in parts:
        coef = ((-1) ** (len(pi) - 1)) * math.factorial(len(pi) - 1)
        seqs = []
        for block in pi:
            nb = len(block)
            mb = sum(is_conj[int(idx)] for idx in block)
            seqs.append(moment_cache[(nb, mb)])
        conv = _conv_k_sequences_torch(seqs)
        conv_center = len(pi) * int(K)
        c_full[center - conv_center : center + conv_center + 1] += float(coef) * conv
    out = c_full[center - int(K) : center + int(K) + 1].clone()
    allowed = cc.get_allowed_k_set(pattern_dict, int(n), int(m))
    for kk in range(-int(K), int(K) + 1):
        if kk not in allowed:
            out[kk + int(K)] = 0.0 + 0.0j
    return out


def build_estimated_cc_matrix_11x24_torch(x, T0, f0, pattern_dict, remove_mean=False, max_k=5):
    mat = torch.zeros((len(cc.CC_K_LIST), len(cc.CC_NM)), device=x.device, dtype=torch.complex128)
    for j, (n, m) in enumerate(cc.CC_NM):
        mat[:, j] = _cyclic_cumulant_sequence_torch(
            x, T0, f0, n, m, int(max_k), pattern_dict, remove_mean
        )
    return mat


def apply_cc_postprocessing_with_power_torch(
    matrix_complex: torch.Tensor,
    power_scale,
    enable_warping=True,
    enable_scaling=True,
    eps=1e-12,
):
    mat = matrix_complex.clone()
    if enable_warping:
        for j, (n, _m) in enumerate(cc.CC_NM):
            mag = torch.abs(mat[:, j])
            phase = torch.angle(mat[:, j])
            mat[:, j] = (mag ** (2.0 / float(n))).to(torch.float64) * torch.exp(
                (1j * phase).to(torch.complex128)
            )
    if enable_scaling:
        power = torch.as_tensor(power_scale, device=mat.device, dtype=torch.float64)
        power = torch.where(
            torch.isfinite(power) & (power > float(eps)),
            power,
            torch.full_like(power, torch.nan),
        )
        if enable_warping:
            mat = mat / (power + float(eps))
        else:
            for j, (n, _m) in enumerate(cc.CC_NM):
                mat[:, j] = mat[:, j] / ((power + float(eps)) ** (float(n) / 2.0))
    return mat, torch.abs(mat)


def _cc_power(prepared: Mapping, x_processed: torch.Tensor, eps: float) -> float:
    report = prepared.get("power_report", {})
    fallback = torch.mean(torch.abs(x_processed) ** 2).to(torch.float64)
    power = fallback
    for key in ("P_total", "P_x_bpf", "P_x_processed", "p_signal_boi_local"):
        value = report.get(key)
        if value is None:
            continue
        value_t = torch.as_tensor(value, device=x_processed.device, dtype=torch.float64)
        valid = torch.isfinite(value_t) & (value_t > float(eps))
        power = torch.where(valid, value_t, power)
    return power


@dataclass
class GpuInferenceDebug:
    keep_intermediates: bool = False


class FullyOnlinePipelineGPU:
    def __init__(
        self,
        dataset_root: Path,
        model_dir: Path,
        split_mode: str,
        cfg: Mapping | None = None,
        device: str | None = None,
        cc_backend: str = "cpu",
    ):
        if split_mode not in {"2018train20", "2022train20"}:
            raise ValueError("split_mode must be '2018train20' or '2022train20'")
        cc_backend = str(cc_backend).lower()
        if cc_backend not in {"cpu", "gpu"}:
            raise ValueError("cc_backend must be 'cpu' or 'gpu'")
        self.dataset_root = Path(dataset_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.split_mode = split_mode
        self.cfg = dict(cfg or make_default_config(self.dataset_root))
        self.cc_backend = cc_backend
        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.lr_path = self.model_dir / f"lr55_{split_mode}.joblib"
        self.cap_path = self.model_dir / f"capnet_{split_mode}.pt"
        self.lr_artifact = joblib.load(self.lr_path)
        self.lr_feature_names = list(self.lr_artifact["feature_names"])
        if len(self.lr_feature_names) != 55:
            raise ValueError(f"Expected 55 LR features, found {len(self.lr_feature_names)}")
        self.lr_model = TorchLogisticPipeline(self.lr_artifact["model"], self.device).eval()

        checkpoint = torch.load(self.cap_path, map_location="cpu", weights_only=False)
        train_cfg = checkpoint.get("train_config", {})
        model_cfg = dict(train_cfg.get("model_config", {}))
        model_cfg.pop("name", None)
        for key in ("shared_k", "shared_s"):
            if key in model_cfg:
                model_cfg[key] = tuple(model_cfg[key])
        self.cap_model = CAPNet(**model_cfg).to(self.device)
        self.cap_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.cap_model.eval()
        self.fine_class_names = list(train_cfg.get("class_names", FINE_CLASS_NAMES))
        if self.fine_class_names != FINE_CLASS_NAMES:
            raise ValueError(
                f"Unexpected CAP class order {self.fine_class_names}; expected {FINE_CLASS_NAMES}"
            )

        cc.configure_cc_runtime(
            USE_CUDA=self.device.type == "cuda",
            CUDA_DEVICE=str(self.device),
            CUDA_COMPLEX_DTYPE="complex128",
        )

    def _blind_vectors(self, x_processed: torch.Tensor, center_shift_hz: float):
        return _extract_vectors_torch(
            x_processed=x_processed,
            center_shift_hz=center_shift_hz,
            cfg=self.cfg,
            lr_names=self.lr_feature_names,
        )

    def _process_loaded_to_vectors(self, x: torch.Tensor):
        x_processed, prepared, center_shift_hz = boi_preprocess_torch(x, self.cfg)
        lr55, rs20, cfo1 = self._blind_vectors(x_processed, center_shift_hz)
        return x_processed, lr55, rs20, cfo1, prepared

    def _finish_qpsklike_loaded(
        self,
        x_processed: torch.Tensor,
        prepared: Mapping,
        rs20: torch.Tensor,
        cfo1: torch.Tensor,
        power: torch.Tensor | None = None,
    ):
        x_processed_cc = x_processed.to(torch.complex64)
        rs_est, t0_est, cfo_est = _estimate_rs_cfo_from_torch_vectors(rs20, cfo1)
        if power is None:
            power = _cc_power(prepared, x_processed, float(self.cfg["EPS"]))

        if self.cc_backend == "cpu":
            x_cpu = x_processed_cc.detach().cpu().numpy().astype(np.complex128, copy=False)
            raw_cc = cc.build_estimated_cc_matrix_11x24_from_signal_pattern(
                x_cpu,
                float(t0_est.detach().cpu().item()),
                float(cfo_est.detach().cpu().item()),
                cc.qpsk,
                remove_mean=False,
                max_k=5,
            )
            power_cpu = float(power.detach().cpu().item())
            _, cc_abs = cc.apply_cc_postprocessing_with_power(
                raw_cc["matrix_complex"],
                power_cpu,
                cc.CC_NM,
                enable_warping=True,
                enable_scaling=False,
                is_theory=False,
                eps=float(self.cfg["EPS"]),
            )
            cap_np = np.asarray(cc_abs[:, :15], dtype=np.float32) / np.float32(power_cpu)
            cap_input = torch.from_numpy(cap_np).to(self.device).reshape(1, 11, 15)
            return cap_input, rs_est, cfo_est

        cc_input = x_processed_cc.to(torch.complex128)
        raw_cc = build_estimated_cc_matrix_11x24_torch(
            cc_input,
            t0_est,
            cfo_est,
            cc.qpsk,
            remove_mean=False,
            max_k=5,
        )
        _, cc_abs = apply_cc_postprocessing_with_power_torch(
            raw_cc,
            power,
            enable_warping=True,
            enable_scaling=False,
            eps=float(self.cfg["EPS"]),
        )
        cap_input = (cc_abs[:, :15].to(torch.float32) / power.to(torch.float32)).reshape(1, 11, 15)
        return cap_input, rs_est, cfo_est

    def process_signal(self, year: int, signal_index: int) -> InferenceResult:
        return self.process_signal_batch(int(year), [int(signal_index)])[0]

    def load_signal_batch(self, year: int, signal_indices: Sequence[int]) -> torch.Tensor:
        paths = [
            _signal_path(
                self.dataset_root,
                int(year),
                int(signal_index),
                int(self.cfg["N_PER_BATCH"]),
            )
            for signal_index in signal_indices
        ]
        return tim_to_complex_iq_batch_torch(
            paths,
            int(self.cfg["SAMPLES_PER_SIG"]),
            self.device,
        )

    def process_signal_batch_loaded(self, x_batch) -> list[InferenceResult]:
        """Run a device-loaded chunk through the same pipeline with batched LR/CAP calls."""
        if torch.is_tensor(x_batch):
            x_batch = x_batch.to(device=self.device, dtype=torch.complex128)
        else:
            x_batch = torch.as_tensor(x_batch, device=self.device, dtype=torch.complex128)
        if x_batch.ndim != 2:
            raise ValueError(f"x_batch must have shape (batch, samples), got {tuple(x_batch.shape)}")

        x_processed_rows = []
        rs20_rows = []
        cfo1_rows = []
        power_rows = []
        lr_rows = []
        for row_idx in range(int(x_batch.shape[0])):
            x_processed, lr55, rs20, cfo1, prepared = self._process_loaded_to_vectors(
                x_batch[row_idx]
            )
            x_processed_rows.append(x_processed)
            rs20_rows.append(rs20)
            cfo1_rows.append(cfo1)
            power_rows.append(_cc_power(prepared, x_processed, float(self.cfg["EPS"])))
            lr_rows.append(lr55)

        x_processed_batch = torch.stack(x_processed_rows, dim=0)
        rs20_batch = torch.stack(rs20_rows, dim=0)
        cfo1_batch = torch.stack(cfo1_rows, dim=0)
        power_batch = torch.stack(power_rows, dim=0)
        lr_matrix = torch.stack(lr_rows, dim=0)
        coarse_indices = self.lr_model.predict_indices(lr_matrix)

        cap_inputs = []
        rs_est_all = torch.full(
            (int(x_batch.shape[0]),),
            torch.nan,
            device=self.device,
            dtype=torch.float64,
        )
        cfo_est_all = torch.full_like(rs_est_all, torch.nan)
        fine_index_all = torch.full(
            (int(x_batch.shape[0]),),
            -1,
            device=self.device,
            dtype=torch.long,
        )
        qpsk_like_index = self.lr_model.classes_.index("QPSK-like")
        qpsk_rows = torch.nonzero(coarse_indices == int(qpsk_like_index), as_tuple=False).reshape(-1)

        for qpos in range(int(qpsk_rows.numel())):
            row_idx_t = qpsk_rows[qpos]
            cap_input, rs_est, cfo_est = self._finish_qpsklike_loaded(
                x_processed_batch[row_idx_t],
                {},
                rs20_batch[row_idx_t],
                cfo1_batch[row_idx_t],
                power=power_batch[row_idx_t],
            )
            cap_inputs.append(cap_input)
            rs_est_all[row_idx_t] = rs_est
            cfo_est_all[row_idx_t] = cfo_est

        if cap_inputs:
            cap_batch = torch.stack(cap_inputs, dim=0)
            with torch.inference_mode():
                fine_indices = self.cap_model(cap_batch).argmax(dim=1)
            fine_index_all[qpsk_rows] = fine_indices

        # Final result return boundary: convert GPU predictions/scalars to Python objects.
        coarse_cpu = coarse_indices.detach().cpu().tolist()
        fine_cpu = fine_index_all.detach().cpu().tolist()
        rs_cpu = rs_est_all.detach().cpu().tolist()
        cfo_cpu = cfo_est_all.detach().cpu().tolist()
        results = []
        for row_idx, class_index in enumerate(coarse_cpu):
            coarse_pred = str(self.lr_model.classes_[int(class_index)])
            if coarse_pred not in COARSE_CLASS_NAMES:
                raise ValueError(f"Unexpected LR prediction: {coarse_pred!r}")
            if coarse_pred != "QPSK-like":
                results.append(
                    InferenceResult(
                        pred_label=coarse_pred,
                        coarse_pred=coarse_pred,
                        route="LR-final",
                    )
                )
                continue
            results.append(
                InferenceResult(
                    pred_label=self.fine_class_names[int(fine_cpu[row_idx])],
                    coarse_pred=coarse_pred,
                    route=(
                        "QPSK-like->CPU-CC-GPU-CAP"
                        if self.cc_backend == "cpu"
                        else "QPSK-like->GPU-CC-CAP"
                    ),
                    rs_est=float(rs_cpu[row_idx]),
                    cfo_est=float(cfo_cpu[row_idx]),
                )
            )
        return results

    def process_signal_batch(
        self,
        year: int,
        signal_indices: Sequence[int],
    ) -> list[InferenceResult]:
        x_batch = self.load_signal_batch(year, signal_indices)
        return self.process_signal_batch_loaded(x_batch)

    def run_dataset(
        self,
        year: int,
        batch_ids: Iterable[int],
        output_csv: Path,
        resume: bool = True,
        checkpoint_every: int = 100,
        fail_fast: bool = True,
    ) -> pd.DataFrame:
        year = int(year)
        batch_ids = [int(batch_id) for batch_id in batch_ids]
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        truth = load_truth_table(self.dataset_root, year)
        label_col = _truth_column(
            truth,
            ["SignalType", "signal_type", "signal type", "modulation", "modulation_type", "label"],
        )
        snr_col = _truth_column(
            truth,
            ["InbandSNRdB", "inband_snr_db_label", "inband snr db label", "snr", "snr_db"],
        )
        indices = []
        for batch_id in batch_ids:
            start = (batch_id - 1) * int(self.cfg["N_PER_BATCH"]) + 1
            stop = batch_id * int(self.cfg["N_PER_BATCH"])
            indices.extend(range(start, stop + 1))

        existing = pd.DataFrame()
        completed: set[int] = set()
        if resume and output_csv.exists():
            existing = pd.read_csv(output_csv)
            if not existing.empty:
                completed = set(existing["signal_index"].astype(int))

        rows = existing.to_dict("records")
        pending = [idx for idx in indices if idx not in completed]
        progress = tqdm(pending, desc=f"{self.split_mode} GPU: {year}", unit="signal")
        since_checkpoint = 0

        for signal_index in progress:
            truth_row = truth.loc[signal_index]
            true_label = _normalize_label(truth_row[label_col])
            snr_db = float(truth_row[snr_col])
            if year == 2018 and snr_db < 0.0:
                continue
            try:
                result = self.process_signal(year, signal_index)
                row = {
                    "year": year,
                    "signal_index": signal_index,
                    "snr_db": snr_db,
                    "true_label": true_label,
                    "pred_label": result.pred_label,
                    "coarse_pred": result.coarse_pred,
                    "route": result.route,
                    "rs_est": result.rs_est,
                    "cfo_est": result.cfo_est,
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "year": year,
                    "signal_index": signal_index,
                    "snr_db": snr_db,
                    "true_label": true_label,
                    "pred_label": "ERROR",
                    "coarse_pred": "",
                    "route": "",
                    "rs_est": np.nan,
                    "cfo_est": np.nan,
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }
                rows.append(row)
                pd.DataFrame(rows).to_csv(output_csv, index=False)
                if fail_fast:
                    raise
                continue

            rows.append(row)
            since_checkpoint += 1
            if since_checkpoint >= int(checkpoint_every):
                pd.DataFrame(rows).to_csv(output_csv, index=False)
                since_checkpoint = 0
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        result_df = pd.DataFrame(rows)
        if not result_df.empty:
            result_df = result_df.sort_values(["year", "signal_index"]).reset_index(drop=True)
        result_df.to_csv(output_csv, index=False)
        return result_df


__all__ = [
    "FullyOnlinePipelineGPU",
    "make_default_config",
    "print_and_plot_cm",
    "tim_to_complex_iq_torch",
    "tim_to_complex_iq_batch_torch",
    "boi_preprocess_torch",
    "build_estimated_cc_matrix_11x24_torch",
    "apply_cc_postprocessing_with_power_torch",
]
