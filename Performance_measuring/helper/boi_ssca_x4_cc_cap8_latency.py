"""BOI + two x(t) SSCA + FFT{x^4} + 11x15 CC + paper CAP timing.

The SSCA implementation and parameters are copied from
``BOI+SCCA+CC_fullyblind-2018-8-14.ipynb``.  Blind Rs/CFO estimation is
intentionally omitted; deterministic random values are supplied to the direct
11x15 original-CC implementation.  The eight-branch CAP follows Table 1 of
Snoap et al., Sensors 2023, with randomly initialized weights.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal.windows import chebwin

from .lr55_cccap.fully_online_pipeline_inference import (
    _boi_preprocess,
    _cc_power,
    _signal_path,
    _tim_to_complex_iq_batch,
    make_default_config,
)
from .lr55_cccap.qpsk_cc_four_way_profile import build_qpsk_cap_input_original_cpu


SSCA_N = 32768
SSCA_NP = 64
SSCA_CHEB_ATTEN_DB = 80
SSCA_PSD_NFFT = 512
SSCA_ENABLE_INPUT_POWER_NORM = False
CAP_TRAINABLE_PARAMETERS = 1_269_072


def normalize_power_local(x, eps=1e-12):
    x = np.asarray(x, dtype=np.complex128)
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + eps)


def make_cheb_window_ssca(np_size: int, atten_db: float = 80):
    window = chebwin(int(np_size), at=float(atten_db), sym=False)
    window = np.asarray(window, dtype=np.float64)
    return window / np.sqrt(np.mean(window**2))


def wrap_freq_ssca(f, fs=1.0):
    return ((f + fs / 2) % fs) - fs / 2


def estimate_psd_welch_ssca(x, fs=1.0, nfft=512, hop=None, eps=1e-15):
    x = np.asarray(x, dtype=np.complex128).ravel()
    if hop is None:
        hop = int(nfft) // 2
    if len(x) < int(nfft):
        padded = np.zeros(int(nfft), dtype=np.complex128)
        padded[: len(x)] = x
        x = padded
    window = np.hanning(int(nfft))
    window_power = np.sum(window**2) + eps
    starts = np.arange(0, len(x) - int(nfft) + 1, int(hop))
    if len(starts) == 0:
        starts = np.array([0])
    psd = np.zeros(int(nfft), dtype=np.float64)
    for start in starts:
        segment = x[start : start + int(nfft)]
        spectrum = np.fft.fftshift(np.fft.fft(segment * window, n=int(nfft)))
        psd += np.abs(spectrum) ** 2 / window_power
    psd /= len(starts)
    psd = np.maximum(psd, eps)
    freq = np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / fs))
    return freq, psd


def interp_psd_periodic_ssca(f_query, f_psd, pxx, fs=1.0, eps=1e-15):
    query = wrap_freq_ssca(f_query, fs)
    order = np.argsort(f_psd)
    freq = f_psd[order]
    power = pxx[order]
    freq_ext = np.concatenate([freq - fs, freq, freq + fs])
    power_ext = np.concatenate([power, power, power])
    out = np.interp(query.ravel(), freq_ext, power_ext)
    return np.maximum(out.reshape(f_query.shape), eps)


def compute_coherence_ssca(
    spectral_correlation,
    spectral_frequency,
    cycle_frequency,
    f_psd,
    pxx,
    branch,
    fs=1.0,
    eps=1e-15,
    denom_floor_rel=1e-3,
    clip_coherence=True,
):
    p1 = interp_psd_periodic_ssca(
        spectral_frequency + cycle_frequency / 2,
        f_psd,
        pxx,
        fs=fs,
        eps=eps,
    )
    if branch == "nonconj":
        p2 = interp_psd_periodic_ssca(
            spectral_frequency - cycle_frequency / 2,
            f_psd,
            pxx,
            fs=fs,
            eps=eps,
        )
    elif branch == "conj":
        p2 = interp_psd_periodic_ssca(
            cycle_frequency / 2 - spectral_frequency,
            f_psd,
            pxx,
            fs=fs,
            eps=eps,
        )
    else:
        raise ValueError("branch must be 'nonconj' or 'conj'")

    finite = pxx[np.isfinite(pxx)]
    finite = finite[finite > eps]
    if finite.size == 0:
        return np.zeros_like(spectral_correlation, dtype=np.complex128)
    denom_floor = max(eps, float(denom_floor_rel) * float(np.max(finite)))
    valid = (
        np.isfinite(p1)
        & np.isfinite(p2)
        & (p1 >= denom_floor)
        & (p2 >= denom_floor)
    )
    coherence = np.zeros_like(spectral_correlation, dtype=np.complex128)
    coherence[valid] = spectral_correlation[valid] / np.sqrt(p1[valid] * p2[valid])
    coherence = np.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_coherence:
        magnitude = np.abs(coherence)
        too_large = magnitude > 1.0
        coherence[too_large] = coherence[too_large] / (magnitude[too_large] + eps)
    return coherence


def chad_style_ssca_fixed_n_torch(
    x,
    fs,
    branch,
    torch_device,
    n=SSCA_N,
    np_size=SSCA_NP,
    cheb_atten_db=SSCA_CHEB_ATTEN_DB,
    psd_nfft=SSCA_PSD_NFFT,
    enable_input_power_norm=SSCA_ENABLE_INPUT_POWER_NORM,
):
    """Exact torch-backend SSCA core used by the source notebook."""
    device = torch.device(torch_device)
    x_np = np.asarray(x, dtype=np.complex64).ravel()
    if enable_input_power_norm:
        x_np = normalize_power_local(x_np).astype(np.complex64)
    needed = int(n) + int(np_size)
    if len(x_np) < needed:
        x_work = np.zeros(needed, dtype=np.complex64)
        x_work[: len(x_np)] = x_np
    else:
        x_work = x_np[:needed].copy()

    window_np = make_cheb_window_ssca(np_size, cheb_atten_db).astype(np.float32)
    x_t = torch.from_numpy(x_work).to(device)
    window_t = torch.from_numpy(window_np).to(device)
    x_windows = x_t.unfold(dimension=0, size=int(np_size), step=1)[: int(n), :]
    x_windows = x_windows * window_t[None, :]
    xf = torch.fft.fftshift(torch.fft.fft(x_windows, n=int(np_size), dim=1), dim=1)

    n_arr = torch.arange(int(n), device=device, dtype=torch.float32)[:, None]
    j_shifted = torch.arange(int(np_size), device=device, dtype=torch.float32)
    j_unshifted = (j_shifted + int(np_size) // 2) % int(np_size)
    phase = torch.exp(-1j * 2.0 * np.pi * n_arr * j_unshifted[None, :] / float(np_size))
    xt = xf * phase
    x_center = x_t[torch.arange(int(n), device=device) + int(np_size) // 2]
    if branch == "nonconj":
        cdp = xt * torch.conj(x_center)[:, None]
    elif branch == "conj":
        cdp = xt * x_center[:, None]
    else:
        raise ValueError("branch must be 'nonconj' or 'conj'")
    s_t = torch.fft.fftshift(torch.fft.fft(cdp, n=int(n), dim=0), dim=0) / float(n)

    q_norm = torch.fft.fftshift(torch.fft.fftfreq(int(n), d=1.0, device=device))
    k_norm = torch.fft.fftshift(torch.fft.fftfreq(int(np_size), d=1.0, device=device))
    q = q_norm[:, None]
    k = k_norm[None, :]
    alpha_t = (k + q) * fs
    freq_t = 0.5 * (k - q) * fs
    spectral_correlation = s_t.detach().cpu().numpy()
    alpha = alpha_t.detach().cpu().numpy()
    freq = freq_t.detach().cpu().numpy()

    psd_nfft_eff = int(min(int(psd_nfft), int(n)))
    f_psd, pxx = estimate_psd_welch_ssca(
        x_work[: int(n)],
        fs=fs,
        nfft=psd_nfft_eff,
        hop=max(1, psd_nfft_eff // 2),
    )
    coherence = compute_coherence_ssca(
        spectral_correlation,
        freq,
        alpha,
        f_psd,
        pxx,
        branch=branch,
        fs=fs,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return spectral_correlation, coherence, freq, alpha, int(n)


def fft_x4_original(x):
    """FFT{x^4} curve computation from the source plotting branch."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    z = x**4
    window = np.hanning(z.size)
    spectrum = np.fft.fftshift(np.fft.fft(z * window))
    magnitude = np.abs(spectrum)
    return 20.0 * np.log10(magnitude / (np.max(magnitude) + 1e-12) + 1e-12)


def _same_pad_2d(x, kernel_size, stride):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    input_h, input_w = x.shape[-2:]
    output_h = math.ceil(input_h / stride_h)
    output_w = math.ceil(input_w / stride_w)
    pad_h = max(0, (output_h - 1) * stride_h + kernel_h - input_h)
    pad_w = max(0, (output_w - 1) * stride_w + kernel_w - input_w)
    return F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))


class SameConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
            bias=True,
        )

    def forward(self, x):
        return self.conv(_same_pad_2d(x, self.kernel_size, self.stride))


class PaperCapsuleBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SameConv2d(56, 56, kernel_size=(4, 4), stride=(1, 2))
        self.bn1 = nn.BatchNorm2d(56)
        self.conv2 = SameConv2d(56, 72, kernel_size=(4, 6), stride=(1, 2))
        self.bn2 = nn.BatchNorm2d(72)
        self.fc = nn.Linear(11 * 2 * 72, 7, bias=True)
        self.bn3 = nn.BatchNorm1d(7)
        self.point_fc = nn.Linear(7, 1, bias=True)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.bn3(self.fc(x)))
        return self.point_fc(x)


class PaperEightBranchCAP(nn.Module):
    """Table 1 CC-trained CAP with random weights."""
    def __init__(self):
        super().__init__()
        self.shared_conv = SameConv2d(1, 56, kernel_size=(6, 4), stride=(1, 2))
        self.shared_bn = nn.BatchNorm2d(56)
        self.branches = nn.ModuleList([PaperCapsuleBranch() for _ in range(8)])

    def forward(self, x):
        shared = torch.tanh(self.shared_bn(self.shared_conv(x)))
        logits = torch.cat([branch(shared) for branch in self.branches], dim=1)
        return torch.softmax(logits, dim=1)


def build_random_paper_cap(device="cuda:1", seed=2026):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    model = PaperEightBranchCAP().to(torch.device(device)).eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != CAP_TRAINABLE_PARAMETERS:
        raise RuntimeError(
            f"CAP parameter count {parameter_count:,} does not match paper value "
            f"{CAP_TRAINABLE_PARAMETERS:,}."
        )
    return model


def _select_existing_indices(dataset_root, year, batch_ids, limit, cfg):
    indices = []
    for batch_id in [int(x) for x in batch_ids]:
        start = (batch_id - 1) * int(cfg["N_PER_BATCH"]) + 1
        stop = min(batch_id * int(cfg["N_PER_BATCH"]), int(cfg["N_TOTAL"]))
        for signal_index in range(start, stop + 1):
            path = _signal_path(dataset_root, int(year), signal_index, int(cfg["N_PER_BATCH"]))
            if path.exists():
                indices.append(signal_index)
            if len(indices) == int(limit):
                return indices
    raise ValueError(f"Only found {len(indices)} signals; need {limit}.")


def benchmark_boi_ssca_cc_cap8(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    ssca_device: str = "cuda:1",
    cap_device: str = "cuda:1",
    random_seed: int = 2026,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Benchmark the three requested inference stages; do not compute accuracy."""
    dataset_root = Path(dataset_root).resolve()
    cfg = make_default_config(dataset_root)
    signal_indices = _select_existing_indices(dataset_root, year, batch_ids, limit, cfg)
    paths = [
        _signal_path(dataset_root, int(year), index, int(cfg["N_PER_BATCH"]))
        for index in signal_indices
    ]
    # File loading, random parameter generation, and random model construction are excluded.
    x_batch = _tim_to_complex_iq_batch(paths, int(cfg["SAMPLES_PER_SIG"]))
    rng = np.random.default_rng(int(random_seed))
    random_rs = rng.uniform(1.0 / 23.0, 0.5, size=len(signal_indices)).astype(np.float32)
    random_cfo = rng.uniform(-0.001, 0.001, size=len(signal_indices)).astype(np.float32)
    cap_model = build_random_paper_cap(device=cap_device, seed=random_seed)
    cap_dev = torch.device(cap_device)
    if cap_dev.type == "cuda":
        torch.cuda.synchronize(cap_dev)

    total_start = time.perf_counter()
    processed = []
    stage_start = time.perf_counter()
    for x in x_batch:
        x_processed, prepared, _center_shift = _boi_preprocess(x, cfg)
        for branch in ("nonconj", "conj"):
            outputs = chad_style_ssca_fixed_n_torch(
                x=x_processed,
                fs=float(cfg["FS_IN"]),
                branch=branch,
                torch_device=ssca_device,
            )
            del outputs
        fft_curve = fft_x4_original(x_processed)
        del fft_curve
        processed.append((x_processed, prepared))
    boi_ssca_fft_s = time.perf_counter() - stage_start

    cc_inputs = []
    stage_start = time.perf_counter()
    for (x_processed, prepared), rs, cfo in zip(processed, random_rs, random_cfo):
        power = _cc_power(prepared, x_processed, float(cfg["EPS"]))
        cc_inputs.append(build_qpsk_cap_input_original_cpu(
            np.asarray(x_processed, dtype=np.complex64),
            np.float32(1.0 / rs),
            np.float32(cfo),
            np.float32(power),
            eps=float(cfg["EPS"]),
        ))
    cc_s = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    cap_array = np.stack(cc_inputs).astype(np.float32, copy=False)
    cap_tensor = torch.from_numpy(cap_array[:, None, :, :]).to(cap_dev)
    with torch.inference_mode():
        probabilities = cap_model(cap_tensor)
        _ = probabilities.argmax(dim=1)
    if cap_dev.type == "cuda":
        torch.cuda.synchronize(cap_dev)
    cap_s = time.perf_counter() - stage_start
    total_s = time.perf_counter() - total_start

    aggregate = pd.DataFrame([{
        "n_samples": len(signal_indices),
        "ssca_device": str(torch.device(ssca_device)),
        "cap_device": str(cap_dev),
        "boi_2ssca_fft_x4_s": boi_ssca_fft_s,
        "cc_11x15_s": cc_s,
        "cap8_s": cap_s,
        "total_s": total_s,
    }])
    per_sample = aggregate[["n_samples", "ssca_device", "cap_device"]].copy()
    for column in ("boi_2ssca_fft_x4_s", "cc_11x15_s", "cap8_s", "total_s"):
        per_sample[f"{column[:-2]}_ms"] = aggregate[column] * 1000.0 / aggregate["n_samples"]
    return aggregate, per_sample, signal_indices


__all__ = [
    "CAP_TRAINABLE_PARAMETERS",
    "PaperEightBranchCAP",
    "benchmark_boi_ssca_cc_cap8",
    "build_random_paper_cap",
    "chad_style_ssca_fixed_n_torch",
]
