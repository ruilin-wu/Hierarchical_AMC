"""Minimal online Snoap2024 CAP inference and latency measurement.

The timed path starts from raw ``.tim`` files and runs BOI detection, BPF/CFO
shift, USP normalisation, feature extraction, and CAP inference.  Ground-truth
labels and accuracy are intentionally not loaded or computed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_NAMES = ["BPSK", "QPSK", "8PSK", "pi/4-DQPSK", "MSK", "16QAM", "64QAM", "256QAM"]
DATA_DIRS = {
    2018: "CSPB_ML_2018_Data",
    2022: "CSPB_ML_2022_Data",
}

EPS_FE = 1e-12


def make_default_config(dataset_root: Path) -> dict:
    dataset_root = Path(dataset_root)
    return {
        "N_PER_BATCH": 4000,
        "N_BATCHES": 28,
        "N_TOTAL": 4000 * 28,
        "SAMPLES_PER_SIG": 32768,
        "FS_IN": 1.0,
        "EPS": 1e-12,
        "DELTA_F0": 0.002,
        "THRESHOLD_DB": 5.0,
        "BOUNDARY_THRESHOLD_DB": 1.5,
        "BOI_SELECT_MODE": "strongest",
        "MERGE_GAP_W_FACTOR": 2.0,
        "MAX_EXPAND_W_FACTOR": 8.0,
        "BOI_GUARD_BINS": 2,
        "ADAPTIVE_FILTER_EXPAND": True,
        "FILTER_EXPAND_RATIO": 0.20,
        "FILTER_EXPAND_RATIO_LOW_SNR": 0.30,
        "FILTER_EXPAND_RATIO_MID_SNR": 0.15,
        "FILTER_EXPAND_RATIO_HIGH_SNR": 0.05,
        "SNR_LOW_TH_DB": 4.0,
        "SNR_HIGH_TH_DB": 8.0,
        "MAX_EXPANDED_FILTER_BW_FRAC": 0.95,
    }


def _signal_path(dataset_root: Path, year: int, signal_index: int, n_per_batch: int) -> Path:
    batch_id = (int(signal_index) - 1) // int(n_per_batch) + 1
    return (
        Path(dataset_root)
        / DATA_DIRS[int(year)]
        / f"Batch_Dir_{batch_id}"
        / f"signal_{int(signal_index)}.tim"
    )


def select_signal_indices(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Iterable[int] = (25,),
    limit: int = 64,
) -> list[int]:
    """Return the first ``limit`` existing raw signals from selected batches."""
    if int(year) not in DATA_DIRS:
        raise ValueError(f"Unsupported dataset year: {year}")
    cfg = make_default_config(dataset_root)
    indices: list[int] = []
    for batch_id in [int(x) for x in batch_ids]:
        start = (batch_id - 1) * int(cfg["N_PER_BATCH"]) + 1
        stop = min(batch_id * int(cfg["N_PER_BATCH"]), int(cfg["N_TOTAL"]))
        for signal_index in range(start, stop + 1):
            path = _signal_path(dataset_root, int(year), signal_index, int(cfg["N_PER_BATCH"]))
            if not path.exists():
                continue
            indices.append(int(signal_index))
            if len(indices) >= int(limit):
                return indices
    if len(indices) < int(limit):
        raise ValueError(f"Only found {len(indices)} eligible signals, need {limit}.")
    return indices


def _real_dtype_for_complex(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    if dtype == torch.complex64:
        return torch.float32
    raise TypeError(f"Expected complex64/complex128, got {dtype}")


def _numpy_complex_dtype(dtype: torch.dtype):
    if dtype == torch.complex128:
        return np.complex128
    if dtype == torch.complex64:
        return np.complex64
    raise TypeError(f"Expected complex64/complex128, got {dtype}")


def load_tim_batch(
    paths: Sequence[Path],
    samples_per_sig: int,
    device: torch.device,
    complex_dtype: torch.dtype = torch.complex128,
) -> torch.Tensor:
    """Read raw ``.tim`` files and transfer one stacked IQ tensor to ``device``.

    The complex128 path intentionally matches boi_shift's loader implementation:
    Path conversion and existence check, one complex128 NumPy batch, float64 I/Q
    conversion, then ``torch.from_numpy(batch).to(device=device)``.
    """
    paths = [Path(path) for path in paths]
    samples_per_sig = int(samples_per_sig)
    need = 2 + 2 * samples_per_sig
    batch = np.empty((len(paths), samples_per_sig), dtype=np.complex128)
    for row, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f".tim file not found: {path}")
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size < need:
            raise ValueError(f"{path}: {raw.size} floats < required {need}")
        iq = raw[2:need]
        batch[row] = iq[0::2].astype(np.float64) + 1j * iq[1::2].astype(np.float64)
    tensor = torch.from_numpy(batch).to(device=device)
    if complex_dtype != torch.complex128:
        tensor = tensor.to(dtype=complex_dtype)
    return tensor


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _wrap_freq_tensor(f: torch.Tensor, fs: float = 1.0) -> torch.Tensor:
    return torch.remainder(f + float(fs) / 2.0, float(fs)) - float(fs) / 2.0


class TorchBoiUspPreprocessor:
    """BOI + BPF/CFO shift + USP normalisation on one torch device."""

    def __init__(
        self,
        cfg: Mapping,
        device: torch.device,
        complex_dtype: torch.dtype = torch.complex64,
    ):
        self.cfg = dict(cfg)
        self.device = torch.device(device)
        self.complex_dtype = complex_dtype
        self.real_dtype = _real_dtype_for_complex(complex_dtype)
        self._freq_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._time_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}

    def _freq_axis(self, n: int) -> torch.Tensor:
        key = (int(n), self.real_dtype)
        cached = self._freq_cache.get(key)
        if cached is None or cached.device != self.device:
            fs = float(self.cfg["FS_IN"])
            cached = (torch.arange(int(n), device=self.device, dtype=self.real_dtype) - int(n) // 2) * (
                fs / float(n)
            )
            self._freq_cache[key] = cached
        return cached

    def _time_axis(self, n: int) -> torch.Tensor:
        key = (int(n), self.real_dtype)
        cached = self._time_cache.get(key)
        if cached is None or cached.device != self.device:
            cached = torch.arange(int(n), device=self.device, dtype=self.real_dtype) / float(
                self.cfg["FS_IN"]
            )
            self._time_cache[key] = cached
        return cached

    @staticmethod
    def _moving_average_same(x: torch.Tensor, width: int) -> torch.Tensor:
        width = int(max(1, width))
        if width == 1:
            return x.clone()
        kernel = torch.ones(1, 1, width, device=x.device, dtype=x.dtype) / float(width)
        return F.conv1d(x.reshape(1, 1, -1), kernel, padding=width // 2).reshape(-1)

    def _boi_shift_one(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        x = x.to(device=self.device, dtype=self.complex_dtype)
        n = int(x.numel())
        fs = float(cfg["FS_IN"])
        eps = float(cfg["EPS"])
        freqs = self._freq_axis(n)

        X = torch.fft.fftshift(torch.fft.fft(x))
        raw_bin_power = (X.abs() ** 2) / float(n * n)

        width = int(np.floor(n * float(cfg["DELTA_F0"])))
        width = max(1, width)
        if width % 2 == 0:
            width += 1
        smooth = self._moving_average_same(raw_bin_power, width)
        half = width // 2
        valid_left = half
        valid_right = n - half
        if valid_right <= valid_left:
            raise ValueError(f"Invalid smoothing width W={width} for n={n}.")

        smooth_valid = smooth[valid_left:valid_right]
        smooth_min = torch.min(smooth_valid)
        threshold = smooth_min * (10.0 ** (float(cfg["THRESHOLD_DB"]) / 10.0))
        boi_valid = smooth_valid >= threshold

        padded = F.pad(boi_valid.to(torch.int8), (1, 1))
        changes = padded[1:] - padded[:-1]
        starts = torch.nonzero(changes == 1, as_tuple=False).reshape(-1)
        stops = torch.nonzero(changes == -1, as_tuple=False).reshape(-1) - 1
        if int(starts.numel()) == 0:
            raise RuntimeError("No BOI detected by fixed-resolution algorithm.")

        lefts = starts + valid_left
        rights = stops + valid_left
        lefts = torch.where(starts == 0, torch.zeros_like(lefts), lefts)
        rights = torch.where(stops == int(boi_valid.numel()) - 1, torch.full_like(rights, n - 1), rights)
        widths = rights - lefts + 1
        keep = widths >= 2
        if not bool(torch.any(keep).item()):
            raise RuntimeError("No BOI detected after min-width filtering.")
        lefts = lefts[keep]
        rights = rights[keep]
        widths = widths[keep].to(self.real_dtype)

        full_boi_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        for left, right in zip(lefts, rights):
            full_boi_mask[int(left.item()) : int(right.item()) + 1] = True

        valid_mask = torch.zeros(n, device=self.device, dtype=torch.bool)
        valid_mask[valid_left:valid_right] = True
        noise_mask = (~full_boi_mask) & valid_mask
        if bool(torch.any(noise_mask).item()):
            noise_floor_raw = torch.mean(raw_bin_power[noise_mask])
            noise_floor_smooth = torch.mean(smooth[noise_mask])
        else:
            noise_floor_raw = torch.min(raw_bin_power[valid_left:valid_right])
            noise_floor_smooth = smooth_min

        zero = torch.zeros(1, device=self.device, dtype=self.real_dtype)
        csum_raw = torch.cat([zero, torch.cumsum(raw_bin_power, dim=0)])
        csum_smooth = torch.cat([zero, torch.cumsum(smooth, dim=0)])
        seg_raw = csum_raw[rights + 1] - csum_raw[lefts]
        seg_smooth = csum_smooth[rights + 1] - csum_smooth[lefts]
        signal_power_raw = torch.clamp(seg_raw - noise_floor_raw * widths, min=0.0)

        selected_idx = torch.argmax(signal_power_raw)
        main_power = torch.clamp(signal_power_raw[selected_idx], min=eps)
        left = lefts[selected_idx]
        right = rights[selected_idx]
        max_gap_bins = int(float(cfg["MERGE_GAP_W_FACTOR"]) * width)
        for _ in range(int(lefts.numel()) + 1):
            inside = (lefts >= left) & (rights <= right)
            gap = torch.where(
                rights < left,
                left - rights - 1,
                torch.where(lefts > right, lefts - right - 1, torch.zeros_like(lefts)),
            )
            eligible = (~inside) & (gap <= max_gap_bins) & (signal_power_raw >= 0.001 * main_power)
            if not bool(torch.any(eligible).item()):
                break
            left = torch.minimum(left, torch.min(lefts[eligible]))
            right = torch.maximum(right, torch.max(rights[eligible]))

        boundary_threshold = smooth_min * (10.0 ** (float(cfg["BOUNDARY_THRESHOLD_DB"]) / 10.0))
        max_expand_bins = int(float(cfg["MAX_EXPAND_W_FACTOR"]) * width)
        left_i = int(left.item())
        right_i = int(right.item())

        lo_i = max(0, left_i - max_expand_bins)
        if left_i > lo_i:
            left_seg = smooth[lo_i:left_i] >= boundary_threshold
            bad = torch.nonzero(~left_seg, as_tuple=False).reshape(-1)
            left_i = lo_i if int(bad.numel()) == 0 else lo_i + int(bad[-1].item()) + 1

        hi_i = min(n - 1, right_i + max_expand_bins)
        if hi_i > right_i:
            right_seg = smooth[right_i + 1 : hi_i + 1] >= boundary_threshold
            bad = torch.nonzero(~right_seg, as_tuple=False).reshape(-1)
            right_i = hi_i if int(bad.numel()) == 0 else right_i + int(bad[0].item())

        left_t = torch.tensor(left_i, device=self.device, dtype=torch.long)
        right_t = torch.tensor(right_i, device=self.device, dtype=torch.long)
        width_t = (right_t - left_t + 1).to(self.real_dtype)
        p_raw = csum_raw[right_t + 1] - csum_raw[left_t]
        p_smooth = csum_smooth[right_t + 1] - csum_smooth[left_t]
        p_signal_boi = torch.clamp(p_raw - noise_floor_raw * width_t, min=0.0)
        f_left = freqs[left_t]
        f_right = freqs[right_t]
        f_center_mid = _wrap_freq_tensor(0.5 * (f_left + f_right), fs=fs)
        local_freqs = freqs[left_i : right_i + 1]
        weights = torch.clamp(raw_bin_power[left_i : right_i + 1] - noise_floor_raw, min=0.0)
        denom = torch.sum(weights)
        f_center = torch.where(
            denom > eps,
            torch.sum(local_freqs * weights) / torch.clamp(denom, min=eps),
            f_center_mid,
        )
        bandwidth_refined = width_t * (fs / float(n))

        p_total = torch.mean(x.abs() ** 2)
        p_noise_inband = noise_floor_raw * width_t
        snr_boi_db = 10.0 * torch.log10((p_signal_boi + eps) / (p_noise_inband + eps))
        if bool(cfg["ADAPTIVE_FILTER_EXPAND"]):
            expand_low = torch.tensor(float(cfg["FILTER_EXPAND_RATIO_LOW_SNR"]), device=self.device)
            expand_mid = torch.tensor(float(cfg["FILTER_EXPAND_RATIO_MID_SNR"]), device=self.device)
            expand_high = torch.tensor(float(cfg["FILTER_EXPAND_RATIO_HIGH_SNR"]), device=self.device)
            expand_ratio = torch.where(
                snr_boi_db < float(cfg["SNR_LOW_TH_DB"]),
                expand_low,
                torch.where(snr_boi_db < float(cfg["SNR_HIGH_TH_DB"]), expand_mid, expand_high),
            ).to(self.real_dtype)
        else:
            expand_ratio = torch.tensor(float(cfg["FILTER_EXPAND_RATIO"]), device=self.device, dtype=self.real_dtype)

        guard = expand_ratio * bandwidth_refined
        filter_left = torch.clamp(f_left - guard, min=-fs / 2.0)
        filter_right = torch.clamp(f_right + guard, max=fs / 2.0)
        filter_bw = filter_right - filter_left
        use_refined = filter_bw >= float(cfg.get("MAX_EXPANDED_FILTER_BW_FRAC", 0.95)) * fs
        boi_left = torch.where(use_refined, f_left, filter_left)
        boi_right = torch.where(use_refined, f_right, filter_right)

        guard_hz = int(cfg["BOI_GUARD_BINS"]) * (fs / float(n))
        lo = boi_left - guard_hz
        hi = boi_right + guard_hz
        lo_w = _wrap_freq_tensor(lo, fs=fs)
        hi_w = _wrap_freq_tensor(hi, fs=fs)
        mask_direct = (freqs >= lo) & (freqs <= hi)
        mask_wrap_normal = (freqs >= lo_w) & (freqs <= hi_w)
        mask_wrap_split = (freqs >= lo_w) | (freqs <= hi_w)
        direct = (lo >= -fs / 2.0) & (hi < fs / 2.0)
        mask = torch.where(direct, mask_direct, torch.where(lo_w <= hi_w, mask_wrap_normal, mask_wrap_split))

        X_filt = torch.where(mask, X, torch.zeros((), device=self.device, dtype=X.dtype))
        x_bpf = torch.fft.ifft(torch.fft.ifftshift(X_filt))
        osc = torch.exp((-1j * 2.0 * torch.pi * f_center * self._time_axis(n)).to(self.complex_dtype))
        x_processed = x_bpf * osc
        scale = torch.sqrt(torch.clamp(p_signal_boi, min=eps))
        x_usp = torch.where(p_signal_boi > eps, x_processed / scale, x_processed)
        return x_usp.to(self.complex_dtype), p_total

    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        outputs = []
        for row in range(int(x_batch.shape[0])):
            x_usp, _ = self._boi_shift_one(x_batch[row])
            outputs.append(x_usp)
        return torch.stack(outputs, dim=0)


def _squaring_layer(i_part: torch.Tensor, q_part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return i_part * i_part - q_part * q_part, 2.0 * i_part * q_part


def _pow3_layer(i_part: torch.Tensor, q_part: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    i2 = i_part * i_part
    q2 = q_part * q_part
    return i2 * i_part - 3.0 * i_part * q2, 3.0 * i2 * q_part - q2 * q_part


def _fft_magnitude(i_part: torch.Tensor, q_part: torch.Tensor, nfft: int = 32768) -> torch.Tensor:
    z = torch.complex(i_part, q_part)
    return torch.fft.fftshift(torch.fft.fft(z, n=int(nfft), dim=-1), dim=-1).abs()


def _td_warp(
    i_part: torch.Tensor,
    q_part: torch.Tensor,
    exponent: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mag = (i_part * i_part + q_part * q_part).sqrt()
    max_mag = mag.amax(dim=-1, keepdim=True).clamp(min=EPS_FE)
    scale = max_mag.pow(float(exponent))
    return i_part / scale, q_part / scale


def _fd_warp(mag: torch.Tensor) -> torch.Tensor:
    max_mag = mag.amax(dim=-1, keepdim=True).clamp(min=EPS_FE)
    return mag / max_mag.pow(0.5)


class FeatureExtractionModule(nn.Module):
    def __init__(self, output_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.output_dtype = output_dtype

    @torch.no_grad()
    def forward(self, x_usp: torch.Tensor) -> list[torch.Tensor]:
        i_usp = x_usp.real.to(torch.float64)
        q_usp = x_usp.imag.to(torch.float64)
        total_pwr = (i_usp**2 + q_usp**2).mean(dim=-1, keepdim=True)
        rms = total_pwr.sqrt().clamp(min=EPS_FE)
        i_utp = i_usp / rms
        q_utp = q_usp / rms

        outputs: list[torch.Tensor] = []
        for i_in, q_in in [(i_utp, q_utp), (i_usp, q_usp)]:
            i_x2, q_x2 = _squaring_layer(i_in, q_in)
            iw, qw = _td_warp(i_x2, q_x2, 0.5)
            outputs.append(torch.stack([iw, qw], dim=1).to(self.output_dtype))
            outputs.append(_fd_warp(_fft_magnitude(i_x2, q_x2)).unsqueeze(1).to(self.output_dtype))

            i_x4, q_x4 = _squaring_layer(i_x2, q_x2)
            iw, qw = _td_warp(i_x4, q_x4, 0.5)
            outputs.append(torch.stack([iw, qw], dim=1).to(self.output_dtype))
            outputs.append(_fd_warp(_fft_magnitude(i_x4, q_x4)).unsqueeze(1).to(self.output_dtype))

            i_x6, q_x6 = _pow3_layer(i_x2, q_x2)
            iw, qw = _td_warp(i_x6, q_x6, 2.0 / 3.0)
            outputs.append(torch.stack([iw, qw], dim=1).to(self.output_dtype))
        return outputs


class _ConvMaxPool1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 23):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ConvAvgPool1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 23):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x)))).squeeze(-1)


class CNNBranch(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            _ConvMaxPool1d(in_channels, 16),
            _ConvMaxPool1d(16, 24),
            _ConvMaxPool1d(24, 32),
            _ConvMaxPool1d(32, 48),
            _ConvMaxPool1d(48, 64),
            _ConvAvgPool1d(64, 96),
        )
        self.fc = nn.Linear(96, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x))


class Snoap2024CAP(nn.Module):
    def __init__(self, n_classes: int = 8):
        super().__init__()
        branch_in_ch = [2, 1, 2, 1, 2, 2, 1, 2, 1, 2]
        self.feature_ext = FeatureExtractionModule()
        self.branches = nn.ModuleList([CNNBranch(in_ch, n_classes) for in_ch in branch_in_ch])
        self.final_fc = nn.Linear(10 * n_classes, n_classes)

    def forward(self, x_usp: torch.Tensor) -> torch.Tensor:
        features = self.feature_ext(x_usp)
        branch_outs = [branch(feat) for branch, feat in zip(self.branches, features)]
        return self.final_fc(torch.cat(branch_outs, dim=-1))


@dataclass
class OnlineTimingResult:
    device: str
    n_samples: int
    load_s: float
    boi_usp_s: float
    model_s: float
    total_s: float
    per_sample_s: float


class SnoapOnlinePipeline:
    """Raw ``.tim`` -> BOI/USP -> Snoap2024CAP inference on one device."""

    def __init__(
        self,
        dataset_root: Path,
        checkpoint_path: Path,
        device: str | torch.device = "cpu",
        cfg: Mapping | None = None,
        complex_dtype: torch.dtype = torch.complex128,
        enable_tf32: bool = True,
        empty_cache_per_signal: bool = False,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = torch.device(device)
        self.cfg = dict(cfg or make_default_config(self.dataset_root))
        self.complex_dtype = complex_dtype
        self.empty_cache_per_signal = bool(empty_cache_per_signal)

        if self.device.type == "cuda" and enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        ckpt_path = self.checkpoint_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model = Snoap2024CAP(n_classes=len(CLASS_NAMES)).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.preprocessor = TorchBoiUspPreprocessor(self.cfg, self.device, complex_dtype=complex_dtype)

    def paths_for_indices(self, year: int, signal_indices: Sequence[int]) -> list[Path]:
        return [
            _signal_path(self.dataset_root, int(year), int(idx), int(self.cfg["N_PER_BATCH"]))
            for idx in signal_indices
        ]

    @torch.inference_mode()
    def predict_indices_sequential(self, year: int, signal_indices: Sequence[int]) -> tuple[np.ndarray, dict]:
        """Load the full chunk once, then run one complete signal at a time.

        This forbids batched BOI/feature/model inference: signal ``i+1`` starts
        only after signal ``i`` has completed BOI/USP, feature extraction, CNN
        forward, argmax, and final prediction transfer.
        """
        paths = self.paths_for_indices(year, signal_indices)

        _sync(self.device)
        total_t0 = time.perf_counter()
        t0 = time.perf_counter()
        x_batch = load_tim_batch(
            paths,
            int(self.cfg["SAMPLES_PER_SIG"]),
            device=self.device,
            complex_dtype=self.complex_dtype,
        )
        _sync(self.device)
        t1 = time.perf_counter()

        boi_usp_s = 0.0
        model_s = 0.0
        empty_cache_s = 0.0
        preds: list[int] = []
        for row_idx in range(int(x_batch.shape[0])):
            _sync(self.device)
            stage_t0 = time.perf_counter()
            x_usp, _ = self.preprocessor._boi_shift_one(x_batch[row_idx])
            x_usp = x_usp.unsqueeze(0)
            _sync(self.device)
            boi_usp_s += time.perf_counter() - stage_t0

            stage_t0 = time.perf_counter()
            logits = self.model(x_usp)
            pred = int(logits.argmax(dim=1).detach().cpu().item())
            _sync(self.device)
            model_s += time.perf_counter() - stage_t0
            preds.append(pred)
            del logits, x_usp

            if self.empty_cache_per_signal and self.device.type == "cuda":
                _sync(self.device)
                stage_t0 = time.perf_counter()
                torch.cuda.empty_cache()
                _sync(self.device)
                empty_cache_s += time.perf_counter() - stage_t0

        total_s = time.perf_counter() - total_t0
        return np.asarray(preds, dtype=np.int64), {
            "load_s": t1 - t0,
            "boi_usp_s": boi_usp_s,
            "model_s": model_s,
            "empty_cache_s": empty_cache_s,
            "total_s": total_s,
        }

    @torch.inference_mode()
    def benchmark_indices_sequential(
        self,
        year: int,
        signal_indices: Sequence[int],
    ) -> OnlineTimingResult:
        """Benchmark strictly sequential BOI/USP and CNN inference.

        The raw files are loaded into one device tensor once, but no BOI,
        feature-extraction, CNN, argmax, or prediction-return operation is
        batched across signals.  Signal ``i+1`` starts only after signal ``i``
        has produced and returned its final prediction.
        """
        _, timing = self.predict_indices_sequential(year, signal_indices)
        total_s = float(timing["total_s"])
        return OnlineTimingResult(
            device=str(self.device),
            n_samples=len(signal_indices),
            load_s=float(timing["load_s"]),
            boi_usp_s=float(timing["boi_usp_s"]),
            model_s=float(timing["model_s"]),
            total_s=total_s,
            per_sample_s=total_s / max(1, len(signal_indices)),
        )


def benchmark_cpu_gpu_sequential(
    dataset_root: Path,
    checkpoint_path: Path,
    year: int = 2022,
    batch_ids: Iterable[int] = (25,),
    limit: int = 64,
    gpu_device: str = "cuda:0",
    complex_dtype: torch.dtype = torch.complex64,
) -> tuple[list[dict], list[int]]:
    """Run the original final-cell benchmark without label/accuracy work."""
    signal_indices = select_signal_indices(
        dataset_root=dataset_root,
        year=year,
        batch_ids=batch_ids,
        limit=limit,
    )
    devices = ["cpu"]
    if torch.cuda.is_available():
        gpu_index = torch.device(gpu_device).index
        if gpu_index is not None and gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested {gpu_device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
            )
        devices.append(gpu_device)

    rows: list[dict] = []
    for device in devices:
        pipeline = SnoapOnlinePipeline(
            dataset_root=dataset_root,
            checkpoint_path=checkpoint_path,
            device=device,
            complex_dtype=complex_dtype,
        )
        result = pipeline.benchmark_indices_sequential(year, signal_indices)
        row = result.__dict__.copy()
        row.update(
            load_ms=row.pop("load_s") * 1000.0,
            boi_usp_ms=row.pop("boi_usp_s") * 1000.0,
            model_ms=row.pop("model_s") * 1000.0,
            total_ms=row.pop("total_s") * 1000.0,
            per_sample_ms=row.pop("per_sample_s") * 1000.0,
            samples_per_second=result.n_samples / result.total_s,
        )
        rows.append(row)
        del pipeline
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return rows, signal_indices
