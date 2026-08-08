"""Fully online, per-signal inference for the hierarchical AMC pipeline.

The inference path uses only the raw IQ signal. Ground-truth labels are read
outside ``process_signal`` and are used only for test-set filtering and final
evaluation.
"""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from . import pre_boi_fft3_generation_base_from_notebook as base
from . import pre_cc_matrix_core_from_old as cc
from . import pre_compact_fft3_vector_generator as compact
from . import pre_qpsklike_rs_cfo_estimator as rc


FULL_CLASS_NAMES = [
    "BPSK",
    "MSK",
    "DQPSK",
    "8PSK",
    "QPSK",
    "16QAM",
    "64QAM",
    "256QAM",
]
COARSE_CLASS_NAMES = ["BPSK", "MSK", "DQPSK", "8PSK", "QPSK-like"]
FINE_CLASS_NAMES = ["QPSK", "16QAM", "64QAM", "256QAM"]

LABEL_FILES = {
    2018: "CSPB_ML_2018_Data/CSPB_ML_2018_Signal_Truth_Labels.txt",
    2022: "CSPB_ML_2022_Data/CSPB_ML_2022_Signal_Truth_Labels.txt",
}
DATA_DIRS = {
    2018: "CSPB_ML_2018_Data",
    2022: "CSPB_ML_2022_Data",
}


def make_default_config(dataset_root: Path) -> dict:
    """Return the BOI/FFT configuration used by the generation notebooks."""
    return {
        "DATASET_ROOT": str(Path(dataset_root).resolve()),
        "N_PER_BATCH": 4000,
        "N_BATCHES": 28,
        "N_TOTAL": 112000,
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
        "FFT_NFFT": 32768,
        "FFT_WINDOW": "hann",
        "FFT_TOP_K": 20,
        "FFT_PEAK_MIN_SEP_BINS": 8,
        "GUIDE_KMAX_INTEGER_RS": 5,
        "GUIDE_KMAX_HALF_RS": 5,
        "GUIDE_SEARCH_BINS": 3,
    }


def _same_pad_2d(x, kernel_size, stride=(1, 1), dilation=(1, 1)):
    if isinstance(kernel_size, int):
        kh, kw = kernel_size, kernel_size
    else:
        kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    in_h, in_w = x.shape[-2], x.shape[-1]
    eff_kh = (kh - 1) * dh + 1
    eff_kw = (kw - 1) * dw + 1
    out_h = (in_h + sh - 1) // sh
    out_w = (in_w + sw - 1) // sw
    pad_h = max(0, (out_h - 1) * sh + eff_kh - in_h)
    pad_w = max(0, (out_w - 1) * sw + eff_kw - in_w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class SameConv2d(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size,
        stride=(1, 1),
        dilation=(1, 1),
        bias=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(_same_pad_2d(x, self.kernel_size, self.stride, self.dilation))


class CAPBranch(nn.Module):
    def __init__(
        self,
        in_ch,
        h_out,
        pool_w=2,
        c1=96,
        c2=160,
        k1=(4, 4),
        s1=(1, 2),
        k2=(4, 6),
        s2=(1, 2),
        fc_mid=16,
    ):
        super().__init__()
        self.conv1 = SameConv2d(in_ch, c1, kernel_size=k1, stride=s1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = SameConv2d(c1, c2, kernel_size=k2, stride=s2, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool = nn.AdaptiveAvgPool2d((h_out, pool_w))
        feat_dim = c2 * h_out * pool_w
        self.fc1 = nn.Linear(feat_dim, fc_mid, bias=False)
        self.bn3 = nn.BatchNorm1d(fc_mid)
        self.fc2 = nn.Linear(fc_mid, 1, bias=True)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


class CAPNet(nn.Module):
    def __init__(
        self,
        in_h=11,
        in_w=15,
        n_classes=4,
        shared_ch=96,
        shared_k=(6, 4),
        shared_s=(1, 2),
        branch_c1=96,
        branch_c2=160,
        branch_fc=16,
        pool_w=2,
    ):
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.n_classes = n_classes
        self.shared_conv = SameConv2d(
            1, shared_ch, kernel_size=shared_k, stride=shared_s, bias=False
        )
        self.shared_bn = nn.BatchNorm2d(shared_ch)
        self.branches = nn.ModuleList(
            [
                CAPBranch(
                    shared_ch,
                    h_out=in_h,
                    pool_w=pool_w,
                    c1=branch_c1,
                    c2=branch_c2,
                    fc_mid=branch_fc,
                )
                for _ in range(n_classes)
            ]
        )

    def forward(self, x):
        x = torch.tanh(self.shared_bn(self.shared_conv(x)))
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class ResBlockLite(nn.Module):
    """Residual block used by the CAPNetLite training notebooks."""

    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


class CAPNetLite(nn.Module):
    """Small CAP classifier matching the LR9+CAPNetLite training notebooks."""

    def __init__(self, n_classes=4, c1=32, c2=64):
        super().__init__()
        self.stem = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c1)
        self.block1 = ResBlockLite(c1)
        self.down = nn.Conv2d(c1, c2, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(c2)
        self.block2 = ResBlockLite(c2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        x = self.block1(x)
        x = F.relu(self.down_bn(self.down(x)), inplace=True)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _normalize_label(value: str) -> str:
    key = str(value).strip().upper().replace("-", "").replace("_", "")
    aliases = {
        "BPSK": "BPSK",
        "MSK": "MSK",
        "DQPSK": "DQPSK",
        "8PSK": "8PSK",
        "QPSK": "QPSK",
        "16QAM": "16QAM",
        "QAM16": "16QAM",
        "64QAM": "64QAM",
        "QAM64": "64QAM",
        "256QAM": "256QAM",
        "QAM256": "256QAM",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported modulation label: {value!r}")
    return aliases[key]


def _truth_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    by_lower = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise KeyError(f"None of the columns {list(candidates)} were found in {list(df.columns)}")


def load_truth_table(dataset_root: Path, year: int) -> pd.DataFrame:
    path = Path(dataset_root) / LABEL_FILES[int(year)]
    truth = base.load_truth_labels(path).copy()
    index_col = _truth_column(
        truth,
        ["SignalIndex", "signal_index", "signal number", "signal_number", "index", "signal_id"],
    )
    truth[index_col] = pd.to_numeric(truth[index_col], errors="raise").astype(int)
    return truth.set_index(index_col, drop=False)


def _signal_path(dataset_root: Path, year: int, signal_index: int, n_per_batch: int) -> Path:
    batch_id = (int(signal_index) - 1) // int(n_per_batch) + 1
    return (
        Path(dataset_root)
        / DATA_DIRS[int(year)]
        / f"Batch_Dir_{batch_id}"
        / f"signal_{int(signal_index)}.tim"
    )


def _tim_to_complex_iq_batch(paths: Sequence[Path], samples_per_sig: int) -> np.ndarray:
    """Load a chunk of .tim files into one CPU array."""
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
    return batch


def _fft_branch(z: np.ndarray, cfg: Mapping) -> dict:
    freq, mag_db = base.compute_fft_mag_db_relative(
        z,
        fs=float(cfg["FS_IN"]),
        nfft=int(cfg["FFT_NFFT"]),
        window=str(cfg["FFT_WINDOW"]),
        eps=float(cfg["EPS"]),
    )
    top_peaks = base.find_top_peaks_1d(
        freq,
        mag_db,
        top_k=int(cfg["FFT_TOP_K"]),
        min_sep_bins=int(cfg["FFT_PEAK_MIN_SEP_BINS"]),
        xlim=(-float(cfg["FS_IN"]) / 2.0, float(cfg["FS_IN"]) / 2.0),
    )
    return {
        "freq": np.asarray(freq),
        # Match the saved FFT payload consumed by the compact-vector notebook.
        "mag_db": np.asarray(mag_db, dtype=np.float16),
        "top_peaks": top_peaks,
    }


def _blind_fft_payload(x_processed: np.ndarray, center_shift_hz: float, cfg: Mapping) -> dict:
    """Build the compact-generator payload without inserting any truth values."""
    abs_x2 = np.abs(x_processed) ** 2
    x2 = x_processed**2
    x4 = x_processed**4
    return {
        "meta": {
            "signal_index": -1,
            "signal_type": "QPSK",
            "year": "online",
        },
        "truth": {
            "inband_snr_db_label": np.nan,
            "symbol_rate": np.nan,
            "carrier_offset_label": np.nan,
        },
        "preprocess": {
            "center_shift_hz": float(center_shift_hz),
        },
        "fft": {
            "abs_x2": _fft_branch(abs_x2, cfg),
            "x2": _fft_branch(x2, cfg),
            "x4": _fft_branch(x4, cfg),
        },
    }


def _required_fft_vectors(
    x_processed: np.ndarray,
    center_shift_hz: float,
    cfg: Mapping,
    lr_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate only LR55/RS20/CFO1 vector fields required for inference."""
    required = set(lr_names) | set(compact.RS20_FEATURES) | set(compact.CFO_FEATURES)
    abs_x2 = np.abs(x_processed) ** 2
    branches = {
        "abs_x2": abs_x2,
        "x2": x_processed**2,
        "x4": x_processed**4,
    }
    fft = {}
    for name, z in branches.items():
        fft[name] = _fft_branch(z, cfg)

    row: dict[str, float] = {}
    abs_peaks = fft["abs_x2"]["top_peaks"]
    candidates = []
    for peak in abs_peaks:
        f = abs(float(peak["freq"]))
        db = float(peak["mag_db"])
        if f > 5 / compact.NFFT and f < 0.49:
            candidates.append((db, f))
    candidates.sort(reverse=True)
    rs = float(candidates[0][1]) if candidates else float("nan")
    row["rs_est"] = rs

    def has_prefix(prefix: str) -> bool:
        return any(name.startswith(prefix) for name in required)

    for branch in ["abs_x2", "x2", "x4"]:
        peaks = fft[branch]["top_peaks"]
        dbs = np.array([float(p["mag_db"]) for p in peaks], dtype=float)
        freqs = np.array([float(p["freq"]) for p in peaks], dtype=float)
        abs_freqs = np.abs(freqs)

        for i in range(20):
            f = float(freqs[i]) if i < len(freqs) else np.nan
            db = float(dbs[i]) if i < len(dbs) else -120.0
            fields = {
                f"{branch}_top{i + 1}_f": f,
                f"{branch}_top{i + 1}_af": abs(f) if np.isfinite(f) else np.nan,
                f"{branch}_top{i + 1}_db": db,
                f"{branch}_top{i + 1}_f_over_rs": (
                    f / rs if np.isfinite(rs) and rs > 0 else np.nan
                ),
                f"{branch}_top{i + 1}_af_over_rs": (
                    abs(f) / rs if np.isfinite(rs) and rs > 0 else np.nan
                ),
            }
            for key, value in fields.items():
                if key in required:
                    row[key] = value

        nonzero_dbs = dbs[abs_freqs > 5 / compact.NFFT]
        nonzero_fields = {
            f"{branch}_nonzero_best_db": (
                float(np.max(nonzero_dbs)) if nonzero_dbs.size else -120.0
            ),
            f"{branch}_nonzero_count_gt_m10": (
                int(np.sum(nonzero_dbs > -10)) if nonzero_dbs.size else 0
            ),
            f"{branch}_nonzero_count_gt_m20": (
                int(np.sum(nonzero_dbs > -20)) if nonzero_dbs.size else 0
            ),
            f"{branch}_nonzero_count_gt_m30": (
                int(np.sum(nonzero_dbs > -30)) if nonzero_dbs.size else 0
            ),
        }
        for key, value in nonzero_fields.items():
            if key in required:
                row[key] = value

        if f"{branch}_top1_minus_top2_db" in required:
            row[f"{branch}_top1_minus_top2_db"] = (
                float(dbs[0] - dbs[1]) if len(dbs) > 1 else 120.0
            )
        if f"{branch}_top1_minus_top5_db" in required:
            row[f"{branch}_top1_minus_top5_db"] = (
                float(dbs[0] - dbs[min(4, len(dbs) - 1)]) if len(dbs) else 120.0
            )

        need_sym = (
            f"{branch}_sym_score_mean" in required
            or f"{branch}_sym_score_min" in required
        )
        if need_sym:
            sym = []
            for f, db in zip(freqs, dbs):
                if abs(f) <= 5 / compact.NFFT:
                    continue
                j = int(np.argmin(np.abs(freqs + f))) if len(freqs) else 0
                sym.append(abs(db - dbs[j]) + 40 * abs(freqs[j] + f))
            if f"{branch}_sym_score_mean" in required:
                row[f"{branch}_sym_score_mean"] = float(np.mean(sym)) if sym else 99.0
            if f"{branch}_sym_score_min" in required:
                row[f"{branch}_sym_score_min"] = float(np.min(sym)) if sym else 99.0

        need_grid = (
            has_prefix(f"{branch}_int_grid_")
            or has_prefix(f"{branch}_half_grid_")
            or f"{branch}_half_minus_int_top3_db" in required
            or f"{branch}_center_f" in required
        )
        if need_grid:
            mag_db = fft[branch]["mag_db"]
            center = 0.0 if branch == "abs_x2" else (float(freqs[0]) if len(freqs) else 0.0)
            if f"{branch}_center_f" in required:
                row[f"{branch}_center_f"] = center
            grid_values = {}
            for mode, offset in [("int", 0.0), ("half", 0.5 * rs)]:
                vals = []
                for k in range(-5, 6):
                    target = k * rs
                    if branch != "abs_x2":
                        target += center + offset
                    value = (
                        compact.sample_mag_db_near(mag_db, target, bins=3)
                        if np.isfinite(rs) and rs > 0
                        else -120.0
                    )
                    vals.append(value)
                    key = f"{branch}_{mode}_grid_k{k:+d}_db"
                    if key in required:
                        row[key] = value
                vals = np.array(vals, dtype=float)
                grid_values[mode] = vals
                vals_sorted = np.sort(vals)[::-1]
                stats = {
                    f"{branch}_{mode}_grid_best_db": float(vals_sorted[0]),
                    f"{branch}_{mode}_grid_2nd_db": float(vals_sorted[1]),
                    f"{branch}_{mode}_grid_3rd_db": float(vals_sorted[2]),
                    f"{branch}_{mode}_grid_mean_top3_db": float(np.mean(vals_sorted[:3])),
                    f"{branch}_{mode}_grid_mean_db": float(np.mean(vals)),
                    f"{branch}_{mode}_grid_count_gt_m10": int(np.sum(vals > -10)),
                    f"{branch}_{mode}_grid_count_gt_m20": int(np.sum(vals > -20)),
                    f"{branch}_{mode}_grid_count_gt_m30": int(np.sum(vals > -30)),
                }
                for key, value in stats.items():
                    if key in required:
                        row[key] = value
            key = f"{branch}_half_minus_int_top3_db"
            if key in required:
                row[key] = float(
                    np.mean(np.sort(grid_values["half"])[::-1][:3])
                    - np.mean(np.sort(grid_values["int"])[::-1][:3])
                )

    def vector(names: Sequence[str]) -> np.ndarray:
        missing = [name for name in names if name not in row]
        if missing:
            raise KeyError(f"Missing required FFT features: {missing[:10]}")
        return np.asarray([row[name] for name in names], dtype=np.float32)

    return vector(lr_names), vector(compact.RS20_FEATURES), vector(compact.CFO_FEATURES)


def _boi_preprocess(x: np.ndarray, cfg: Mapping) -> tuple[np.ndarray, dict, float]:
    boi_pack = base.run_fixed_boi_detector(x, cfg)
    prepared = boi_pack["prepared"]
    boi_filter = prepared["selected_filter"]
    max_bw = float(cfg["MAX_EXPANDED_FILTER_BW_FRAC"]) * float(cfg["FS_IN"])
    if float(boi_filter["bandwidth"]) >= max_bw:
        boi_filter = prepared["selected_refined"]
    shifted = base.extract_filter_shift_noresample(
        x=x,
        boi=boi_filter,
        fs=float(cfg["FS_IN"]),
        guard_bins=int(cfg["BOI_GUARD_BINS"]),
    )
    center_shift_hz = float(prepared["selected_refined"]["f_center"])
    return np.asarray(shifted["x_processed"]), prepared, center_shift_hz


def _cc_power(prepared: Mapping, x_processed: np.ndarray, eps: float) -> float:
    report = prepared.get("power_report", {})
    candidates = [
        report.get("p_signal_boi_local"),
        report.get("P_x_processed"),
        report.get("P_x_bpf"),
        report.get("P_total"),
        float(np.mean(np.abs(x_processed) ** 2)),
    ]
    for value in candidates:
        try:
            power = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(power) and power > eps:
            return power
    raise ValueError("No positive finite BOI power was available for CC normalization")


@dataclass
class InferenceResult:
    pred_label: str
    coarse_pred: str
    route: str
    rs_est: float = np.nan
    cfo_est: float = np.nan


class FullyOnlinePipeline:
    def __init__(
        self,
        dataset_root: Path,
        model_dir: Path,
        split_mode: str,
        cfg: Mapping | None = None,
        device: str | None = None,
        lr_vector_key: str = "lr55",
        cap_model_key: str = "capnet",
        cc_algorithm: str = "simplified_double",
    ):
        if split_mode not in {"2018train20", "2022train20"}:
            raise ValueError("split_mode must be '2018train20' or '2022train20'")
        lr_vector_key = str(lr_vector_key).lower()
        expected_lr_feature_counts = {"lr7": 7, "lr9": 9, "lr55": 55}
        if lr_vector_key not in expected_lr_feature_counts:
            raise ValueError("lr_vector_key must be 'lr7', 'lr9', or 'lr55'")
        cap_model_key = str(cap_model_key).lower()
        if cap_model_key not in {"capnet", "capnet_lite"}:
            raise ValueError("cap_model_key must be 'capnet' or 'capnet_lite'")
        cc_algorithm = str(cc_algorithm).lower()
        if cc_algorithm not in {"original_double", "simplified_double"}:
            raise ValueError(
                "cc_algorithm must be 'original_double' or 'simplified_double'"
            )
        self.dataset_root = Path(dataset_root).resolve()
        self.model_dir = Path(model_dir).resolve()
        self.split_mode = split_mode
        self.lr_vector_key = lr_vector_key
        self.cap_model_key = cap_model_key
        self.cc_algorithm = cc_algorithm
        self.cfg = dict(cfg or make_default_config(self.dataset_root))
        self.device = torch.device(
            device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.lr_path = self.model_dir / f"{self.lr_vector_key}_{split_mode}.joblib"
        if self.cap_model_key == "capnet_lite":
            self.cap_path = self.model_dir / f"capnet_lite_c32_64_{split_mode}.pt"
        else:
            self.cap_path = self.model_dir / f"capnet_{split_mode}.pt"
        self.lr_artifact = joblib.load(self.lr_path)
        self.lr_model = self.lr_artifact["model"]
        self.lr_feature_names = list(self.lr_artifact["feature_names"])
        artifact_vector_key = str(
            self.lr_artifact.get("vector_key", self.lr_vector_key)
        ).lower()
        if artifact_vector_key != self.lr_vector_key:
            raise ValueError(
                f"LR artifact vector_key={artifact_vector_key!r} does not match "
                f"requested lr_vector_key={self.lr_vector_key!r}"
            )
        expected_lr_features = expected_lr_feature_counts[self.lr_vector_key]
        if len(self.lr_feature_names) != expected_lr_features:
            raise ValueError(
                f"Expected {expected_lr_features} {self.lr_vector_key.upper()} features, "
                f"found {len(self.lr_feature_names)}"
            )
        forbidden = ("truth", "label", "snr", "signal_type", "symbol_rate")
        leaked = [
            name
            for name in self.lr_feature_names
            if any(token in str(name).lower() for token in forbidden)
        ]
        if leaked:
            raise ValueError(f"LR feature list contains evaluation metadata: {leaked}")

        checkpoint = torch.load(self.cap_path, map_location="cpu", weights_only=False)
        if self.cap_model_key == "capnet_lite":
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            c1 = int(state_dict["stem.weight"].shape[0])
            c2 = int(state_dict["down.weight"].shape[0])
            n_classes = int(state_dict["fc.weight"].shape[0])
            self.cap_model = CAPNetLite(
                n_classes=n_classes,
                c1=c1,
                c2=c2,
            ).to(self.device)
            self.cap_model.load_state_dict(state_dict, strict=True)
            train_cfg = checkpoint.get("train_config", {}) if "model_state_dict" in checkpoint else {}
        else:
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

    def _blind_vectors(
        self, x_processed: np.ndarray, center_shift_hz: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return _required_fft_vectors(
            x_processed,
            center_shift_hz,
            self.cfg,
            self.lr_feature_names,
        )

    def _process_loaded_to_vectors(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        x_processed, prepared, center_shift_hz = _boi_preprocess(x, self.cfg)
        lr55, rs20, cfo1 = self._blind_vectors(x_processed, center_shift_hz)
        return x_processed, lr55, rs20, cfo1, prepared

    def _finish_qpsklike_loaded(
        self,
        x_processed: np.ndarray,
        prepared: Mapping,
        rs20: np.ndarray,
        cfo1: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        x_processed_cc = np.asarray(x_processed, dtype=np.complex64)
        compact_sample = {
            "x_processed": x_processed_cc,
            "vectors": {
                "rs20": rs20,
                "cfo1": cfo1,
            },
        }
        estimates = rc.estimate_rs_cfo_from_sample(
            compact_sample,
            compact.RS20_FEATURES,
            compact.CFO_FEATURES,
            set(),
            rs_vector_key="rs20",
        )
        rs_est = float(estimates["rs_est"])
        t0_est = float(estimates["T0_est"])
        cfo_est = float(estimates["cfo_est"])

        x_processed_cc_double = np.asarray(x_processed_cc, dtype=np.complex128)
        if self.cc_algorithm == "simplified_double":
            raw_cc = cc.build_estimated_qpsk_cc_matrix_11x15_simplified_double(
                x_processed_cc_double,
                t0_est,
                cfo_est,
                remove_mean=False,
                max_k=5,
            )
            cc_nm = cc.CC_NM[:15]
        else:
            raw_cc = cc.build_estimated_cc_matrix_11x24_from_signal_pattern(
                x_processed_cc_double,
                t0_est,
                cfo_est,
                cc.qpsk,
                remove_mean=False,
                max_k=5,
            )
            cc_nm = cc.CC_NM
        power = _cc_power(prepared, x_processed, float(self.cfg["EPS"]))
        _, cc_abs = cc.apply_cc_postprocessing_with_power(
            raw_cc["matrix_complex"],
            power,
            cc_nm,
            enable_warping=True,
            enable_scaling=False,
            is_theory=False,
            eps=float(self.cfg["EPS"]),
        )
        cap_input = np.asarray(cc_abs[:, :15], dtype=np.float32) / np.float32(power)
        return cap_input, rs_est, cfo_est

    def process_signal(self, year: int, signal_index: int) -> InferenceResult:
        """Run one raw-IQ signal through the complete blind pipeline."""
        path = _signal_path(
            self.dataset_root,
            int(year),
            int(signal_index),
            int(self.cfg["N_PER_BATCH"]),
        )
        x = base.tim_to_complex_iq(path, int(self.cfg["SAMPLES_PER_SIG"]))
        x_processed, lr55, rs20, cfo1, prepared = self._process_loaded_to_vectors(x)

        coarse_pred = str(self.lr_model.predict(lr55.reshape(1, -1))[0])
        if coarse_pred not in COARSE_CLASS_NAMES:
            raise ValueError(f"Unexpected LR prediction: {coarse_pred!r}")
        if coarse_pred != "QPSK-like":
            return InferenceResult(
                pred_label=coarse_pred,
                coarse_pred=coarse_pred,
                route="LR-final",
            )

        cap_input, rs_est, cfo_est = self._finish_qpsklike_loaded(
            x_processed,
            prepared,
            rs20,
            cfo1,
        )
        cap_tensor = torch.from_numpy(cap_input[None, None, :, :]).to(self.device)
        with torch.inference_mode():
            fine_index = int(self.cap_model(cap_tensor).argmax(dim=1).item())
        fine_pred = self.fine_class_names[fine_index]
        return InferenceResult(
            pred_label=fine_pred,
            coarse_pred=coarse_pred,
            route="QPSK-like->CC-CAP",
            rs_est=rs_est,
            cfo_est=cfo_est,
        )

    def load_signal_batch(self, year: int, signal_indices: Sequence[int]) -> np.ndarray:
        paths = [
            _signal_path(
                self.dataset_root,
                int(year),
                int(signal_index),
                int(self.cfg["N_PER_BATCH"]),
            )
            for signal_index in signal_indices
        ]
        return _tim_to_complex_iq_batch(paths, int(self.cfg["SAMPLES_PER_SIG"]))

    def process_signal_batch_loaded(self, x_batch: np.ndarray) -> list[InferenceResult]:
        """Run a CPU-loaded chunk through the same pipeline with batched LR/CAP calls."""
        x_batch = np.asarray(x_batch)
        if x_batch.ndim != 2:
            raise ValueError(f"x_batch must have shape (batch, samples), got {x_batch.shape}")

        staged = []
        lr_rows = []
        for x in x_batch:
            x_processed, lr55, rs20, cfo1, prepared = self._process_loaded_to_vectors(x)
            staged.append(
                {
                    "x_processed": x_processed,
                    "rs20": rs20,
                    "cfo1": cfo1,
                    "prepared": prepared,
                }
            )
            lr_rows.append(lr55)

        lr_matrix = np.stack(lr_rows, axis=0).astype(np.float32, copy=False)
        coarse_preds = [str(pred) for pred in self.lr_model.predict(lr_matrix)]
        results: list[InferenceResult | None] = [None] * len(coarse_preds)
        cap_inputs = []
        cap_meta = []

        for row_idx, coarse_pred in enumerate(coarse_preds):
            if coarse_pred not in COARSE_CLASS_NAMES:
                raise ValueError(f"Unexpected LR prediction: {coarse_pred!r}")
            if coarse_pred != "QPSK-like":
                results[row_idx] = InferenceResult(
                    pred_label=coarse_pred,
                    coarse_pred=coarse_pred,
                    route="LR-final",
                )
                continue

            item = staged[row_idx]
            cap_input, rs_est, cfo_est = self._finish_qpsklike_loaded(
                item["x_processed"],
                item["prepared"],
                item["rs20"],
                item["cfo1"],
            )
            cap_inputs.append(cap_input)
            cap_meta.append((row_idx, coarse_pred, rs_est, cfo_est))

        if cap_inputs:
            cap_array = np.stack(cap_inputs, axis=0).astype(np.float32, copy=False)
            cap_tensor = torch.from_numpy(cap_array[:, None, :, :]).to(self.device)
            with torch.inference_mode():
                fine_indices = self.cap_model(cap_tensor).argmax(dim=1).detach().cpu().numpy()
            for fine_index, (row_idx, coarse_pred, rs_est, cfo_est) in zip(fine_indices, cap_meta):
                results[row_idx] = InferenceResult(
                    pred_label=self.fine_class_names[int(fine_index)],
                    coarse_pred=coarse_pred,
                    route="QPSK-like->CC-CAP",
                    rs_est=float(rs_est),
                    cfo_est=float(cfo_est),
                )

        missing = [row_idx for row_idx, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"Missing batch inference results at rows {missing[:10]}")
        return [result for result in results if result is not None]

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
        """Run all eligible signals in the requested batches and checkpoint results."""
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
        progress = tqdm(pending, desc=f"{self.split_mode}: {year}", unit="signal")
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
                    "error": "".join(
                        traceback.format_exception_only(type(exc), exc)
                    ).strip(),
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

    def run_compact_csv(
        self,
        dataset_batches: Mapping[int, Iterable[int]],
        output_csv: Path,
        resume: bool = False,
        checkpoint_every: int = 100,
    ) -> pd.DataFrame:
        """Run several evaluation sets and save only the requested five CSV fields.

        The CSV schema is fixed to:
        ``dataset,index,snr_db,true_label,lr_prediction,nn_prediction``.
        ``nn_prediction`` is empty unless LR routes the signal to ``QPSK-like``.
        Negative-SNR 2018 rows keep their negative ``snr_db`` while both
        prediction fields remain empty; no raw signal is loaded and no
        inference stage is run for those rows.
        """
        columns = [
            "dataset",
            "index",
            "snr_db",
            "true_label",
            "lr_prediction",
            "nn_prediction",
        ]
        legacy_columns = [
            "dataset",
            "index",
            "true_label",
            "lr_prediction",
            "nn_prediction",
        ]
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        existing = pd.DataFrame(columns=columns)
        completed: set[tuple[int, int]] = set()
        if resume and output_csv.exists():
            existing = pd.read_csv(output_csv, keep_default_na=False)
            if list(existing.columns) == legacy_columns:
                snr_values = pd.Series(index=existing.index, dtype=np.float64)
                for existing_year in existing["dataset"].astype(int).unique():
                    truth = load_truth_table(self.dataset_root, int(existing_year))
                    snr_col = _truth_column(
                        truth,
                        [
                            "InbandSNRdB",
                            "inband_snr_db_label",
                            "inband snr db label",
                            "snr",
                            "snr_db",
                        ],
                    )
                    year_mask = existing["dataset"].astype(int).eq(int(existing_year))
                    signal_indices = existing.loc[year_mask, "index"].astype(int)
                    snr_values.loc[year_mask] = (
                        truth.loc[signal_indices, snr_col].astype(float).to_numpy()
                    )
                existing.insert(2, "snr_db", snr_values.astype(float))
                existing = existing.loc[:, columns]
                existing.to_csv(output_csv, index=False)
            elif list(existing.columns) == [
                "dataset",
                "index",
                "snr_db",
                "status",
                "true_label",
                "lr_prediction",
                "nn_prediction",
            ]:
                existing = existing.drop(columns=["status"]).loc[:, columns]
                existing.to_csv(output_csv, index=False)
            elif list(existing.columns) != columns:
                raise ValueError(
                    f"{output_csv} has columns {list(existing.columns)}, expected {columns}"
                )
            completed = {
                (int(row.dataset), int(row.index))
                for row in existing.itertuples(index=False)
            }
        elif not resume:
            existing.to_csv(output_csv, index=False)

        rows = existing.to_dict("records")
        since_checkpoint = 0

        for year_value, batch_ids_value in dataset_batches.items():
            year = int(year_value)
            truth = load_truth_table(self.dataset_root, year)
            label_col = _truth_column(
                truth,
                [
                    "SignalType",
                    "signal_type",
                    "signal type",
                    "modulation",
                    "modulation_type",
                    "label",
                ],
            )
            snr_col = _truth_column(
                truth,
                [
                    "InbandSNRdB",
                    "inband_snr_db_label",
                    "inband snr db label",
                    "snr",
                    "snr_db",
                ],
            )

            indices = []
            for batch_id in [int(value) for value in batch_ids_value]:
                start = (batch_id - 1) * int(self.cfg["N_PER_BATCH"]) + 1
                stop = batch_id * int(self.cfg["N_PER_BATCH"])
                indices.extend(range(start, stop + 1))

            pending = [
                signal_index
                for signal_index in indices
                if (year, signal_index) not in completed
            ]
            progress = tqdm(
                pending,
                desc=f"{self.split_mode}: {year}",
                unit="signal",
            )

            for signal_index in progress:
                truth_row = truth.loc[signal_index]
                true_label = _normalize_label(truth_row[label_col])
                snr_db = float(truth_row[snr_col])
                if year == 2018 and snr_db < 0.0:
                    rows.append(
                        {
                            "dataset": year,
                            "index": int(signal_index),
                            "snr_db": snr_db,
                            "true_label": true_label,
                            "lr_prediction": "",
                            "nn_prediction": "",
                        }
                    )
                    completed.add((year, int(signal_index)))
                    since_checkpoint += 1
                    if since_checkpoint >= int(checkpoint_every):
                        pd.DataFrame(rows, columns=columns).to_csv(
                            output_csv,
                            index=False,
                        )
                        since_checkpoint = 0
                    continue

                result = self.process_signal(year, signal_index)
                nn_prediction = (
                    result.pred_label
                    if result.coarse_pred == "QPSK-like"
                    else ""
                )
                rows.append(
                    {
                        "dataset": year,
                        "index": int(signal_index),
                        "snr_db": snr_db,
                        "true_label": true_label,
                        "lr_prediction": result.coarse_pred,
                        "nn_prediction": nn_prediction,
                    }
                )
                completed.add((year, int(signal_index)))
                since_checkpoint += 1

                if since_checkpoint >= int(checkpoint_every):
                    pd.DataFrame(rows, columns=columns).to_csv(
                        output_csv,
                        index=False,
                    )
                    since_checkpoint = 0
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

        result_df = pd.DataFrame(rows, columns=columns)
        if not result_df.empty:
            result_df = result_df.sort_values(
                ["dataset", "index"],
            ).reset_index(drop=True)
        result_df.to_csv(output_csv, index=False)
        return result_df


def print_and_plot_cm(results: pd.DataFrame, title: str) -> pd.DataFrame:
    if results.empty:
        raise ValueError(f"No evaluated signals are available for {title}")
    errors = results["pred_label"].eq("ERROR")
    if errors.any():
        raise RuntimeError(
            f"{title} contains {int(errors.sum())} failed signals. "
            "Inspect the error column before reporting a confusion matrix."
        )

    y_true = results["true_label"].astype(str)
    y_pred = results["pred_label"].astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=FULL_CLASS_NAMES)
    cm_df = pd.DataFrame(cm, index=FULL_CLASS_NAMES, columns=FULL_CLASS_NAMES)
    accuracy = float(np.mean(y_true.to_numpy() == y_pred.to_numpy()))
    q_fraction = float(np.mean(results["coarse_pred"].eq("QPSK-like")))

    print(f"{title}: N={len(results):,}, accuracy={accuracy:.6f}, QPSK-like route={q_fraction:.4f}")
    display(cm_df)

    denom = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_norm = cm / denom
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    image = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        title=f"{title}\nrow-normalized confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(FULL_CLASS_NAMES)),
        yticks=np.arange(len(FULL_CLASS_NAMES)),
        xticklabels=FULL_CLASS_NAMES,
        yticklabels=FULL_CLASS_NAMES,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.tight_layout()
    plt.show()
    return cm_df
