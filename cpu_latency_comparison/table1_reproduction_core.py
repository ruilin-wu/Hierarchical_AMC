"""Self-contained computations used by the recomputed Table-I notebook.

Inputs are generated in memory from raw IQ and dataset metadata.
Only one checkpoint per model and training year is used; no result cache is read.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS8 = ("BPSK", "QPSK", "8PSK", "DQPSK", "MSK", "16QAM", "64QAM", "256QAM")
CLASS4 = ("QPSK", "16QAM", "64QAM", "256QAM")
LABEL_MAP = {name: index for index, name in enumerate(CLASS8)}
QPSK_LIKE = {"QPSK", "16QAM", "64QAM", "256QAM"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_pad_2d(x, kernel_size, stride=(1, 1), dilation=(1, 1)):
    kh, kw = kernel_size
    sh, sw = stride
    dh, dw = dilation
    in_h, in_w = x.shape[-2:]
    effective_h, effective_w = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    out_h, out_w = (in_h + sh - 1) // sh, (in_w + sw - 1) // sw
    pad_h = max(0, (out_h - 1) * sh + effective_h - in_h)
    pad_w = max(0, (out_w - 1) * sw + effective_w - in_w)
    return F.pad(x, (pad_w // 2, pad_w - pad_w // 2,
                     pad_h // 2, pad_h - pad_h // 2))


class SameConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 dilation=(1, 1), bias=False):
        super().__init__()
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):
        return self.conv(_same_pad_2d(x, self.kernel_size, self.stride, self.dilation))


class CAPBranch(nn.Module):
    def __init__(self, in_channels=56, h_out=11, pool_w=2, c1=56, c2=72,
                 k1=(4, 4), s1=(1, 2), k2=(4, 6), s2=(1, 2), fc_mid=7):
        super().__init__()
        self.conv1 = SameConv2d(in_channels, c1, k1, s1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv2 = SameConv2d(c1, c2, k2, s2, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.pool = nn.AdaptiveAvgPool2d((h_out, pool_w))
        self.fc1 = nn.Linear(c2 * h_out * pool_w, fc_mid, bias=False)
        self.bn3 = nn.BatchNorm1d(fc_mid)
        self.fc2 = nn.Linear(fc_mid, 1)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc2(F.relu(self.bn3(self.fc1(x))))


class CAPNet(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.shared_conv = SameConv2d(1, 56, (6, 4), (1, 2), bias=False)
        self.shared_bn = nn.BatchNorm2d(56)
        self.branches = nn.ModuleList([CAPBranch() for _ in range(n_classes)])

    def forward(self, x):
        shared = torch.tanh(self.shared_bn(self.shared_conv(x)))
        return torch.cat([branch(shared) for branch in self.branches], dim=1)


class ModernResBlock(nn.Module):
    def __init__(self, channels, kernel=5):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        value = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(self.bn2(self.conv2(value)) + x)


class ModernStage(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ModernResBlock(channels) for _ in range(blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ModernCAPNetLite(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.stem = nn.Conv2d(1, 20, 5, padding=2, bias=False)
        self.stem_bn = nn.BatchNorm2d(20)
        self.block1 = ModernStage(20, 3)
        self.down = nn.Conv2d(20, 20, 5, stride=(1, 2), padding=2, bias=False)
        self.down_bn = nn.BatchNorm2d(20)
        self.block2 = ModernStage(20, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(20, n_classes)

    def forward(self, x):
        x = self.block1(F.gelu(self.stem_bn(self.stem(x))))
        x = self.block2(F.gelu(self.down_bn(self.down(x))))
        return self.fc(self.pool(x).flatten(1))


class LegacyResBlock(nn.Module):
    def __init__(self, channels, kernel=5):
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        value = F.gelu(self.bn1(self.conv1(x)))
        return F.gelu(self.bn2(self.conv2(value)) + x)


class LegacyCAPNetLite(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.stem = nn.Conv2d(1, 20, 5, padding=2, bias=False)
        self.stem_bn = nn.BatchNorm2d(20)
        self.block1 = nn.Sequential(*[LegacyResBlock(20) for _ in range(3)])
        self.down = nn.Conv2d(20, 20, 5, stride=(1, 2), padding=2, bias=False)
        self.down_bn = nn.BatchNorm2d(20)
        self.block2 = nn.Sequential(*[LegacyResBlock(20) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(20, n_classes)

    def forward(self, x):
        x = self.block1(F.gelu(self.stem_bn(self.stem(x))))
        x = self.block2(F.gelu(self.down_bn(self.down(x))))
        return self.fc(self.pool(x).flatten(1))


EPS_FE = 1e-12


def _squaring_layer(i_value, q_value):
    return i_value * i_value - q_value * q_value, 2.0 * i_value * q_value


def _pow3_layer(i_value, q_value):
    i2, q2 = i_value * i_value, q_value * q_value
    return i2 * i_value - 3.0 * i_value * q2, 3.0 * i2 * q_value - q2 * q_value


def _fft_magnitude(i_value, q_value, nfft=32768):
    value = torch.complex(i_value, q_value)
    return torch.fft.fftshift(torch.fft.fft(value, n=nfft, dim=-1), dim=-1).abs()


def _td_warp(i_value, q_value, exponent):
    magnitude = (i_value * i_value + q_value * q_value).sqrt()
    scale = magnitude.amax(dim=-1, keepdim=True).clamp(min=EPS_FE).pow(exponent)
    return i_value / scale, q_value / scale


def _fd_warp(magnitude):
    scale = magnitude.amax(dim=-1, keepdim=True).clamp(min=EPS_FE)
    return magnitude / scale.pow(0.5)


class USPFeatureExtraction(nn.Module):
    @torch.no_grad()
    def forward(self, signal):
        i_usp, q_usp = signal.real.float(), signal.imag.float()
        rms = (i_usp.square() + q_usp.square()).mean(dim=-1, keepdim=True).sqrt().clamp(min=EPS_FE)
        pairs = ((i_usp / rms, q_usp / rms), (i_usp, q_usp))
        outputs = []
        for i_value, q_value in pairs:
            i2, q2 = _squaring_layer(i_value, q_value)
            iw, qw = _td_warp(i2, q2, 0.5)
            outputs.append(torch.stack([iw, qw], dim=1))
            outputs.append(_fd_warp(_fft_magnitude(i2, q2)).unsqueeze(1))
            i4, q4 = _squaring_layer(i2, q2)
            iw, qw = _td_warp(i4, q4, 0.5)
            outputs.append(torch.stack([iw, qw], dim=1))
            outputs.append(_fd_warp(_fft_magnitude(i4, q4)).unsqueeze(1))
            i6, q6 = _pow3_layer(i2, q2)
            iw, qw = _td_warp(i6, q6, 2.0 / 3.0)
            outputs.append(torch.stack([iw, qw], dim=1))
        return outputs


class ConvMaxPool1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=23):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(2, 2))

    def forward(self, x):
        return self.block(x)


class ConvAvgPool1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=23):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x)))).squeeze(-1)


class USPCNNBranch(nn.Module):
    def __init__(self, in_channels, n_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            ConvMaxPool1d(in_channels, 16), ConvMaxPool1d(16, 24),
            ConvMaxPool1d(24, 32), ConvMaxPool1d(32, 48),
            ConvMaxPool1d(48, 64), ConvAvgPool1d(64, 96))
        self.fc = nn.Linear(96, n_classes)

    def forward(self, x):
        return self.fc(self.net(x))


class USPNet(nn.Module):
    def __init__(self, n_classes=8):
        super().__init__()
        self.feature_ext = USPFeatureExtraction()
        channels = (2, 1, 2, 1, 2, 2, 1, 2, 1, 2)
        self.branches = nn.ModuleList([USPCNNBranch(value, n_classes) for value in channels])
        self.final_fc = nn.Linear(10 * n_classes, n_classes)

    def forward(self, signal):
        features = self.feature_ext(signal)
        return self.final_fc(torch.cat([branch(value) for branch, value in
                                       zip(self.branches, features)], dim=-1))


class USPRealInput(nn.Module):
    """Deployment wrapper accepting contiguous real/imag pairs."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, signal):
        return self.model(torch.view_as_complex(signal))


def count_macs(model: nn.Module, dummy: torch.Tensor) -> dict:
    counts = {"conv_macs": 0, "linear_macs": 0}
    hooks = []

    def count(module, inputs, output):
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            kernel = int(np.prod(module.kernel_size))
            counts["conv_macs"] += int(output.numel() * module.in_channels // module.groups * kernel)
        elif isinstance(module, nn.Linear):
            counts["linear_macs"] += int(output.numel() * module.in_features)

    for layer in model.modules():
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            hooks.append(layer.register_forward_hook(count))
    model.eval()
    with torch.inference_mode():
        model(dummy)
    for hook in hooks:
        hook.remove()
    counts["total_macs"] = counts["conv_macs"] + counts["linear_macs"]
    counts["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return counts


def native_kernels(assets: Path):
    modules = []
    for name in ("spl_boi_exact", "spl_fft3_exact", "spl_cc_current"):
        spec = importlib.util.spec_from_file_location(name, assets / "native" / f"{name}.so")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def dataset_metadata(project: Path, year: int) -> pd.DataFrame:
    """Read dataset truth and the split manifest, never evaluation results."""
    root = project / "../../CSPB.ML_dataset" / f"CSPB_ML_{year}_Data"
    truth = pd.read_csv(root / f"CSPB_ML_{year}_Signal_Truth_Labels.txt", sep=r"\s+")
    split = pd.read_csv(project / "group_split_70_5_25alter/splits" / f"{year}_group_split.csv")
    frame = truth.merge(split[["SignalIndex", "split"]], on="SignalIndex", validate="one_to_one")
    frame["label"] = frame.SignalType.str.upper().map(LABEL_MAP)
    if frame.label.isna().any():
        raise ValueError("Unknown modulation label")
    return frame.sort_values("SignalIndex")


def select_rows(frame: pd.DataFrame, samples_per_class: int | None) -> pd.DataFrame:
    """Select evenly across SNR within each class; None selects the full population."""
    if samples_per_class is None:
        return frame.sort_values("SignalIndex")
    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive or None")
    rows = []
    for label in range(8):
        group = frame.loc[frame.label == label].sort_values(["InbandSNRdB", "SignalIndex"])
        if len(group) < samples_per_class:
            raise ValueError("Not enough samples in a class")
        positions = np.linspace(0, len(group)-1, samples_per_class+2)[1:-1].round().astype(int)
        rows.append(group.iloc[positions])
    return pd.concat(rows).sort_values("SignalIndex")


def raw_path(project: Path, year: int, index: int) -> Path:
    batch = (int(index)-1)//4000+1
    return (project / "../../CSPB.ML_dataset" / f"CSPB_ML_{year}_Data" /
            f"Batch_Dir_{batch}" / f"signal_{int(index)}.tim")


@torch.inference_mode()
def generate_inputs(project: Path, assets: Path, samples_per_class: int = 32) -> dict:
    """Generate all evaluation tensors from raw IQ in RAM without writing caches."""
    import sys
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from helper_function import pre_cc_matrix_core_from_old as oracle
    boi, fft, cc = native_kernels(assets)
    data = {}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for year in (2018, 2022):
            metadata = dataset_metadata(project, year)
            for population in ("same_test", "external_all"):
                frame = metadata.loc[metadata.split == "test"] if population == "same_test" else metadata
                selected = select_rows(frame, samples_per_class)
                lr, mixed, blind, iq = [], [], [], []
                for item in selected.itertuples(index=False):
                    raw = cc.read_tim(str(raw_path(project, year, item.SignalIndex)), 32768)
                    x, power, center, *_ = boi.boi_exact(raw)
                    _, _, db4, dbe, features, *_ = fft.fft3_exact(x, False)
                    rs, cfo = fft.estimate_rs_cfo(db4, dbe)
                    blind_cc = cc.cc_matrix_current(x, float(rs), float(cfo), float(power))
                    if str(item.SignalType).upper() in QPSK_LIKE:
                        mixed_cc = blind_cc
                    else:
                        # Preserve the original class-specific oracle CC pattern.
                        up = float(item.UpsampleFactor) or 1.0
                        down = float(item.DownsampleFactor) or 1.0
                        t0 = up * float(item.BaseSymbolPeriod) / down
                        residual = float(item.CarrierOffset) - float(center)
                        pattern = oracle.CC_PATTERN_MAP[str(item.SignalType).lower()]
                        moments = {}
                        matrix = np.column_stack([
                            oracle.cyclic_cumulant_sequence_tau0_pattern(
                                x.numpy(), t0, residual, n, m, 5, pattern,
                                False, moment_cache=moments)
                            for n, m in oracle.CC_NM[:15]])
                        orders = np.array([n for n, _ in oracle.CC_NM[:15]])
                        mixed_cc = torch.from_numpy((np.abs(matrix)**(2.0/orders) /
                                                     (float(power)+1e-12)).astype(np.float32))
                    lr.append(features)
                    mixed.append(mixed_cc.unsqueeze(0))
                    blind.append(blind_cc.unsqueeze(0))
                    iq.append((x / np.sqrt(power)).to(torch.complex64))
                pack = dict(signal_index=torch.tensor(selected.SignalIndex.to_numpy(), dtype=torch.int64),
                            label=torch.tensor(selected.label.to_numpy(), dtype=torch.int64),
                            lr8_32=torch.stack(lr), mixed_cc=torch.stack(mixed),
                            blind_cc=torch.stack(blind), signal=torch.stack(iq))
                for key in ("lr8_32", "mixed_cc", "blind_cc", "signal"):
                    if not torch.isfinite(pack[key]).all():
                        raise ValueError(f"Nonfinite {key}: {year} {population}")
                data[year, population] = pack
                print(f"Raw IQ inputs generated in memory: {year} {population}, {len(selected)} samples", flush=True)
    finally:
        torch.set_num_threads(previous_threads)
    return data


def select_population(data: dict, train_year: int, test_year: int) -> dict:
    return data[test_year, "same_test" if train_year == test_year else "external_all"]


@torch.inference_mode()
def predict_batches(model: nn.Module, x: torch.Tensor, device: torch.device,
                    batch_size: int = 1024) -> np.ndarray:
    model = model.to(device).eval()
    output = []
    for start in range(0, len(x), batch_size):
        output.append(model(x[start:start + batch_size].to(device, non_blocking=True)).argmax(1).cpu())
    model.cpu()
    return torch.cat(output).numpy().astype(np.int64)


class LRRuntime(nn.Module):
    def __init__(self, pipeline):
        super().__init__()
        imputer = pipeline.named_steps["simpleimputer"]
        scaler = pipeline.named_steps["standardscaler"]
        classifier = pipeline.named_steps["logisticregression"]
        self.register_buffer("fill", torch.as_tensor(imputer.statistics_, dtype=torch.float32))
        self.register_buffer("mean", torch.as_tensor(scaler.mean_, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer("coef", torch.as_tensor(classifier.coef_, dtype=torch.float32))
        self.register_buffer("intercept", torch.as_tensor(classifier.intercept_, dtype=torch.float32))

    def forward(self, x):
        x = torch.where(torch.isfinite(x), x, self.fill)
        return ((x - self.mean) / self.scale) @ self.coef.T + self.intercept


def frozen_runtime(model: nn.Module, example: torch.Tensor):
    """Trace, freeze, serialize and reload exactly as the LibTorch deployment."""
    model = model.cpu().eval()
    with torch.inference_mode(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(model, example, check_trace=False).eval()
        frozen = torch.jit.freeze(traced)
        with tempfile.NamedTemporaryFile(suffix=".ts") as handle:
            frozen.save(handle.name)
            runtime = torch.jit.load(handle.name, map_location="cpu").eval()
    return torch.jit.optimize_for_inference(torch.jit.freeze(runtime))


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    return (per_seed.groupby(["method", "train_year", "test_year"], sort=False)
            .agg(accuracy_percent=("accuracy_percent", "first"),
                 n_seeds=("seed", "nunique"), n=("n", "first"))
            .reset_index())


def route_predictions(assets: Path, train_year: int, lr_features: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    artifact = joblib.load(assets / "weights/lr" / f"lr8_32_{train_year}train20.joblib")
    prediction = artifact["model"].predict(lr_features.numpy())
    gate = np.asarray(prediction) == "QPSK-like"
    coarse = np.full(len(prediction), -1, dtype=np.int64)
    for index, value in enumerate(prediction):
        if value != "QPSK-like":
            coarse[index] = LABEL_MAP[str(value).upper()]
    return gate, coarse


def evaluate_accuracies(assets: Path, device: torch.device, data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records, route_rows = [], []
    mixed_root = assets / "weights/mixed_cc"
    method_map = {
        "FFT_CC_CAP8": ("cap8_always", "CAP8"),
        "SSCA_CC_CAP8": ("ssca_reference", "CAP8"),
        "FFT_CC_CNN8": ("lite8_always", "CNN8"),
        "SSCA_CC_CNN8": ("ssca_lite8_always", "CNN8"),
    }
    for folder in sorted(path for path in mixed_root.iterdir() if path.is_dir()):
        prefix = folder.name.split("_mixed_")[0]
        method, kind = method_map[prefix]
        train_year = int(folder.name.split("_mixed_")[1].split("train")[0])
        for test_year in (2018, 2022):
            population = select_population(data, train_year, test_year)
            truth = population["label"].numpy().astype(np.int64)
            x = population["mixed_cc"]
            for path in sorted(folder.glob("seed*_best.pt")):
                seed = int(path.stem.split("seed")[1].split("_")[0])
                model = CAPNet() if kind == "CAP8" else ModernCAPNetLite(8)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
                pred = predict_batches(model, x, device, 1024)
                records.append(dict(method=method, train_year=train_year, test_year=test_year,
                                    seed=seed, n=len(truth),
                                    accuracy_percent=100.0 * float(np.mean(pred == truth))))
            print(f"{method} {train_year}->{test_year}: one weight complete", flush=True)

    selected = {
        year: tuple(int(path.stem.split("_seed")[-1]) for path in
                    sorted((assets / "weights/proposed_lite4").glob(f"*_{year}train20_seed*.pt")))
        for year in (2018, 2022)
    }
    for train_year, seeds in selected.items():
        for test_year in (2018, 2022):
            population = select_population(data, train_year, test_year)
            truth = population["label"].numpy().astype(np.int64)
            gate, coarse = route_predictions(assets, train_year, population["lr8_32"])
            true_like = np.isin(truth, [LABEL_MAP[name] for name in QPSK_LIKE])
            route_rows.append(dict(train_year=train_year, test_year=test_year, n=len(truth),
                predicted_refinement_rate=float(gate.mean()), true_qpsklike_rate=float(true_like.mean()),
                true_qpsklike_n=int(true_like.sum()), other_n=int((~true_like).sum()),
                tpr=float(gate[true_like].mean()), fpr=float(gate[~true_like].mean()),
                false_family_rejections=int(np.sum(true_like & ~gate)),
                unnecessary_refinements=int(np.sum(~true_like & gate))))
            x_gate = population["mixed_cc"][torch.from_numpy(gate)]
            for seed in seeds:
                name = ("capnet_lite_lr8_32_xprocrawcc_warpscale_k5_c20_20_nb3_2_" 
                        f"{train_year}train20_seed{seed}.pt")
                model = ModernCAPNetLite(4)
                model.load_state_dict(torch.load(assets / "weights/proposed_lite4" / name,
                                                 map_location="cpu", weights_only=True), strict=True)
                refined = predict_batches(model, x_gate, device, 2048)
                pred = coarse.copy()
                pred[gate] = np.asarray([LABEL_MAP[CLASS4[value]] for value in refined])
                records.append(dict(method="proposed", train_year=train_year, test_year=test_year,
                                    seed=seed, n=len(truth),
                                    accuracy_percent=100.0 * float(np.mean(pred == truth)),
                                    qpsklike_accuracy=float(np.mean(pred[true_like] == truth[true_like])),
                                    other_accuracy=float(np.mean(pred[~true_like] == truth[~true_like]))))
            print(f"proposed {train_year}->{test_year}: one weight complete", flush=True)

            direct = joblib.load(assets / "weights/lr" / f"direct_lr8_train{train_year}.joblib")
            direct_pred = np.asarray([LABEL_MAP[str(value).upper()] for value in
                                      direct.predict(population["lr8_32"].numpy())])
            accuracy = 100.0 * float(np.mean(direct_pred == truth))
            for seed in range(1):
                records.append(dict(method="direct", train_year=train_year, test_year=test_year,
                                    seed=seed, n=len(truth), accuracy_percent=accuracy))

    for train_year in (2018, 2022):
        model_paths = [assets / "weights/usp" / f"final_model_{train_year}_seed{seed}.pt"
                       for seed in range(101, 102)]
        for test_year in (2018, 2022):
            pack = select_population(data, train_year, test_year)
            signal = pack["signal"]
            truth = pack["label"].numpy()
            for seed, path in zip(range(101, 102), model_paths):
                model = USPNet(8)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False), strict=True)
                pred = predict_batches(model, signal, device, 64)
                records.append(dict(method="usp", train_year=train_year, test_year=test_year,
                                    seed=seed, n=len(truth),
                                    accuracy_percent=100.0 * float(np.mean(pred == truth))))
            print(f"usp(sample) {train_year}->{test_year}: one weight complete", flush=True)

    per_seed = pd.DataFrame(records)
    summary = summarize(per_seed)
    return per_seed, summary, pd.DataFrame(route_rows)


def evaluate_matched_controls(assets: Path, device: torch.device, data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for train_year in (2018, 2022):
        for test_year in (2018, 2022):
            population = select_population(data, train_year, test_year)
            truth = population["label"].numpy().astype(np.int64)
            gate, coarse = route_predictions(assets, train_year, population["lr8_32"])
            x = population["blind_cc"]
            for seed in range(101, 102):
                cap = CAPNet()
                cap_pack = torch.load(assets / "weights/matched_cap8" /
                                      f"best_{train_year}_capnet_seed{seed}.pt",
                                      map_location="cpu", weights_only=False)
                cap.load_state_dict(cap_pack["model_state_dict"], strict=True)
                lite = LegacyCAPNetLite(8)
                lite.load_state_dict(torch.load(assets / "weights/matched_lite8" /
                                                f"train{train_year}_seed{seed}.pt",
                                                map_location="cpu", weights_only=True), strict=True)
                for label, model in (("cap8", cap), ("lite8", lite)):
                    full = predict_batches(model, x, device, 1024)
                    conditional = np.where(gate, full, coarse)
                    for mode, pred in (("always", full), ("gated", conditional)):
                        records.append(dict(method=f"{label}_{mode}", train_year=train_year,
                                            test_year=test_year, seed=seed, n=len(truth),
                                            accuracy_percent=100.0 * float(np.mean(pred == truth)),
                                            predicted_refinement_rate=float(gate.mean())))
            print(f"matched controls {train_year}->{test_year}: one paired weight complete", flush=True)
    per_seed = pd.DataFrame(records)
    summary = (per_seed.groupby(["method", "train_year", "test_year"], sort=False)
               .agg(accuracy_percent=("accuracy_percent", "first"),
                    n_seeds=("seed", "nunique"), n=("n", "first"),
                    predicted_refinement_rate=("predicted_refinement_rate", "first"))
               .reset_index())
    return per_seed, summary


def architecture_counts() -> pd.DataFrame:
    models = {
        "CC_CAP8": (CAPNet(), torch.zeros(1, 1, 11, 15)),
        "CAPNetLite4": (ModernCAPNetLite(4), torch.zeros(1, 1, 11, 15)),
        "CAPNetLite8": (ModernCAPNetLite(8), torch.zeros(1, 1, 11, 15)),
        "IQ_BOI_USP8": (USPNet(8), torch.zeros(1, 32768, dtype=torch.complex64)),
    }
    return pd.DataFrame.from_dict({name: count_macs(model, dummy)
                                   for name, (model, dummy) in models.items()}, orient="index")


@torch.inference_mode()
def benchmark_torch_model(model: nn.Module, example: torch.Tensor, repeats=100, trials=3) -> dict:
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = frozen_runtime(model, example)
    for _ in range(3):
        model(example)
    values = []
    for _ in range(trials):
        start = time.perf_counter_ns()
        for _ in range(repeats):
            model(example)
        values.append((time.perf_counter_ns() - start) / 1e6 / repeats)
    torch.set_num_threads(original_threads)
    return {"median_ms": float(np.median(values)), "mean_ms": float(np.mean(values)),
            "trial_ms": values, "repeats": repeats}


def fresh_classifier_timings(assets: Path) -> pd.DataFrame:
    rows = []
    for year in (2018, 2022):
        specifications = []
        cap_path = assets / "weights/matched_cap8" / f"best_{year}_capnet_seed101.pt"
        cap_pack = torch.load(cap_path, map_location="cpu", weights_only=False)
        cap = CAPNet(); cap.load_state_dict(cap_pack["model_state_dict"])
        specifications.append(("ref4_cap8", cap, torch.randn(1, 1, 11, 15), 100))
        lite_path = sorted((assets / "weights/mixed_cc" /
                            f"FFT_CC_CNN8_mixed_{year}train").glob("*.pt"))[0]
        lite = ModernCAPNetLite(8); lite.load_state_dict(torch.load(lite_path, map_location="cpu", weights_only=True))
        specifications.append(("control_lite8", lite, torch.randn(1, 1, 11, 15), 200))
        proposed_path = sorted((assets / "weights/proposed_lite4").glob(f"*_{year}train20_seed*.pt"))[0]
        proposed = ModernCAPNetLite(4); proposed.load_state_dict(torch.load(proposed_path, map_location="cpu", weights_only=True))
        specifications.append(("proposed_lite4", proposed, torch.randn(1, 1, 11, 15), 200))
        usp = USPNet(8); usp.load_state_dict(torch.load(assets / "weights/usp" /
                                                       f"final_model_{year}_seed101.pt",
                                                       map_location="cpu", weights_only=False))
        specifications.append(("usp_cnn_core", usp, torch.randn(1, 32768, dtype=torch.complex64), 3))
        for component, model, example, repeats in specifications:
            result = benchmark_torch_model(model, example, repeats=repeats)
            rows.append(dict(year=year, component=component, **result))
            print(f"CPU {year} {component}: {result['median_ms']:.6f} ms", flush=True)
        route = joblib.load(assets / "weights/lr" / f"lr8_32_{year}train20.joblib")["model"]
        direct = joblib.load(assets / "weights/lr" / f"direct_lr8_train{year}.joblib")
        for component, pipeline in (("route5", route), ("direct8", direct)):
            result = benchmark_torch_model(
                LRRuntime(pipeline), torch.zeros((1, 32), dtype=torch.float32), repeats=5000)
            rows.append(dict(year=year, component=component, **result))
    return pd.DataFrame(rows)


def fresh_scalar_timings(project: Path) -> pd.DataFrame:
    """Measure the four arithmetic proxy stages from the local C++ library."""
    import ctypes
    import math
    library = project / "cpu_latency_comparison/helper/cpp_build/libscalar_mac_inference.so"
    if not library.is_file():
        raise FileNotFoundError(library)
    native = ctypes.CDLL(str(library))
    native.scalar_mac_seconds.argtypes = [ctypes.c_uint64, ctypes.c_uint64,
                                          ctypes.POINTER(ctypes.c_float)]
    native.scalar_mac_seconds.restype = ctypes.c_double
    budgets = {"ssca_frontend": 53018624, "fft3_frontend": 1736704,
               "usp_frontend": 2228224, "cc_full": 9076736}
    target_ops = 1_500_000_000
    affinity = os.sched_getaffinity(0)
    core = min(affinity)
    os.sched_setaffinity(0, {core})
    rows = []
    try:
        for name, operations in budgets.items():
            logical = math.ceil(target_ops / operations)
            scheduled = logical * operations
            trials, checksums = [], []
            for trial in range(2):
                checksum = ctypes.c_float()
                seconds = float(native.scalar_mac_seconds(
                    scheduled, 0x20260807 + trial, ctypes.byref(checksum)))
                trials.append(seconds)
                checksums.append(float(checksum.value))
            best = min(trials)
            rows.append(dict(stage=name, operation_budget=operations,
                             logical_inferences=logical, scheduled_ops=scheduled,
                             scalar_proxy_ms=1000.0 * best / logical,
                             trials_seconds=trials, checksum=checksums[trials.index(best)],
                             pinned_core=core))
            print(f"scalar proxy {name}: {rows[-1]['scalar_proxy_ms']:.6f} ms", flush=True)
    finally:
        os.sched_setaffinity(0, affinity)
    return pd.DataFrame(rows)


def weight_inventory(assets: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((assets / "weights").rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(assets)), "bytes": path.stat().st_size,
                         "sha256": sha256(path)})
    return pd.DataFrame(rows)


def _load_cpu_registry(assets: Path, train_year: int) -> dict:
    proposed_path = sorted((assets / "weights/proposed_lite4").glob(
        f"*_{train_year}train20_seed*.pt"))[0]
    proposed = ModernCAPNetLite(4)
    proposed.load_state_dict(torch.load(proposed_path, map_location="cpu", weights_only=True))
    cap = CAPNet()
    cap_pack = torch.load(assets / "weights/matched_cap8" /
                          f"best_{train_year}_capnet_seed101.pt",
                          map_location="cpu", weights_only=False)
    cap.load_state_dict(cap_pack["model_state_dict"])
    lite = LegacyCAPNetLite(8)
    lite.load_state_dict(torch.load(assets / "weights/matched_lite8" /
                                    f"train{train_year}_seed101.pt",
                                    map_location="cpu", weights_only=True))
    usp = USPNet(8)
    usp.load_state_dict(torch.load(assets / "weights/usp" /
                                   f"final_model_{train_year}_seed101.pt",
                                   map_location="cpu", weights_only=False))
    sample_cc = torch.zeros(1, 1, 11, 15)
    sample_iq = torch.zeros(1, 32768, 2, dtype=torch.float32)
    route_pipeline = joblib.load(assets / "weights/lr" /
                                 f"lr8_32_{train_year}train20.joblib")["model"]
    direct_pipeline = joblib.load(
        assets / "weights/lr" / f"direct_lr8_train{train_year}.joblib")
    if isinstance(direct_pipeline, dict):
        direct_pipeline = direct_pipeline["model"]
    sample_lr = torch.zeros(1, 32, dtype=torch.float32)
    return {
        "proposed": frozen_runtime(proposed, sample_cc),
        "cap": frozen_runtime(cap, sample_cc),
        "lite": frozen_runtime(lite, sample_cc),
        "usp": frozen_runtime(USPRealInput(usp), sample_iq),
        "route": frozen_runtime(LRRuntime(route_pipeline), sample_lr),
        "route_classes": tuple(route_pipeline.classes_),
        "direct": frozen_runtime(LRRuntime(direct_pipeline), sample_lr),
        "direct_classes": tuple(direct_pipeline.classes_),
    }


def raw_iq_cpu_measurements(project: Path, assets: Path, samples_per_class: int = 1) -> pd.DataFrame:
    """Fresh small-sample raw-IQ timing for all seven Table-I execution modes."""
    boi_cpp, fft_cpp, cc_cpp = native_kernels(assets)

    methods = (
        "proposed_lite4_gated", "ref4_cap8_always", "ref14_iq_boi_usp_always",
        "cap8_gated_control", "lite8_always_control", "lite8_gated_control",
        "fft32_lr8_direct_control",
    )
    torch.set_num_threads(1)
    registries = {year: _load_cpu_registry(assets, year) for year in (2018, 2022)}

    selected = {}
    for year in (2018, 2022):
        frame = dataset_metadata(project, year)
        selected[year] = select_rows(frame.loc[frame.split == "test"], samples_per_class)

    records = []
    with torch.inference_mode():
        for test_year in (2018, 2022):
            raw_cache = {int(index): cc_cpp.read_tim(str(raw_path(project, test_year, index)), 32768)
                         for index in selected[test_year].SignalIndex}
            for train_year in (2018, 2022):
                registry = registries[train_year]
                for method in methods:
                    for item in selected[test_year].itertuples(index=False):
                        start_total = time.perf_counter_ns()
                        stage = {name: 0.0 for name in
                                 ("boi_ms", "fft3_ms", "lr_ms", "rs_cfo_cc_ms", "model_ms")}
                        start = time.perf_counter_ns()
                        x_processed, power, *_ = boi_cpp.boi_exact(
                            raw_cache[int(item.SignalIndex)])
                        stage["boi_ms"] = (time.perf_counter_ns() - start) / 1e6
                        if method == "ref14_iq_boi_usp_always":
                            start = time.perf_counter_ns()
                            iq = torch.view_as_real(
                                (x_processed / np.sqrt(power)).to(torch.complex64)).unsqueeze(0)
                            pred = int(registry["usp"](iq).argmax(1))
                            stage["model_ms"] = (time.perf_counter_ns() - start) / 1e6
                            refined = False
                        else:
                            start = time.perf_counter_ns()
                            _, _, db4, dbe, vector, *_ = fft_cpp.fft3_exact(
                                x_processed, False)
                            stage["fft3_ms"] = (time.perf_counter_ns() - start) / 1e6
                            start = time.perf_counter_ns()
                            if method == "fft32_lr8_direct_control":
                                output = registry["direct"](vector.unsqueeze(0))
                                label = str(registry["direct_classes"][int(output.argmax(1))]).upper()
                                pred = LABEL_MAP[label]
                                refined = False
                            else:
                                output = registry["route"](vector.unsqueeze(0))
                                route = str(registry["route_classes"][int(output.argmax(1))])
                                refined = route == "QPSK-like"
                                pred = -1 if refined else LABEL_MAP[route.upper()]
                            stage["lr_ms"] = (time.perf_counter_ns() - start) / 1e6
                            always = method in ("ref4_cap8_always", "lite8_always_control")
                            if always:
                                refined = True
                            if refined and method != "fft32_lr8_direct_control":
                                start = time.perf_counter_ns()
                                rs, cfo = fft_cpp.estimate_rs_cfo(db4, dbe)
                                cc = cc_cpp.cc_matrix_current(
                                    x_processed, float(rs), float(cfo), float(power))
                                stage["rs_cfo_cc_ms"] = (time.perf_counter_ns() - start) / 1e6
                                start = time.perf_counter_ns()
                                cc = cc.view(1, 1, 11, 15)
                                if method == "proposed_lite4_gated":
                                    local = int(registry["proposed"](cc).argmax(1))
                                    pred = LABEL_MAP[CLASS4[local]]
                                elif method in ("ref4_cap8_always", "cap8_gated_control"):
                                    pred = int(registry["cap"](cc).argmax(1))
                                else:
                                    pred = int(registry["lite"](cc).argmax(1))
                                stage["model_ms"] = (time.perf_counter_ns() - start) / 1e6
                        total_ms = (time.perf_counter_ns() - start_total) / 1e6
                        records.append(dict(train_year=train_year, test_year=test_year,
                            method=method, signal_index=int(item.SignalIndex),
                            true_label=int(item.label), pred_label=pred, correct=pred == int(item.label),
                            predicted_refinement=bool(refined), total_ms=total_ms, **stage))
                    print(f"raw-IQ CPU {train_year}->{test_year} {method} complete", flush=True)
    return pd.DataFrame(records)
