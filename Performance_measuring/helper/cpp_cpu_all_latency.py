"""One CPU C++ benchmark for latency-comparison pipelines 1--5.

The timed DSP kernels live in ``cpp/cpu_latency_kernels.cpp`` and are compiled
as a PyTorch C++ extension.  CNN/CAP inference uses frozen TorchScript graphs,
so the timed model operators execute in LibTorch/oneDNN rather than Python.
Python in this module is deliberately limited to workload selection, stage
orchestration, clocks, and result-table construction.

No labels are read during a timed run and no accuracy is computed.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


HERE = Path(__file__).resolve().parent
CPP_SOURCE = HERE / "cpp" / "cpu_latency_kernels.cpp"
CPP_BUILD_DIR = HERE / "cpp_build"
CPP_MODEL_DIR = HERE / "cpp_models"
SAMPLES_PER_SIGNAL = 32768
PIPELINE_STAGE_COLUMNS = (
    "load_s",
    "boi_s",
    "ssca_nonconj_s",
    "ssca_conj_s",
    "fft_x4_s",
    "feature_s",
    "lr_s",
    "rs_cfo_s",
    "cc_s",
    "nn_s",
)


def load_cpp_extension(verbose: bool = False):
    """Compile once (incrementally cached) and load the native CPU kernels."""
    from torch.utils.cpp_extension import load

    CPP_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MAX_JOBS", "1")
    return load(
        name="latency_cpp_cpu_v2",
        sources=[str(CPP_SOURCE)],
        build_directory=str(CPP_BUILD_DIR),
        extra_cflags=["-O3", "-march=native", "-DNDEBUG"],
        verbose=bool(verbose),
    )


class _SnoapCNNCore(nn.Module):
    """CNN-only half of Snoap2024; feature extraction is a native C++ stage."""

    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.branches = full_model.branches
        self.final_fc = full_model.final_fc

    def forward(self, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
        features = (f0, f1, f2, f3, f4, f5, f6, f7, f8, f9)
        outputs = [self.branches[i](features[i]) for i in range(10)]
        return self.final_fc(torch.cat(outputs, dim=-1))


def _trace_and_save(module: nn.Module, example, path: Path) -> None:
    module = module.cpu().eval()
    with torch.inference_mode():
        traced = torch.jit.trace(module, example, check_trace=False)
        frozen = torch.jit.freeze(traced.eval())
        frozen.save(str(path))


def prepare_cpp_models(force: bool = False) -> dict[str, Path]:
    """Create frozen LibTorch graphs outside every timed inference run.

    Pipeline 1 intentionally uses random CAP weights, as specified.  Pipelines
    2, 3, and 5 preserve their existing trained weights.  Weight preparation,
    graph tracing, and file I/O are never included in inference latency.
    """
    from .boi_ssca_x4_cc_cap8_latency import build_random_paper_cap
    from .lr55_cccap.fully_online_pipeline_inference import CAPNet
    from .snoap2024_latency import Snoap2024CAP

    CPP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_cap8": CPP_MODEL_DIR / "paper_cap8_random_cpu.ts",
        "qpsklike_cap": CPP_MODEL_DIR / "qpsklike_cap_2018train20_cpu.ts",
        "snoap_core": CPP_MODEL_DIR / "snoap2024_cnn_core_2022train20_cpu.ts",
    }

    if force or not paths["paper_cap8"].exists():
        model = build_random_paper_cap(device="cpu", seed=2026)
        _trace_and_save(model, torch.zeros(1, 1, 11, 15), paths["paper_cap8"])

    if force or not paths["qpsklike_cap"].exists():
        checkpoint_path = HERE / "lr55_cccap" / "models" / "capnet_2018train20.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        train_cfg = checkpoint.get("train_config", {})
        model_cfg = dict(train_cfg.get("model_config", {}))
        model_cfg.pop("name", None)
        for key in ("shared_k", "shared_s"):
            if key in model_cfg:
                model_cfg[key] = tuple(model_cfg[key])
        model = CAPNet(**model_cfg)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        _trace_and_save(model, torch.zeros(1, 1, 11, 15), paths["qpsklike_cap"])

    if force or not paths["snoap_core"].exists():
        checkpoint_path = HERE / "snoap2024_2022train20_final_model.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        full = Snoap2024CAP(n_classes=8)
        full.load_state_dict(state, strict=True)
        core = _SnoapCNNCore(full).eval()
        channels = (2, 1, 2, 1, 2, 2, 1, 2, 1, 2)
        example = tuple(torch.zeros(1, ch, SAMPLES_PER_SIGNAL) for ch in channels)
        _trace_and_save(core, example, paths["snoap_core"])
    return paths


def _load_lr_tensors() -> tuple[torch.Tensor, ...]:
    artifact_path = HERE / "lr55_cccap" / "models" / "lr55_2018train20.joblib"
    artifact = joblib.load(artifact_path)
    pipeline = artifact["model"]
    scaler = pipeline.named_steps["standardscaler"]
    lr = pipeline.named_steps["logisticregression"]
    return tuple(
        torch.as_tensor(value, dtype=torch.float32).contiguous()
        for value in (scaler.mean_, scaler.scale_, lr.coef_, lr.intercept_)
    )


def _path(dataset_root: Path, year: int, index: int) -> Path:
    batch = (int(index) - 1) // 4000 + 1
    return (
        Path(dataset_root)
        / f"CSPB_ML_{int(year)}_Data"
        / f"Batch_Dir_{batch}"
        / f"signal_{int(index)}.tim"
    )


def _first_existing_indices(dataset_root: Path, year: int, limit: int) -> list[int]:
    indices: list[int] = []
    for index in range(96001, 100001):
        if _path(dataset_root, year, index).exists():
            indices.append(index)
            if len(indices) == int(limit):
                return indices
    raise FileNotFoundError(f"Only found {len(indices)} year-{year} signals; need {limit}")


def _qpsklike_indices(dataset_root: Path, limit: int) -> list[int]:
    # Truth is used only once to select the fixed workload, before timing.
    from .lr55_cccap.fully_online_pipeline_inference import make_default_config
    from .lr55_cccap.qpsk_cc_four_way_profile import eligible_qpsklike_indices

    return eligible_qpsklike_indices(
        dataset_root=Path(dataset_root),
        cfg=make_default_config(Path(dataset_root)),
        year=2018,
        batch_ids=(25,),
        limit=int(limit),
        drop_2018_negative_snr=True,
    )


def _clock(function: Callable):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def _empty_row(pipeline: str, implementation: str, variant: str, n: int) -> dict:
    row = {
        "pipeline": pipeline,
        "implementation": implementation,
        "variant": variant,
        "n_samples": int(n),
    }
    row.update({column: 0.0 for column in PIPELINE_STAGE_COLUMNS})
    return row


def _finish_row(row: dict, wall_start: float) -> dict:
    row["stage_sum_s"] = sum(float(row[key]) for key in PIPELINE_STAGE_COLUMNS)
    row["total_s"] = time.perf_counter() - wall_start
    row["orchestration_s"] = max(0.0, row["total_s"] - row["stage_sum_s"])
    row["per_sample_ms"] = 1000.0 * row["total_s"] / max(1, row["n_samples"])
    return row


@torch.inference_mode()
def _warm_up(kernels, sample_path: Path, models: dict[str, torch.jit.ScriptModule], lr_tensors) -> None:
    """Prime FFT plans, oneDNN primitives, and TorchScript graphs."""
    x = kernels.read_tim(str(sample_path), SAMPLES_PER_SIGNAL)
    y = kernels.boi_shift(x, False)
    _ = kernels.ssca_checksum(y, False)
    _ = kernels.ssca_checksum(y, True)
    _ = kernels.fft_x4_checksum(y)
    f55, aux = kernels.lr55_features(y)
    _ = kernels.lr_predict(f55, *lr_tensors)
    est = kernels.estimate_rs_cfo(aux)
    cc = kernels.cc_matrix(y, float(est[0]), float(est[1]), True)
    _ = models["qpsklike_cap"](cc.view(1, 1, 11, 15)).argmax(1)
    _ = models["paper_cap8"](cc.view(1, 1, 11, 15)).argmax(1)
    z = kernels.boi_shift(x, True)
    features = kernels.snoap_features(z)
    _ = models["snoap_core"](*features).argmax(1)


@torch.inference_mode()
def _pipeline_1(k, model, paths: Sequence[Path], random_seed: int) -> dict:
    row = _empty_row("1", "BOI + 2xSSCA + FFTx4 + CC11x15 + CAP8", "original_cc", len(paths))
    wall = time.perf_counter()
    raw, row["load_s"] = _clock(lambda: [k.read_tim(str(p), SAMPLES_PER_SIGNAL) for p in paths])
    boi, row["boi_s"] = _clock(lambda: [k.boi_shift(x, False) for x in raw])
    _, row["ssca_nonconj_s"] = _clock(lambda: [k.ssca_checksum(x, False) for x in boi])
    _, row["ssca_conj_s"] = _clock(lambda: [k.ssca_checksum(x, True) for x in boi])
    _, row["fft_x4_s"] = _clock(lambda: [k.fft_x4_checksum(x) for x in boi])
    rng = np.random.default_rng(int(random_seed))
    rs = rng.uniform(1.0 / 23.0, 0.5, size=len(boi)).astype(np.float32)
    cfo = rng.uniform(-0.001, 0.001, size=len(boi)).astype(np.float32)
    cc, row["cc_s"] = _clock(
        lambda: [k.cc_matrix(x, float(r), float(f), False) for x, r, f in zip(boi, rs, cfo)]
    )
    _, row["nn_s"] = _clock(lambda: [model(x.view(1, 1, 11, 15)).argmax(1) for x in cc])
    return _finish_row(row, wall)


@torch.inference_mode()
def _pipeline_2(k, model, paths: Sequence[Path]) -> dict:
    row = _empty_row("2", "Snoap2024 BOI/USP + 10-branch CNN", "trained_weights", len(paths))
    wall = time.perf_counter()
    raw, row["load_s"] = _clock(lambda: [k.read_tim(str(p), SAMPLES_PER_SIGNAL) for p in paths])
    boi, row["boi_s"] = _clock(lambda: [k.boi_shift(x, True) for x in raw])
    features, row["feature_s"] = _clock(lambda: [k.snoap_features(x) for x in boi])
    _, row["nn_s"] = _clock(lambda: [model(*items).argmax(1) for items in features])
    return _finish_row(row, wall)


@torch.inference_mode()
def _pipeline_3(k, model, paths: Sequence[Path], simplified: bool) -> dict:
    variant = "simplified_cc" if simplified else "original_cc"
    row = _empty_row("3", "QPSK-like FFT3 + CC11x15 + CAP", variant, len(paths))
    wall = time.perf_counter()
    raw, row["load_s"] = _clock(lambda: [k.read_tim(str(p), SAMPLES_PER_SIGNAL) for p in paths])
    boi, row["boi_s"] = _clock(lambda: [k.boi_shift(x, False) for x in raw])
    packs, row["feature_s"] = _clock(lambda: [k.lr55_features(x) for x in boi])
    estimates, row["rs_cfo_s"] = _clock(lambda: [k.estimate_rs_cfo(aux) for _, aux in packs])
    cc, row["cc_s"] = _clock(
        lambda: [
            k.cc_matrix(x, float(est[0]), float(est[1]), bool(simplified))
            for x, est in zip(boi, estimates)
        ]
    )
    _, row["nn_s"] = _clock(lambda: [model(x.view(1, 1, 11, 15)).argmax(1) for x in cc])
    return _finish_row(row, wall)


@torch.inference_mode()
def _pipeline_4(k, lr_tensors, paths: Sequence[Path]) -> dict:
    row = _empty_row("4", "LR55 coarse classifier", "trained_lr", len(paths))
    wall = time.perf_counter()
    raw, row["load_s"] = _clock(lambda: [k.read_tim(str(p), SAMPLES_PER_SIGNAL) for p in paths])
    boi, row["boi_s"] = _clock(lambda: [k.boi_shift(x, False) for x in raw])
    packs, row["feature_s"] = _clock(lambda: [k.lr55_features(x) for x in boi])
    _, row["lr_s"] = _clock(lambda: [k.lr_predict(f, *lr_tensors) for f, _ in packs])
    return _finish_row(row, wall)


@torch.inference_mode()
def _pipeline_5(k, model, lr_tensors, paths: Sequence[Path], simplified: bool) -> dict:
    variant = "simplified_cc" if simplified else "original_cc"
    row = _empty_row("5", "LR55 + Rs/CFO + CC11x15 + CAP", variant, len(paths))
    wall = time.perf_counter()
    raw, row["load_s"] = _clock(lambda: [k.read_tim(str(p), SAMPLES_PER_SIGNAL) for p in paths])
    boi, row["boi_s"] = _clock(lambda: [k.boi_shift(x, False) for x in raw])
    packs, row["feature_s"] = _clock(lambda: [k.lr55_features(x) for x in boi])
    _, row["lr_s"] = _clock(lambda: [k.lr_predict(f, *lr_tensors) for f, _ in packs])
    estimates, row["rs_cfo_s"] = _clock(lambda: [k.estimate_rs_cfo(aux) for _, aux in packs])
    cc, row["cc_s"] = _clock(
        lambda: [
            k.cc_matrix(x, float(est[0]), float(est[1]), bool(simplified))
            for x, est in zip(boi, estimates)
        ]
    )
    _, row["nn_s"] = _clock(lambda: [model(x.view(1, 1, 11, 15)).argmax(1) for x in cc])
    return _finish_row(row, wall)


def environment_report(threads: int) -> pd.DataFrame:
    try:
        compiler = subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]
    except Exception:
        compiler = "unknown"
    return pd.DataFrame(
        [
            {
                "CPU": platform.processor() or platform.machine(),
                "logical_CPUs": os.cpu_count(),
                "intraop_threads": int(threads),
                "PyTorch_LibTorch": torch.__version__,
                "MKL-DNN": bool(torch.backends.mkldnn.is_available()),
                "compiler": compiler,
            }
        ]
    )


def aggregate_seconds_table(results: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    columns = [
        "pipeline", "implementation", "variant", "n_samples", *PIPELINE_STAGE_COLUMNS,
        "orchestration_s", "stage_sum_s", "total_s",
    ]
    return results[columns].round(int(digits))


def per_sample_ms_table(results: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = results[["pipeline", "implementation", "variant", "n_samples"]].copy()
    for column in PIPELINE_STAGE_COLUMNS:
        out[column.removesuffix("_s") + "_ms"] = 1000.0 * results[column] / results["n_samples"]
    out["orchestration_ms"] = 1000.0 * results["orchestration_s"] / results["n_samples"]
    out["total_ms"] = 1000.0 * results["total_s"] / results["n_samples"]
    return out.round(int(digits))


def speed_ranking(results: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    ranked = per_sample_ms_table(results, digits=digits)[
        ["pipeline", "implementation", "variant", "total_ms"]
    ].sort_values("total_ms", ignore_index=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def benchmark_all_cpp_cpu(
    dataset_root: str | Path,
    limit: int = 64,
    threads: int = 8,
    random_seed: int = 2026,
    warmup: bool = True,
    verbose_build: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Run all seven CPU rows derived from notebooks 1--5.

    Compilation, model graph creation/loading, warm-up, label-based workload
    selection, and random-weight initialisation are outside the timed region.
    Timed inference is batch-size 1 throughout.
    """
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be positive")

    kernels = load_cpp_extension(verbose=verbose_build)
    threads = max(1, int(threads))
    torch.set_num_threads(threads)
    kernels.configure_threads(threads)
    model_paths = prepare_cpp_models(force=False)
    models = {name: torch.jit.load(str(path), map_location="cpu").eval() for name, path in model_paths.items()}
    lr_tensors = _load_lr_tensors()

    ordinary18 = _first_existing_indices(dataset_root, 2018, limit)
    ordinary22 = _first_existing_indices(dataset_root, 2022, limit)
    qpsk18 = _qpsklike_indices(dataset_root, limit)
    paths18 = [_path(dataset_root, 2018, index) for index in ordinary18]
    paths22 = [_path(dataset_root, 2022, index) for index in ordinary22]
    qpaths = [_path(dataset_root, 2018, index) for index in qpsk18]

    if warmup:
        _warm_up(kernels, paths18[0], models, lr_tensors)
    gc.collect()

    rows = [
        _pipeline_1(kernels, models["paper_cap8"], paths18, random_seed),
        _pipeline_2(kernels, models["snoap_core"], paths22),
        _pipeline_3(kernels, models["qpsklike_cap"], qpaths, False),
        _pipeline_3(kernels, models["qpsklike_cap"], qpaths, True),
        _pipeline_4(kernels, lr_tensors, paths18),
        _pipeline_5(kernels, models["qpsklike_cap"], lr_tensors, qpaths, False),
        _pipeline_5(kernels, models["qpsklike_cap"], lr_tensors, qpaths, True),
    ]
    return pd.DataFrame(rows), {
        "year_2018": ordinary18,
        "year_2022": ordinary22,
        "qpsklike_2018": qpsk18,
    }


__all__ = [
    "PIPELINE_STAGE_COLUMNS",
    "aggregate_seconds_table",
    "benchmark_all_cpp_cpu",
    "environment_report",
    "load_cpp_extension",
    "per_sample_ms_table",
    "prepare_cpp_models",
    "speed_ranking",
]
