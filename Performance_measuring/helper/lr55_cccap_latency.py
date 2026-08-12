"""Latency entry points for the LR55 + CC + CAP pipeline.

This wrapper preserves the four paths benchmarked by the source notebook's
last cell, while returning only whole-pipeline latency metrics.  Model loading
and selection of the fixed 64-signal workload happen outside the timed region.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from .lr55_cccap.fully_online_pipeline_inference import (
    FullyOnlinePipeline as FullyOnlinePipelineCPU,
    make_default_config,
)
from .lr55_cccap.fully_online_pipeline_inference_gpu import FullyOnlinePipelineGPU
from .lr55_cccap.qpsk_cc_four_way_profile import (
    STAGE_COLUMNS,
    eligible_qpsklike_indices,
    profile_four_schemes,
)


def _profile_lr55_cccap(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    gpu_device: str = "cuda:0",
    drop_2018_negative_snr: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Run the shared four-way profiler once and return its detailed output."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires both CPU and GPU paths.")
    gpu = torch.device(gpu_device)
    if gpu.type != "cuda":
        raise ValueError(f"gpu_device must be a CUDA device, got {gpu_device!r}")
    if gpu.index is not None and gpu.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested {gpu_device}, but only {torch.cuda.device_count()} CUDA device(s) are visible."
        )

    dataset_root = Path(dataset_root).resolve()
    model_dir = Path(__file__).resolve().parent / "lr55_cccap" / "models"
    cfg = make_default_config(dataset_root)
    signal_indices = eligible_qpsklike_indices(
        dataset_root=dataset_root,
        cfg=cfg,
        year=int(year),
        batch_ids=[int(x) for x in batch_ids],
        limit=int(limit),
        drop_2018_negative_snr=bool(drop_2018_negative_snr),
    )

    cpu_pipeline = FullyOnlinePipelineCPU(
        dataset_root=dataset_root,
        model_dir=model_dir,
        split_mode="2018train20",
        cfg=dict(cfg),
        device="cpu",
    )
    gpu_pipeline = FullyOnlinePipelineGPU(
        dataset_root=dataset_root,
        model_dir=model_dir,
        split_mode="2018train20",
        cfg=dict(cfg),
        device=gpu_device,
        cc_backend="gpu",
    )

    detailed = profile_four_schemes(
        cpu_pipeline=cpu_pipeline,
        gpu_pipeline=gpu_pipeline,
        year=int(year),
        signal_indices=signal_indices,
    )
    return detailed, signal_indices


def benchmark_lr55_cccap_end_to_end(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    gpu_device: str = "cuda:0",
    drop_2018_negative_snr: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """Benchmark the four paths and return only end-to-end metrics."""
    detailed, signal_indices = _profile_lr55_cccap(
        dataset_root=dataset_root,
        year=year,
        batch_ids=batch_ids,
        limit=limit,
        gpu_device=gpu_device,
        drop_2018_negative_snr=drop_2018_negative_snr,
    )
    result = detailed[["device", "n_samples", "total_s", "per_sample_s"]].copy()
    result["per_sample_ms"] = result.pop("per_sample_s") * 1000.0
    result["samples_per_second"] = result["n_samples"] / result["total_s"]
    return result, signal_indices


def benchmark_lr55_cccap_stages(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    gpu_device: str = "cuda:0",
    drop_2018_negative_snr: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Return aggregate seconds and mean per-sample milliseconds by stage."""
    detailed, signal_indices = _profile_lr55_cccap(
        dataset_root=dataset_root,
        year=year,
        batch_ids=batch_ids,
        limit=limit,
        gpu_device=gpu_device,
        drop_2018_negative_snr=drop_2018_negative_snr,
    )
    seconds_columns = ["device", "n_samples", *STAGE_COLUMNS, "total_s"]
    aggregate_seconds = detailed[seconds_columns].copy()

    per_sample_ms = detailed[["device", "n_samples"]].copy()
    for column in (*STAGE_COLUMNS, "total_s"):
        output_column = f"{column[:-2]}_ms"
        per_sample_ms[output_column] = detailed[column] * 1000.0 / detailed["n_samples"]
    per_sample_ms["samples_per_second"] = detailed["n_samples"] / detailed["total_s"]
    return aggregate_seconds, per_sample_ms, signal_indices


__all__ = [
    "benchmark_lr55_cccap_end_to_end",
    "benchmark_lr55_cccap_stages",
]
