"""End-to-end inference timing for the fully online LR55-only path."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd

from .lr55_cccap.fully_online_pipeline_inference import (
    _boi_preprocess,
    _required_fft_vectors,
    _signal_path,
    _tim_to_complex_iq_batch,
    _truth_column,
    load_truth_table,
    make_default_config,
)


def _select_signal_indices(
    dataset_root: Path,
    cfg: dict,
    year: int,
    batch_ids: Sequence[int],
    limit: int,
    drop_2018_negative_snr: bool,
) -> list[int]:
    truth = load_truth_table(dataset_root, year)
    snr_col = _truth_column(
        truth,
        ["InbandSNRdB", "inband_snr_db_label", "inband snr db label", "snr", "snr_db"],
    )
    indices: list[int] = []
    for batch_id in [int(x) for x in batch_ids]:
        start = (batch_id - 1) * int(cfg["N_PER_BATCH"]) + 1
        stop = min(batch_id * int(cfg["N_PER_BATCH"]), int(cfg["N_TOTAL"]))
        for signal_index in range(start, stop + 1):
            if year == 2018 and drop_2018_negative_snr:
                if float(truth.loc[signal_index, snr_col]) < 0.0:
                    continue
            path = _signal_path(dataset_root, year, signal_index, int(cfg["N_PER_BATCH"]))
            if not path.exists():
                continue
            indices.append(signal_index)
            if len(indices) == int(limit):
                return indices
    raise ValueError(f"Only found {len(indices)} eligible signals; need {limit}.")


def benchmark_lr55_stage_times(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    drop_2018_negative_snr: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Return aggregate seconds and per-sample milliseconds for LR55 stages."""
    dataset_root = Path(dataset_root).resolve()
    cfg = make_default_config(dataset_root)
    model_path = Path(__file__).resolve().parent / "lr55_cccap" / "models" / "lr55_2018train20.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])
    signal_indices = _select_signal_indices(
        dataset_root=dataset_root,
        cfg=cfg,
        year=int(year),
        batch_ids=batch_ids,
        limit=int(limit),
        drop_2018_negative_snr=bool(drop_2018_negative_snr),
    )
    paths = [
        _signal_path(dataset_root, int(year), index, int(cfg["N_PER_BATCH"]))
        for index in signal_indices
    ]

    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    x_batch = _tim_to_complex_iq_batch(paths, int(cfg["SAMPLES_PER_SIG"]))
    timings["load_s"] = time.perf_counter() - stage_start

    processed = []
    stage_start = time.perf_counter()
    for x in x_batch:
        x_processed, _prepared, center_shift_hz = _boi_preprocess(x, cfg)
        processed.append((x_processed, center_shift_hz))
    timings["boi_shift_s"] = time.perf_counter() - stage_start

    lr_rows = []
    stage_start = time.perf_counter()
    for x_processed, center_shift_hz in processed:
        lr55, _, _ = _required_fft_vectors(
            x_processed,
            center_shift_hz,
            cfg,
            feature_names,
            include_auxiliary=False,
        )
        lr_rows.append(lr55)
    timings["fft3_lr_feature_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    _ = model.predict(np.stack(lr_rows).astype(np.float32, copy=False))
    timings["lr_s"] = time.perf_counter() - stage_start
    total_s = time.perf_counter() - total_start

    aggregate_seconds = pd.DataFrame([{
        "device": "cpu_lr55",
        "n_samples": len(signal_indices),
        **timings,
        "total_s": total_s,
    }])
    per_sample_ms = aggregate_seconds[["device", "n_samples"]].copy()
    for column in ("load_s", "boi_shift_s", "fft3_lr_feature_s", "lr_s", "total_s"):
        per_sample_ms[f"{column[:-2]}_ms"] = (
            aggregate_seconds[column] * 1000.0 / aggregate_seconds["n_samples"]
        )
    per_sample_ms["samples_per_second"] = aggregate_seconds["n_samples"] / aggregate_seconds["total_s"]
    return aggregate_seconds, per_sample_ms, signal_indices


def benchmark_lr55_inference_time(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    drop_2018_negative_snr: bool = True,
) -> float:
    """Return total seconds from raw file loading through LR55 predictions."""
    aggregate, _, _ = benchmark_lr55_stage_times(
        dataset_root=dataset_root,
        year=year,
        batch_ids=batch_ids,
        limit=limit,
        drop_2018_negative_snr=drop_2018_negative_snr,
    )
    return float(aggregate.loc[0, "total_s"])


__all__ = ["benchmark_lr55_inference_time", "benchmark_lr55_stage_times"]
