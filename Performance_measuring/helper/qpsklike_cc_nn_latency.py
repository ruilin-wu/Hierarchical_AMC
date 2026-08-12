"""Four-way QPSK-like CC-to-NN latency benchmark with no LR stage."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from .lr55_cccap import pre_compact_fft3_vector_generator as compact
from .lr55_cccap import pre_qpsklike_rs_cfo_estimator as rc
from .lr55_cccap.fully_online_pipeline_inference import (
    CAPNet,
    _boi_preprocess as _boi_preprocess_cpu,
    _cc_power as _cc_power_cpu,
    _required_fft_vectors,
    _signal_path,
    _tim_to_complex_iq_batch,
    make_default_config,
)
from .lr55_cccap.fully_online_pipeline_inference_gpu import (
    _cc_power as _cc_power_gpu,
    _estimate_rs_cfo_from_torch_vectors,
    _extract_vectors_torch,
    boi_preprocess_torch,
    tim_to_complex_iq_batch_torch,
)
from .lr55_cccap.qpsk_cc_four_way_profile import (
    build_qpsk_cap_input_original_cpu,
    build_qpsk_cap_input_original_gpu,
    build_qpsk_cap_input_simplified_cpu,
    build_qpsk_cap_input_simplified_gpu,
    eligible_qpsklike_indices,
)


STAGE_COLUMNS = (
    "load_s",
    "boi_shift_s",
    "fft3_rs_cfo_feature_s",
    "rs_cfo_s",
    "cc_s",
    "nn_s",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_cap_model(model_path: Path, device: torch.device) -> CAPNet:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    train_cfg = checkpoint.get("train_config", {})
    model_cfg = dict(train_cfg.get("model_config", {}))
    model_cfg.pop("name", None)
    for key in ("shared_k", "shared_s"):
        if key in model_cfg:
            model_cfg[key] = tuple(model_cfg[key])
    model = CAPNet(**model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval()


def _profile_cpu(
    dataset_root: Path,
    cfg: dict,
    cap_model: CAPNet,
    year: int,
    signal_indices: Sequence[int],
    cc_mode: str,
) -> dict:
    gc.collect()
    timings: dict[str, float] = {}
    paths = [
        _signal_path(dataset_root, year, index, int(cfg["N_PER_BATCH"]))
        for index in signal_indices
    ]
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    x_batch = _tim_to_complex_iq_batch(paths, int(cfg["SAMPLES_PER_SIG"]))
    timings["load_s"] = time.perf_counter() - stage_start

    staged = []
    stage_start = time.perf_counter()
    for x in x_batch:
        x_processed, prepared, center_shift_hz = _boi_preprocess_cpu(x, cfg)
        staged.append((x_processed, prepared, center_shift_hz))
    timings["boi_shift_s"] = time.perf_counter() - stage_start

    auxiliary_vectors = []
    stage_start = time.perf_counter()
    for x_processed, _prepared, center_shift_hz in staged:
        _, rs20, cfo1 = _required_fft_vectors(
            x_processed,
            center_shift_hz,
            cfg,
            (),
            include_auxiliary=True,
        )
        auxiliary_vectors.append((rs20, cfo1))
    timings["fft3_rs_cfo_feature_s"] = time.perf_counter() - stage_start

    estimates = []
    stage_start = time.perf_counter()
    for (x_processed, _prepared, _center), (rs20, cfo1) in zip(staged, auxiliary_vectors):
        sample = {
            "x_processed": np.asarray(x_processed, dtype=np.complex64),
            "vectors": {"rs20": rs20, "cfo1": cfo1},
        }
        estimates.append(rc.estimate_rs_cfo_from_sample(
            sample,
            compact.RS20_FEATURES,
            compact.CFO_FEATURES,
            set(),
            rs_vector_key="rs20",
        ))
    timings["rs_cfo_s"] = time.perf_counter() - stage_start

    cc_builder = (
        build_qpsk_cap_input_original_cpu
        if cc_mode == "original"
        else build_qpsk_cap_input_simplified_cpu
    )
    cap_inputs = []
    stage_start = time.perf_counter()
    for (x_processed, prepared, _center), estimate in zip(staged, estimates):
        power = _cc_power_cpu(prepared, x_processed, float(cfg["EPS"]))
        cap_inputs.append(cc_builder(
            np.asarray(x_processed, dtype=np.complex64),
            np.float32(estimate["T0_est"]),
            np.float32(estimate["cfo_est"]),
            np.float32(power),
            eps=float(cfg["EPS"]),
        ))
    timings["cc_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    cap_array = np.stack(cap_inputs).astype(np.float32, copy=False)
    with torch.inference_mode():
        _ = cap_model(torch.from_numpy(cap_array[:, None, :, :])).argmax(dim=1)
    timings["nn_s"] = time.perf_counter() - stage_start

    total_s = time.perf_counter() - total_start
    return {
        "device": f"cpu_{cc_mode}_cc",
        "n_samples": len(signal_indices),
        **timings,
        "total_s": total_s,
    }


def _profile_gpu(
    dataset_root: Path,
    cfg: dict,
    cap_model: CAPNet,
    device: torch.device,
    year: int,
    signal_indices: Sequence[int],
    cc_mode: str,
) -> dict:
    gc.collect()
    torch.cuda.empty_cache()
    timings: dict[str, float] = {}
    paths = [
        _signal_path(dataset_root, year, index, int(cfg["N_PER_BATCH"]))
        for index in signal_indices
    ]
    _sync(device)
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    x_batch = tim_to_complex_iq_batch_torch(paths, int(cfg["SAMPLES_PER_SIG"]), device).to(torch.complex64)
    _sync(device)
    timings["load_s"] = time.perf_counter() - stage_start

    staged = []
    stage_start = time.perf_counter()
    for row in range(int(x_batch.shape[0])):
        x_processed, prepared, center_shift_hz = boi_preprocess_torch(x_batch[row], cfg)
        staged.append((x_processed, prepared, center_shift_hz))
    _sync(device)
    timings["boi_shift_s"] = time.perf_counter() - stage_start

    auxiliary_vectors = []
    stage_start = time.perf_counter()
    for x_processed, _prepared, center_shift_hz in staged:
        _, rs20, cfo1 = _extract_vectors_torch(
            x_processed,
            center_shift_hz,
            cfg,
            (),
            include_lr=False,
        )
        auxiliary_vectors.append((rs20, cfo1))
    _sync(device)
    timings["fft3_rs_cfo_feature_s"] = time.perf_counter() - stage_start

    estimates = []
    stage_start = time.perf_counter()
    for rs20, cfo1 in auxiliary_vectors:
        estimates.append(_estimate_rs_cfo_from_torch_vectors(rs20, cfo1))
    _sync(device)
    timings["rs_cfo_s"] = time.perf_counter() - stage_start

    cc_builder = (
        build_qpsk_cap_input_original_gpu
        if cc_mode == "original"
        else build_qpsk_cap_input_simplified_gpu
    )
    cap_inputs = []
    stage_start = time.perf_counter()
    for (x_processed, prepared, _center), estimate in zip(staged, estimates):
        power = _cc_power_gpu(prepared, x_processed, float(cfg["EPS"]))
        cap_inputs.append(cc_builder(
            x_processed.to(torch.complex64),
            estimate[1].to(device=device, dtype=torch.float32),
            estimate[2].to(device=device, dtype=torch.float32),
            power.to(device=device, dtype=torch.float32),
            eps=float(cfg["EPS"]),
        ))
    _sync(device)
    timings["cc_s"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    with torch.inference_mode():
        _ = cap_model(torch.stack(cap_inputs)[:, None, :, :]).argmax(dim=1)
    _sync(device)
    timings["nn_s"] = time.perf_counter() - stage_start

    total_s = time.perf_counter() - total_start
    return {
        "device": f"{device}_{cc_mode}_cc",
        "n_samples": len(signal_indices),
        **timings,
        "total_s": total_s,
    }


def benchmark_qpsklike_cc_nn_stages(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    gpu_device: str = "cuda:0",
    drop_2018_negative_snr: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """Run CPU/GPU original/simplified CC paths without any LR stage."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark requires CPU and GPU paths.")
    device = torch.device(gpu_device)
    if device.type != "cuda":
        raise ValueError(f"gpu_device must be CUDA, got {gpu_device!r}")
    dataset_root = Path(dataset_root).resolve()
    cfg = make_default_config(dataset_root)
    signal_indices = eligible_qpsklike_indices(
        dataset_root=dataset_root,
        cfg=cfg,
        year=int(year),
        batch_ids=[int(x) for x in batch_ids],
        limit=int(limit),
        drop_2018_negative_snr=bool(drop_2018_negative_snr),
    )
    model_path = Path(__file__).resolve().parent / "lr55_cccap" / "models" / "capnet_2018train20.pt"
    cpu_model = _load_cap_model(model_path, torch.device("cpu"))
    gpu_model = _load_cap_model(model_path, device)

    rows = [
        _profile_cpu(dataset_root, cfg, cpu_model, int(year), signal_indices, "original"),
        _profile_gpu(dataset_root, cfg, gpu_model, device, int(year), signal_indices, "original"),
        _profile_cpu(dataset_root, cfg, cpu_model, int(year), signal_indices, "simplified"),
        _profile_gpu(dataset_root, cfg, gpu_model, device, int(year), signal_indices, "simplified"),
    ]
    aggregate_seconds = pd.DataFrame(rows)[["device", "n_samples", *STAGE_COLUMNS, "total_s"]]
    per_sample_ms = aggregate_seconds[["device", "n_samples"]].copy()
    for column in (*STAGE_COLUMNS, "total_s"):
        per_sample_ms[f"{column[:-2]}_ms"] = (
            aggregate_seconds[column] * 1000.0 / aggregate_seconds["n_samples"]
        )
    per_sample_ms["samples_per_second"] = aggregate_seconds["n_samples"] / aggregate_seconds["total_s"]
    return aggregate_seconds, per_sample_ms, signal_indices


__all__ = ["benchmark_qpsklike_cc_nn_stages"]
