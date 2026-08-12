"""Fully online LR55-only coarse-classification accuracy evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from .lr55_cccap.fully_online_pipeline_inference import (
    _boi_preprocess,
    _normalize_label,
    _required_fft_vectors,
    _signal_path,
    _tim_to_complex_iq_batch,
    _truth_column,
    load_truth_table,
    make_default_config,
)


def _coarse_label(label: str) -> str:
    normalized = _normalize_label(label)
    if normalized in {"QPSK", "16QAM", "64QAM", "256QAM"}:
        return "QPSK-like"
    return normalized


def _select_indices_and_labels(
    dataset_root: Path,
    cfg: dict,
    year: int,
    batch_ids: Sequence[int],
    limit: int,
    drop_2018_negative_snr: bool,
) -> tuple[list[int], list[str]]:
    truth = load_truth_table(dataset_root, year)
    label_col = _truth_column(
        truth,
        ["SignalType", "signal_type", "signal type", "modulation", "modulation_type", "label"],
    )
    snr_col = _truth_column(
        truth,
        ["InbandSNRdB", "inband_snr_db_label", "inband snr db label", "snr", "snr_db"],
    )
    indices: list[int] = []
    labels: list[str] = []
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
            labels.append(_coarse_label(str(truth.loc[signal_index, label_col])))
            if len(indices) == int(limit):
                return indices, labels
    raise ValueError(f"Only found {len(indices)} eligible signals; need {limit}.")


def evaluate_lr55_coarse_accuracy(
    dataset_root: Path,
    year: int = 2018,
    batch_ids: Sequence[int] = (25,),
    limit: int = 64,
    drop_2018_negative_snr: bool = True,
) -> float:
    """Return accuracy from raw IQ through BOI/FFT3 and the LR55 output."""
    dataset_root = Path(dataset_root).resolve()
    cfg = make_default_config(dataset_root)
    model_path = Path(__file__).resolve().parent / "lr55_cccap" / "models" / "lr55_2018train20.joblib"
    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_names = list(artifact["feature_names"])

    signal_indices, true_labels = _select_indices_and_labels(
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
    x_batch = _tim_to_complex_iq_batch(paths, int(cfg["SAMPLES_PER_SIG"]))

    lr_rows = []
    for x in x_batch:
        x_processed, _prepared, center_shift_hz = _boi_preprocess(x, cfg)
        lr55, _, _ = _required_fft_vectors(
            x_processed,
            center_shift_hz,
            cfg,
            feature_names,
            include_auxiliary=False,
        )
        lr_rows.append(lr55)
    predictions = np.asarray(model.predict(np.stack(lr_rows).astype(np.float32, copy=False)), dtype=str)
    truth = np.asarray(true_labels, dtype=str)
    return float(np.mean(predictions == truth))


__all__ = ["evaluate_lr55_coarse_accuracy"]
