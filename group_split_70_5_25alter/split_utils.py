"""Split all eight modulations together by their shared generation parameters.

SignalIndex and SignalType are excluded from the group key. Negative-SNR 2018
rows are removed before grouping. No eligible IQ waveform is discarded.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


SEED = 20260926
LABEL_FILES = {
    2018: "CSPB_ML_2018_Data/CSPB_ML_2018_Signal_Truth_Labels.txt",
    2022: "CSPB_ML_2022_Data/CSPB_ML_2022_Signal_Truth_Labels.txt",
}
SPLITS = ("train", "val", "test")


def make_manifest(dataset_root: Path, year: int, seed: int = SEED) -> pd.DataFrame:
    year = int(year)
    truth = pd.read_csv(Path(dataset_root) / LABEL_FILES[year], sep=r"\s+", dtype=str)
    required = {
        "SignalIndex", "SignalType", "BaseSymbolPeriod", "CarrierOffset",
        "ExcessBandwidth", "UpsampleFactor", "DownsampleFactor",
        "InbandSNRdB", "NoiseSpectralDensitydB",
    }
    if set(truth.columns) != required:
        raise ValueError(f"Unexpected {year} truth columns: {list(truth.columns)}")
    truth["SignalIndex"] = truth["SignalIndex"].astype(int)
    if truth["SignalIndex"].duplicated().any():
        raise ValueError(f"Duplicate {year} SignalIndex")
    if year == 2018:
        truth = truth.loc[truth["InbandSNRdB"].astype(float) >= 0].copy()

    group_fields = [c for c in truth.columns if c not in ("SignalIndex", "SignalType")]
    truth["group_id"] = truth.groupby(group_fields, sort=True, dropna=False).ngroup()
    counts_by_type = truth.groupby(["group_id", "SignalType"]).size().unstack(fill_value=0)
    if counts_by_type.shape[1] != 8 or not counts_by_type.eq(counts_by_type.iloc[:, 0], axis=0).all().all():
        raise ValueError("Each parameter group must contain all eight signal types equally")
    groups = truth.groupby("group_id", sort=True).size().rename("size").reset_index()
    if (groups["size"] % 8).any():
        raise AssertionError("Parameter-group sizes must be multiples of eight")

    group_split = {}
    # One unit is one sample of each of the eight modulation types. Splitting
    # in these units gives the nearest attainable 70/5/25 sample counts.
    total_units = len(truth) // 8
    train_target = round(0.70 * total_units)
    val_target = round(0.05 * total_units)
    targets = {"train": train_target, "val": val_target,
               "test": total_units - train_target - val_target}
    remaining = targets.copy()
    groups["units"] = groups["size"] // 8
    rng = np.random.default_rng(seed + year)
    # Assign large groups first; one-unit groups fill the last integer gaps.
    for units in sorted(groups["units"].unique(), reverse=True):
        ids = rng.permutation(groups.loc[groups["units"] == units, "group_id"].to_numpy())
        for group_id in ids:
            choices = [split for split in SPLITS if remaining[split] >= units]
            if not choices:
                raise AssertionError(f"Cannot fit {year} parameter group {group_id}")
            split = max(choices, key=lambda name: (remaining[name] / targets[name], remaining[name]))
            group_split[int(group_id)] = split
            remaining[split] -= int(units)
    if any(remaining.values()):
        raise AssertionError(f"Sample quotas not filled: {year} {remaining}")

    truth["split"] = truth["group_id"].map(group_split)
    if truth["split"].isna().any() or truth.groupby("group_id")["split"].nunique().max() != 1:
        raise AssertionError("Missing split or parameter-group leakage")
    counts = truth["split"].value_counts()
    expected = {split: 8 * units for split, units in targets.items()}
    if any(int(counts.get(split, 0)) != target for split, target in expected.items()):
        raise AssertionError(f"Sample quota mismatch: {counts.to_dict()} != {expected}")
    return truth[["SignalIndex", "SignalType", "InbandSNRdB", "group_id", "split"]].sort_values("SignalIndex").reset_index(drop=True)


def ensure_manifests(dataset_root: Path, experiment_dir: Path,
                     replace_existing: bool = False) -> dict[int, pd.DataFrame]:
    split_dir = Path(experiment_dir) / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for year in (2018, 2022):
        expected = make_manifest(dataset_root, year)
        path = split_dir / f"{year}_group_split.csv"
        if path.exists() and not replace_existing:
            stored = pd.read_csv(path, dtype={"InbandSNRdB": str})
            pd.testing.assert_frame_equal(stored, expected, check_dtype=False)
        else:
            temp_path = path.with_suffix(".csv.tmp")
            expected.to_csv(temp_path, index=False)
            temp_path.replace(path)
        manifests[year] = expected
    metadata = {
        "seed": SEED,
        "group_fields": [
            "BaseSymbolPeriod", "CarrierOffset", "ExcessBandwidth",
            "UpsampleFactor", "DownsampleFactor", "InbandSNRdB", "NoiseSpectralDensitydB",
        ],
        "group_excludes_signal_type": True,
        "split_seed": SEED,
        "sample_ratio_target": {"train": 0.70, "val": 0.05, "test": 0.25},
        "negative_2018_snr_excluded": True,
        "counts": {
            str(year): {
                split: {
                    "samples": int((manifest["split"] == split).sum()),
                    "groups": int(manifest.loc[manifest["split"] == split, "group_id"].nunique()),
                }
                for split in SPLITS
            }
            for year, manifest in manifests.items()
        },
    }
    metadata_path = split_dir / "metadata.json"
    temp_metadata = metadata_path.with_suffix(".json.tmp")
    temp_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temp_metadata.replace(metadata_path)
    return manifests


def split_index_sets(manifests: dict[int, pd.DataFrame], train_year: int) -> dict[str, set[int]]:
    train_year = int(train_year)
    ext_year = 2022 if train_year == 2018 else 2018
    own = manifests[train_year]
    return {
        **{
            split: set(own.loc[own["split"] == split, "SignalIndex"].astype(int))
            for split in SPLITS
        },
        "ext": set(manifests[ext_year]["SignalIndex"].astype(int)),
    }


def run_manifest_inference(pipeline, manifests, train_year: int, output_csv: Path,
                           resume: bool = True, checkpoint_every: int = 100) -> pd.DataFrame:
    """Infer on the same-year held-out samples and every eligible other-year sample."""
    import csv
    from tqdm.auto import tqdm

    columns = ["dataset", "index", "snr_db", "true_label", "lr_prediction", "nn_prediction"]
    train_year = int(train_year)
    ext_year = 2022 if train_year == 2018 else 2018
    targets = {
        train_year: manifests[train_year].loc[manifests[train_year]["split"] == "test"],
        ext_year: manifests[ext_year],
    }
    expected = {(int(year), int(index)) for year, frame in targets.items() for index in frame["SignalIndex"]}
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if resume and output_csv.exists():
        existing = pd.read_csv(output_csv, keep_default_na=False)
        if list(existing.columns) != columns:
            raise ValueError(f"Unexpected columns in {output_csv}: {list(existing.columns)}")
        completed = set(zip(existing["dataset"].astype(int), existing["index"].astype(int)))
        if len(completed) != len(existing) or not completed <= expected:
            raise ValueError("Existing inference CSV has duplicate or out-of-split rows")
    else:
        with output_csv.open("w", newline="") as file:
            csv.DictWriter(file, fieldnames=columns).writeheader()
        completed = set()

    with output_csv.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        for year, frame in targets.items():
            pending = frame.loc[~frame["SignalIndex"].isin(
                index for old_year, index in completed if old_year == year
            )]
            for row in tqdm(pending.itertuples(index=False), total=len(pending), desc=f"{train_year}: {year}"):
                signal_index = int(row.SignalIndex)
                snr_db = float(row.InbandSNRdB)
                if year == 2018 and snr_db < 0:
                    raise AssertionError("Negative-SNR 2018 row reached inference")
                result = pipeline.process_signal(year, signal_index)
                writer.writerow({
                    "dataset": year,
                    "index": signal_index,
                    "snr_db": snr_db,
                    "true_label": str(row.SignalType).upper(),
                    "lr_prediction": result.coarse_pred,
                    "nn_prediction": result.pred_label if result.coarse_pred == "QPSK-like" else "",
                })
                if len(completed) % checkpoint_every == 0:
                    file.flush()
                completed.add((year, signal_index))
            file.flush()
    result = pd.read_csv(output_csv, keep_default_na=False)
    if len(result) != len(expected):
        raise AssertionError(f"Incomplete inference CSV: {len(result)} / {len(expected)}")
    return result


if __name__ == "__main__":
    experiment_dir = Path(__file__).resolve().parent
    project_dir = experiment_dir.parents[2]
    dataset_root = (project_dir / "../../CSPB.ML_dataset").resolve()
    manifests = ensure_manifests(dataset_root, experiment_dir, replace_existing=True)
    for year, manifest in manifests.items():
        summary = manifest.groupby("split").agg(
            samples=("SignalIndex", "size"), groups=("group_id", "nunique")
        ).reindex(SPLITS)
        print(f"{year}: {len(manifest):,} eligible samples, {manifest['group_id'].nunique():,} parameter groups")
        print(summary.to_string())
