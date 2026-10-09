"""Seed / hyperparameter sweep for the group-split CAPNetLite QPSK-like trainer.

Loads the shared LR55 vectors / CC matrices / group-split masks ONCE per
process, then retrains CAPNetLite for each requested (split_mode, seed) pair
under a configurable set of TRAINING hyperparameters (LR, weight decay, batch
size, epochs/patience, label smoothing, optimizer), and appends the resulting
full-8-class same-domain and external accuracies to a CSV. The model
architecture (CAPNetLite c1=32/c2=64, 111,716 params) is never changed --
only training-process hyperparameters are, per the "hyperparameters only"
constraint for this experiment.

2018train20 and 2022train20 may use different (and different-length) seed
lists -- they no longer need to share a seed.

Designed to be launched once per physical GPU with a disjoint slice of the
work so a large sweep can run on 4 GPUs (or several processes per GPU, since
CAPNetLite is tiny and one RTX 3090 barely registers load from one run).

This intentionally duplicates (rather than imports) the logic in
Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-{2018,2022}train20.ipynb
cells 1/2/3/5/7/9 so the heavy LR55-vector and CC-matrix loading happens once
per process instead of once per notebook execution.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--seeds", type=str, default="", help="comma-separated seed list applied to BOTH split modes")
parser.add_argument("--seeds-2018", type=str, default="", help="comma-separated seed list for 2018train20 only")
parser.add_argument("--seeds-2022", type=str, default="", help="comma-separated seed list for 2022train20 only")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--tag", type=str, default="baseline", help="label for this hyperparameter config, stored in the CSV")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=3e-4)
parser.add_argument("--batch-size", type=int, default=250)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=45)
parser.add_argument("--plateau-patience", type=int, default=8)
parser.add_argument("--plateau-factor", type=float, default=0.5)
parser.add_argument("--label-smoothing", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
parser.add_argument("--config-json", type=str, default="", help="path to a JSON list of hyperparameter-config dicts; "
                     "overrides all single-config hyperparameter flags above")
args = parser.parse_args()


def _parse_seeds(s):
    return [int(v) for v in s.split(",") if v.strip()]


SEEDS_COMMON = _parse_seeds(args.seeds)
SEEDS_BY_SPLIT = {
    "2018train20": _parse_seeds(args.seeds_2018) or SEEDS_COMMON,
    "2022train20": _parse_seeds(args.seeds_2022) or SEEDS_COMMON,
}
if not args.config_json and not SEEDS_BY_SPLIT["2018train20"] and not SEEDS_BY_SPLIT["2022train20"]:
    raise SystemExit("No seeds given: use --seeds, or --seeds-2018/--seeds-2022, or --config-json")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parents[2]
EXPERIMENT_DIR = HERE
sys.path.insert(0, str(EXPERIMENT_DIR))
from split_utils import ensure_manifests, split_index_sets

torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible CUDA device for GPU {args.gpu}.")
DEVICE = torch.device("cuda:0")
print(f"[gpu{args.gpu}] visible device: {torch.cuda.get_device_name(0)}", flush=True)

# =========================
# Architecture is FIXED (never swept): CAPNetLite c1=32/c2=64, 111,716 params,
# matching the manuscript's Table I op-count. Only training hyperparameters
# below are configurable, either via CLI flags (single config) or via
# --config-json (a list of configs, all sharing this process's loaded data).
# =========================
C1, C2 = 32, 64

DEFAULT_CFG = dict(
    tag=args.tag, lr=args.lr, wd=args.wd, batch_size=args.batch_size, epochs=args.epochs,
    patience=args.patience, plateau_patience=args.plateau_patience, plateau_factor=args.plateau_factor,
    label_smoothing=args.label_smoothing, optimizer=args.optimizer,
    seeds_2018=SEEDS_BY_SPLIT["2018train20"], seeds_2022=SEEDS_BY_SPLIT["2022train20"],
)

if args.config_json:
    import json
    with open(args.config_json) as fh:
        CONFIGS = json.load(fh)
    for cfg in CONFIGS:
        for key, default in DEFAULT_CFG.items():
            if key not in cfg:
                cfg[key] = default
else:
    CONFIGS = [DEFAULT_CFG]

N_BATCHES = 28
DROP_2018_NEGATIVE_SNR = True
CLASS_NAMES = ["QPSK", "16QAM", "64QAM", "256QAM"]
LABEL_MAP = {"qpsk": 0, "16qam": 1, "64qam": 2, "256qam": 3}
N_CLASSES = len(CLASS_NAMES)
LR_LABEL5_ORDER = ["BPSK", "MSK", "DQPSK", "8PSK", "QPSK-like"]
FULL8_LABEL_ORDER = ["BPSK", "MSK", "DQPSK", "8PSK", "QPSK", "16QAM", "64QAM", "256QAM"]
QPSKLIKE_SIGNAL_TYPES = {"qpsk", "16qam", "64qam", "256qam"}
LR_VECTOR_KEY = "lr55"

PT_ROOT = PROJECT_DIR / "qpsklike_cc_matrix_11x24_blind_rs_cfo_pt_dataset"
PREFIX = {
    "2018": "2018_qpsklike_cc_11x24_blindrs_cfo_batch_",
    "2022": "2022_qpsklike_cc_11x24_blindrs_cfo_batch_",
}


def label5_from_signal_type(signal_type):
    sig = str(signal_type).strip().lower()
    if sig == "bpsk":
        return "BPSK"
    if sig == "msk":
        return "MSK"
    if sig == "dqpsk":
        return "DQPSK"
    if sig == "8psk":
        return "8PSK"
    if sig in QPSKLIKE_SIGNAL_TYPES:
        return "QPSK-like"
    raise ValueError(f"Unsupported signal_type={signal_type!r}")


def label8_from_signal_type(signal_type):
    sig = str(signal_type).strip().lower()
    mapping = {
        "bpsk": "BPSK", "msk": "MSK", "dqpsk": "DQPSK", "8psk": "8PSK",
        "qpsk": "QPSK", "16qam": "16QAM", "64qam": "64QAM", "256qam": "256QAM",
    }
    if sig not in mapping:
        raise ValueError(f"Unsupported signal_type={signal_type!r}")
    return mapping[sig]


def list_pt_files(year):
    return [PT_ROOT / f"{PREFIX[year]}{i:02d}.pt" for i in range(1, N_BATCHES + 1)]


def list_compact_vector_files(year):
    files = sorted((PROJECT_DIR / str(year)).glob(f"{year}_batch_*_boi_shift_fft3_compact_vectors.pt"))
    if len(files) != N_BATCHES:
        print(f"[gpu{args.gpu}] WARNING: expected {N_BATCHES} compact vector files for {year}, found {len(files)}")
    return files


# =========================
# One-time load: LR55 vectors (both years)
# =========================
def load_lr55_vectors(years=("2018", "2022")):
    meta_parts, x_parts = [], []
    feature_names = None
    t0 = time.time()
    for year in years:
        files = list_compact_vector_files(str(year))
        for file_idx, path in enumerate(files, start=1):
            pack = torch.load(path, map_location="cpu", weights_only=False)
            samples = pack["samples"]
            names = list(pack["vector_feature_names"][LR_VECTOR_KEY])
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError(f"{LR_VECTOR_KEY} feature-name mismatch in {path}")
            rows, vectors = [], []
            for sample in samples:
                meta = sample["meta"]
                truth = sample["truth"]
                signal_type = str(meta["signal_type"]).strip().lower()
                batch_id = int(meta.get("batch_id", pack.get("batch_id", -1)))
                signal_index = int(meta.get("signal_index", meta.get("global_index", -1)))
                rows.append({
                    "year": str(year), "pt_name": path.name, "batch_id": batch_id,
                    "signal_index": signal_index, "signal_type": signal_type,
                    "label5": label5_from_signal_type(signal_type),
                    "label8": label8_from_signal_type(signal_type),
                    "snr": float(truth.get("inband_snr_db_label", truth.get("snr_truth_db", np.nan))),
                    "cc_key": (str(year), batch_id, signal_index),
                })
                vectors.append(np.asarray(sample["vectors"][LR_VECTOR_KEY], dtype=np.float32))
            meta_parts.append(pd.DataFrame(rows))
            x_parts.append(np.vstack(vectors).astype(np.float32, copy=False))
            print(f"[gpu{args.gpu}] LR55 load {year} {file_idx:02d}/{len(files):02d} elapsed={time.time()-t0:.1f}s", flush=True)
            del pack, samples
    meta_df = pd.concat(meta_parts, ignore_index=True)
    X = np.vstack(x_parts).astype(np.float32, copy=False)
    if DROP_2018_NEGATIVE_SNR:
        keep = ~((meta_df["year"] == "2018") & (meta_df["snr"] < 0))
        meta_df = meta_df.loc[keep].reset_index(drop=True)
        X = X[keep.to_numpy()]
    X = np.where(np.isfinite(X), X, np.nan)
    print(f"[gpu{args.gpu}] LR55 rows: {len(meta_df)} X: {X.shape}", flush=True)
    return meta_df, X, feature_names


# =========================
# One-time load: raw CC-matrix cache (both years, all 56 files)
# =========================
def load_raw_cc_file(path, year):
    pack = torch.load(path, map_location="cpu", weights_only=False)
    X_all = pack["estimate_abs"].float()
    if X_all.ndim != 3 or tuple(X_all.shape[1:]) != (11, 24):
        raise ValueError(f"{path.name}: expected estimate_abs [N,11,24], got {tuple(X_all.shape)}")
    status = pack.get("status", torch.ones(X_all.shape[0], dtype=torch.bool)).bool()
    signal_type = [str(s).strip().lower() for s in list(pack["signal_type"])]
    label_ok = torch.tensor([s in LABEL_MAP for s in signal_type], dtype=torch.bool)
    power_all = pack["boi_inband_signal"].float()
    power_ok = torch.isfinite(power_all) & (power_all > 0)
    snr_ok = torch.ones(X_all.shape[0], dtype=torch.bool)
    if DROP_2018_NEGATIVE_SNR and path.name.startswith("2018_") and "snr" in pack:
        snr_all = pack["snr"].float()
        snr_ok = torch.isfinite(snr_all) & (snr_all >= 0)
    X_11x15_all = X_all[:, :, :15]
    X_ok = torch.isfinite(X_11x15_all.reshape(X_11x15_all.shape[0], -1)).all(dim=1)
    valid = status & label_ok & power_ok & snr_ok & X_ok
    valid_idx = torch.where(valid)[0]
    X = X_11x15_all[valid] / power_all[valid].view(-1, 1, 1)
    scaled_ok = torch.isfinite(X.reshape(X.shape[0], -1)).all(dim=1)
    X = X[scaled_ok]
    valid_idx = valid_idx[scaled_ok]
    signal_indices = pack["signal_index"].long()
    batch_id = int(pack.get("meta", {}).get("batch_id", -1))
    if batch_id < 0:
        batch_id = int(path.stem.split("batch_")[-1].split(".")[0].split("_")[0])
    signal_type_valid = [signal_type[i] for i in valid_idx.tolist()]
    signal_index_valid = signal_indices[valid_idx].tolist()
    y = torch.tensor([LABEL_MAP[s] for s in signal_type_valid], dtype=torch.long)
    return {
        "year": str(year), "batch_id": batch_id, "X": X, "y": y,
        "signal_type": signal_type_valid, "signal_index": signal_index_valid,
    }


def load_raw_cc_cache():
    cache = {"2018": [], "2022": []}
    t0 = time.time()
    for year in ("2018", "2022"):
        for i, path in enumerate(list_pt_files(year), start=1):
            cache[year].append(load_raw_cc_file(path, year))
            print(f"[gpu{args.gpu}] CC load {year} {i:02d}/{N_BATCHES} elapsed={time.time()-t0:.1f}s", flush=True)
    return cache


def build_cc_lookup(raw_cache):
    lookup = {}
    for year, entries in raw_cache.items():
        for entry in entries:
            for i, sidx in enumerate(entry["signal_index"]):
                key = (entry["year"], entry["batch_id"], int(sidx))
                lookup[key] = {
                    "X": entry["X"][i].unsqueeze(0),
                    "signal_type": entry["signal_type"][i],
                    "label8": label8_from_signal_type(entry["signal_type"][i]),
                }
    return lookup


def build_split_dataset(raw_cache, year, allowed_indices):
    allowed = set(int(v) for v in allowed_indices)
    Xs, ys = [], []
    for entry in raw_cache[year]:
        keep = [i for i, sidx in enumerate(entry["signal_index"]) if int(sidx) in allowed]
        if not keep:
            continue
        keep_t = torch.tensor(keep, dtype=torch.long)
        Xs.append(entry["X"][keep_t])
        ys.append(entry["y"][keep_t])
    X = torch.cat(Xs, dim=0).unsqueeze(1)
    y = torch.cat(ys, dim=0)
    return TensorDataset(X, y)


# =========================
# CAPNetLite
# =========================
class ResBlock(nn.Module):
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
        out = out + identity
        return F.relu(out, inplace=True)


class CAPNetLite(nn.Module):
    def __init__(self, n_classes=4, c1=32, c2=64):
        super().__init__()
        self.stem = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(c1)
        self.block1 = ResBlock(c1)
        self.down = nn.Conv2d(c1, c2, kernel_size=3, stride=(1, 2), padding=1, bias=False)
        self.down_bn = nn.BatchNorm2d(c2)
        self.block2 = ResBlock(c2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c2, n_classes)

    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem(x)), inplace=True)
        x = self.block1(x)
        x = F.relu(self.down_bn(self.down(x)), inplace=True)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@torch.no_grad()
def eval_one(model, loader, criterion):
    model.eval()
    ys, ps, losses = [], [], []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        logits = model(X)
        losses.append(criterion(logits, y).item())
        ys.append(y.cpu())
        ps.append(logits.argmax(dim=1).cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return float(np.mean(losses)), float(accuracy_score(y_true, y_pred))


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train_capnetlite(seed, ds_train, ds_val, cfg):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch_size = cfg["batch_size"]
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = CAPNetLite(n_classes=N_CLASSES, c1=C1, c2=C2).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=cfg["plateau_factor"], patience=cfg["plateau_patience"], min_lr=1e-5)

    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    best_state = None
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        train_one_epoch(model, train_loader, optimizer, criterion)
        _, va_acc = eval_one(model, val_loader, criterion)
        if va_acc > best_val_acc + 1e-4:
            best_val_acc = va_acc
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        scheduler.step(va_acc)
        if no_improve >= cfg["patience"]:
            break
    elapsed_min = (time.time() - t0) / 60.0
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val_acc, best_epoch, epoch, elapsed_min


def evaluate_full_pipeline(model, lr55_model, mask, lr_meta_df, X_lr55, cc_lookup, batch_size):
    rows = lr_meta_df.loc[mask].copy().reset_index(drop=True)
    X_lr = X_lr55[mask]
    true8 = rows["label8"].astype(str).to_numpy()
    pred5 = lr55_model.predict(X_lr)

    final_pred = np.empty(len(rows), dtype=object)
    nn_positions, nn_tensors = [], []
    for i, (row, coarse_pred) in enumerate(zip(rows.itertuples(index=False), pred5)):
        if coarse_pred != "QPSK-like":
            final_pred[i] = coarse_pred
            continue
        if row.label5 != "QPSK-like":
            final_pred[i] = "QPSK"
            continue
        item = cc_lookup.get(row.cc_key)
        if item is None:
            final_pred[i] = "QPSK"
            continue
        nn_positions.append(i)
        nn_tensors.append(item["X"])

    if nn_tensors:
        X_nn = torch.cat(nn_tensors, dim=0).unsqueeze(1)
        loader = DataLoader(TensorDataset(X_nn), batch_size=batch_size, shuffle=False)
        preds = []
        with torch.no_grad():
            model.eval()
            for (Xb,) in loader:
                Xb = Xb.to(DEVICE, non_blocking=True)
                preds.append(model(Xb).argmax(dim=1).cpu())
        pred4 = torch.cat(preds).numpy()
        for pos, pred_idx in zip(nn_positions, pred4):
            final_pred[pos] = CLASS_NAMES[int(pred_idx)]

    full8_acc = accuracy_score(true8, final_pred)
    return float(full8_acc * 100)


def main():
    manifests = ensure_manifests((PROJECT_DIR / "../../CSPB.ML_dataset").resolve(), EXPERIMENT_DIR)
    lr_meta_df, X_lr55, _ = load_lr55_vectors()
    for year in (2018, 2022):
        expected = set(manifests[year]["SignalIndex"].astype(int))
        actual = set(lr_meta_df.loc[lr_meta_df["year"] == str(year), "signal_index"].astype(int))
        if actual != expected:
            raise ValueError(f"Incomplete {year} LR55 cache: missing={len(expected - actual)}, extra={len(actual - expected)}")
    y_lr5 = lr_meta_df["label5"].astype(str).to_numpy()

    raw_cc_cache = load_raw_cc_cache()
    cc_lookup = build_cc_lookup(raw_cc_cache)

    per_year = {}
    for split_mode, train_year in (("2018train20", "2018"), ("2022train20", "2022")):
        ext_year = "2022" if train_year == "2018" else "2018"
        split_indices = split_index_sets(manifests, int(train_year))
        own = lr_meta_df["year"].eq(train_year)
        train_mask, val_mask, test_mask = (
            (own & lr_meta_df["signal_index"].isin(split_indices[name])).to_numpy()
            for name in ("train", "val", "test")
        )
        ext_mask = (lr_meta_df["year"].eq(ext_year) & lr_meta_df["signal_index"].isin(split_indices["ext"])).to_numpy()

        lr55_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced", C=3, random_state=None),
        )
        lr55_model.fit(X_lr55[train_mask], y_lr5[train_mask])

        ds_train = build_split_dataset(raw_cc_cache, train_year, split_indices["train"])
        ds_val = build_split_dataset(raw_cc_cache, train_year, split_indices["val"])

        per_year[split_mode] = dict(
            train_year=train_year, ext_year=ext_year,
            test_mask=test_mask, ext_mask=ext_mask,
            lr55_model=lr55_model, ds_train=ds_train, ds_val=ds_val,
        )
        print(f"[gpu{args.gpu}] {split_mode}: train={int(train_mask.sum())} val={int(val_mask.sum())} "
              f"test={int(test_mask.sum())} ext={int(ext_mask.sum())}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_path.exists()
    fieldnames = [
        "gpu", "tag", "seed", "split_mode", "same_domain_acc", "external_acc",
        "best_epoch", "total_epochs", "train_min",
        "lr", "wd", "batch_size", "epochs", "patience", "plateau_patience",
        "plateau_factor", "label_smoothing", "optimizer",
    ]
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new:
            writer.writeheader()
        for cfg in CONFIGS:
            seeds_by_split = {"2018train20": cfg["seeds_2018"], "2022train20": cfg["seeds_2022"]}
            for split_mode, ctx in per_year.items():
                for seed in seeds_by_split[split_mode]:
                    model, best_val_acc, best_epoch, total_epochs, train_min = train_capnetlite(
                        seed, ctx["ds_train"], ctx["ds_val"], cfg
                    )
                    same_acc = evaluate_full_pipeline(model, ctx["lr55_model"], ctx["test_mask"], lr_meta_df, X_lr55, cc_lookup, cfg["batch_size"])
                    ext_acc = evaluate_full_pipeline(model, ctx["lr55_model"], ctx["ext_mask"], lr_meta_df, X_lr55, cc_lookup, cfg["batch_size"])
                    row = {
                        "gpu": args.gpu, "tag": cfg["tag"], "seed": seed, "split_mode": split_mode,
                        "same_domain_acc": f"{same_acc:.4f}", "external_acc": f"{ext_acc:.4f}",
                        "best_epoch": best_epoch, "total_epochs": total_epochs, "train_min": f"{train_min:.2f}",
                        "lr": cfg["lr"], "wd": cfg["wd"], "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
                        "patience": cfg["patience"], "plateau_patience": cfg["plateau_patience"],
                        "plateau_factor": cfg["plateau_factor"], "label_smoothing": cfg["label_smoothing"],
                        "optimizer": cfg["optimizer"],
                    }
                    writer.writerow(row)
                    f.flush()
                    print(f"[gpu{args.gpu}] tag={cfg['tag']} seed={seed} {split_mode}: same={same_acc:.4f}% ext={ext_acc:.4f}% "
                          f"(best_val={best_val_acc*100:.2f}%@{best_epoch}, {total_epochs} epochs, {train_min:.1f} min)", flush=True)
                    del model


if __name__ == "__main__":
    main()
