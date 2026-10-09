"""Create isolated R1-C1 notebook copies with parameter-group data splits."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f"Expected exactly one occurrence of {old[:90]!r}; found {text.count(old)}")
    return text.replace(old, new)


def source(cell):
    return "".join(cell["source"])


def set_source(cell, value):
    cell["source"] = value.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def train_notebook(nb):
    cells = nb["cells"]
    year = 2018 if "2018train20" in source(cells[1]) else 2022
    c = source(cells[1])
    c = replace_once(c, 'MODEL_DIR = BASE_DIR / "fully_online_pipeline_models"',
                     'EXPERIMENT_DIR = PROJECT_DIR / "response_to_spl" / "r1c1" / "group_split_70_5_25alter"\n'
                     'MODEL_DIR = EXPERIMENT_DIR / "models"\n'
                     'sys.path.insert(0, str(EXPERIMENT_DIR))\n'
                     'from split_utils import ensure_manifests, split_index_sets')
    c = replace_once(c, 'PT_ROOT = Path("qpsklike_cc_matrix_11x24_blind_rs_cfo_pt_dataset")',
                     'PT_ROOT = PROJECT_DIR / "qpsklike_cc_matrix_11x24_blind_rs_cfo_pt_dataset"')
    start = c.index('# Choose one:')
    end = len(c)
    c = c[:start] + '''# Preserve the original model names for the matching inference notebook.
# The experiment directory identifies this group-based split.
DROP_2018_NEGATIVE_SNR = True

CLASS_NAMES = ["QPSK", "16QAM", "64QAM", "256QAM"]
LABEL_MAP = {"qpsk": 0, "16qam": 1, "64qam": 2, "256qam": 3}
N_CLASSES = len(CLASS_NAMES)

def list_pt_files(year: str):
    return [PT_ROOT / f"{PREFIX[year]}{i:02d}.pt" for i in range(1, N_BATCHES + 1)]

pt_files = {"2018": list_pt_files("2018"), "2022": list_pt_files("2022")}
for year, files in pt_files.items():
    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} {year} CC files; first: {missing[0]}")

DATASET_ROOT = (PROJECT_DIR / "../../CSPB.ML_dataset").resolve()
manifests = ensure_manifests(DATASET_ROOT, EXPERIMENT_DIR)
TRAIN_SET_NAME = "2018" if SPLIT_MODE == "2018train20" else "2022"
EXT_SET_NAME = "2022" if TRAIN_SET_NAME == "2018" else "2018"
SPLIT_INDICES = split_index_sets(manifests, int(TRAIN_SET_NAME))
train_files = val_files = test_files = pt_files[TRAIN_SET_NAME]
ext_files = pt_files[EXT_SET_NAME]
print("Group split seed: 20260926")
print("Train/val/test year:", TRAIN_SET_NAME, "external year:", EXT_SET_NAME)
for name, indices in SPLIT_INDICES.items():
    print(name, "samples:", len(indices))''' + c[end:]
    set_source(cells[1], c)

    c = source(cells[2])
    c = replace_once(c, 'y_lr5 = lr_meta_df["label5"].astype(str).to_numpy()',
                     '''for year in (2018, 2022):
    expected = set(manifests[year]["SignalIndex"].astype(int))
    actual = set(lr_meta_df.loc[lr_meta_df["year"] == str(year), "signal_index"].astype(int))
    if actual != expected:
        raise ValueError(f"Incomplete {year} LR55 cache: missing={len(expected - actual)}, extra={len(actual - expected)}")
y_lr5 = lr_meta_df["label5"].astype(str).to_numpy()''')
    start = c.index('def make_lr_masks(split_mode):')
    end = c.index('\n\nlr_train_mask,', start)
    c = c[:start] + '''def make_lr_masks(split_mode):
    train_year = "2018" if split_mode == "2018train20" else "2022"
    ext_year = "2022" if train_year == "2018" else "2018"
    own = lr_meta_df["year"].eq(train_year)
    masks = tuple((own & lr_meta_df["signal_index"].isin(SPLIT_INDICES[name])).to_numpy()
                  for name in ("train", "val", "test"))
    ext_mask = (lr_meta_df["year"].eq(ext_year) &
                lr_meta_df["signal_index"].isin(SPLIT_INDICES["ext"])).to_numpy()
    if np.any(sum(mask.astype(int) for mask in (*masks, ext_mask)) != 1):
        raise AssertionError("LR55 data split overlap or uncovered rows")
    return *masks, ext_mask''' + c[end:]
    c = replace_once(c, 'MODEL_DIR = PROJECT_DIR / "fully_online_pipeline_models"',
                     'MODEL_DIR = EXPERIMENT_DIR / "models"')
    set_source(cells[2], c)

    c = source(cells[3])
    c = replace_once(c, 'def __init__(self, pt_files, label_map=LABEL_MAP):',
                     'def __init__(self, pt_files, allowed_indices, label_map=LABEL_MAP):')
    c = replace_once(c, 'self.pt_files = list(pt_files)',
                     'self.pt_files = list(pt_files)\n        self.allowed_indices = set(allowed_indices)')
    c = replace_once(c, 'valid = status & label_ok & power_ok & snr_ok & X_ok',
                     'index_ok = torch.tensor([int(i) in self.allowed_indices for i in pack["signal_index"]], dtype=torch.bool)\n'
                     '            valid = status & label_ok & power_ok & snr_ok & X_ok & index_ok')
    c = replace_once(c, 'f"X min/max=({float(X.min()):.4e}, {float(X.max()):.4e})"',
                     'f"X min/max=({float(X.min()) if X.numel() else float(\'nan\'):.4e}, "\n'
                     '                f"{float(X.max()) if X.numel() else float(\'nan\'):.4e})"')
    for split in ('train', 'val', 'test', 'ext'):
        c = replace_once(c, f'QPSKLikeCCDataset({split}_files)',
                         f'QPSKLikeCCDataset({split}_files, SPLIT_INDICES["{split}"])')
    set_source(cells[3], c)

    c = source(cells[7])
    c = replace_once(c, 'BEST_MODEL_PATH = BASE_DIR / "fully_online_pipeline_models" /',
                     'BEST_MODEL_PATH = MODEL_DIR /')
    set_source(cells[7], c)
    c = source(cells[11])
    c = replace_once(c, 'save_path = f"snr_acc_samples_{SPLIT_MODE}.pt"',
                     'save_path = EXPERIMENT_DIR / "results" / f"snr_acc_samples_{SPLIT_MODE}.pt"\n'
                     'save_path.parent.mkdir(parents=True, exist_ok=True)')
    set_source(cells[11], c)
    intro = source(cells[0]).replace('saved under `fully_online_pipeline_models`',
                                   'saved under `group_split_70_5_25alter/models`')
    set_source(cells[0], intro +
               '\n\nThis R1-C1 copy assigns **signal samples** at 70/5/25 while keeping each parameter group intact; 2018 negative-SNR samples are excluded.\n')


def inference_notebook(nb):
    cells = nb["cells"]
    year = 2018 if "2018train20" in source(cells[1]) else 2022
    c = source(cells[1])
    c = replace_once(c, 'MODEL_DIR = PROJECT_DIR / "fully_online_pipeline_models"\n'
                     'RESULT_DIR = PROJECT_DIR / "fully_online_inference_results"',
                     'EXPERIMENT_DIR = PROJECT_DIR / "response_to_spl" / "r1c1" / "group_split_70_5_25alter"\n'
                     'sys.path.insert(0, str(EXPERIMENT_DIR))\n'
                     'from split_utils import ensure_manifests, run_manifest_inference\n'
                     'manifests = ensure_manifests(DATASET_ROOT, EXPERIMENT_DIR)\n'
                     'MODEL_DIR = EXPERIMENT_DIR / "models"\n'
                     'RESULT_DIR = EXPERIMENT_DIR / "results"')
    set_source(cells[1], c)
    c = source(cells[3])
    start = c.index('compact_results = pipeline.run_compact_csv(')
    end = c.index('\n\nprint("Saved:', start)
    c = c[:start] + f'''compact_results = run_manifest_inference(
    pipeline, manifests, train_year={year}, output_csv=RESULT_CSV,
    resume=RESUME_FROM_EXISTING_CSV, checkpoint_every=100,
)''' + c[end:]
    set_source(cells[3], c)
    set_source(cells[2], '## Run inference and save the CSV\n\n'
               'Evaluate the same-year held-out parameter groups and every eligible other-year sample. '
               'All 2018 negative-SNR rows are excluded.\n')
    intro = source(cells[0])
    intro = intro.replace('It evaluates 2018 held-out batches 25-28 and the complete 2022 cross-domain set.',
                          'It evaluates the 2018 held-out parameter groups and the complete 2022 cross-domain set.')
    intro = intro.replace('It evaluates 2022 held-out batches 25-28 and the complete 2018 cross-domain set.',
                          'It evaluates the 2022 held-out parameter groups and every nonnegative-SNR 2018 cross-domain sample.')
    intro = intro.replace('fully_online_inference_results/LR55-CAPNetLite-',
                          'group_split_70_5_25alter/results/LR55-CAPNetLite-')
    set_source(cells[0], intro +
               '\n\nThis R1-C1 copy loads models trained using a 70/5/25 **sample-count** split with whole parameter groups assigned to a single subset.\n')


def main():
    for path in sorted(SOURCE.glob('*.ipynb')):
        if path.name.startswith('Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-'):
            transform = train_notebook
        elif path.name.startswith('FullyOnline-Inference-LR55-CAPNetLite-'):
            transform = inference_notebook
        else:
            continue
        nb = json.loads(path.read_text())
        transform(nb)
        # Stored outputs from the old batch split must not appear in this experiment.
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        target = HERE / path.name
        target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
        print(target)


if __name__ == '__main__':
    main()
