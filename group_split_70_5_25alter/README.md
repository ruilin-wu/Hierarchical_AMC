# R1-C1 parameter-group experiment

The four notebooks here are copies of the two year-specific trainers and two
matching online inference notebooks in the parent `r1c1` directory. The model
architecture, input preprocessing, and training hyperparameters are unchanged.
Model files are written to `models/`; results are written to `results/`.
The `2018train20` and `2022train20` suffixes are retained for the existing
model-loading convention; these copies use sample-based group splits, not the
original 20-batch training split.

`split_utils.py` groups rows by the seven generation-parameter fields, excluding
both `SignalIndex` and `SignalType`. It excludes negative-SNR 2018 rows, then
assigns every modulation type with the same parameters to one split. Whole
groups are assigned to train/validation/test so that **sample counts** are as
close as possible to 70/5/25. The fixed split seed is `20260926`.
The CSV manifests in `splits/` contain every eligible
signal index and are checked against the original truth files when loaded.

| Train year | Train samples | Validation samples | Same-year test samples | Other-year test samples |
| --- | ---: | ---: | ---: | ---: |
| 2018 | 78,144 | 5,584 | 27,904 | 112,000 |
| 2022 | 78,400 | 5,600 | 28,000 | 111,632 |

The 2018 sample counts differ by at most four from the ideal integer quotas
because each parameter group contains all eight modulation types. Existing
models and search results in this copied directory were produced with the old
split and must be retrained before they can represent this new split.

Run each training notebook before its corresponding inference notebook. For a
command-line run with per-cell checkpoints:

```bash
python run_training.py 2018
python run_training.py 2022
```

The inference notebook selects the same-year test indices and all eligible
other-year indices from the manifests. Its CSV can resume an interrupted run.
