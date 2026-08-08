# LR55 + CNN Modulation Classification Pipeline

This directory implements an eight-class modulation-classification pipeline for CSPB.ML.2018 and CSPB.ML.2022. Starting from raw complex-IQ `.tim` files, the pipeline detects and frequency-shifts the band of interest (BOI), extracts FFT features, and applies LR55 for five-class coarse classification. Samples classified as `QPSK-like` are passed through blind symbol-rate/carrier-frequency-offset estimation, a cyclic-cumulant (CC) matrix, and CNN to obtain the final eight-class prediction.

```text
.tim IQ
  -> BOI detection and frequency shift
  -> FFT3 features
  -> LR55: BPSK / MSK / DQPSK / 8PSK / QPSK-like
  -> QPSK-like route
  -> blind Rs/CFO estimation
  -> 11x15 CC matrix
  -> CNN: QPSK / 16QAM / 64QAM / 256QAM
```

## Dataset

The dataset entry page is [Cyclostationary Signal Processing — Datasets](https://cyclostationary.blog/data-sets/). 

### Raw-data location

The main workflow treats this directory as `PROJECT_DIR`. Place the raw dataset at:

```text
PROJECT_DIR/../../CSPB.ML_dataset
```

The intended layout is:

```text
Project/
├── CSPB.ML_dataset/
│   ├── CSPB_ML_2018_Data/
│   │   ├── CSPB_ML_2018_Signal_Truth_Labels.txt
│   │   ├── Batch_Dir_1/
│   │   │   ├── signal_1.tim
│   │   │   └── ...
│   │   ├── Batch_Dir_2/
│   │   └── ... Batch_Dir_28/
│   └── CSPB_ML_2022_Data/
│       ├── CSPB_ML_2022_Signal_Truth_Labels.txt
│       ├── Batch_Dir_1/
│       └── ... Batch_Dir_28/
└── SPL_code/
    └── spl_sourcecode/          # This directory: PROJECT_DIR
```

Regardless of the names used inside the downloaded archives, the final directory and filenames must match this structure. The `signal_N.tim` files use global indices: Batch 1 contains signals 1–4000, Batch 2 contains signals 4001–8000, and so on.

Run these checks from `PROJECT_DIR`:

```bash
test -f ../../CSPB.ML_dataset/CSPB_ML_2018_Data/CSPB_ML_2018_Signal_Truth_Labels.txt
test -f ../../CSPB.ML_dataset/CSPB_ML_2022_Data/CSPB_ML_2022_Signal_Truth_Labels.txt

find ../../CSPB.ML_dataset/CSPB_ML_2018_Data -type f -name '*.tim' | wc -l
find ../../CSPB.ML_dataset/CSPB_ML_2022_Data -type f -name '*.tim' | wc -l
```

Both count commands should print `112000` for a complete dataset.

## Python environment

Linux, Python 3.10, and an NVIDIA CUDA GPU are recommended. The code has been used with these main package versions:

| Software | Tested version |
|---|---:|
| Python | 3.10.14 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| Matplotlib | 3.10.7 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| tqdm | 4.67.1 |
| PyTorch | 2.5.1+cu121 |
| IPython | 8.37.0 |
| ipykernel | 6.29.5 |

Create the Python environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy==1.26.4 \
  pandas==2.3.3 \
  matplotlib==3.10.7 \
  scikit-learn==1.7.2 \
  joblib==1.5.3 \
  tqdm==4.67.1 \
  IPython==8.37.0 \
  ipykernel==6.29.5 \
  jupyterlab
```

## Recommended execution order

Run each notebook from top to bottom in a fresh kernel. If the required intermediate files already exist, start from the corresponding downstream step.

### 1. Generate compact BOI + FFT3 vectors

Run in this order:

1. `pre_BOI+FFT-Generate-2018-compact-vectors-final.ipynb`
2. `pre_BOI+FFT-Generate-2022-compact-vectors-final.ipynb`

These notebooks:

- read the raw `.tim` files and truth-label files;
- perform BOI detection, frequency shifting, and three-branch FFT feature extraction;
- save `x_processed`, LR vectors, Rs/CFO vectors, truth metadata, and power information.

### 2. Generate QPSK-like blind-Rs/CFO CC matrices

Run in this order:

1. `pre_QPSK_like_CC_Matrix_generate_blind_Rs_CFO_ptvectors-2018.ipynb`
2. `pre_QPSK_like_CC_Matrix_generate_blind_Rs_CFO_ptvectors-2022.ipynb`

These notebooks:

- read the compact PT files from Step 1;
- process only `QPSK`, `16QAM`, `64QAM`, and `256QAM` samples;
- estimate Rs/CFO from `rs20` and `cfo1`;
- generate 11x24 CC matrices, of which the first 15 columns are used for training.

Both notebooks use `cuda:0` when CUDA is available. The CC helper can fall back to the CPU, but CPU execution is substantially slower.

### 3. Optional: run the FFT-feature ablation study

Run:

```text
Ablation-6baselines-FFT-ablation-5class-8class-LR-MLP.ipynb
```

This notebook uses the 56 compact PT files from Step 1 to compare six baselines:

- 55D five-class LR;
- five-class LR without `FFT{|x|²}`;
- five-class LR without `FFT{x²}`;
- five-class LR without `FFT{x⁴}`;
- 55D eight-class LR;
- 55D eight-class MLP.

It displays a 6×4 accuracy table and does not save models or result files. It does not require the CC dataset.

### 4. Train LR55 + CNN

Run sequentially, or run each notebook on its assigned GPU:

1. `Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-2018train20.ipynb`
2. `Classfier-LR55-CAPNetLite-QPSKlike-blind-RsCFO-11x15-2022train20.ipynb`

Each notebook performs three tasks:

1. trains a five-class Logistic Regression model from the `lr55` compact vectors;
2. trains a four-class CNN model from the QPSK-like CC PT files;
3. simulates the complete eight-class pipeline using the precomputed intermediate data.

GPU configuration:

- the 2018train20 notebook uses physical GPU 2 through `CUDA_VISIBLE_DEVICES="2"`;
- the 2022train20 notebook uses physical GPU 3 through `CUDA_VISIBLE_DEVICES="3"`;
- each notebook uses only one GPU.


### 5. Run fully-online inference

After the model files have been saved, run:

1. `FullyOnline-Inference-LR55-CAPNetLite-2018train20.ipynb`
2. `FullyOnline-Inference-LR55-CAPNetLite-2022train20.ipynb`

These notebooks execute the complete online pipeline directly from raw `.tim` files. They do not read the precomputed compact PT or CC PT files.

- The 2018train20 model evaluates 2018 batches 25–28 and performs cross-domain testing on all 2022 data.
- The 2022train20 model evaluates 2022 batches 25–28 and performs cross-domain testing on all nonnegative-SNR 2018 data.

