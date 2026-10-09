# LR8_32 + CNN

Eight-class modulation classification using a 32-feature logistic regression router and a four-class CNN.

```text
Raw IQ → BOI filtering/shift → P96-clipped z → 3 FFTs → LR
                                                     ├─ BPSK / MSK / DQPSK / 8PSK
                                                     └─ QPSK-like → blind Rs/CFO → CC → CNN
```


## Environment

Reference environment: **Python 3.10.14, Linux, PyTorch 2.5.1+cu121**. Install the recorded package versions:

```bash
conda create -n lr8_32_capnet python=3.10.14 pip -y
conda activate lr8_32_capnet
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install numpy==2.1.2 scipy==1.15.3 pandas==2.3.3 scikit-learn==1.7.2 joblib==1.5.3 matplotlib==3.10.7 tqdm==4.67.1 ipython==8.37.0 ipykernel==6.29.5
python -m ipykernel install --user --name lr8_32_capnet --display-name "Python (LR8_32 CNN)"
```

Training requires CUDA. The 2018/2022 training and inference notebooks default to physical GPUs **2/3**; change `CUDA_VISIBLE_DEVICES` to your GPU number before starting a fresh kernel.

## Dataset and layout

Download **`CSPB.ML.2018 and CSPB.ML.2022.zip`** from [IEEE DataPort](https://ieee-dataport.org/documents/synthetic-digitally-modulated-signal-datasets-automatic-modulation-classification), following its login/access requirements. DOI: [10.21227/pcj5-zr38](https://doi.org/10.21227/pcj5-zr38).

Each year contains 112,000 signals: 28 batches × 4,000 signals, covering eight modulation classes. Each `.tim` contains 32,768 complex IQ samples; the reader skips two float32 header values, then reads interleaved I/Q.

```text
Project/
├── CSPB.ML_dataset/
│   ├── CSPB_ML_2018_Data/
│   │   ├── CSPB_ML_2018_Signal_Truth_Labels.txt
│   │   └── Batch_Dir_1/ ... Batch_Dir_28/
│   │       └── signal_<SignalIndex>.tim
│   └── CSPB_ML_2022_Data/
│       ├── CSPB_ML_2022_Signal_Truth_Labels.txt
│       └── Batch_Dir_1/ ... Batch_Dir_28/
│           └── signal_<SignalIndex>.tim
└── Code/
    └── boi_p96_fft3_github/
        ├── README.md
        ├── *.ipynb                         # Eight notebooks listed below
        ├── helper_function/                # BOI, FFT, blind estimation and CC
        ├── group_split_70_5_25alter/
        │   ├── split_utils.py
        │   └── splits/
        │       ├── 2018_group_split.csv
        │       ├── 2022_group_split.csv
        │       └── metadata.json
        ├── fully_online_pipeline_models/   # LR .joblib and CNN .pt weights
        ├── generated_boi_p96_fft2_pt/       # Generated feature caches
        ├── generated_qpsklike_cc_11x15_boi_p96_fft2_pt/  # Generated CC caches
        ├── snr_acc_samples_lr8_32_xproccc_<year>train20.pt # Simulated results
        └── fully_online_inference_results/ # End-to-end CSV results
```


**Run with the kernel working directory set to `boi_p96_fft3_github/`.** Training/inference use `../../CSPB.ML_dataset`. 

## Dataset split

[split_utils.py](group_split_70_5_25alter/split_utils.py) assigns complete parameter groups to approximately **70% train / 5% validation / 25% test**, with base seed `20260926`. Grouping uses:

```text
BaseSymbolPeriod, CarrierOffset, ExcessBandwidth, UpsampleFactor,
DownsampleFactor, InbandSNRdB, NoiseSpectralDensitydB
```

`SignalIndex` and `SignalType` are excluded from the group key, keeping all modulations with identical generation parameters in one split. Negative-SNR 2018 samples are excluded.


## Run order

Execute each notebook from top to bottom.

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | [2018 BOI/FFT](pre_BOI%2BFFT-Generate-2018-compact-vectors-final.ipynb) | Raw IQ → waveforms, LR features, blind Rs/CFO PT files |
| 2 | [2022 BOI/FFT](pre_BOI%2BFFT-Generate-2022-compact-vectors-final.ipynb) | Same preprocessing for 2022 |
| 3 | [2018 CC](pre_QPSK_like_CC_Matrix_generate_blind_Rs_CFO_ptvectors-2018.ipynb) | Step 1 outputs → raw complex 11×15 CC for QPSK-like samples |
| 4 | [2022 CC](pre_QPSK_like_CC_Matrix_generate_blind_Rs_CFO_ptvectors-2022.ipynb) | Step 2 outputs → 2022 CC files |
| 5 | [2018 training](Classfier-LR8_32-CNN-QPSKlike-blind-RsCFO-11x15-2018train20.ipynb) | Train LR/CNN; save weights and simulated evaluation results |
| 6 | [2022 training](Classfier-LR8_32-CNN-QPSKlike-blind-RsCFO-11x15-2022train20.ipynb) | Train the corresponding 2022 models |
| 7 | [2018-model inference](FullyOnline-Inference-LR8_32-CNN-2018train20.ipynb) | Raw IQ → predictions on 2018 test and all eligible 2022 samples |
| 8 | [2022-model inference](FullyOnline-Inference-LR8_32-CNN-2022train20.ipynb) | Raw IQ → predictions on 2022 test and all eligible 2018 samples |

**Both training notebooks require both years' feature and CC caches.** With existing weights, raw IQ and split manifests, skip directly to steps **7–8**; online inference does not need cached features or CC matrices.

## Models and outputs

Each training notebook runs one CNN seed for 200 epochs and saves its best-validation checkpoint. Default model names are:

```text
lr8_32_<year>train20.joblib
capnet_lite_lr8_32_xprocrawcc_warpscale_k5_c20_20_nb3_2_<year>train20.pt
```

Additional `_seed<number>.pt` files are not loaded automatically; select one through `CAP_MODEL_PATH` if needed.

Inference saves `fully_online_inference_results/LR8_32-CAPNetLite-<year>train20.csv`, reports accuracy and displays one combined two-year confusion matrix.

## Optional: CPU latency evaluation

CPU latency tests are available in [`cpu_latency_comparison/`](cpu_latency_comparison/). Only one notebook needs to be run: [`TableI_current_Conditional_analysis_and_CPU_tests_recomputed.ipynb`](cpu_latency_comparison/TableI_current_Conditional_analysis_and_CPU_tests_recomputed.ipynb).

The CPU package includes **16 PT checkpoints (8 configurations × 2 training years, one seed each)**, four LR models, and the required helpers. Inputs are generated in memory from the raw dataset at `../../CSPB.ML_dataset` and the existing split manifests; no input caches or historical result files are required. The default accuracy check uses 32 samples per class, configurable via `SAMPLES_PER_CLASS`, and reports single-checkpoint sample accuracy rather than five-seed statistics. CPU timings are measured during execution, and results remain embedded in the notebook. The complete distributed project is under **100 MB**, excluding the separately downloaded dataset and files generated by the main training workflow.

