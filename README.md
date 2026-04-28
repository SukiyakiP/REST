# REST — Rodent EEG Sleep-stage Transformer

Automated sleep-stage scoring for rodent EEG/EMG recordings. Classifies every 4-second epoch into **Wake**, **NREM**, or **REM** and saves results as a MATLAB-compatible `.mat` file.

---

## Installation

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate REST
   ```

---

## GUI — Quick Start

The GUI is the easiest way to score recordings. No coding required.

**Launch:**
```bash
conda activate REST
python Inference_GUI.py
```

Wait for **Model Status: Loaded ✅** before doing anything else.

---

### Single File Mode

1. Click **Browse** and select your `.edf` file.
2. Choose your **EEG** and **EMG** channels from the dropdowns.
3. Leave **HMM Smoothing** enabled (recommended).
4. Click **Score** and wait for the progress bar to finish.
5. A hypnogram will appear. Click **Save Results** to export the `.mat` file.

> **Tip:** Press **Ctrl+P** to save a 300 DPI screenshot of the GUI.

---

### Batch Mode

Scores an entire folder of EDF files automatically.

1. Click **Browse** and select your root folder (subfolders are included).
2. Set the **Channel Keywords** to match your recording setup:
   | Field | Default | Change if… |
   |---|---|---|
   | EEG 1 Keyword | `RF` | Your EEG channel isn't named with `RF` |
   | EMG Keyword | `EMG` | Your EMG channel has a different name |
3. Click **Scan & Score**. Progress is shown in the log box.
4. One `_REST_V1.5.mat` file is saved next to each scored EDF.

---

### Output Format

Each `.mat` file contains two variables:

| Variable | Description |
|---|---|
| `score` | Sleep stage per epoch — **1 = Wake, 2 = NREM, 3 = REM** |
| `power` | Spectral band powers per epoch (Delta, Theta, Alpha, Beta, Sigma, Gamma, Full EEG, EMG) |

Load in MATLAB:
```matlab
data  = load('MyRecording_REST_V1.5.mat');
score = data.score;   % [N × 1]
power = data.power;   % [N × 8]
```

Load in Python:
```python
from scipy.io import loadmat
data  = loadmat('MyRecording_REST_V1.5.mat')
score = data['score'].flatten()
power = data['power']
```

---

## Script Pipeline

For command-line / high-throughput use, or for retraining the model on new data.

### Step 1 — Compile Training Data (`Data_compile.py`)

Converts raw EDF + manual score file pairs into the format expected by the trainer.

**Edit these paths at the top of the script:**
```python
OUTPUT_DIR    = r"D:\Training data V1.5_tensor"   # where to save compiled data

FP_EDF_ORIG   = [r"...\Animal1.edf", ...]         # EDF files
FP_SCORE_ORIG = [r"...\Animal1_RM.txt", ...]      # matching score files (same order)

# For Sirenia .tsv scored recordings, point to their parent directories:
ADDITIONAL_DIRS = [r"...\fmr1\5mo", ...]
```

**Run:**
```bash
python Data_compile.py
```

Results are saved to `OUTPUT_DIR`. Already-compiled recordings are skipped automatically.

---

### Step 2 — Train the Model (`Training.py`)

**Edit these paths:**
```python
DATA_DIR   = r"D:\Training data V1.5_tensor"   # output from Step 1
Model_path = r"...\my_new_model.pth"           # where to save trained weights
```

**Run:**
```bash
python Training.py
```

The best model (by validation accuracy) is saved automatically. Training stops early if accuracy stops improving.

> ⚠️ **Important:** The `Use_BatchNorm` flag must be the same in `Training.py` **and** `Inference.py`. Mismatching will produce garbage predictions.

---

### Step 3 — Run Inference (`Inference.py`)

Scores a large set of EDF files from the command line.

**Edit these settings:**
```python
Model_path        = r"...\model_BatchNorm_1.pth"
edf_folder        = [r"M:\EEG files\2026\Cohort15\sham",
                     r"M:\EEG files\2026\Cohort15\TBI"]
Skip_processed    = False   # set True to skip files already scored
HMM_smoothing     = True    # recommended
```

**Run:**
```bash
python Inference.py
```

One `_REST_V1.5.mat` file is saved next to each EDF. Errors (e.g., missing channels, server disconnects) are logged and skipped without stopping the run.

---

### Step 4 — Evaluate Accuracy (`Test_EDF.py`)

Compares model predictions against manual reference score files and generates an Excel report.

**Edit these settings:**
```python
MODEL_PATH = r"...\model_BatchNorm_1.pth"

edf_folder_config = [
    ("GroupA", r"M:\REST-Testing\GroupA"),   # folder must contain paired .edf + .txt files
    ("GroupB", r"M:\REST-Testing\GroupB"),
]
```

**Run:**
```bash
python Test_EDF.py
```

Results (accuracy, Cohen's Kappa, confusion matrix) are printed to the console and saved as an Excel workbook next to the model file.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Model Status: Failed ❌ | Check the model path. Make sure the `.pth` file exists. |
| "Skipping: Missing 'RF' channel" | Change the EEG keyword in Batch Mode, or edit `Inference.py` to match your channel names. |
| GPU out of memory | Reduce `batch_size` in the script (try 64 or 32). |
| Hypnogram looks wrong | Make sure `Use_BatchNorm` matches between training and inference. Also verify correct channel selection. |
| Slow on CPU | A CUDA-capable GPU is strongly recommended. CPU inference works but is slow (~10 min/recording). |
