# REST — Rodent EEG Sleep-stage Transformer

Automated sleep-stage scoring for rodent EEG/EMG recordings. Classifies every 4-second epoch into **Wake**, **NREM**, **REM**, or **Artifact** and saves results as a MATLAB-compatible `.mat` file.

Artifact detection uses a **two-layer approach**: the REST transformer flags artifacts as part of normal classification, and a second rule-based EEG signal-quality filter runs as a final override pass. See [Artifact Detection](#artifact-detection) for details.

---

## Option A — Standalone Executable (No Python Required)

A pre-built Windows executable is available on the [GitHub Releases](../../releases/latest) page.

1. Go to **Releases** and download **REST_Inference_GUI.exe** under **V1.5**.
2. Double-click the downloaded file to launch the GUI.

No installation needed. Skip to [GUI — Quick Start](#gui--quick-start) below.

---

## Option B — Python Installation

Required if you want to retrain the model or run the inference scripts.

1. Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create the environment:
   ```bash
   conda env create -f environment.yml
   conda activate REST
   ```

---

## GUI — Quick Start

The GUI is the easiest way to score recordings. No coding required.

**Launch (EXE):** Double-click `dist/SeizureDetector/SeizureDetector.exe`.

**Launch (Python):**
```bash
conda activate REST
python Inference_GUI.py
```

Wait for **Model Status: Loaded ✅** before doing anything else.

---

### Single File Mode

1. Click **Browse** and select your `.edf` file.
2. Choose your **EEG** and **EMG** channels from the dropdowns.
3. Leave **HMM Smoothing** and **Artifact Filter** enabled (both recommended).
4. Click **Score** and wait for the progress bar to finish.
5. A hypnogram will appear. Click **Save Results** to export the `.mat` file.

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
| `score` | Sleep stage per epoch — **1 = Wake, 2 = NREM, 3 = REM, 4 = Artifact** |
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

## Artifact Detection

REST uses two independent layers for artifact detection. Both must agree an epoch is clean for it to retain its sleep-stage label.

### Layer 1 — Transformer model

The REST model is trained on 4-class output (Wake / NREM / REM / Artifact). It learns temporal patterns across a 120-epoch (8-minute) context window and can identify artifacts based on their relationship to surrounding epochs. Because real artifact labels are rare in training data, the model is conservative — it captures artifacts that have clear spectral signatures within their broader context.

### Layer 2 — Rule-based EEG signal-quality filter

A signal-processing filter (`ArtifactFilter.py`) runs after the model as a final override step. It works on the raw EEG channel only and applies two rules:

**Smooth-PTP rule (primary):** Computes the peak-to-peak amplitude of each epoch and ranks it against all other epochs in the same recording (within-recording percentile). Epochs that are substantially above the recording's own baseline, and whose elevation is sustained over a ~20-second window, are flagged. The within-recording normalization is essential — mouse strains differ by ~2× in baseline EEG amplitude, so a global threshold would not generalise across cohorts. The temporal smoothing distinguishes sustained artifact bursts from isolated high-amplitude arousal peaks during Wake.

**Saturation rule (secondary):** Flags epochs where more than 10% of samples are clipped at the ADC rail. Catches hard-clipping events that the Smooth-PTP rule can miss (the waveform is stuck at a ceiling rather than swinging through a large range).

Any epoch flagged by either rule is overridden to **Artifact (4)** regardless of the model's prediction.

### Performance (rule-based layer, EEG only)

Evaluated on 5 held-out test datasets (parameters tuned on ~460 separate training recordings):

| Dataset | F1 | Notes |
|---|---|---|
| CD1 | 0.88 | Primary use case — artifacts are highly separable by amplitude |
| DBAKA | 0.61 | Good separation |
| DBASA | 0.52 | Moderate separation |
| C57KA | 0.09 | Artifacts close to active-Wake amplitude |
| C57SA | 0.02 | Known limitation — artifacts overlap with Wake in EEG; model layer carries this |

The rule-based layer adds the most value for CD1 recordings, where artifact PTP is typically 10–50× the normal signal. C57 saline recordings are the known limitation and rely primarily on Layer 1.

### Toggling

- **`Inference.py`:** `ARTIFACT_FILTER = True / False`
- **`Inference_GUI.py`:** "Artifact Filter" checkbox in both Single File and Batch mode

Parameters are stored in `artifact_params.json`. See `filter_function_note.md` for a full explanation of the algorithm and tuning procedure.

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
ARTIFACT_FILTER   = True    # rule-based EEG override pass (Layer 2); recommended
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
