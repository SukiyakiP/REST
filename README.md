# REST: Transformer-Based Sleep Scoring for Mice

This repository contains code for training, evaluating, and deploying a deep learning model to automatically classify sleep stages (Wake, NREM, REM) from EEG and EMG signals in mice. The model is based on a Transformer architecture.

---

### 1. **`Compile_Data.py`**
Preprocess raw `.edf` and `.tsv` files into a `.npz` dataset for training/testing.
- **Input folder:** contains matched `subject_x.edf` and `subject_x.tsv` files, if other format for score file is used, section on score loader need to be changed. recommend a simple NP or Mat file containing only the score array.
- **Output:** `.npz` file with EEG, EMG, and score arrays
- **Set paths:**
  - `folder_path` → input EDF & TSV files
  - `save_path` → output `.npz` dataset

### 2. **`Model_Training.py`**
Train the SleepTransformer model on the preprocessed `.npz` data.
- **Set paths:**
  - `ds_path` → input `.npz` dataset
  - `model_path` → output path for saving model weights

### 3. **`Model_Testing.py`**
Evaluate a trained model on test data.
- **Set paths:**
  - `ds_path` → test `.npz` dataset
  - `model_path` → `.pth` file from training

### 4. **`Score_new_EDF.py`**
Use a trained model to predict sleep stages on new `.edf` files.
- **Input:** folder of new EDF files
- **Output:** `.mat` file for each EDF containing predicted scores and power features
- **Set paths:**
  - `model_path` → `.pth` trained model
  - `file_folder` → folder of EDFs to score
  - `save_path` → folder to save `.mat` results

### 5. **`quality_screening.m`**
Matlab script used for manual screening for the sleep score predicted by REST
- **Input:** UI will ask for the folder containing the mat file produced by REST
- **Output:** Figures containing the Sleep hypnogram and EEG/EMG power. for Sleep score, 1=wake, 2=NREM, 3=REM

Workflow 1: if user prefer training their own model using their own recordings. user shoud first compile a training dataset using `Compile_Data.py`, then training using `Model_Training.py`, `Model_Training.py` will produce a pth file containing model structure and weight. Using `Score_new_EDF.py` and the pth file, user can score new EDFs. `Model_Testing.py` is optional if user want to test the model on unseen and prelabeled recordings.

Workflow 2:A pretrained model and weight is provided: 'model_and weight.pth' this model is well generalize and should work with most mice. If user want to use the pretrained model, download the pth file and use it in 'Score_new_EDF.py' to score new EDF, thus skipping the training process.' However, it is recommended that user use some prelabeled recording to test the usability of the model by first compile a training dataset using `Compile_Data.py`. Followed by  verifiying the model performance using `Model_Testing.py`.

Note: `Score_new_EDF.py` and `Compile_Data.py` require user to preset the keyword for the EEG and EMG channel name in the script for the correct channel to be detected and chosen.
