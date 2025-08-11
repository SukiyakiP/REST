# REST: Transformer-Based Sleep Scoring for Mice

This repository contains code for training, evaluating, and deploying a deep learning model to automatically classify sleep stages (Wake, NREM, REM) from EEG and EMG signals in mice. The model is based on a Transformer architecture.

---

### 1. **`Labeled_data_processing.ipynb`**
Preprocess raw `.edf` and `.txt` files into a `.npz` dataset for training/testing.
- **Input folder:** contains matched `subject_x.edf` and `subject_x.txt` files, if other format for score file is used, section on score loader need to be changed. recommend a simple NP or Mat file containing only the score array.
- **Output:** `.npz` file with STFT of EEG, EMG along with the score arrays
- **Set paths:**
  - `folder_path` → input EDF & txt files
  - `save_path` → output `.npz` dataset

### 2. **`Training_General.ipynb`**
Train the SleepTransformer model on the preprocessed `.npz` data from step1.
- **Set paths (optional):**
  - `ds_path` → input `.npz` dataset
  - `Model_path` → output path for saving model weights
  - by default the script will try to find a npz file in the same folder and save the trained weights in the same folder as well.

### 3. **`Accuracy_Test.ipynb`**
Evaluate a trained model on test data.
- **Set paths:**
  - `ds_path` → test `.npz` dataset
  - `model_path` → `.pth` file from training, by default it will try to find the pth in the same folder, change to manual directory if prefer.

### 4. **`Inference_General_ScanFolder.ipynb`**
Use the trained model to predict sleep stages on new(unseen) `.edf` files.
- **Input:** folder of new EDF files
- **Output:** `.mat` file for each EDF containing predicted scores and power features
- **Set paths:**
  - `Model_path` → `.pth` trained model
  - `file_folder` → folder of EDFs to score
  - by default mat file will be save to the same folder as EDF files with same file name + _REST.mat

### 5. **`quality_screening.m`**
Matlab script used for manual screening for the sleep score predicted by REST
- **Input:** UI will ask for the folder containing the mat file produced by REST
- **Output:** Figures containing the Sleep hypnogram and EEG/EMG power. for Sleep score, 1=wake, 2=NREM, 3=REM

Workflow 1: if user prefer training their own model using their own recordings. user shoud first compile a training dataset using 'Labeled_data_processing.ipynb`, then training using `Training_General.ipynb`, `Training_General.ipynb` will produce a pth file containing model structure and weight. Using `Inference_General_ScanFolder.ipynb` and the pth file, user can score new EDFs. `Accuracy_Test.ipynb` is optional, user can test the model accuracy on unseen and prelabeled recordings.

Workflow 2:A pretrained model and weight is provided: 'Model_general.pth' this model is well generalize and should work with most mice. If user want to use the pretrained model, download the pth file and use it in 'Inference_General_ScanFolder.ipynb' to score new EDF, thus skipping the training process.' However, it is recommended that user use some prelabeled recording to test the usability of the model

Note: `Inference_General_ScanFolder.ipynb` and `CLabeled_data_processing.ipynb` require user to preset the keyword for the EEG and EMG channel name in the script for the correct channel to be detected and chosen.
Note: it is recommended to do a fast manual qualityscreen with quality_screening.m (require matlab) to ensure classification quality is up to user's satisfation before performing any analysis.
Note: Please feel free to contact me at jwang3276@wisc.edu for any questions, suggestions ,or tech support.
