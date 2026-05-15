# %%
"""
Rapid Model Testing Script
=========================
This script combines inference and evaluation for quick model weight testing.
It processes EDF files through the REST model and compares against reference scores.
Saves an Excel report with per-dataset sheets including full analysis per dataset.

Usage:
    1. Set MODEL_PATH to the model weight file you want to test
    2. Run the script - it will infer all recordings and save results
"""

import os
import glob
import mne
import torch
import gc
import time
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

from RESTCORE import REST
from RESTutils import data_process_tensor, create_sequences, viterbi_smooth, find_data_start

# =============================================================================
# MODEL CONFIGURATION - CHANGE THIS TO TEST DIFFERENT WEIGHTS
# =============================================================================
MODEL_PATH = r"M:\Alex\Python\REST\checkpoints\20260514_1309_w120_artrepeat2\best_artf1.pth"

# =============================================================================
# Parameters (should match training configuration)
# =============================================================================
HMM_smoothing = True
BOUT_FILTER   = False  # set True to remove short bouts; see BOUT_MIN_EPOCHS below
Use_LayerNorm = True
fs = 512
epoch_length = 4
window_size = 120
step = 90
batch_size = 200
n_classes = 4
f_bin = 130
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"Model path: {MODEL_PATH}")

# =============================================================================
# Load Model
# =============================================================================
model = REST(
    in_feat=f_bin,
    n_classes=n_classes,
    win_len=window_size,
    d_model=384,
    nhead=8,
    nlayers_epoch=4,
    nlayers_seq=6,
    ff=768,
    fc_hidden1=256,
    fc_hidden2=128,
    dropout=0.15,
    use_layernorm=Use_LayerNorm
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
model.eval()

# =============================================================================
# Testing Data Folders (5 datasets)
# =============================================================================
edf_folder_config = [
    ("C57KA", r"M:\Alex\REST-Testing\C57KA"),
    ("C57SA", r"M:\Alex\REST-Testing\C57SA"),
    ("CD1",   r"M:\Alex\REST-Testing\CD1"),
    ("DBAKA", r"M:\Alex\REST-Testing\DBAKA"),
    ("DBASA", r"M:\Alex\REST-Testing\DBASA"),
]

edf_files = []
for folder_name, folder_path in edf_folder_config:
    a = glob.glob(os.path.join(folder_path, "**", "*.edf"), recursive=True)
    edf_files.extend([(f, folder_name) for f in a])

print(f"Found {len(edf_files)} EDF files to process")

# Minimum consecutive epochs per stage before a bout is kept (4 s/epoch).
# 1-based: 1=Wake, 2=NREM, 3=REM, 4=Artifact
BOUT_MIN_EPOCHS = {1: 1, 2: 3, 3: 3, 4: 2}

def filter_short_bouts(score, min_epochs):
    """Reassign bouts shorter than min_epochs[stage] to the surrounding stage."""
    out = score.copy()
    i = 0
    while i < len(out):
        stage = out[i]
        j = i
        while j < len(out) and out[j] == stage:
            j += 1
        bout_len = j - i
        if bout_len < min_epochs.get(int(stage), 1):
            neighbor = out[i - 1] if i > 0 else (out[j] if j < len(out) else stage)
            out[i:j] = neighbor
        i = j
    return out

# =============================================================================
# Score Loading Functions
# =============================================================================
def load_txt_score(score_path):
    """Read RM .txt score file (comma-delimited, score in column 3).
    Returns int32 array: 1=Wake, 2=NREM, 3=REM, 4=Artifact, -100=ignore."""
    df = pd.read_csv(score_path, delimiter=',')
    score = df.iloc[:, 3].to_numpy().astype(np.int32)
    score[score > 3] = -100
    score[score == 0] = 4
    return score

def load_tsv_score(tsv_path):
    """Read Sirenia .tsv score file (tab-delimited, score in column 4)."""
    start_line = find_data_start(tsv_path, sep='\t', expected_columns=5)
    df = pd.read_csv(tsv_path, sep='\t', skiprows=start_line + 1, header=None)
    score = df.iloc[:, 4].to_numpy().astype(np.int32)
    score[score > 3] = -100
    score[score == 0] = 4
    return score

# =============================================================================
# Inference Function
# =============================================================================
def process_edf(fp_edf, model, window_size, step, batch_size, device, HMM_smoothing):
    """Run inference on a single EDF file and return predictions."""
    try:
        file_name = os.path.splitext(os.path.basename(fp_edf))[0]
        
        raw = None
        for attempt in range(3):
            try:
                raw = mne.io.read_raw_edf(fp_edf, preload=True, verbose=False)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"Failed to load {fp_edf} (Attempt {attempt + 1}/3). Waiting 5 seconds...")
                    time.sleep(5)
                else:
                    raise e
        
        with torch.no_grad():
            channel_name = raw.info.ch_names
            EEG_channel = [i for i, name in enumerate(channel_name) if 'RF' in name and 'LP' not in name]
            EMG_channel_list = [index for index, name in enumerate(channel_name) if 'EMG' in name]
            
            if not EEG_channel:
                print(f"Skipping {fp_edf}: Missing 'RF' channel")
                if hasattr(raw, 'close'): raw.close()
                return None, None
            
            if not EMG_channel_list:
                print(f"Skipping {fp_edf}: Missing 'EMG' channel")
                if hasattr(raw, 'close'): raw.close()
                return None, None
            
            EMG_channel = EMG_channel_list[0]
            
            EEG = raw.get_data(EEG_channel) 
            EMG = raw.get_data(EMG_channel) 
            
            if hasattr(raw, 'close'):
                raw.close()
            
            EEG_STFT, EMG_STFT = data_process_tensor(EEG, EMG, fs=512, device=device)
            STFT = np.concatenate((EEG_STFT, EMG_STFT), axis=-1)
            
            X = create_sequences(data=STFT, window_size=window_size, step=step)
            sequences_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            sequences_batch = DataLoader(sequences_tensor, batch_size=batch_size, shuffle=False)
            
            all_preds = []
            for batch_X in sequences_batch:
                batch_X = batch_X.to(device)
                output = model(batch_X)
                probs = F.softmax(output, dim=2)
                first_epoch_probs = probs[:, :step, :].cpu().numpy()
                all_preds.append(first_epoch_probs)
            
            probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, n_classes)
            if HMM_smoothing:
                score = viterbi_smooth(probs_flat) + 1
            else:
                score = np.argmax(probs_flat, axis=1) + 1

            if BOUT_FILTER:
                score = filter_short_bouts(score, BOUT_MIN_EPOCHS)

            return file_name, score
            
    except Exception as e:
        print(f"Error processing {fp_edf}: {e}")
        return None, None

# =============================================================================
# Main Execution
# =============================================================================
print("\n" + "="*60)
print("RAPID MODEL TESTING")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"HMM Smoothing: {HMM_smoothing}")
print("="*60 + "\n")

print("Running Inference & Evaluation...")
print("-" * 40)

# Track results per dataset
dataset_data = {name: {'true': [], 'pred': [], 'files': []} for name, _ in edf_folder_config}
skipped_files = []

for fp_edf, dataset_name in tqdm(edf_files, desc="Processing"):
    base_name = os.path.splitext(os.path.basename(fp_edf))[0]
    edf_folder_path = os.path.dirname(fp_edf)
    score_p = os.path.join(edf_folder_path, base_name + ".txt")
    
    # Check if reference score exists
    if not os.path.exists(score_p):
        skipped_files.append(base_name)
        continue
    
    try:
        file_name, pred_score = process_edf(fp_edf, model, window_size, step, batch_size, device, HMM_smoothing)
        if file_name is None:
            continue
        
        true_score = load_txt_score(score_p)
        
        final_len = min(len(pred_score), len(true_score))
        pred_score = pred_score[:final_len]
        true_score = true_score[:final_len]
        
        valid_mask = true_score != -100
        pred_score_valid = pred_score[valid_mask]
        true_score_valid = true_score[valid_mask]
        
        if len(true_score_valid) == 0:
            continue
        
        # Store per-dataset
        dataset_data[dataset_name]['true'].extend(true_score_valid.tolist())
        dataset_data[dataset_name]['pred'].extend(pred_score_valid.tolist())
        dataset_data[dataset_name]['files'].append(base_name)
        
    except Exception as e:
        print(f"\nError processing {fp_edf}: {e}")
    
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'cufft_plan_cache'):
        try:
            torch.backends.cuda.cufft_plan_cache.clear()
        except:
            pass

# =============================================================================
# Report Results - Per Dataset
# =============================================================================
print("\n" + "="*60)
print("CLASSIFICATION ACCURACY ANALYSIS - BY DATASET")
print("="*60)

if skipped_files:
    print(f"\nSkipped {len(skipped_files)} files (no reference score found)")

# Prepare Excel data
all_sheets_data = {}
model_basename = os.path.splitext(os.path.basename(MODEL_PATH))[0]

# Process each dataset
for dataset_name in sorted(dataset_data.keys()):
    data = dataset_data[dataset_name]
    if len(data['true']) == 0:
        print(f"\n{dataset_name}: No valid data")
        continue
    
    y_true = np.array(data['true']) - 1
    y_pred = np.array(data['pred']) - 1
    y_true[y_true > 3] = 0
    y_pred[y_pred > 3] = 0
    
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name}")
    print(f"{'='*60}")
    print(f"  Files: {len(data['files'])}")
    print(f"  Total epochs: {len(y_true)}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Cohen's Kappa: {kappa:.3f}")
    
    print(f"\n  Classification Report:")
    report = classification_report(y_true, y_pred, target_names=["Wake", "NREM", "REM", "Artifact"], labels=[0, 1, 2, 3], zero_division=0)
    for line in report.split('\n'):
        print(f"    {line}")
    
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"    Wake:   {cm[0].tolist()}")
    print(f"    NREM:   {cm[1].tolist()}")
    print(f"    REM:    {cm[2].tolist()}")
    print(f"    Artifact: {cm[3].tolist()}")
    
    # Store for Excel
    all_sheets_data[dataset_name] = {
        'metrics': {
            'Dataset': dataset_name,
            'Files': len(data['files']),
            'Total Epochs': len(y_true),
            'Accuracy': f"{acc:.4f}",
            'Cohen Kappa': f"{kappa:.3f}"
        },
        'files': data['files'],
        'y_true': y_true,
        'y_pred': y_pred
    }

# =============================================================================
# Save Excel Report
# =============================================================================
excel_path = os.path.join(os.path.dirname(MODEL_PATH), f"RapidTest_{model_basename}_results.xlsx")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # One sheet per dataset with full analysis
    for dataset_name, data in all_sheets_data.items():
        metrics = data['metrics']
        y_true = data['y_true']
        y_pred = data['y_pred']
        
        # Truncate sheet name to 31 chars (Excel limit)
        sheet_name = dataset_name[:31]
        
        # Metrics summary
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
        cm_labels = ['Wake', 'NREM', 'REM', 'Artifact']
        cm_df = pd.DataFrame(cm, index=cm_labels, columns=cm_labels)
        cm_df.index.name = 'True \\ Pred'
        # Append to same sheet below metrics
        start_row = len(metrics_df) + 2
        cm_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row)
        
        # Per-file accuracies
        file_acc_data = []
        for fname in sorted(data['files']):
            file_acc_data.append({'File': fname})
        if file_acc_data:
            start_row += len(cm_df) + 3
            file_acc_df = pd.DataFrame(file_acc_data)
            file_acc_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=start_row)

print(f"\n\nExcel report saved: {excel_path}")

print("\n" + "="*60)
print("TESTING COMPLETE")
print("="*60 + "\n")
