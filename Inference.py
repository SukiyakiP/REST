# %%
import os
import glob
import mne
import torch
import gc
import time
import torch.nn.functional as F
import numpy as np
from scipy.io import savemat
from torch.utils.data import DataLoader
from tqdm import tqdm
from RESTCORE import REST
from RESTutils import data_process_tensor, create_sequences, compute_powers_welch_tensor, viterbi_smooth
from ArtifactFilter import flag_artifacts


# %%
# Parameters
HMM_smoothing   = True  # Enable/Disable Viterbi/HMM smoothing
Skip_processed  = False # Skip files that already have a scoring file generated
BOUT_FILTER     = True  # Remove biologically implausible short bouts after scoring
ARTIFACT_FILTER = True  # Override REST artifact predictions with rule-based EEG detector (last step)

# Minimum consecutive epochs per stage (4 s/epoch) — 1-based scoring
# 1=Wake, 2=NREM, 3=REM, 4=Artifact
BOUT_MIN_EPOCHS = {1: 1, 2: 3, 3: 3, 4: 2}
Use_LayerNorm = True  # input LayerNorm (must match Training.py)
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
window_size = 120 # Number of epochs in a sequence
step = 90         # overlapping step size for sequences
batch_size = 256  # Batch size for training
n_classes = 4   # Number of sleep stages (Wake, NREM, REM, Artifact)
f_bin=130 # Frequency bin for PSD computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
CHECKPOINT_DIR = r"M:\Alex\Python\REST\checkpoints\20260514_1309_w120_artrepeat2"
WEIGHT_FILE    = "best_acc.pth"   # or "best_artf1.pth"

import json as _json
_config_path = os.path.join(CHECKPOINT_DIR, 'config.json')
if os.path.exists(_config_path):
    with open(_config_path) as _f:
        print(f"[Model] config: {_json.load(_f)}")

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
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, WEIGHT_FILE),
                                 weights_only=True, map_location=device))
model.to(device)  # Move the model to the GPU
model.eval()  # Set the model to evaluation mode

# %%
# length=fs*60*60*24
# edf_folder = r"M:\Alex\Python\GrandClassifier\Seizure EDF
# edf_folder = (r"M:\EEG files\2024\DOD Cohort 2\sham",r"M:\EEG files\2024\DOD Cohort 2\TBI",
#     r"M:\EEG files\2024\DOD Cohort 3\sham",r"M:\EEG files\2024\DOD Cohort 3\TBI",r"M:\EEG files\2024\DOD Cohort 4\sham",r"M:\EEG files\2024\DOD Cohort 4\TBI",
#     r"M:\EEG files\2024\DOD Cohort 5\sham",r"M:\EEG files\2024\DOD Cohort 5\TBI",r"M:\EEG files\2024\DOD Cohort 6\sham",r"M:\EEG files\2024\DOD Cohort 6\TBI",
#     r"M:\EEG files\2024\DOD Cohort 7\sham",r"M:\EEG files\2024\DOD Cohort 7\TBI",r"M:\EEG files\2024\DOD Cohort 8\sham",r"M:\EEG files\2024\DOD Cohort 8\TBI",
#     r"M:\EEG files\2025\DOD Cohort 9\TBI",r"M:\EEG files\2025\DOD Cohort 10\TBI",r"M:\EEG files\2025\DOD Cohort 11\headcap",r"M:\EEG files\2025\DOD Cohort 11\TBI",
#     r"M:\EEG files\2025\DOD Cohort 12\headcap",r"M:\EEG files\2025\DOD Cohort 13\headcap",r"M:\EEG files\2025\DOD Cohort 13\TBI",r"M:\EEG files\2025\DOD Cohort 14\headcap",
#     r"M:\EEG files\2025\DOD Cohort 14\TBI",r"M:\EEG files\2026\DOD Cohort 15\headcap",r"M:\EEG files\2026\DOD Cohort 15\TBI")
# edf_folder = [r"M:\Alex\REST-Testing"]
edf_folder = [r"M:\EEG files\2026\DBA\Reduced"]
# edf_folder = (r"M:\EEG files\2026\DOD Cohort 16\headcap",r"M:\EEG files\2026\DOD Cohort 16\TBI")
edf_files = []  # Initialize edf_files as an empty list
for folder in edf_folder:
    a = glob.glob(os.path.join(folder, "**", "*.edf"), recursive=True)
    edf_files.extend(a)  # Append the found files to edf_files
score_file_header = "_REST_V1.5.mat"
# score_file_header = "_Full_Labels.mat"

# %%

def filter_short_bouts(score, min_epochs):
    """Reassign bouts shorter than min_epochs[stage] to the surrounding stage."""
    out = score.copy()
    i = 0
    while i < len(out):
        stage = out[i]
        j = i
        while j < len(out) and out[j] == stage:
            j += 1
        if (j - i) < min_epochs.get(int(stage), 1):
            neighbor = out[i - 1] if i > 0 else (out[j] if j < len(out) else stage)
            out[i:j] = neighbor
        i = j
    return out


def process_edf(fp_edf, model, window_size, step, batch_size, device, HMM_smoothing, score_file_header, skip_processed):
    try:
        file_name = os.path.splitext(os.path.basename(fp_edf))[0]
        save_folder = os.path.dirname(fp_edf)
        save_path = os.path.join(save_folder, file_name + score_file_header)
        
        # Check if the file has already been processed and the skip toggle is on
        if skip_processed and os.path.exists(save_path):
            print(f"Skipping {fp_edf}: Already scored ({save_path} exists)")
            return
        
        # Retry mechanism for data server disconnection
        raw = None
        for attempt in range(3):
            try:
                with torch.no_grad():
                    raw = mne.io.read_raw_edf(fp_edf, preload=True, verbose=False)
                    if raw.info['sfreq'] != fs:
                        raw.resample(fs)
                break # Successfully loaded
            except Exception as e:
                if attempt < 2:
                    print(f"Failed to load {fp_edf} (Attempt {attempt + 1}/3). Waiting 5 seconds...")
                    time.sleep(5)
                else:
                    raise e # Let the outer try-except block catch and skip the file
                    
        with torch.no_grad():                           
            channel_name = raw.info.ch_names

            EEG_channel = [i for i, name in enumerate(channel_name) if 'RF' in name and 'LP' not in name]
            EMG_channel_list = [index for index, name in enumerate(channel_name) if 'EMG' in name]
            
            if not EEG_channel:
                print(f"Skipping {fp_edf}: Missing 'RF' channel")
                if hasattr(raw, 'close'): raw.close()
                return
                
            if not EMG_channel_list:
                print(f"Skipping {fp_edf}: Missing 'EMG' channel")
                if hasattr(raw, 'close'): raw.close()
                return
                
            EMG_channel = EMG_channel_list[0]
            
            EEG = raw.get_data(EEG_channel) 
            EMG = raw.get_data(EMG_channel) 
            
            if hasattr(raw, 'close'):
                raw.close()
            
            power = compute_powers_welch_tensor(EEG*1e6, EMG*1e6, sfreq=512, s=4, device=device) # Convert to microvolts
            EEG_STFT, EMG_STFT = data_process_tensor(EEG, EMG, fs=512, device=device)
            STFT = np.concatenate((EEG_STFT, EMG_STFT), axis=-1)
            
            n_epochs = len(STFT)
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

            probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, n_classes) if all_preds else np.empty((0, n_classes), dtype=np.float32)

            # Score any tail epochs not covered by the regular windows.
            # create_sequences drops epochs that don't start a new full window,
            # so recordings whose length isn't a multiple of step lose up to (step-1) tail epochs.
            n_scored = probs_flat.shape[0]
            if n_scored < n_epochs:
                tail_len = n_epochs - n_scored
                tail_window = STFT[max(0, n_epochs - window_size):n_epochs]
                if len(tail_window) < window_size:
                    pad = np.zeros((window_size - len(tail_window), STFT.shape[1]), dtype=np.float32)
                    tail_window = np.concatenate([pad, tail_window], axis=0)
                tail_tensor = torch.tensor(tail_window[np.newaxis], dtype=torch.float32).to(device)
                with torch.no_grad():
                    tail_probs = F.softmax(model(tail_tensor), dim=2)[0, -tail_len:, :].cpu().numpy()
                probs_flat = np.concatenate([probs_flat, tail_probs], axis=0)
            if HMM_smoothing:
                score = viterbi_smooth(probs_flat) + 1
            else:
                score = np.argmax(probs_flat, axis=1) + 1

            if BOUT_FILTER:
                score = filter_short_bouts(score, BOUT_MIN_EPOCHS)

            if ARTIFACT_FILTER:
                art_mask = flag_artifacts(EEG[0], fs=fs, epoch_seconds=epoch_length)
                n = min(len(art_mask), len(score))
                score[:n][art_mask[:n]] = 4

            savemat(save_path, {'score': score, 'power': power})
            
    except Exception as e:
        print(f"Error processing {fp_edf}: {e}")
        # Typical errors include FileNotFoundError or OSError if the data server abruptly disconnects.
        # The function will exit here cleanly and the loop will simply advance to the next file.

for fp_edf in tqdm(edf_files):
    process_edf(fp_edf, model, window_size, step, batch_size, device, HMM_smoothing, score_file_header, Skip_processed)
    # Explicit garbage collection and cache clearing after every file or error
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available() and hasattr(torch.backends.cuda, 'cufft_plan_cache'):
        try:
            torch.backends.cuda.cufft_plan_cache.clear()
        except:
            pass



