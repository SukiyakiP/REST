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
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from tqdm import tqdm
from RESTCORE import REST
from RESTutils import compute_powers,data_process,data_process_tensor,smooth_label,create_sequences,compute_powers_welch,compute_powers_welch_tensor,viterbi_smooth


# %%
# Parameters
HMM_smoothing = True # Enable/Disable Viterbi/HMM smoothing
Skip_processed = False # Skip files that already have a scoring file generated
Use_BatchNorm = True # Use 1D Batch Normalization inside the Neural Network (WARNING: Must match the toggle used during Training.py!)
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
window_size = 90 # Number of epochs in a sequence
step=60 # overlapping step size for sequences
batch_size = 256  # Batch size for training
n_classes = 4   # Number of sleep stages (Wake, NREM, REM, Artifact)
f_bin=130 # Frequency bin for PSD computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
Model_path=r"M:\Alex\Python\REST V1.5\model_artifact.pth"  # 4-class with Artifact head (no-inject baseline)
model = REST(
    in_feat=f_bin,
    n_classes=n_classes,
    win_len=window_size,
    d_model=256,
    nhead=8,
    nlayers_epoch=4,
    nlayers_seq=4,
    ff=512,
    fc_hidden1=128,
    fc_hidden2=64,
    dropout=0.1,
    use_batchnorm=Use_BatchNorm
).to(device)
model.load_state_dict(torch.load(Model_path))  # Load the trained weights
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

            # Viterbi Smoothing (Updated)
            probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, n_classes)
            if HMM_smoothing:
                score = viterbi_smooth(probs_flat) + 1
            else:
                # Raw Argmax (1-based)
                score = np.argmax(probs_flat, axis=1) + 1

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



