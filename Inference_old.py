# %%
import os
import glob
import mne
import torch
import torch.nn.functional as F
import numpy as np
from scipy.io import savemat
from scipy.signal import medfilt
from torch.utils.data import DataLoader
from tqdm import tqdm
from RESTCORE import REST
from RESTutils import compute_powers,data_process,smooth_label,create_sequences,compute_powers_welch, viterbi_smooth


# %%
# Parameters
HMM_smoothing = True # Enable/Disable Viterbi/HMM smoothing
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
window_size = 90 # Number of epochs in a sequence
step=60 # overlapping step size for sequences
batch_size = 256  # Batch size for training
n_classes = 3   # Number of sleep stages (e.g., Wake, NREM, REM)
f_bin=130 # Frequency bin for PSD computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
Model_path=r"M:\Alex\Python\REST V1.5\model_V1.pth"
model = REST(
    in_feat=f_bin,
    n_classes=3,
    win_len=window_size,
    d_model=256,
    nhead=8,
    nlayers_epoch=4,
    nlayers_seq=4,
    ff=512,
    fc_hidden1=128,
    fc_hidden2=64,
    dropout=0.1
).to(device)
model.load_state_dict(torch.load(Model_path))  # Load the trained weights
model.to(device)  # Move the model to the GPU
model.eval()  # Set the model to evaluation mode

# %%
# length=fs*60*60*24
# edf_folder = r"M:\Alex\Python\GrandClassifier\Seizure EDF
edf_folder = (r"M:\EEG files\2024\DOD Cohort 2\sham",r"M:\EEG files\2024\DOD Cohort 2\TBI",
    r"M:\EEG files\2024\DOD Cohort 3\sham",r"M:\EEG files\2024\DOD Cohort 3\TBI",r"M:\EEG files\2024\DOD Cohort 4\sham",r"M:\EEG files\2024\DOD Cohort 4\TBI",
    r"M:\EEG files\2024\DOD Cohort 5\sham",r"M:\EEG files\2024\DOD Cohort 5\TBI",r"M:\EEG files\2024\DOD Cohort 6\sham",r"M:\EEG files\2024\DOD Cohort 6\TBI",
    r"M:\EEG files\2024\DOD Cohort 7\sham",r"M:\EEG files\2024\DOD Cohort 7\TBI",r"M:\EEG files\2024\DOD Cohort 8\sham",r"M:\EEG files\2024\DOD Cohort 8\TBI",
    r"M:\EEG files\2025\DOD Cohort 9\TBI",r"M:\EEG files\2025\DOD Cohort 10\TBI",r"M:\EEG files\2025\DOD Cohort 11\headcap",r"M:\EEG files\2025\DOD Cohort 11\TBI",
    r"M:\EEG files\2025\DOD Cohort 12\headcap",r"M:\EEG files\2025\DOD Cohort 13\headcap",r"M:\EEG files\2025\DOD Cohort 13\TBI",r"M:\EEG files\2025\DOD Cohort 14\headcap",
    r"M:\EEG files\2025\DOD Cohort 14\TBI",r"M:\EEG files\2026\DOD Cohort 15\headcap",r"M:\EEG files\2026\DOD Cohort 15\TBI")
# edf_folder = [r"M:\Alex\REST-Testing\C57SA"]
edf_files = []  # Initialize edf_files as an empty list
for folder in edf_folder:
    a = glob.glob(os.path.join(folder, "**", "*.edf"), recursive=True)
    edf_files.extend(a)  # Append the found files to edf_files
score_file_header = "_REST_V1.5.mat"


# %%

for fp_edf in tqdm(edf_files):
    try:
        file_name = os.path.splitext(os.path.basename(fp_edf))[0]
        save_folder = os.path.dirname(fp_edf)
        save_path = os.path.join(save_folder, file_name + score_file_header)
        
        raw = mne.io.read_raw_edf(fp_edf, preload=True, verbose=False)                            
        channel_name = raw.info.ch_names

        EEG_channel = [i for i, name in enumerate(channel_name) if 'RF' in name and 'LP' not in name]
        EMG_channel_list = [index for index, name in enumerate(channel_name) if 'EMG' in name]
        
        if not EEG_channel:
            print(f"Skipping {fp_edf}: Missing 'RF' channel")
            continue
            
        if not EMG_channel_list:
            print(f"Skipping {fp_edf}: Missing 'EMG' channel")
            continue
            
        EMG_channel = EMG_channel_list[0]
        
        EEG = raw.get_data(EEG_channel) 
        EMG = raw.get_data(EMG_channel) 
        
        # power = compute_powers(EEG, EMG, sfreq=512)
        power = compute_powers_welch(EEG*1e6, EMG*1e6, sfreq=512) # Convert to microvolts
        EEG_STFT, EMG_STFT = data_process(EEG, EMG)
        STFT = np.concatenate((EEG_STFT, EMG_STFT), axis=-1)
        
        X = create_sequences(data=STFT, window_size=window_size, step=step)
        sequences_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        sequences_batch = DataLoader(sequences_tensor, batch_size=batch_size, shuffle=False)

        all_preds = []
        with torch.no_grad():
            for batch_X in sequences_batch:
                batch_X = batch_X.to(device)
                output = model(batch_X)
                probs = F.softmax(output, dim=2)
                first_epoch_probs = probs[:, :step, :].cpu().numpy()
                all_preds.append(first_epoch_probs)

        # Viterbi Smoothing (Updated)
        probs_flat = np.concatenate(all_preds, axis=0).reshape(-1, 3)
        if HMM_smoothing:
            score = viterbi_smooth(probs_flat) + 1
        else:
            # Raw Argmax (1-based)
            score = np.argmax(probs_flat, axis=1) + 1
        # predictions = score - 1 # 0-indexed for consistency if needed
        # score = np.array(predictions, dtype=np.int64) # Replaced by Viterbi
        # score = medfilt(score, 5) # Replaced by Viterbi

        # savemat(save_path, {'score': score, 'power': power, 'power_welch': power_welch})
        savemat(save_path, {'score': score, 'power': power})

        # print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error processing {fp_edf}: {e}")
        continue



