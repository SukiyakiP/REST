# %%
# import packages and functions
import os
import mne
from scipy.signal import butter, filtfilt, iirnotch,resample_poly,stft,resample
import numpy as np
import pandas as pd
from math import gcd

# %%
# Parameters
fs=512
epoch_length=4 # seconds
# Folder path
folder_path = r'' # Specify the folder path containing the both EDF and score files, the files should be named like 'subject_1.edf' and 'subject_1.tsv'
save_path = r'' # Specify the folder path to save the processed data
EEG_channel_name = 'RF' # Name of the EEG channel to use, change if needed
EMG_channel_name = 'EMG' # Name of the EMG channel to use, change if needed

# %%
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, freq, fs, quality=30):
    nyquist = 0.5 * fs
    f0 = freq / nyquist
    b, a = iirnotch(f0, quality)
    return filtfilt(b, a, data)

def find_data_start(file_path, sep='\t', expected_columns=None):
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # Split the line using the delimiter
            columns = line.strip().split(sep)
            
            # Check if the number of columns matches the expected format
            if expected_columns and len(columns) == expected_columns:
                return i  # Return the first valid data line index
    return 0  # Default to start at the beginning if no valid line is found

def resample_to_target(signal, original_fs, target_fs):
    """
    Resamples a 1D or 2D signal from original_fs to target_fs using polyphase filtering.

    Parameters:
    - signal: np.ndarray, 1D or 2D array where resampling is applied along axis=0
    - original_fs: int or float, original sampling frequency
    - target_fs: int or float, target sampling frequency

    Returns:
    - Resampled signal as np.ndarray
    """
    # Convert to integers and compute resample ratio
    orig = int(original_fs * 1000)
    target = int(target_fs * 1000)
    factor = gcd(orig, target)
    up = target // factor
    down = orig // factor

    return resample_poly(signal, up=up, down=down, axis=0)

def data_process(EEG,EMG):
    #EEG
    EEG = np.asarray(EEG).flatten() # flatten the signal
    EEG=bandpass_filter(EEG, 0.1, 30, fs,4) # bandpass filter
    EEG=resample_to_target(EEG, fs, 64) # resample the signal, 64hz, 266 samples per epoch
    EEG = (EEG - np.mean(EEG)) / np.std(EEG) # normalize the signal
    
    n_epochs = len(EEG) // 256
    EEG = EEG[:n_epochs * 256] # truncate to full epochs
    EEG=EEG.reshape(-1, 256) # reshape to epochs
      
    #EMG
    EMG = np.asarray(EMG).flatten() # flatten the signal
    EMG=bandpass_filter(EMG, 10, 250, fs,4) # bandpass filter
    EMG=notch_filter(EMG, 60, fs, 30) # notch filter
    
    EMG = (EMG - np.mean(EMG)) / np.std(EMG) # normalize the signal
    n_epochs = len(EMG) // 2048
    EMG = EMG[:n_epochs * 2048] # truncate to full epochs
    EMG=EMG.reshape(-1, 2048) # reshape to epochs

    # --- STFT parameters ---------------------------------------------------------
    EEG_fs,  EEG_nperseg  =  fs/8, fs/4        # 0.5 Hz resolution  •  5 time frames 
    EMG_fs,  EMG_nperseg  = fs, fs*2       # 0.5 Hz resolution  •  5 time frames

    def epoch_to_spectrogram(epoch, fs, nperseg):
        _, _, Zxx = stft(epoch, fs=fs, nperseg=nperseg, noverlap=nperseg//2, padded=False)
        return np.abs(Zxx).T                 # → [frames, freq_bins]

    EEG_STFT = np.stack([epoch_to_spectrogram(ep, EEG_fs, EEG_nperseg) for ep in EEG])
    EMG_STFT = np.stack([epoch_to_spectrogram(ep, EMG_fs, EMG_nperseg) for ep in EMG])

    # Optional:  resample EMG freqs so both have 65 bins → makes concatenation trivial
    EMG_STFT = resample(EMG_STFT, EEG_STFT.shape[-1], axis=-1)
    return EEG_STFT, EMG_STFT

# %%

# Initialize lists
fp_tsv = []
fp_edf = []

# Scan the folder and add files to the lists
for file in sorted(os.listdir(folder_path)):
    if file.endswith('.tsv'):
        fp_tsv.append(os.path.join(folder_path, file))
    elif file.endswith('.edf'):
        fp_edf.append(os.path.join(folder_path, file))

# Ensure the files are in pairs and in correct order
fp_tsv.sort()
fp_edf.sort()

# Check if the pairs are correct
for edf_file, tsv_file in zip(fp_edf, fp_tsv):
    edf_base = os.path.basename(edf_file).rsplit('_', 1)[0]
    tsv_base = os.path.basename(tsv_file).rsplit('_', 1)[0]
    if edf_base != tsv_base:
        print(f"Mismatch: {edf_file} and {tsv_file}")

pair_lengths = []
for tsv_file, edf_file in zip(fp_tsv, fp_edf):
    start_line = find_data_start(tsv_file, sep='\t', expected_columns=5)
    df = pd.read_csv(tsv_file, sep='\t', skiprows=start_line + 1, header=None)
    score = df.iloc[:, 4].to_numpy()
    score_length = len(score)  # This is the number of score epochs (21600 after padding)
    raw = mne.io.read_raw_edf(edf_file, preload=False)
    n_samples=raw.n_times
    eeg_epochs = n_samples // (fs * 4) 
    pair_lengths.append((score_length, eeg_epochs))

# %%
score_additional = []
EEG_additional = []
EMG_additional = []
for i, (score_length, eeg_epochs) in enumerate(pair_lengths):        
    # Read the TSV file for score, if your score file has a different structure, you may need to adjust this part
    start_line = find_data_start(fp_tsv[i], sep='\t', expected_columns=5)
    df = pd.read_csv(fp_tsv[i], sep='\t', skiprows=start_line+1, header=None)#questionable, may need to change
    score = df.iloc[:, 4].to_numpy()
    score[score == 0] = 1
    score[score > 3] = 1
    # read the edf file    
    raw = mne.io.read_raw_edf(fp_edf[i], preload=True) 
    channel_name=raw.info.ch_names # get the channel names
    idx=[index for index, name in enumerate (channel_name) if EEG_channel_name in name] # find the index of the RF channel
    EEG=raw.get_data(idx) # get the RF channel       
    idx=[index for index, name in enumerate (channel_name) if EMG_channel_name in name] # find the index of the EMG channel
    EMG=raw.get_data(idx)   # get the EMG channel
    EEG,EMG=data_process(EEG,EMG) # process the data
    # Compare lengths and cut the longer one to match the shorter one
    min_length = min(len(EEG), len(EMG), len(score))
    EEG = EEG[:min_length]
    EMG = EMG[:min_length]
    score = score[:min_length]  
    EEG=EEG.astype(np.float32)
    EMG=EMG.astype(np.float32)
    score=score.astype(np.float32)

    if len(score_additional) == 0:
        score_additional = score
    else:
        score_additional = np.concatenate((score_additional, score), axis=0)
        
    if len(EEG_additional) == 0:
        EEG_additional = EEG
    else:
        EEG_additional = np.concatenate((EEG_additional, EEG), axis=0)
        
    if len(EMG_additional) == 0:
        EMG_additional = EMG
    else:
        EMG_additional = np.concatenate((EMG_additional, EMG), axis=0)

np.savez(save_path , EEG=EEG_additional, EMG=EMG_additional, score=score_additional)


