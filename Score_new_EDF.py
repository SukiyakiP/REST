# %%
import os
import glob
import math 
import mne
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, resample_poly,stft,resample
from scipy.io import savemat
from scipy.ndimage import median_filter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# %%
# Parameters
fs = 512  # Sampling frequency
epoch_length = 4  # Epoch length in seconds
sequence_length = 90 # Number of epochs in a sequence
step=30 # overlapping step size for sequences
batch_size = 128  # Batch size for training
n_classes = 3   # Number of sleep stages (e.g., Wake, NREM, REM)
WeightedLoss = False # Use weighted loss function
f_bin=130 # Frequency bin for PSD computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = r'' # Path to the trained model
file_folder = r'' # folder containing EDF files
save_path = r'' # Path to save the score files
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

def hampel_filter(signal, window_size=5, n_sigma=3):
    """
    Removes spikes using Hampel filtering.
    
    Parameters:
    signal (array-like): Input 1D signal.
    window_size (int): Number of points to consider in the moving window.
    n_sigma (float): Threshold for identifying outliers.

    Returns:
    array: Filtered signal.
    """
    median = median_filter(signal, size=window_size, mode='nearest')
    diff = np.abs(signal - median)
    threshold = n_sigma * np.median(diff)

    # Replace outliers with median
    signal[diff > threshold] = median[diff > threshold]
    
    return signal

def remove_spikes(features, threshold=4, max_replacements=3):
    """
    Removes extreme spikes from multi-feature data using Z-score thresholding.
    
    Parameters:
    features (ndarray): 2D array (samples × features) e.g., (21600×16).
    threshold (float): Z-score threshold for spike detection (default: 4).

    Returns:
    ndarray: Filtered features with spikes replaced by local median.
    """
    features = np.array(features)  # Ensure it's a NumPy array
    is_1d = features.ndim == 1  # Check if it's a single feature

    if is_1d:
        features = features.reshape(-1, 1)  # Convert to 2D for processing

    filtered_features = features.copy()  # Copy to avoid modifying the original

    for feature_idx in range(filtered_features.shape[1]):  # Process each feature independently
        column = filtered_features[:, feature_idx]  # Extract one feature column
        
        for _ in range(max_replacements):  # Multiple passes for strong spikes
            mean = np.mean(column)
            std = np.std(column)

            z_scores = np.abs((column - mean) / std)  # Compute Z-scores
            spike_indices = np.where(z_scores > threshold)[0]  # Detect spikes
            
            if len(spike_indices) == 0:  # Stop early if no spikes found
                break
            
            # Replace spikes with the median of surrounding clean data
            for i in spike_indices:
                left = max(0, i - 50)  # Expand neighborhood to 5 points
                right = min(len(column), i + 51)

                clean_data = column[left:right]
                clean_data = clean_data[np.abs((clean_data - mean) / std) < threshold]  # Keep only non-spike values

                if len(clean_data) > 0:  # Only replace if clean data exists
                    filtered_features[i, feature_idx] = np.median(clean_data)

    if is_1d:
        return filtered_features.flatten()  # Convert back to 1D if input was 1D
    return filtered_features  # Return 2D for multi-feature input

def compute_powers(EEG, EMG, sfreq=512,s=4): #input should be raw EEG and EMG signals, this is for analysis and visual validation only, the extracted powers are not used in scoring
    EEG = np.asarray(EEG).flatten() # flatten the signal
    EMG = np.asarray(EMG).flatten() # flatten the signal
    
    epoch_length = int(sfreq * s)    # Define epoch length (s seconds)
    n_epochs =  len(EEG) // epoch_length# Determine the number of complete epochs in the signal
    
    # Trim the signal to only include complete epochs
    trimmed_EEG = EEG[:n_epochs * epoch_length]
    trimmed_EMG = EMG[:n_epochs * epoch_length]
    
    # Reshape the signal into epochs (each row is one epoch)
    EEG_epochs = trimmed_EEG.reshape(n_epochs, epoch_length)
    EMG_epochs = trimmed_EMG.reshape(n_epochs, epoch_length)
    
    features = []
    for EEG_epoch,EMG_epoch in zip(EEG_epochs, EMG_epochs):
        Delta=np.mean(bandpass_filter(EEG_epoch, 0.5, 4, sfreq, order=4) **2 )
        Theta=np.mean(bandpass_filter(EEG_epoch, 5, 7, sfreq, order=4) **2 )
        Alpha=np.mean(bandpass_filter(EEG_epoch, 8, 13, sfreq, order=4) **2 )
        Beta=np.mean(bandpass_filter(EEG_epoch, 14, 23, sfreq, order=4) **2 )
        Gamma=np.mean(bandpass_filter(EEG_epoch, 30, 70, sfreq, order=4) **2 )
        EMG_f=notch_filter(EMG_epoch, 60, sfreq, quality=30) # AC filter
        EMG_POW=np.mean(bandpass_filter(EMG_f, 10, 250, sfreq, order=4) **2 )
        
        # Append all features
        features.append([Delta, Theta, Alpha, Beta, Gamma, EMG_POW])
    
    return np.array(features)  # Shape: [n_epochs, 8]

def data_process(EEG,EMG,fs=512): #STFT for classification, input should be raw EEG and EMG signals
    ## downsample, filter, and normaliztion
    EEG = np.asarray(EEG).flatten() # flatten the signal
    EEG=bandpass_filter(EEG, 0.1, 30, fs,4) # bandpass filter
    EEG=resample_poly(EEG, up=1, down=8) # resample the signal, 64hz, 266 samples per epoch
    EEG = (EEG - np.mean(EEG)) / np.std(EEG) # normalize the signal
    
    EMG = np.asarray(EMG).flatten() # flatten the signal
    EMG=bandpass_filter(EMG, 10, 250, fs,4) # bandpass filter
    EMG=notch_filter(EMG, 60, fs, 30) # notch filter
    EMG = (EMG - np.mean(EMG)) / np.std(EMG) # normalize the signal
    
    ## reshape the signal to epochs
    n_epochs = len(EEG) // 256
    EEG = EEG[:n_epochs * 256] # truncate to full epochs
    EEG=EEG.reshape(-1, 256) # reshape to epochs
    n_epochs = len(EMG) // 2048
    EMG = EMG[:n_epochs * 2048] # truncate to full epochs
    EMG=EMG.reshape(-1, 2048) # reshape to epochs
    
    ## STFT
    EEG_fs,  EEG_nperseg  =  64, 128        # 0.5 Hz resolution  •  5 time frames
    EMG_fs,  EMG_nperseg  = 512, 1024       # 0.5 Hz resolution  •  5 time frames

    def epoch_to_spectrogram(epoch, fs, nperseg):
        _, _, Zxx = stft(epoch, fs=fs, nperseg=nperseg, noverlap=nperseg//2, padded=False)
        return np.abs(Zxx).T                 # → [frames, freq_bins]

    EEG_STFT = np.stack([epoch_to_spectrogram(ep, EEG_fs, EEG_nperseg) for ep in EEG])
    EMG_STFT = np.stack([epoch_to_spectrogram(ep, EMG_fs, EMG_nperseg) for ep in EMG])

    # resample EMG freqs to 64hz so both have 65 bins, resolution compression. this is mainly done to save RAM and VRAM usage.
    EMG_STFT = resample(EMG_STFT, EEG_STFT.shape[-1], axis=-1)
    return EEG_STFT, EMG_STFT

# %%
# -- Positional encoding ------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):                             # x:[B, L, D]
        return x + self.pe[:, :x.size(1)]
# -- Attention pooling --------------------------------------------------------
class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, 1)

    def forward(self, x):                             # x:[B, L, D]
        w = torch.softmax(self.q(x).squeeze(-1), dim=1)   # [B, L]
        return torch.sum(w.unsqueeze(-1) * x, dim=1)      # [B, D]

# -- Epoch‑level encoder ------------------------------------------------------
class EpochEncoder(nn.Module):
    def __init__(self, in_feat, d_model=256, nhead=8, nlayers=2, ff=256, dropout=0.1):
        super().__init__()
        self.in_proj  = nn.Linear(in_feat, d_model)
        self.pos_enc  = PositionalEncoding(d_model, max_len=10)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True)
        self.enc      = nn.TransformerEncoder(encoder_layer, nlayers)
        self.pool     = AttnPool(d_model)

    def forward(self, x):                 # x:[B, frames, feat]
        x = self.in_proj(x)
        x = self.pos_enc(x)
        x = self.enc(x)                   # [B, frames, d_model]
        return self.pool(x)               # [B, d_model]

# -- Sequence‑level model -----------------------------------------------------
class SleepTransformer(nn.Module):
    def __init__(self, in_feat, n_classes, win_len,    # win_len = window_size
                 d_model=256, nhead=8, nlayers_epoch=4, nlayers_seq=4,
                 ff=512, fc_hidden=64, dropout=0.1):
        super().__init__()
        self.epoch_encoder = EpochEncoder(in_feat, d_model, nhead,
                                          nlayers_epoch, ff, dropout)

        self.pos_enc_seq   = PositionalEncoding(d_model, max_len=win_len)
        seq_layer          = nn.TransformerEncoderLayer(d_model, nhead,
                                                        ff, dropout, batch_first=True)
        self.seq_encoder   = nn.TransformerEncoder(seq_layer, nlayers_seq)

        self.fc = nn.Sequential(
            nn.Linear(d_model, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, n_classes)
        )

    def forward(self, x):                 # x:[B, win_len, frames, feat]
        B, W, F, C = x.shape
        x = x.view(B * W, F, C)           # merge batch & window
        epoch_vec = self.epoch_encoder(x) # [B*W, d_model]
        epoch_vec = epoch_vec.view(B, W, -1)

        x = self.pos_enc_seq(epoch_vec)
        x = self.seq_encoder(x)           # [B, W, d_model]

        logits = self.fc(x)               # [B, W, n_classes]
        return logits

# Reshape features into overlapping sequences
def create_sequences(data, window_size, step): # data: [n_samples, n_features], labels: [n_samples]
    X= []
    max_start = len(data) - window_size + 1 # last start index for a whole window
    for start in range(0, max_start, step): # step through data
        end = start + window_size # end index for current window
        X.append(data[start:end])       # shape [window_size, frame_seq_len, n_features]
    return np.array(X) # shape [n_windows, window_size, frame_seq_len, n_features], [n_windows, window_size]

# %%
model = SleepTransformer(f_bin, n_classes, sequence_length).to(device)
model.load_state_dict(torch.load(model_path))  # Load the trained weights
model.to(device)  # Move the model to the GPU
model.eval()  # Set the model to evaluation mode

edf_files = glob.glob(os.path.join(file_folder, '*.edf')) # Get all .edf files in the folder
# Process each animal's files                  
for file_path in tqdm(edf_files, desc="Scoring EDFs"):
    basename = os.path.splitext(os.path.basename(file_path))[0]
    file_name = f"{basename}_score_and_feature.mat"
    fp_edf=file_path
    raw = mne.io.read_raw_edf(fp_edf, preload=True) # read the edf file                           
    channel_name=raw.info.ch_names # get the channel names
    EEG_channel=[index for index, name in enumerate (channel_name) if EEG_channel_name in name] # find the index of the RF channel
    EEG=raw.get_data(EEG_channel) # get the RF channel as raw EEG signal
    EMG_channel=[index for index, name in enumerate (channel_name) if EMG_channel_name in name] # find the index of the EMG channel
    EMG=raw.get_data(EMG_channel)   # get the EMG channel as raw EMG signal
    power=compute_powers(EEG, EMG, sfreq=512) #[Delta, Theta, Alpha, Beta, Gamma, EMG_POW]
    EEG_STFT,EMG_STFT=data_process(EEG,EMG) # STFT for classification,shape    
    STFT = np.concatenate((EEG_STFT, EMG_STFT), axis=-1) # shape [n_epochs, 5, 130]
    X = create_sequences(data=STFT, window_size=sequence_length, step=step)
    sequences_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    sequences_batch=DataLoader(sequences_tensor, batch_size=batch_size, shuffle=False)
    all_preds=[]
    with torch.no_grad():
        for batch_X in sequences_batch:
            batch_X= batch_X.to(device)
            output = model(batch_X)  # Shape: [batch_size, sequence_length, n_classes]
            predicted = torch.argmax(output.data, 2)  # Shape: [batch_size, sequence_length] 
            first_epoch_preds = predicted[:,:step].cpu().numpy() # first n=step of each sequence
            all_preds.append(first_epoch_preds) # concatenate predictions from each sequence
    predictions = np.concatenate(all_preds, axis=0).flatten() + 1  # Adjust labels if needed
    score = np.array(predictions, dtype=np.int64)
            
    # Save the result in the designated folder structure.
    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, file_name)
    savemat(out_path, {'score': score,'power':power})


