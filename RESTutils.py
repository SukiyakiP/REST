import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.signal import welch, butter, filtfilt, iirnotch, stft, resample, resample_poly, medfilt as median_filter
from math import gcd


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
    
    epoch_length = int(sfreq * s)    # Define epoch length (s seconds* sample frequency)
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
        Full_EEG=np.mean(bandpass_filter(EEG_epoch, 0.5, 30, sfreq, order=4) **2 ) # Full band power
        EMG_f=notch_filter(EMG_epoch, 60, sfreq, quality=30) # AC filter
        EMG_POW=np.mean(bandpass_filter(EMG_f, 10, 250, sfreq, order=4) **2 )
        
        # Append all features
        features.append([Delta, Theta, Alpha, Beta, Gamma, Full_EEG, EMG_POW])
    
    return np.array(features)  # Shape: [n_epochs, 8]

def band_power(freqs, spectra, fmin=0.5, fmax=4):
    """Extract delta power from spectra matrix."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    power = spectra[mask].sum()
    return power

def compute_powers_welch(EEG,EMG, sfreq=512,s=4): #input should be raw EEG and EMG signals, this is for analysis and visual validation only, the extracted powers are not used in scoring
    EEG = np.asarray(EEG).flatten() # flatten the signal
    EMG = np.asarray(EMG).flatten() # flatten the signal
    epoch_length = int(sfreq * s)    # Define epoch length (s seconds* sample frequency)
    n_epochs =  len(EEG) // epoch_length# Determine the number of complete epochs in the signal
    
    # Trim the signal to only include complete epochs
    trimmed_EEG = EEG[:n_epochs * epoch_length]
    trimmed_EMG = EMG[:n_epochs * epoch_length]
    
    # Reshape the signal into epochs (each row is one epoch)
    EEG_epochs = trimmed_EEG.reshape(n_epochs, epoch_length)
    EMG_epochs = trimmed_EMG.reshape(n_epochs, epoch_length)
    features = []
    nperseg = EEG_epochs.shape[1]
    noverlap = 0
    for EEG_epoch,EMG_epoch in zip(EEG_epochs, EMG_epochs):
        EEG_epoch=notch_filter(EEG_epoch, 60, sfreq, quality=30) # AC filter
        freqs, spectra = welch(EEG_epoch, fs=sfreq, nperseg=nperseg, noverlap=noverlap, scaling="spectrum")
        Delta=band_power(freqs, spectra, fmin=0.5, fmax=4)
        Theta=band_power(freqs, spectra, fmin=5, fmax=7)
        Alpha=band_power(freqs, spectra, fmin=8, fmax=13)
        Beta=band_power(freqs, spectra, fmin=14, fmax=23)
        Gamma=band_power(freqs, spectra, fmin=30, fmax=70)
        Full_EEG=band_power(freqs, spectra, fmin=0.5, fmax=30) # Full band power
        EMG_f=notch_filter(EMG_epoch, 60, sfreq, quality=30) # AC filter
        EMG_POW=np.mean(bandpass_filter(EMG_f, 10, 250, sfreq, order=4) **2 )      
        # Append all features
        features.append([Delta, Theta, Alpha, Beta, Gamma, Full_EEG, EMG_POW])
    
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

def smooth_label(preds, min_length=4, target_state=3, replace_with=2):
    smoothed = preds.copy()
    i = 0
    while i < len(smoothed):
        if smoothed[i] == target_state:
            # Find contiguous segment of REM
            j = i
            while j < len(smoothed) and smoothed[j] == target_state:
                j += 1
            # If length of REM bout is less than min_length, set them to NREM
            if (j - i) < min_length:
                smoothed[i:j] = replace_with
            i = j  # Continue from end of the segment
        else:
            i += 1
    return smoothed

def create_sequences(window_size, step, data, labels=None ): # data: [n_samples, n_features], labels: [n_samples]
    X, y = [], []
    max_start = len(data) - window_size + 1 # last start index for a whole window
    for start in range(0, max_start, step): # step through data
        end = start + window_size # end index for current window
        X.append(data[start:end]) # shape [n_windows, window_size, n_features]
        if labels is not None:
            y.append(labels[start:end]) # shape [n_windows, window_size]
    if labels is not None:
        return np.array(X), np.array(y) # shape [n_windows, window_size, n_features], [n_windows, window_size]
    else:
        return np.array(X)

def get_oversampled_indices(Y, rem_label=2, repeat_factor=3):
    """
    Returns a list of indices for oversampling REM-containing sequences.
    
    Arguments:
    - Y: ndarray shape [n_sequences, win_len], your training label matrix.
    - rem_label: int, the REM class label.
    - repeat_factor: int, how many times to upsample REM sequences.
    
    Returns:
    - oversampled_indices: shuffled array of indices (with REM sequences repeated)
    """
    # Find REM-containing sequences
    rem_mask = np.any(Y == rem_label, axis=1)
    rem_indices = np.where(rem_mask)[0]
    non_rem_indices = np.where(~rem_mask)[0]

    # Repeat REM indices only
    repeated_rem_indices = np.tile(rem_indices, repeat_factor)

    # Combine and shuffle
    all_indices = np.concatenate([non_rem_indices, repeated_rem_indices])
    np.random.shuffle(all_indices)

    return all_indices


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-100):
        """
        alpha: class weights (Tensor of shape [n_classes] or None)
        gamma: focusing parameter
        reduction: 'mean', 'sum', or 'none'
        ignore_index: label to ignore in loss computation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        inputs: [B, C] logits
        targets: [B] integer labels
        """
        # Mask out ignored targets
        valid_mask = targets != self.ignore_index
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        if targets.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = probability of the true class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # shape: [N]
        
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


