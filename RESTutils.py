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
        EMG_POW=np.mean(bandpass_filter(EMG_f, 5, 50, sfreq, order=4) **2 )
        
        # Append all features
        features.append([Delta, Theta, Alpha, Beta, Gamma, Full_EEG, EMG_POW])
    
    return np.array(features)  # Shape: [n_epochs, 8]

def band_power(freqs, spectra, fmin=0.5, fmax=4):
    """Extract delta power from spectra matrix."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    df = np.mean(np.diff(freqs))       # average frequency bin width (Hz)
    power = spectra[mask].sum() * df
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
        sigma=band_power(freqs, spectra, fmin=24, fmax=29)
        Gamma=band_power(freqs, spectra, fmin=30, fmax=70)
        Full_EEG=band_power(freqs, spectra, fmin=0.5, fmax=30) # Full band power
        EMG_f=notch_filter(EMG_epoch, 60, sfreq, quality=30) # AC filter
        EMG_POW=np.mean(bandpass_filter(EMG_f, 5, 50, sfreq, order=4) **2 )      
        # Append all features
        features.append([Delta, Theta, Alpha, Beta,sigma, Gamma, Full_EEG, EMG_POW])
    
    return np.array(features)  # Shape: [n_epochs, 8]

def data_process(EEG,EMG,fs=512): #STFT for classification, input should be raw EEG and EMG signals
    ## downsample, filter, and normaliztion
    EEG = np.asarray(EEG).flatten() # flatten the signal
    EEG=bandpass_filter(EEG, 0.1, 30, fs,4) # bandpass filter
    EEG=resample_poly(EEG, up=1, down=8) # resample the signal, 64hz, 266 samples per epoch
    EEG = (EEG - np.mean(EEG)) / np.std(EEG) # normalize the signal
    
    EMG = np.asarray(EMG).flatten() # flatten the signal
    # EMG=bandpass_filter(EMG, 10, 250, fs,4) # bandpass filter
    EMG=bandpass_filter(EMG, 5, 50, fs,4) # bandpass filter
    EMG=notch_filter(EMG, 60, fs, 30) # notch filter
    EMG = (EMG - np.mean(EMG)) / np.std(EMG) # normalize the signal
    
    ## reshape the signal to epochs
    n_epochs = len(EEG) // 256
    EEG = EEG[:n_epochs * 256] # truncate to full epochs
    EEG=EEG.reshape(-1, 256) # reshape to epochs
    # EMG: downsample 512 Hz → 128 Hz (bandpass already limits to 5-50 Hz, anti-aliasing safe)
    EMG = resample_poly(EMG, up=1, down=4)  # 512 Hz → 128 Hz
    n_epochs = len(EMG) // 512
    EMG = EMG[:n_epochs * 512] # truncate to full epochs
    EMG=EMG.reshape(-1, 512) # reshape to epochs (128 Hz × 4 s = 512 samples)

    # Align epoch counts — EEG (÷8) and EMG (÷4) can differ by 1 due to integer division
    n_epochs = min(len(EEG), len(EMG))
    EEG = EEG[:n_epochs]
    EMG = EMG[:n_epochs]
    
    ## STFT
    EEG_fs,  EEG_nperseg  =  64, 128        # 0.5 Hz resolution  •  bins: 65 (0-32 Hz)
    EMG_fs,  EMG_nperseg  = 128, 256        # 0.5 Hz resolution  •  bins: 129 (0-64 Hz)

    def epoch_to_spectrogram(epoch, fs, nperseg):
        _, _, Zxx = stft(epoch, fs=fs, nperseg=nperseg, noverlap=nperseg//2, padded=False)
        return np.abs(Zxx).T                 # → [frames, freq_bins]

    EEG_STFT = np.stack([epoch_to_spectrogram(ep, EEG_fs, EEG_nperseg) for ep in EEG])
    EMG_STFT = np.stack([epoch_to_spectrogram(ep, EMG_fs, EMG_nperseg) for ep in EMG])

    EMG_STFT = resample(EMG_STFT, EEG_STFT.shape[-1], axis=-1)
    return EEG_STFT, EMG_STFT

def fft_bandpass_filter(data_tensor, lowcut, highcut, fs):
    """
    Lightning-fast FFT-based bandpass filter on GPU.
    data_tensor: [1, L]
    """
    N = data_tensor.shape[-1]
    freqs = torch.fft.rfftfreq(N, 1/fs).to(data_tensor.device)
    mask = (freqs >= lowcut) & (freqs <= highcut)
    fft_data = torch.fft.rfft(data_tensor)
    fft_data = fft_data * mask.unsqueeze(0)
    filtered = torch.fft.irfft(fft_data, n=N)
    return filtered

def compute_powers_welch_tensor(EEG, EMG, sfreq=512, s=4, device='cuda'):
    """
    GPU-accelerated Welch's method.
    EEG, EMG: 1D numpy arrays or tensors of raw data
    """
    if not isinstance(EEG, torch.Tensor):
        EEG = torch.tensor(EEG, dtype=torch.float32, device=device).view(1, -1)
        EMG = torch.tensor(EMG, dtype=torch.float32, device=device).view(1, -1)
    else:
        if EEG.dim() == 1:
            EEG = EEG.unsqueeze(0)
            EMG = EMG.unsqueeze(0)
        
    epoch_length = int(sfreq * s)
    n_epochs = EEG.shape[-1] // epoch_length
    
    # We must explicitly slice using the final calculated sequence length to reshape properly
    valid_len = n_epochs * epoch_length
    EEG = EEG[:, :valid_len]
    EMG = EMG[:, :valid_len]
    
    # Force the memory array to be contiguous if it's sliced, otherwise view might throw error on reshape
    EEG_epochs = EEG.contiguous().view(n_epochs, epoch_length)
    EMG_epochs = EMG.contiguous().view(n_epochs, epoch_length)
    
    nperseg = epoch_length
    window = torch.hann_window(nperseg, device=device)
    
    EEG_stft_complex = torch.stft(EEG_epochs, n_fft=nperseg, hop_length=nperseg, window=window, return_complex=True, center=False)
    Sxx = (EEG_stft_complex.abs() ** 2) / (window.sum() ** 2)
    Sxx[:, 1:-1, :] *= 2.0
    EEG_spectra = Sxx.mean(dim=-1)
    
    freqs = torch.fft.rfftfreq(nperseg, 1/sfreq).to(device)
    df = freqs[1] - freqs[0]
    
    def band_power_tensor(spectra, f_min, f_max):
        mask = (freqs >= f_min) & (freqs <= f_max)
        return spectra[:, mask].sum(dim=1) * df

    Delta = band_power_tensor(EEG_spectra, 0.5, 4)
    Theta = band_power_tensor(EEG_spectra, 5, 7)
    Alpha = band_power_tensor(EEG_spectra, 8, 13)
    Beta  = band_power_tensor(EEG_spectra, 14, 23)
    Sigma = band_power_tensor(EEG_spectra, 24, 29)
    Gamma = band_power_tensor(EEG_spectra, 30, 70)
    Full_EEG = band_power_tensor(EEG_spectra, 0.5, 30)
    
    EMG_f = fft_bandpass_filter(EMG_epochs, 5, 50, sfreq)
    EMG_POW = (EMG_f ** 2).mean(dim=1)
    
    features = torch.stack([Delta, Theta, Alpha, Beta, Sigma, Gamma, Full_EEG, EMG_POW], dim=1)
    return features.cpu().numpy()

def data_process_tensor(EEG, EMG, fs=512, device='cuda'):
    """
    GPU-accelerated STFT feature extraction for the Neural Network.
    """
    if not isinstance(EEG, torch.Tensor):
        EEG = torch.tensor(EEG, dtype=torch.float32, device=device).view(1, -1)
        EMG = torch.tensor(EMG, dtype=torch.float32, device=device).view(1, -1)
    else:
        if EEG.dim() == 1:
            EEG = EEG.unsqueeze(0)
            EMG = EMG.unsqueeze(0)
        
    EEG = fft_bandpass_filter(EEG, 0.1, 30, fs)
    EEG = F.interpolate(EEG.unsqueeze(0), scale_factor=1/8, mode='linear', align_corners=False).squeeze(0).squeeze(0)
    EEG = (EEG - EEG.mean()) / EEG.std()
    
    EMG = fft_bandpass_filter(EMG, 5, 50, fs)
    EMG = F.interpolate(EMG.unsqueeze(0), scale_factor=1/4, mode='linear', align_corners=False).squeeze(0).squeeze(0)
    EMG = (EMG - EMG.mean()) / EMG.std()
    
    eeg_epoch_len = 256 # 64 Hz * 4 sec
    emg_epoch_len = 512 # 128 Hz * 4 sec
    
    n_epochs_eeg = len(EEG) // eeg_epoch_len
    n_epochs_emg = len(EMG) // emg_epoch_len
    n_epochs = min(n_epochs_eeg, n_epochs_emg)
    
    EEG = EEG[:n_epochs * eeg_epoch_len].reshape(n_epochs, eeg_epoch_len)
    EMG = EMG[:n_epochs * emg_epoch_len].reshape(n_epochs, emg_epoch_len)
    
    EEG_fs, EEG_nperseg = 64, 128
    EMG_fs, EMG_nperseg = 128, 256
    
    def get_stft_tensor(epochs, nperseg):
        window = torch.hann_window(nperseg, device=device)
        pad_len = nperseg // 2
        epochs_padded = F.pad(epochs, (pad_len, pad_len), mode='constant', value=0.0)
        Zxx = torch.stft(epochs_padded, n_fft=nperseg, hop_length=nperseg//2, window=window, return_complex=True, center=False)
        return (Zxx.abs() / (nperseg / 2)).transpose(1, 2)

    EEG_STFT = get_stft_tensor(EEG, EEG_nperseg) 
    EMG_STFT = get_stft_tensor(EMG, EMG_nperseg) 
    
    B, F_T, Freqs = EMG_STFT.shape
    EMG_STFT_flat = EMG_STFT.reshape(B*F_T, 1, Freqs)
    EMG_STFT_flat = F.interpolate(EMG_STFT_flat, size=EEG_STFT.shape[-1], mode='linear', align_corners=False)
    EMG_STFT = EMG_STFT_flat.reshape(B, F_T, EEG_STFT.shape[-1])
    
    return EEG_STFT.cpu().numpy(), EMG_STFT.cpu().numpy()

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

def sliding_window_inference(
    model,
    epochs,
    win_len,
    step,
    device,
    batch_size=16,
):
    """
    Run REST v2 on a single recording of shape [N, 2, L] using sliding windows.

    Inputs:
        model   : REST model (no boundary head, returns logits only)
        epochs  : [N, 2, L] float32 (EEG, EMG per epoch)
        win_len : window length in epochs (e.g. 900)
        step    : step size in epochs (e.g. 150)
        device  : torch.device
        batch_size : how many windows per forward pass

    Returns:
        avg_logits: [N, C]  (per-epoch averaged logits across all windows)
    """
    model.eval()
    N, C_in, L = epochs.shape
    assert C_in == 2, "Expected 2 channels (EEG, EMG)"

    # Build window start indices
    starts = list(range(0, max(0, N - win_len), step))
    if len(starts) == 0:  # recording shorter than window
        starts = [0]
        win_len = N

    n_windows = len(starts)

    # Number of classes from final FC layer
    n_classes = model.fc[-1].out_features

    # Accumulators
    logit_sum   = np.zeros((N, n_classes), dtype=np.float32)
    logit_count = np.zeros(N, dtype=np.int32)

    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            b_starts = starts[i:i + batch_size]
            B = len(b_starts)

            # Build batch [B, W, 2, L]
            max_W = win_len
            batch = np.zeros((B, max_W, 2, L), dtype=np.float32)
            valid_W = []

            for bi, s in enumerate(b_starts):
                e = min(s + win_len, N)
                wlen = e - s
                valid_W.append(wlen)
                batch[bi, :wlen] = epochs[s:e]

            batch_t = torch.from_numpy(batch).to(device)

            # Model output: logits only [B, W, C]
            logits_t = model(batch_t)
            logits_np = logits_t.cpu().numpy()  # [B, W, C]

            # Accumulate per-epoch
            for bi, s in enumerate(b_starts):
                wlen = valid_W[bi]
                e = s + wlen

                logit_sum[s:e]   += logits_np[bi, :wlen]
                logit_count[s:e] += 1

    # Average logits across all windows that covered each epoch
    avg_logits = np.zeros_like(logit_sum)
    mask = logit_count > 0
    avg_logits[mask] = logit_sum[mask] / logit_count[mask, None]

    return avg_logits

def postprocess_hypnogram(
    avg_logits,
    median_kernel=5,
    entropy_high=None,
):
    """
    Post-processing pipeline:
      1) Softmax → raw predictions
      2) Confidence/entropy fix for 1-epoch islands
      3) Boundary-based suppression / permission of transitions (optional)
      4) Median filter

    Inputs:
        avg_logits:     [N, C] float32
        avg_boundaries:[N] float32 (logits), optional
    Returns:
        final_pred: [N] int
    """

    N, C = avg_logits.shape

    # --- 1) softmax + raw predictions ---
    probs = torch.softmax(torch.from_numpy(avg_logits), dim=-1).numpy()
    pred = np.argmax(probs, axis=-1)

    # entropy per epoch
    ent = -np.sum(probs * np.log(probs + 1e-8), axis=-1)

    if entropy_high is None:
        entropy_high = np.percentile(ent, 70)

    # --- 2) confidence-based 1-epoch island fix ---
    for i in range(1, N - 1):
        if ent[i] > entropy_high:
            if pred[i-1] == pred[i+1] and pred[i] != pred[i-1]:
                pred[i] = pred[i-1]

    # --- 3) median filter ---
    if median_kernel is not None and median_kernel > 1:
        if median_kernel % 2 == 0:
            median_kernel += 1
        pred = median_filter(pred, kernel_size=median_kernel)

    return pred.astype(np.int64)

def apply_artifact_only_on_wake(pred, artifact_mask, wake_label=1, artifact_label=0):
    pred2 = pred.copy()
    pred2[(pred2 == wake_label) & artifact_mask] = artifact_label
    return pred2

def detect_artifact_eeg_V2(
    eeg_epochs,                 # [N, L]
    Thr=8              # raise to be less aggressive
):
    eeg = eeg_epochs.astype(np.float32)
    scale = np.median(np.abs(eeg_epochs)) / 0.6745  # robust std
    eeg = eeg / (scale + 1e-8)

    # Feature 1: peak-to-peak
    ptp = eeg.max(axis=1) - eeg.min(axis=1)

    # Feature 2: sharpness / pops
    diff_rms = np.sqrt(np.mean(np.diff(eeg, axis=1)**2, axis=1))
    rms        = np.sqrt(np.mean(eeg**2, axis=1))
    art = ((ptp > np.mean(ptp)*Thr) | (diff_rms > np.mean(diff_rms)*Thr) | (rms > np.mean(rms)*Thr))

    return art

def sliding_window_logits_average(model, epochs_np, win_len, step, device, batch_size=32):
    """
    epochs_np: [N, frames, feat] float32
    returns:   logits_avg [N, n_classes] float32
    """
    model.eval()
    N = epochs_np.shape[0]

    # window start indices; ensure tail coverage
    starts = list(range(0, max(N - win_len + 1, 1), step))
    if N >= win_len and (len(starts) == 0 or starts[-1] != N - win_len):
        starts.append(N - win_len)
    if N < win_len:
        # pad by repeating last epoch so we can still run one window
        pad = win_len - N
        epochs_pad = np.concatenate([epochs_np, np.repeat(epochs_np[-1:], pad, axis=0)], axis=0)
        starts = [0]
    else:
        epochs_pad = epochs_np

    # infer n_classes from a tiny forward pass
    with torch.no_grad():
        tmp = torch.from_numpy(epochs_pad[starts[0]:starts[0]+win_len]).unsqueeze(0).to(device)  # [1,W,F,C]
        n_classes = model(tmp).shape[-1]

    logits_sum = np.zeros((N, n_classes), dtype=np.float32)
    counts     = np.zeros((N,), dtype=np.float32)

    batch_starts = []
    batch_windows = []

    def flush_batch():
        nonlocal batch_starts, batch_windows, logits_sum, counts
        if not batch_windows:
            return
        x = np.stack(batch_windows, axis=0)  # [B, W, F, C]
        x = torch.from_numpy(x).to(device)
        with torch.no_grad():
            out = model(x)  # [B, W, K]
        out = out.detach().cpu().numpy()

        for b, s in enumerate(batch_starts):
            # map window positions back to epoch indices (clipped to original N)
            for i in range(win_len):
                idx = s + i
                if idx >= N:
                    break
                logits_sum[idx] += out[b, i]
                counts[idx]     += 1.0

        batch_starts = []
        batch_windows = []

    for s in starts:
        w = epochs_pad[s:s+win_len]  # [W,F,C]
        batch_starts.append(s)
        batch_windows.append(w.astype(np.float32))
        if len(batch_windows) >= batch_size:
            flush_batch()

    flush_batch()

    # avoid divide by zero (shouldn't happen with tail coverage, but safe)
    counts[counts == 0] = 1.0
    logits_avg = logits_sum / counts[:, None]
    return logits_avg

def viterbi_smooth(probs, transition_matrix=None):
    """
    Apply Viterbi algorithm to find the most likely sequence of states.
    
    Parameters:
    - probs: [N, n_classes] array of probabilities (from softmax)
    - transition_matrix: [n_classes, n_classes] transition probabilities (A_ij = P(j|i))
    
    Returns:
    - path: [N] array of integer states
    """
    N, n_classes = probs.shape
    
    # Default transition matrix (Wake, NREM, REM, Art) if None
    # 0=Wake, 1=NREM, 2=REM, 3=Art
    if transition_matrix is None:
        if n_classes == 3:
            # 3-State Model: Wake, NREM, REM
            # W -> W (high), N (med), R (low)
            # N -> W (low), N (high), R (med)
            # R -> W (low), N (low), R (high)
            A = np.array([
                [100, 20, 1], # Wake -> W, N, R
                [10, 100, 10], # NREM -> W, N, R
                [10, 5, 100], # REM -> W, N, R
            ])
        else:
            # Default 4-State: Wake, NREM, REM, Art
            # 0=Wake, 1=NREM, 2=REM, 3=Art
            # Real motion artifacts often span 2-5 consecutive epochs.
            # Previous Art self=2 made artifact transient → isolated predictions
            # got dissolved by smoothing.  Bumping Art self to 30 (P_stay≈0.60)
            # gives expected cluster length ≈2.5 epochs while still allowing
            # short artifact bursts to be filtered out as noise.
            A = np.array([
                [100, 20, 1, 8],   # Wake -> W, N, R, A
                [10, 100, 10, 3],  # NREM -> W, N, R, A
                [20, 5, 100, 3],   # REM  -> W, N, R, A
                [10, 5, 5, 30],    # Art  -> W, N, R, A  (P_stay 0.60)
            ])
        
        # Normalize rows
        A = A / A.sum(axis=1, keepdims=True)
        transition_matrix = np.log(A + 1e-10) # Log-probability for numerical stability
    else:
        transition_matrix = np.log(transition_matrix + 1e-10)
        
    # Initial state probabilities (assume equal or Wake-heavy)
    # Log probabilities
    dp = np.zeros((N, n_classes))
    path = np.zeros((N, n_classes), dtype=int)
    
    # Init first step
    dp[0] = np.log(probs[0] + 1e-10)
    
    # Forward pass
    for t in range(1, N):
        # Emission log-prob for this step
        emission = np.log(probs[t] + 1e-10)
        
        # for each next state j
        # dp[t, j] = max_i (dp[t-1, i] + A[i, j]) + emission[j]
        # Vectorized:
        # dp[t-1] is [C], A is [C, C]. 
        # (dp[t-1][:, None] + A) is [C, C] where (i,j) is previous i transition to j
        
        scores = dp[t-1][:, None] + transition_matrix
        best_prev = np.argmax(scores, axis=0)
        path[t] = best_prev
        dp[t] = scores[best_prev, np.arange(n_classes)] + emission
        
    # Backward pass (Backtracking)
    best_path = np.zeros(N, dtype=int)
    best_path[-1] = np.argmax(dp[-1])
    
    for t in range(N-2, -1, -1):
        best_path[t] = path[t+1, best_path[t+1]]
        
    return best_path