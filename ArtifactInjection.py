"""
ArtifactInjection.py  —  REST V2.0
═══════════════════════════════════════════════════════════════════════════════
Synthetic artifact augmentation for training the 4-class REST model.

Injects fake artifact patterns into clean STFT epochs and overwrites their
label to Artifact (3).  Operates on the same shape the model consumes:
  X : float32  [win_len, frames=5, feat=130]   (z-scored signal → STFT magnitude)
  Y : int64    [win_len]                        (0=W, 1=N, 2=R, 3=Art, -100=ignore)

Frequency mapping (set by data_process_tensor):
  EEG bins  0..64   →  0.5 Hz/bin   (range 0–32 Hz)
  EMG bins 65..129  →  1.0 Hz/bin   (range 0–64 Hz)

Magnitude scaling is per-window: each generator returns values in "scaled
units" where 1.0 ≈ a typical loud-but-clean STFT bin.  inject_artifacts()
multiplies by the window's 95th-percentile to translate to real STFT units.
"""

import numpy as np

EEG_LO, EEG_HI = 0, 65        # EEG bin slice
EMG_LO, EMG_HI = 65, 130      # EMG bin slice
EEG_DF = 0.5                  # Hz per EEG bin
EMG_DF = 1.0                  # Hz per EMG bin
N_FRAMES = 5                  # frames per 4-s epoch (1.25 Hz frame rate)
N_FEAT   = 130


# ─────────────────────────────────────────────────────────────────────────────
# Per-type generators.  Each returns np.float32 [5, 130] in scaled units.
# ─────────────────────────────────────────────────────────────────────────────

def _motion_burst(rng):
    """Broadband EEG+EMG burst, 1–2 frames, loud."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    center = rng.randint(0, N_FRAMES)
    width  = rng.uniform(0.5, 1.5)
    mag    = rng.uniform(3.0, 8.0)

    t = np.arange(N_FRAMES, dtype=np.float32)
    time_env = mag * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    eeg_profile = (1.0 + 0.4 * np.linspace(0, 1, 65, dtype=np.float32))
    eeg_profile *= rng.uniform(0.7, 1.3, 65).astype(np.float32)
    emg_profile = (1.0 + 0.6 * np.linspace(0, 1, 65, dtype=np.float32))
    emg_profile *= rng.uniform(0.7, 1.3, 65).astype(np.float32)
    emg_profile *= rng.uniform(0.8, 1.4)   # EMG often louder than EEG in motion

    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _chewing(rng):
    """Rhythmic 2–5 Hz EMG with mild EEG bleed at high freq, sustained."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    chew_hz = rng.uniform(2.0, 5.0)
    eeg_mag = rng.uniform(1.0, 3.0)
    emg_mag = rng.uniform(2.0, 8.0)

    eeg_profile = np.zeros(65, dtype=np.float32)
    eeg_profile[40:65] = eeg_mag * rng.uniform(0.4, 1.2, 25).astype(np.float32)

    emg_profile = emg_mag * 0.3 * rng.uniform(0.4, 1.2, 65).astype(np.float32)
    for harm in (1, 2, 3):
        bin_idx = int(round(harm * chew_hz / EMG_DF))
        if bin_idx >= 65:
            break
        for d in range(-2, 3):
            j = bin_idx + d
            if 0 <= j < 65:
                emg_profile[j] += emg_mag * np.exp(-d * d / 2.0) * rng.uniform(0.6, 1.4)

    time_env = (1.0 + 0.3 * rng.uniform(-1, 1, N_FRAMES)).astype(np.float32)

    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _electrode_pop(rng):
    """Sharp DC/low-freq spike, concentrated in lowest EEG bins."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    frame  = rng.randint(0, N_FRAMES)
    mag    = rng.uniform(5.0, 12.0)
    cutoff = rng.randint(3, 7)

    eeg_profile = np.zeros(65, dtype=np.float32)
    eeg_profile[:cutoff] = mag * np.exp(-np.arange(cutoff) * 0.3) \
                              * rng.uniform(0.7, 1.3, cutoff)

    epoch[frame, EEG_LO:EEG_HI] += eeg_profile
    epoch[frame, EMG_LO:EMG_LO + 5] += eeg_profile[:5] * rng.uniform(0.1, 0.4)
    return epoch


def _saturation(rng):
    """Flat-spectrum elevation, brief, both channels."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    center = rng.randint(0, N_FRAMES)
    width  = rng.uniform(0.4, 1.0)
    mag    = rng.uniform(3.0, 7.0)

    t = np.arange(N_FRAMES, dtype=np.float32)
    time_env = mag * np.exp(-((t - center) ** 2) / (2 * width ** 2))

    eeg_profile = rng.uniform(0.8, 1.2, 65).astype(np.float32)
    emg_profile = rng.uniform(0.8, 1.2, 65).astype(np.float32)

    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _emg_bleed(rng):
    """High-freq EEG mirror of EMG envelope (transitional / muscle leak)."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    eeg_mag = rng.uniform(2.0, 6.0)
    emg_mag = rng.uniform(3.0, 10.0)

    time_env = np.cumsum(rng.uniform(0.0, 1.0, N_FRAMES)).astype(np.float32)
    time_env /= max(time_env.max(), 1e-6)
    time_env *= rng.uniform(0.7, 1.3)

    eeg_profile = np.zeros(65, dtype=np.float32)
    eeg_profile[40:65] = eeg_mag * rng.uniform(0.5, 1.3, 25).astype(np.float32)
    emg_profile = emg_mag * rng.uniform(0.7, 1.3, 65).astype(np.float32)

    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _cardiac(rng):
    """Periodic 5–10 Hz pulse (rodent HR), low-amplitude, sustained."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    hr_hz = rng.uniform(5.0, 10.0)
    mag   = rng.uniform(1.0, 3.0)

    eeg_profile = np.zeros(65, dtype=np.float32)
    emg_profile = np.zeros(65, dtype=np.float32)
    for harm in (1, 2):
        eeg_bin = int(round(harm * hr_hz / EEG_DF))
        emg_bin = int(round(harm * hr_hz / EMG_DF))
        for d in range(-1, 2):
            if 0 <= eeg_bin + d < 65:
                eeg_profile[eeg_bin + d] += mag * np.exp(-d * d) * rng.uniform(0.7, 1.3)
            if 0 <= emg_bin + d < 65:
                emg_profile[emg_bin + d] += mag * 0.7 * np.exp(-d * d) * rng.uniform(0.7, 1.3)

    time_env = (1.0 + 0.2 * rng.uniform(-1, 1, N_FRAMES)).astype(np.float32)
    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _respiratory(rng):
    """Slow 1–3 Hz periodic, low amplitude, both channels."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    resp_hz = rng.uniform(1.0, 3.0)
    mag     = rng.uniform(0.5, 2.0)

    eeg_bin = int(round(resp_hz / EEG_DF))
    emg_bin = int(round(resp_hz / EMG_DF))

    eeg_profile = np.zeros(65, dtype=np.float32)
    emg_profile = np.zeros(65, dtype=np.float32)
    for d in range(-1, 2):
        if 0 <= eeg_bin + d < 65:
            eeg_profile[eeg_bin + d] = mag * np.exp(-d * d) * rng.uniform(0.7, 1.3)
        if 0 <= emg_bin + d < 65:
            emg_profile[emg_bin + d] = mag * 0.4 * np.exp(-d * d) * rng.uniform(0.7, 1.3)

    phase = rng.uniform(0, 2 * np.pi)
    time_env = (1.0 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES) + phase)).astype(np.float32)

    epoch[:, EEG_LO:EEG_HI] += time_env[:, None] * eeg_profile[None, :]
    epoch[:, EMG_LO:EMG_HI] += time_env[:, None] * emg_profile[None, :]
    return epoch


def _twitch(rng):
    """Brief sleep-time motion (REM phasic / NREM body movement) — like motion_burst but milder."""
    epoch = np.zeros((N_FRAMES, N_FEAT), dtype=np.float32)
    frame = rng.randint(0, N_FRAMES)
    mag   = rng.uniform(3.0, 8.0)

    eeg_profile = (1.0 + 0.3 * np.linspace(0, 1, 65, dtype=np.float32))
    eeg_profile *= rng.uniform(0.7, 1.3, 65).astype(np.float32)
    emg_profile = (1.0 + 0.5 * np.linspace(0, 1, 65, dtype=np.float32))
    emg_profile *= rng.uniform(0.7, 1.3, 65).astype(np.float32)

    epoch[frame, EEG_LO:EEG_HI] += mag * eeg_profile
    epoch[frame, EMG_LO:EMG_HI] += mag * emg_profile
    return epoch


_GENERATORS = {
    'motion_burst':  _motion_burst,
    'chewing':       _chewing,
    'electrode_pop': _electrode_pop,
    'saturation':    _saturation,
    'emg_bleed':     _emg_bleed,
    'cardiac':       _cardiac,
    'respiratory':   _respiratory,
    'twitch':        _twitch,
}

# Stage-conditional pools.  Wake gets the "loud / muscle-driven" types;
# sleep gets the subtler / physiologically-coupled types plus rare twitches.
WAKE_TYPES  = ('motion_burst', 'chewing', 'electrode_pop', 'saturation', 'emg_bleed')
SLEEP_TYPES = ('electrode_pop', 'saturation', 'cardiac', 'respiratory', 'twitch')


# ─────────────────────────────────────────────────────────────────────────────
# Window-level injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_artifacts(X, Y, inject_p=0.05, wake_share=0.8, rng=None):
    """
    Stamp synthetic artifacts onto a window in-place and relabel those epochs to 3.

    Real artifacts (Y==3) and ignored epochs (Y==-100) are never overwritten.
    Magnitude is auto-scaled by the window's 95th-percentile STFT value so the
    same generator code works regardless of the underlying signal scale.

    Parameters
    ----------
    X : ndarray  [win_len, frames, feat]   STFT window (mutated in place)
    Y : ndarray  [win_len]                  0-based labels (mutated in place)
    inject_p   : float   target post-injection artifact rate (e.g. 0.05 = 5%)
    wake_share : float   fraction of injected epochs placed on Wake
    rng        : np.random.RandomState     (defaults to module global)

    Returns
    -------
    int — number of epochs actually injected.
    """
    if rng is None:
        rng = np.random
    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    win_len = X.shape[0]
    n_target = int(round(inject_p * win_len))
    if n_target <= 0:
        return 0

    wake_idx  = np.flatnonzero(Y == 0)
    sleep_idx = np.flatnonzero((Y == 1) | (Y == 2))

    n_wake  = int(round(n_target * wake_share))
    n_sleep = n_target - n_wake
    n_wake  = min(n_wake,  len(wake_idx))
    n_sleep = min(n_sleep, len(sleep_idx))

    scale = float(np.percentile(np.abs(X), 95))
    if scale < 1e-6:
        scale = 1.0

    n_done = 0

    if n_wake > 0:
        chosen = rng.choice(wake_idx, size=n_wake, replace=False)
        for i in chosen:
            kind = WAKE_TYPES[rng.randint(len(WAKE_TYPES))]
            X[i] += scale * _GENERATORS[kind](rng)
            Y[i] = 3
        n_done += n_wake

    if n_sleep > 0:
        chosen = rng.choice(sleep_idx, size=n_sleep, replace=False)
        for i in chosen:
            kind = SLEEP_TYPES[rng.randint(len(SLEEP_TYPES))]
            X[i] += scale * _GENERATORS[kind](rng)
            Y[i] = 3
        n_done += n_sleep

    return n_done
