"""
ArtifactFilter.py
=================
Rule-based EEG artifact detector. No torch/MNE dependencies — numpy + scipy only.

Key insight: artifact only matters when the EEG itself is corrupted. EMG activity
alone (active Wake) is not artifact. Primary rules operate on the EEG channel;
an optional EMG rule (high EMG beyond even typical Wake activity) can be added.

Algorithm (empirically validated against 5 mouse datasets, 690k epochs):
  Three-rule OR design:

  1. SMOOTH-PTP (primary): within-recording PTP percentile rank converted to a soft
     score, convolved over a short temporal window, then thresholded.
     - Adapts to each recording's own amplitude baseline (no global threshold needed).
     - Temporal smoothing requires the elevation to be somewhat sustained, which
       separates clustered artifact bursts from isolated Wake movement peaks.
     - Works well for CD1 (F1≈0.84), DBAKA (F1≈0.61), DBASA (F1≈0.38).

  2. SATURATION (secondary): fraction of samples at ≥99% of recording's abs-max.
     - 97% precision on labeled data; catches the most obvious clipping events.
     - Adds recall for saturation artifacts that are too brief to cluster.

  3. EMG-ELEVATION (optional): EMG RMS within-recording percentile rank, thresholded.
     - EMG during artifacts is 2–10x above typical Wake EMG (per-dataset analysis).
     - Uses within-recording percentile rank so it adapts to each mouse's muscle tone.
     - Helps C57SA where EEG amplitude alone barely separates artifacts from Wake.
     - Only active when EMG_1d is provided to flag_artifacts / compute_features.

Performance summary (5 datasets, default parameters, EEG-only):
  ALL F1=0.270  R=0.183  P=0.512  W-flag=0.05%
  CD1 F1=0.810  CD1 is highly separable (very extreme PTP amplitudes).
  C57SA F1≈0.02 — artifacts there are near-normal EEG amplitude; EMG rule helps.

Usage
-----
from ArtifactFilter import flag_artifacts, compute_features, DEFAULT_PARAMS

# Quick use — EEG only — returns bool[n_epochs], one per 4-second epoch
mask = flag_artifacts(EEG_1d_volts, fs=512, epoch_seconds=4)

# With EMG channel — enables the EMG-elevation rule
mask = flag_artifacts(EEG_1d_volts, EMG_1d_volts=EMG_1d, fs=512, epoch_seconds=4)

# With soft scores for inspection
mask, soft = flag_artifacts(EEG_1d_volts, return_scores=True)
# soft is float[n_epochs]: the smoothed PTP score before thresholding

Tuned parameters are loaded from artifact_params.json if that file exists next
to this module (written by tune_artifact_filter.py after tuning).
"""

import json
import os
import numpy as np
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_PARAMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "artifact_params.json")

DEFAULT_PARAMS = {
    # Primary rule: temporal smooth of within-recording PTP percentile score
    "ptp_percentile":  99.0,  # reference percentile within each recording
    "smooth_window":    5,    # epoch window for temporal convolution (5 × 4s = 20s)
    "score_threshold":  0.4,  # smoothed score threshold to flag (tuned on 5 datasets)

    # Secondary rule: saturation (high precision, catches obvious clipping)
    "saturation_threshold": 0.10,  # fraction of samples at ≥99% of recording abs-max

    # Tertiary rule: EMG elevation (optional — only used when EMG is provided)
    # Within-recording EMG RMS percentile threshold. Epochs with EMG RMS at or above
    # this percentile of the recording are flagged. Only OR'd in when emg_rms is present.
    "emg_percentile": 99.5,  # within-recording EMG percentile threshold
}


def _load_params():
    if os.path.exists(_PARAMS_FILE):
        with open(_PARAMS_FILE) as f:
            saved = json.load(f)
        merged = dict(DEFAULT_PARAMS)
        merged.update(saved)
        return merged
    return dict(DEFAULT_PARAMS)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(EEG_1d, EMG_1d=None, fs=512, epoch_seconds=4):
    """
    Compute per-epoch signal-quality features for one EEG (and optionally EMG) recording.

    Parameters
    ----------
    EEG_1d : 1D array-like
        Raw EEG in **volts** (matches mne raw.get_data() output).
        Internally converted to µV.
    EMG_1d : 1D array-like or None
        Raw EMG in **volts**. If provided, emg_rms and emg_pct are included in output.
    fs : int
        Sampling frequency in Hz.
    epoch_seconds : float
        Epoch duration in seconds.

    Returns
    -------
    dict with:
        ptp_raw     — peak-to-peak amplitude in µV per epoch
        abs_uv      — max |EEG| in µV per epoch
        saturation  — fraction of samples at ≥99% of recording abs-max
        ptp_pct     — within-recording percentile rank of ptp_raw (0–100)
        emg_rms     — (if EMG_1d provided) RMS amplitude in µV per epoch
        emg_pct     — (if EMG_1d provided) within-recording percentile rank (0–100)
    """
    eeg_uv = np.asarray(EEG_1d, dtype=np.float64).ravel() * 1e6
    epoch_samples = int(fs * epoch_seconds)
    n_epochs = len(eeg_uv) // epoch_samples

    _empty = np.zeros(0, dtype=np.float64)
    if n_epochs == 0:
        result = {k: _empty for k in ("ptp_raw", "abs_uv", "saturation", "ptp_pct")}
        if EMG_1d is not None:
            result["emg_rms"] = _empty
            result["emg_pct"] = _empty
        return result

    epochs = eeg_uv[:n_epochs * epoch_samples].reshape(n_epochs, epoch_samples)

    ptp_raw = epochs.max(axis=1) - epochs.min(axis=1)
    abs_uv  = np.max(np.abs(epochs), axis=1)

    # Saturation: fraction of samples within 1% of the recording's absolute peak.
    rec_absmax  = np.max(np.abs(eeg_uv))
    rail_thresh = rec_absmax * 0.99
    saturation  = np.mean(np.abs(epochs) >= rail_thresh, axis=1)

    # Within-recording percentile rank of PTP (0=lowest, 100=highest).
    order    = np.argsort(ptp_raw)
    ptp_pct  = np.empty_like(ptp_raw)
    ptp_pct[order] = np.linspace(0, 100, n_epochs)

    result = {
        "ptp_raw":    ptp_raw,
        "abs_uv":     abs_uv,
        "saturation": saturation,
        "ptp_pct":    ptp_pct,
    }

    if EMG_1d is not None:
        emg_uv = np.asarray(EMG_1d, dtype=np.float64).ravel() * 1e6
        n_emg  = min(n_epochs, len(emg_uv) // epoch_samples)
        if n_emg > 0:
            emg_ep  = emg_uv[:n_emg * epoch_samples].reshape(n_emg, epoch_samples)
            emg_rms = np.sqrt(np.mean(emg_ep ** 2, axis=1))
            # Pad with zeros if EMG shorter than EEG
            if n_emg < n_epochs:
                emg_rms = np.concatenate([emg_rms, np.zeros(n_epochs - n_emg)])
        else:
            emg_rms = np.zeros(n_epochs, dtype=np.float64)

        order_e  = np.argsort(emg_rms)
        emg_pct  = np.empty_like(emg_rms)
        emg_pct[order_e] = np.linspace(0, 100, n_epochs)

        result["emg_rms"] = emg_rms
        result["emg_pct"] = emg_pct

    return result


# ---------------------------------------------------------------------------
# Soft score computation (exposed for inspection and tuning)
# ---------------------------------------------------------------------------

def smooth_ptp_score(ptp_raw, ptp_percentile=99.0, smooth_window=5):
    """
    Convert a per-epoch PTP array (µV) into a temporally smoothed artifact score.

    Returns float[n_epochs]:
        - 0 for epochs at or below the ptp_percentile threshold.
        - Positive and rising for epochs above threshold.
        - Smoothed over smooth_window consecutive epochs.

    Epochs that are locally extreme AND sustained get a high score;
    isolated single spikes are diluted by the smoothing.
    """
    ref = np.percentile(ptp_raw, ptp_percentile)
    soft = np.clip((ptp_raw - ref) / (ref + 1e-9), 0.0, 2.0)
    return uniform_filter1d(soft.astype(np.float64), size=int(smooth_window))


# ---------------------------------------------------------------------------
# Artifact flagging
# ---------------------------------------------------------------------------

def flag_artifacts(EEG_1d, EMG_1d=None, fs=512, epoch_seconds=4,
                   params=None, return_scores=False):
    """
    Flag epochs as artifact.

    Three rules (OR logic):
      1. Smooth-PTP: within-recording PTP percentile → temporal soft score ≥ threshold
      2. Saturation: fraction of samples at ADC rail ≥ threshold
      3. EMG-elevation (only if EMG_1d provided): within-recording EMG RMS percentile ≥ threshold

    Parameters
    ----------
    EEG_1d : 1D array-like
        Raw EEG in **volts** (from mne raw.get_data()).
    EMG_1d : 1D array-like or None
        Raw EMG in **volts**. If provided, enables the EMG-elevation rule.
    fs : int
    epoch_seconds : float
    params : dict or None
        Keys from DEFAULT_PARAMS. If None, loads artifact_params.json if present,
        else uses DEFAULT_PARAMS.
    return_scores : bool
        If True, also return the raw soft score array (before thresholding).

    Returns
    -------
    flagged : bool[n_epochs]
    scores : float[n_epochs]  (only if return_scores=True)
    """
    if params is None:
        params = _load_params()

    feats  = compute_features(EEG_1d, EMG_1d=EMG_1d, fs=fs, epoch_seconds=epoch_seconds)
    n      = len(feats["ptp_raw"])

    scores = smooth_ptp_score(
        feats["ptp_raw"],
        ptp_percentile = params["ptp_percentile"],
        smooth_window  = params["smooth_window"],
    )

    rule_ptp = scores >= params["score_threshold"]
    rule_sat = feats["saturation"] >= params["saturation_threshold"]
    flagged  = rule_ptp | rule_sat

    if "emg_pct" in feats and "emg_percentile" in params:
        rule_emg = feats["emg_pct"] >= params["emg_percentile"]
        flagged  = flagged | rule_emg

    if return_scores:
        return flagged, scores
    return flagged
