"""
Data_compile.py  —  REST V2.0
═══════════════════════════════════════════════════════════════════════════════
Preprocesses raw EDF + score files into per-recording .npy files for lazy-
loading during training.  This replaces the old approach of loading everything
into RAM at once and saving a single monolithic .npz.

Output layout (all files go to OUTPUT_DIR):
  <recording_name>.npy         float32  [n_epochs, frames*feat]   (flattened STFT)
  <recording_name>_score.npy   int32    [n_epochs]                (per-recording labels)

Score encoding (stored):   1=Wake  2=NREM  3=REM  4=Artefact
  -100 = ignored (unscored or out-of-range)

Usage:
    python Data_compile.py
    (idempotent — skips recordings whose .npy + _score.npy already exist)
"""

import os
import gc
import re
import time
import numpy as np
import mne
import pandas as pd
import torch
from RESTutils import find_data_start, data_process_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration  —  edit these paths as needed
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = r"D:\Training data Extended"

# Add training data folders here (non-recursive — only the immediate folder is scanned).
# Each EDF is matched to the best .txt (RM format) or .tsv (Sirenia format) score file
# in the same folder using fuzzy name matching (handles spaces, _reduced suffixes, etc.).
TRAINING_DIRS = [r"M:\Alex\Python\Mouse Sleep Data\fmr1\5mo",
                 r"M:\Alex\Python\Mouse Sleep Data\fmr1\Adult",
                 r"M:\Alex\Python\Mouse Sleep Data\Additional_ELI",
                 r"M:\Alex\Python\Mouse Sleep Data\Additional_RM",
                 r"M:\Alex\Python\Mouse Sleep Data\Additional_TBI"
]

FS = 512  # native EDF sampling rate

# Trailing suffixes stripped from file stems before matching.
# Covers both EDF-side (_data, _reduced, …) and score-side (_scores) variants.
_EDF_SUFFIXES = ('_data', '_scores', '_reduced', '_export', '_screenprint_export', '_screenprint')


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _with_retry(label, fn, *args, retries=3, delay=5, **kwargs):
    """Retry a network-drive I/O call on transient failure.

    Catches FileNotFoundError / OSError / PermissionError, waits `delay`
    seconds, and retries up to `retries` times.  Re-raises on final failure.
    """
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except (FileNotFoundError, OSError, IOError, PermissionError) as e:
            if attempt < retries - 1:
                print(f"  [retry] {label} failed (attempt {attempt+1}/{retries}): {e}. "
                      f"Waiting {delay}s ...")
                time.sleep(delay)
            else:
                raise


def _load_rm_score(score_path):
    """Read RM .txt score file.  Returns int32 array (1=W, 2=N, 3=R, 4=Art, -100=ignore).

    Handles three formats:
      - Tab-separated (standard RM export)
      - Comma-separated (standard CSV)
      - Outer-quoted CSV: each row wrapped in double-quotes — "col1,col2,..."
    """
    with open(score_path, 'r', errors='replace') as fh:
        lines = fh.readlines()

    non_empty = [l.rstrip('\n') for l in lines if l.strip()]
    first = non_empty[0] if non_empty else ''

    # Detect outer-quoted format: every row is "col1,col2,...,colN"
    if first.startswith('"') and first.endswith('"') and '\t' not in first:
        non_empty = [l.strip('"') for l in non_empty]

    sep = '\t' if '\t' in non_empty[0] else ','
    from io import StringIO
    df = pd.read_csv(StringIO('\n'.join(non_empty)), delimiter=sep)
    score = df.iloc[:, 3].to_numpy().astype(np.int32)
    score[score > 3] = -100   # unscored
    score[score == 0] = 4     # artefact
    return score


def _load_bout_tsv_score(tsv_path):
    """Read Sirenia bout-summary .tsv (one row per bout, not per epoch).

    Columns: Start Epoch | Start Time | Score Number | Score Name | Length (s) | ...
    Score numbers: 0=Artifact→4, 1=Wake→1, 2=NREM→2, 3=REM→3.
    Expands bouts into a per-epoch int32 array (-100 = gap/unscored).
    """
    label = os.path.basename(tsv_path)
    with open(tsv_path, 'r', errors='replace') as fh:
        lines = fh.readlines()

    header_idx = next((i for i, l in enumerate(lines)
                       if l.strip().startswith('Start Epoch')), None)
    if header_idx is None:
        raise ValueError(f"Cannot find 'Start Epoch' header in {label}")

    rows = []
    for line in lines[header_idx + 1:]:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        try:
            rows.append((int(parts[0]), int(parts[2]), float(parts[4])))
        except ValueError:
            continue

    if not rows:
        raise ValueError(f"No bout data found in {label}")

    last_start, _, last_len = rows[-1]
    total_epochs = last_start + int(round(last_len / 4))
    score = np.full(total_epochs, -100, dtype=np.int32)

    for start_ep, score_num, length_sec in rows:
        n_ep = int(round(length_sec / 4))
        mapped = 4 if score_num == 0 else score_num   # 0=Artifact→4
        score[start_ep: start_ep + n_ep] = mapped

    return score


def _load_tsv_score(tsv_path):
    """Read Sirenia .tsv score file (auto-detects epoch-by-epoch vs bout-summary).
    Returns int32 array (1=W, 2=N, 3=R, 4=Art, -100=ignore)."""
    label = os.path.basename(tsv_path)

    # Peek at the first 60 lines to detect the bout-summary format
    try:
        with open(tsv_path, 'r', errors='replace') as fh:
            for _ in range(60):
                line = fh.readline()
                if not line:
                    break
                if line.strip().startswith('Start Epoch'):
                    return _load_bout_tsv_score(tsv_path)
    except Exception:
        pass

    # Epoch-by-epoch format
    start_line = _with_retry(f"scan {label}",
                             find_data_start, tsv_path, sep='\t', expected_columns=5)
    df = _with_retry(f"read {label}",
                     pd.read_csv, tsv_path, sep='\t',
                     skiprows=start_line + 1, header=None)
    score = df.iloc[:, 4].to_numpy().astype(np.int32)
    score[score > 3] = -100   # unscored / out-of-range
    score[score == 0] = 4     # artefact
    return score


def _process_edf(edf_path):
    """Read EDF → return (EEG_STFT, EMG_STFT) via data_process_tensor()."""
    mne.set_log_level('ERROR')
    for attempt in range(3):
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            break
        except Exception as e:
            if attempt < 2:
                print(f"  Failed to open (attempt {attempt+1}/3): {e}. Waiting 5 s ...")
                time.sleep(5)
            else:
                raise

    ch_names = raw.ch_names
    sfreq    = raw.info['sfreq']
    dur_min  = raw.times[-1] / 60
    print(f"  loaded  : {dur_min:.1f} min  |  sfreq: {sfreq:.0f} Hz  |  channels: {ch_names}")

    # Find EEG: prefer RF\d* (e.g. RF, RF3); fall back to bare F\d* (e.g. F, F3).
    # Lookbehind/lookahead keeps 'F3' but blocks 'RF3' and 'LF3' in the fallback.
    eeg_candidates = [n for n in ch_names if re.search(r'RF\d*', n, re.I)]
    if not eeg_candidates:
        eeg_candidates = [n for n in ch_names
                          if re.search(r'(?<![A-Za-z])F\d*(?![A-Za-z])', n, re.I)]
    emg_candidates = [n for n in ch_names if re.search(r'EMG', n, re.I)]

    if not eeg_candidates:
        raise ValueError(f"No EEG channel (RF/F) found in {edf_path}. Available: {ch_names}")
    if not emg_candidates:
        raise ValueError(f"No EMG channel found in {edf_path}. Available: {ch_names}")

    eeg_name = eeg_candidates[0]
    emg_name = emg_candidates[0]
    if len(eeg_candidates) > 1:
        print(f"  [warn] multiple EEG candidates {eeg_candidates}; using '{eeg_name}'")
    if len(emg_candidates) > 1:
        print(f"  [warn] multiple EMG candidates {emg_candidates}; using '{emg_name}'")
    print(f"  channels: EEG='{eeg_name}'  |  EMG='{emg_name}'")

    # Load only the two channels we need
    raw.pick([eeg_name, emg_name])
    raw.load_data(verbose=False)

    if int(round(sfreq)) != FS:
        print(f"  resample: {sfreq:.0f} → {FS} Hz ...")
        raw.resample(FS, verbose=False)

    # Retrieve by name — safe regardless of channel order in EDF
    EEG = raw.get_data(picks=[eeg_name])   # [1, n_samples]  float64
    EMG = raw.get_data(picks=[emg_name])   # [1, n_samples]  float64

    with torch.no_grad():
        EEG_STFT, EMG_STFT = data_process_tensor(EEG, EMG, fs=FS, device=device)
    return EEG_STFT, EMG_STFT


def _save_recording(name, edf_path, score, out_dir):
    """
    Process one recording and save <name>.npy and <name>_score.npy to out_dir.
    Returns n_epochs or None on failure.
    """
    out_path   = os.path.join(out_dir, f"{name}.npy")
    score_path = os.path.join(out_dir, f"{name}_score.npy")

    # Skip if both files already exist (idempotent)
    if os.path.exists(out_path) and os.path.exists(score_path):
        try:
            existing = np.load(out_path, mmap_mode='r')
            n = existing.shape[0]
            print(f"  [skip] {name}  ({n} epochs already compiled)")
            return n
        except Exception:
            pass  # corrupted — re-process

    try:
        EEG_STFT, EMG_STFT = _process_edf(edf_path)
    except Exception as e:
        print(f"  [ERROR] Failed to process {name}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # Concatenate EEG + EMG along feature dim → [n_epochs, frames, 2*bins]
    STFT = np.concatenate([EEG_STFT, EMG_STFT], axis=-1)  # [N, frames, feat]

    # Flatten to [N, frames*feat] for compact on-disk storage
    N, frames, feat = STFT.shape
    flat = STFT.reshape(N, frames * feat).astype(np.float32)

    # Align with score
    n_epochs      = min(N, len(score))
    flat          = flat[:n_epochs]
    score_trimmed = score[:n_epochs]

    np.save(out_path, flat)
    np.save(score_path, score_trimmed)
    del EEG_STFT, EMG_STFT, STFT, flat
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.backends.cuda, 'cufft_plan_cache'):
            try:
                torch.backends.cuda.cufft_plan_cache.clear()
            except:
                pass

    return n_epochs


def _normalize_stem(stem):
    """Normalize a file stem for fuzzy matching.

    Removes spaces, lowercases, and strips known EDF-only trailing suffixes so
    that e.g. 'Day 4DMSO5122_reduced' → 'day4dmso5122'.
    """
    s = stem.replace(' ', '').lower()
    for suf in _EDF_SUFFIXES:
        if s.endswith(suf.lower()):
            s = s[: -len(suf)]
            break  # strip at most one suffix
    return s


def _scan_dirs(dirs):
    """Scan each folder in dirs for EDF+score pairs using fuzzy name matching.

    For each EDF the function looks for a companion .txt (RM format) or .tsv
    (Sirenia format) score file in the same folder.  Matching is done on
    normalized stems (spaces removed, lower-cased, EDF suffixes stripped).

    Returns sorted list of (edf_stem, edf_path, score_path, fmt) where
    fmt is 'rm' or 'tsv'.
    """
    pairs = []
    for folder in dirs:
        try:
            entries = _with_retry(f"listdir {folder}", os.listdir, folder)
        except Exception as e:
            print(f"  [ERROR] Cannot list {folder}: {e}")
            continue

        # Build score lookup: normalized_key → (path, fmt)
        # .txt (RM) takes priority over .tsv when both exist for the same key.
        score_map = {}
        for f in sorted(entries):
            stem = os.path.splitext(f)[0]
            nkey = _normalize_stem(stem)
            fl = f.lower()
            if fl.endswith('.txt'):
                score_map[nkey] = (os.path.join(folder, f), 'rm')
            elif fl.endswith('.tsv') and nkey not in score_map:
                score_map[nkey] = (os.path.join(folder, f), 'tsv')

        # Match each EDF to a score file
        for f in sorted(entries):
            if not f.lower().endswith('.edf'):
                continue
            edf_path = os.path.join(folder, f)
            edf_stem = os.path.splitext(f)[0]
            edf_key  = _normalize_stem(edf_stem)

            # 1. Exact match after normalization
            if edf_key in score_map:
                score_path, fmt = score_map[edf_key]
                pairs.append((edf_stem, edf_path, score_path, fmt))
                continue

            # 2. Prefix match — score name is a leading substring of EDF key
            #    (or vice-versa).  Pick the longest matching score key.
            best = None
            for skey, (sp, fmt) in score_map.items():
                if edf_key.startswith(skey) or skey.startswith(edf_key):
                    if best is None or len(skey) > len(best[0]):
                        best = (skey, sp, fmt)
            if best:
                _, score_path, fmt = best
                pairs.append((edf_stem, edf_path, score_path, fmt))
            else:
                print(f"  [warn] No score file found for {f!r} in {folder}")

    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    # ── Discover all EDF+score pairs ──────────────────────────────────────────
    pairs = _scan_dirs(TRAINING_DIRS)
    print(f"{'─'*60}")
    print(f"Found {len(pairs)} EDF+score pair(s) across {len(TRAINING_DIRS)} folder(s).")
    print(f"{'─'*60}\n")

    names   = []
    lengths = []

    # ── Process each pair ────────────────────────────────────────────────────
    n_pairs = len(pairs)
    for idx, (edf_stem, edf_path, score_path, fmt) in enumerate(pairs):
        size_mb = os.path.getsize(edf_path) / 1e6
        print(f"\n[{idx+1}/{n_pairs}] {edf_stem}")
        print(f"  edf  : {edf_path}  ({size_mb:.0f} MB)")
        print(f"  score: {score_path}  [{fmt}]")

        if fmt == 'rm':
            score = _load_rm_score(score_path)
        else:
            score = _load_tsv_score(score_path)

        n = _save_recording(edf_stem, edf_path, score, OUTPUT_DIR)
        if n is not None:
            names.append(edf_stem)
            lengths.append(n)
        time.sleep(0.05)

    if not names:
        print("\nNo recordings compiled successfully — nothing to save.")
        return

    print(f"\n{'═'*60}")
    print(f"Compilation complete!")
    print(f"  Recordings  : {len(names)}")
    print(f"  Total epochs: {sum(lengths):,}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
