"""
Data_compile.py  —  REST V1.5
═══════════════════════════════════════════════════════════════════════════════
Preprocesses raw EDF + score files into per-recording .npy files for lazy-
loading during training.  This replaces the old approach of loading everything
into RAM at once and saving a single monolithic .npz.

Output layout (all files go to OUTPUT_DIR):
  <recording_name>.npy   float32  [n_epochs, frames*feat]   (flattened STFT)
  all_scores.npy         int32    [total_epochs]             (concatenated labels)
  manifest.npz                    names, offsets, lengths

Score encoding (stored):   1=Wake  2=NREM  3=REM  (original Sirenia / RM .txt)
  -100 = ignored (unscored, artefact, or out-of-range)

Usage:
    python Data_compile.py
    (idempotent — skips recordings whose .npy already exists)
"""

import os
import gc
import time
import numpy as np
import mne
import pandas as pd
from tqdm import tqdm
import torch
from RESTutils import find_data_start, data_process_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration  —  edit these paths as needed
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = r"D:\Training data V1.5_tensor"

# ── Original 7 animals (scored with RM .txt files) ──────────────────────────
FP_EDF_ORIG = [
    r"M:\EEG files\selected data for manual score\Sham\2A1_CD1M_Cohort4_5-25-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\Sham\2A2_CD1M_Cohort7_Week4_10-12-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\Sham\4A1_CD1F_Cohort6_Week4_08-23-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\TBI\3B1_CD1M_Cohort7_Week3_10-4-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\TBI\3B2_CD1M_Cohort5_Week4_7-14-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\TBI\5B1_CD1M_Cohort4_Week3_5-25-24_reduced.edf",
    r"M:\EEG files\selected data for manual score\TBI\5B2_CD1F_Cohort6_Week4_08-25-24_reduced.edf",
]
FP_SCORE_ORIG = [
    r"M:\EEG files\selected data for manual score\Sham\2A1_CD1M_Cohort4_RM.txt",
    r"M:\EEG files\selected data for manual score\Sham\2A2_CD1M_Cohort7_RM.txt",
    r"M:\EEG files\selected data for manual score\Sham\4A1_CD1F_Cohort6_RM.txt",
    r"M:\EEG files\selected data for manual score\TBI\3B1_CD1M_Cohort7_RM.txt",
    r"M:\EEG files\selected data for manual score\TBI\3B2_CD1M_Cohort5_RM.txt",
    r"M:\EEG files\selected data for manual score\TBI\5B1_CD1M_Cohort4_RM.txt",
    r"M:\EEG files\selected data for manual score\TBI\5B2_CD1F_Cohort6_RM.txt",
]

# ── Additional Westmark animals (scored with Sirenia .tsv files) ─────────────
ADDITIONAL_DIRS = [
    r"M:\Alex\Python\Mouse Sleep Data\fmr1\5mo",
    r"M:\Alex\Python\Mouse Sleep Data\fmr1\Adult",
]

# Epoch count guard for additional recordings (sanity check)
MIN_EPOCHS = 21550
MAX_EPOCHS = 21610

FS = 512  # native EDF sampling rate


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_orig_score(score_path):
    """Read RM .txt score file.  Returns int32 array with -100 for ignored."""
    df = pd.read_csv(score_path, delimiter=',')
    score = df.iloc[:, 3].to_numpy().astype(np.int32)
    score[score > 3] = -100   # unscored
    score[score == 0] = -100  # artefact → ignore
    return score


def _load_tsv_score(tsv_path):
    """Read Sirenia .tsv score file.  Returns int32 array with -100 for ignored."""
    start_line = find_data_start(tsv_path, sep='\t', expected_columns=5)
    df = pd.read_csv(tsv_path, sep='\t', skiprows=start_line + 1, header=None)
    score = df.iloc[:, 4].to_numpy().astype(np.int32)
    score[score > 3] = -100   # unscored / out-of-range
    score[score == 0] = -100  # artefact → ignore
    return score


def _process_edf(edf_path):
    """Read EDF → return (EEG_STFT, EMG_STFT) via data_process()."""
    raw = None
    for attempt in range(3):
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            break
        except Exception as e:
            if attempt < 2:
                print(f"Failed to load {edf_path} (Attempt {attempt + 1}/3). Waiting 5 seconds...")
                time.sleep(5)
            else:
                raise e
                
    ch = raw.info['ch_names']
    eeg_idx = [i for i, n in enumerate(ch) if 'RF' in n]
    emg_idx = [i for i, n in enumerate(ch) if 'EMG' in n]
    EEG = raw.get_data(eeg_idx)
    EMG = raw.get_data(emg_idx)
    raw.close()
    
    with torch.no_grad():
        EEG_STFT, EMG_STFT = data_process_tensor(EEG, EMG, fs=512, device=device)
    return EEG_STFT, EMG_STFT   # returns (EEG_STFT, EMG_STFT)


def _save_recording(name, edf_path, score, out_dir):
    """
    Process one recording and save to <out_dir>/<name>.npy.
    Returns (n_epochs, score_trimmed) or (None, None) on failure.
    """
    out_path = os.path.join(out_dir, f"{name}.npy")

    # Skip if already processed (idempotent)
    if os.path.exists(out_path):
        try:
            existing = np.load(out_path, mmap_mode='r')
            n = existing.shape[0]
            print(f"  [skip] {name}  ({n} epochs already compiled)")
            return n, score[:n]
        except Exception:
            pass  # corrupted — re-process

    try:
        EEG_STFT, EMG_STFT = _process_edf(edf_path)
    except Exception as e:
        print(f"  [ERROR] Failed to process {name}: {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

    # Concatenate EEG + EMG along feature dim → [n_epochs, frames, 2*bins]
    STFT = np.concatenate([EEG_STFT, EMG_STFT], axis=-1)  # [N, frames, feat]

    # Flatten to [N, frames*feat] for compact on-disk storage
    N, frames, feat = STFT.shape
    flat = STFT.reshape(N, frames * feat).astype(np.float32)

    # Align with score
    n_epochs = min(N, len(score))
    flat = flat[:n_epochs]
    score = score[:n_epochs]

    np.save(out_path, flat)
    del EEG_STFT, EMG_STFT, STFT, flat
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.backends.cuda, 'cufft_plan_cache'):
            try:
                torch.backends.cuda.cufft_plan_cache.clear()
            except:
                pass

    return n_epochs, score


def _scan_additional_dirs(dirs):
    """Return sorted list of (name, edf_path, tsv_path) pairs from additional dirs."""
    pairs = []
    for folder in dirs:
        tsv_files = {}
        edf_files = {}
        for f in sorted(os.listdir(folder)):
            fp = os.path.join(folder, f)
            if f.endswith('.tsv'):
                key = f.rsplit('_', 1)[0]
                tsv_files[key] = fp
            elif f.endswith('.edf'):
                key = f.rsplit('_', 1)[0]
                edf_files[key] = fp
        for key in sorted(set(edf_files) & set(tsv_files)):
            pairs.append((key, edf_files[key], tsv_files[key]))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    names = []
    lengths = []
    all_scores_list = []

    # ── 1. Original 7 animals ─────────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"Processing {len(FP_EDF_ORIG)} original animals ...")
    for edf_path, score_path in tqdm(zip(FP_EDF_ORIG, FP_SCORE_ORIG), total=len(FP_EDF_ORIG), desc="Original"):
        name = os.path.splitext(os.path.basename(edf_path))[0]
        score = _load_orig_score(score_path)

        n, sc = _save_recording(name, edf_path, score, OUTPUT_DIR)
        if n is not None:
            names.append(name)
            lengths.append(n)
            all_scores_list.append(sc)
        time.sleep(0.05)

    # ── 2. Additional Westmark animals ────────────────────────────────────────
    additional_pairs = _scan_additional_dirs(ADDITIONAL_DIRS)
    print(f"\n{'─'*60}")
    print(f"Processing {len(additional_pairs)} additional animals ...")

    for name, edf_path, tsv_path in tqdm(additional_pairs, desc="Additional"):
        score = _load_tsv_score(tsv_path)

        # Sanity-check epoch count against EDF before full load
        try:
            raw_probe = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            eeg_epochs = raw_probe.n_times // (FS * 4)
            raw_probe.close()
        except Exception as e:
            print(f"  [ERROR] Cannot probe {name}: {e}")
            continue

        score_epochs = len(score)
        if not (MIN_EPOCHS <= score_epochs <= MAX_EPOCHS and
                MIN_EPOCHS <= eeg_epochs  <= MAX_EPOCHS):
            print(f"  [skip] {name}: epoch count mismatch "
                  f"(score={score_epochs}, eeg={eeg_epochs})")
            continue

        n, sc = _save_recording(name, edf_path, score, OUTPUT_DIR)
        if n is not None:
            names.append(name)
            lengths.append(n)
            all_scores_list.append(sc)
        time.sleep(0.05)

    # ── 3. Save aggregate metadata ────────────────────────────────────────────
    if not names:
        print("\nNo recordings compiled successfully — nothing to save.")
        return

    all_scores = np.concatenate(all_scores_list).astype(np.int32)
    offsets = np.concatenate([[0], np.cumsum(lengths[:-1])]).astype(np.int64)

    scores_path   = os.path.join(OUTPUT_DIR, 'all_scores.npy')
    manifest_path = os.path.join(OUTPUT_DIR, 'manifest.npz')

    np.save(scores_path, all_scores)
    np.savez(manifest_path,
             names   = np.array(names, dtype=object),
             offsets = offsets,
             lengths = np.array(lengths, dtype=np.int64))

    print(f"\n{'═'*60}")
    print(f"Compilation complete!")
    print(f"  Recordings : {len(names)}")
    print(f"  Total epochs: {len(all_scores):,}")
    print(f"  Scores     : {scores_path}")
    print(f"  Manifest   : {manifest_path}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
