"""
tune_artifact_filter.py
=======================
Interactive tuning script for ArtifactFilter.py.

Tuning is done on the training data (TRAINING_DIRS, ~490 recordings).
The 5 held-out test datasets are used only for final evaluation — they are
never seen during parameter selection.

Algorithm:
  Two rules (OR):
    1. Smooth-PTP: per-recording PTP percentile rank, temporally smoothed.
    2. Saturation: fraction of samples at ADC rail.

Parameters (artifact_params.json):
    ptp_percentile        — within-recording reference percentile (default 99)
    smooth_window         — convolution window in epochs (default 5 = 20 s)
    score_threshold       — smoothed score flag threshold (default 0.4)
    saturation_threshold  — fraction at ADC rail (default 0.10)

Cells:
  0 — Config & imports
  1 — Build TRAINING feature cache (slow on first run; skips cached files)
  2 — Training-data stats: artifact count per source folder
  3 — Baseline: per-folder F1 at default params on training data
  4 — Grid sweep on training data
  5 — Per-folder breakdown with tuned params
  6 — HELD-OUT evaluation on 5 test datasets (honest F1)
  7 — Visual inspection: sample flagged epoch timestamps
  8 — Feature histograms
  9 — Save tuned params to JSON
"""

# %% ── Cell 0: Config & imports ──────────────────────────────────────────────
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

from ArtifactFilter import compute_features, smooth_ptp_score, DEFAULT_PARAMS
from Data_compile import _load_rm_score, _load_tsv_score, _scan_dirs, _with_retry

# ── Training data (used for parameter tuning) ─────────────────────────────────
TRAINING_DIRS = [
    r"M:\Alex\Python\Mouse Sleep Data\fmr1\5mo",
    r"M:\Alex\Python\Mouse Sleep Data\fmr1\Adult",
    r"M:\Alex\Python\Mouse Sleep Data\Additional_ELI",
    r"M:\Alex\Python\Mouse Sleep Data\Additional_RM",
    r"M:\Alex\Python\Mouse Sleep Data\Additional_TBI",
]
TRAIN_CACHE_DIR = r"M:\Alex\Python\REST\artifact_cache_train"

# Skip recordings with more than this fraction of artifact labels —
# files with >5% zeros are almost certainly unscored blanks, not real artifacts.
MAX_ART_RATE = 0.05

# ── Held-out test datasets (evaluation only — never used for tuning) ──────────
TEST_DATASETS = [
    ("C57KA", r"M:\Alex\REST-Testing\C57KA"),
    ("C57SA", r"M:\Alex\REST-Testing\C57SA"),
    ("CD1",   r"M:\Alex\REST-Testing\CD1"),
    ("DBAKA", r"M:\Alex\REST-Testing\DBAKA"),
    ("DBASA", r"M:\Alex\REST-Testing\DBASA"),
]
TEST_CACHE_DIR = r"M:\Alex\Python\REST\artifact_cache"

FS            = 512
EPOCH_SECONDS = 4
EPOCH_SAMPLES = FS * EPOCH_SECONDS

os.makedirs(TRAIN_CACHE_DIR, exist_ok=True)
os.makedirs(TEST_CACHE_DIR, exist_ok=True)


def _load_all_cache(cache_dir, folders=None):
    """Load all .npz files from cache_dir. Optionally filter by folder_name."""
    records = []
    for fp in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        d = dict(np.load(fp, allow_pickle=True))
        ds = str(d["dataset_name"][0])
        if folders is not None and ds not in folders:
            continue
        d["dataset"] = ds
        d["file"]    = str(d["file_name"][0])
        records.append(d)
    return records


def eval_params(records, params):
    """Evaluate params dict → (recall, precision, f1, w_flag)."""
    tp = fp = fn = w_fp = w_tot = 0
    for d in records:
        ts    = d["true_score"]
        valid = ts != -100
        tv    = ts[valid]
        art   = tv == 4
        ptp   = d["ptp_raw"][valid]
        sat   = d["saturation"][valid]

        scores = smooth_ptp_score(ptp,
                                  ptp_percentile=params["ptp_percentile"],
                                  smooth_window=params["smooth_window"])
        pred   = (scores >= params["score_threshold"]) | (sat >= params["saturation_threshold"])

        tp    += int(( pred &  art).sum())
        fp    += int(( pred & ~art).sum())
        fn    += int((~pred &  art).sum())
        w_fp  += int(( pred & (tv == 1)).sum())
        w_tot += int((tv == 1).sum())

    r  = tp / max(tp + fn, 1)
    p  = tp / max(tp + fp, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return r, p, f1, w_fp / max(w_tot, 1)


def _load_eeg_raw(edf_path):
    """Load raw EEG (volts, 1D) from EDF. Returns None if no suitable channel found."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    ch  = raw.ch_names
    # Match Data_compile.py channel detection logic
    eeg_cands = [n for n in ch if re.search(r'RF\d*', n, re.I) and 'LP' not in n]
    if not eeg_cands:
        eeg_cands = [n for n in ch if re.search(r'(?<![A-Za-z])F\d*(?![A-Za-z])', n, re.I)]
    if not eeg_cands:
        raw.close()
        return None
    # Resample if needed
    if int(round(raw.info['sfreq'])) != FS:
        raw.resample(FS, verbose=False)
    EEG = raw.get_data(picks=[eeg_cands[0]])[0]
    raw.close()
    return EEG


# %% ── Cell 1: Build TRAINING feature cache ───────────────────────────────────
# Scans TRAINING_DIRS for EDF+score pairs, extracts EEG features, caches as .npz.
# Files with artifact rate > MAX_ART_RATE are skipped (likely unscored blanks).
# Already-cached files are skipped on re-runs.

print("=== Building TRAINING feature cache ===")
print(f"Source folders: {len(TRAINING_DIRS)}")
print(f"Max artifact rate filter: {MAX_ART_RATE:.0%}\n")

pairs   = _scan_dirs(TRAINING_DIRS)
print(f"Found {len(pairs)} EDF+score pairs\n")

skipped_no_eeg   = []
skipped_art_rate = []
errors           = []
cached_new       = 0

for edf_stem, edf_path, score_path, fmt in tqdm(pairs, desc="Training cache"):
    folder_name = os.path.basename(os.path.dirname(edf_path))
    cache_fp    = os.path.join(TRAIN_CACHE_DIR, f"{folder_name}_{edf_stem}.npz")

    if os.path.exists(cache_fp):
        continue

    try:
        # Load score
        score = _load_rm_score(score_path) if fmt == 'rm' else _load_tsv_score(score_path)

        # Check artifact rate on scored epochs
        scored = score[score != -100]
        art_rate = float((scored == 4).sum()) / max(len(scored), 1)
        if art_rate > MAX_ART_RATE:
            skipped_art_rate.append((folder_name, edf_stem, f"{art_rate:.1%}"))
            continue

        # Load raw EEG
        EEG = _load_eeg_raw(edf_path)
        if EEG is None:
            skipped_no_eeg.append((folder_name, edf_stem))
            continue

        feats = compute_features(EEG, fs=FS, epoch_seconds=EPOCH_SECONDS)
        n     = min(len(feats["ptp_raw"]), len(score))

        save_dict = {k: v[:n] for k, v in feats.items()}
        save_dict["true_score"]   = score[:n]
        save_dict["dataset_name"] = np.array([folder_name])
        save_dict["file_name"]    = np.array([edf_stem])
        np.savez_compressed(cache_fp, **save_dict)
        cached_new += 1

    except Exception as e:
        errors.append((folder_name, edf_stem, str(e)))

print(f"\nDone.  New files cached: {cached_new}")
print(f"Skipped (no EEG channel): {len(skipped_no_eeg)}")
print(f"Skipped (art rate >{MAX_ART_RATE:.0%}): {len(skipped_art_rate)}")
if skipped_art_rate:
    for fn, fs_, rate in skipped_art_rate[:10]:
        print(f"  {fn}/{fs_}  art={rate}")
print(f"Errors: {len(errors)}")
for fn, fs_, err in errors[:5]:
    print(f"  {fn}/{fs_}: {err}")


# %% ── Cell 2: Training cache stats ──────────────────────────────────────────
train_records = _load_all_cache(TRAIN_CACHE_DIR)
print(f"Training cache: {len(train_records)} recordings\n")

folder_counts = {}
for d in train_records:
    ds = d["dataset"]
    ts = d["true_score"]
    n_scored = int((ts != -100).sum())
    n_art    = int((ts == 4).sum())
    if ds not in folder_counts:
        folder_counts[ds] = {"recs": 0, "epochs": 0, "art": 0}
    folder_counts[ds]["recs"]   += 1
    folder_counts[ds]["epochs"] += n_scored
    folder_counts[ds]["art"]    += n_art

print(f"{'Folder':<25} {'Recs':>5} {'Epochs':>9} {'Art':>6} {'Art%':>7}")
print("-" * 58)
tot_r = tot_e = tot_a = 0
for ds, c in sorted(folder_counts.items()):
    pct = 100 * c["art"] / max(c["epochs"], 1)
    print(f"{ds:<25} {c['recs']:>5} {c['epochs']:>9,} {c['art']:>6} {pct:>6.3f}%")
    tot_r += c["recs"]; tot_e += c["epochs"]; tot_a += c["art"]
print("-" * 58)
print(f"{'TOTAL':<25} {tot_r:>5} {tot_e:>9,} {tot_a:>6} {100*tot_a/max(tot_e,1):>6.3f}%")


# %% ── Cell 3: Baseline on training data ─────────────────────────────────────
print(f"=== Baseline (EEG-only defaults) on TRAINING data ===\n")
print(f"Params: {DEFAULT_PARAMS}\n")
print(f"{'Folder':<25}  {'Recall':>7} {'Prec':>7} {'F1':>7}  {'W-flag%':>8}")
print("-" * 60)

folders = sorted(folder_counts.keys())
for ds in folders + ["ALL"]:
    recs = train_records if ds == "ALL" else [d for d in train_records if d["dataset"] == ds]
    if not recs:
        continue
    r, p, f1, wf = eval_params(recs, DEFAULT_PARAMS)
    print(f"{ds:<25}  {r:>7.3f} {p:>7.3f} {f1:>7.3f}  {wf:>7.2%}")


# %% ── Cell 4: Grid sweep on TRAINING data ────────────────────────────────────
PERCENTILE_GRID   = [95, 97, 98, 99, 99.5]
SMOOTH_WINDOW_GRID = [3, 5, 7]
SCORE_THRESH_GRID  = [0.3, 0.4, 0.5, 0.6, 0.8]
SAT_THRESHOLD      = 0.10

print("=== Grid sweep on TRAINING data (sorted by ALL F1) ===\n")
results = []
for pct in PERCENTILE_GRID:
    for sw in SMOOTH_WINDOW_GRID:
        for st in SCORE_THRESH_GRID:
            params = {**DEFAULT_PARAMS,
                      "ptp_percentile": pct,
                      "smooth_window":  sw,
                      "score_threshold": st,
                      "saturation_threshold": SAT_THRESHOLD}
            r, p, f1, wf = eval_params(train_records, params)
            results.append((f1, r, p, wf, pct, sw, st))

results.sort(reverse=True)
print(f"{'pct':>6} {'sw':>4} {'st':>5}  {'F1':>7} {'R':>7} {'P':>7}  {'W-flag':>8}")
print("-" * 56)
for f1, r, p, wf, pct, sw, st in results[:20]:
    print(f"{pct:>6.1f} {sw:>4d} {st:>5.2f}  {f1:>7.3f} {r:>7.3f} {p:>7.3f}  {wf:>7.2%}")


# %% ── Cell 5: Per-folder breakdown with tuned params (training data) ─────────
# Edit TUNED_PARAMS to the best row from Cell 4, then re-run this cell.

TUNED_PARAMS = {
    "ptp_percentile":      99.0,
    "smooth_window":        5,
    "score_threshold":      0.4,
    "saturation_threshold": 0.10,
}

print("=== Per-folder breakdown — TRAINING data ===\n")
print(f"Params: {TUNED_PARAMS}\n")
print(f"{'Folder':<25}  {'Recall':>7} {'Prec':>7} {'F1':>7}  {'W-flag%':>8}  Recs")
print("-" * 68)
for ds in folders + ["ALL"]:
    recs = train_records if ds == "ALL" else [d for d in train_records if d["dataset"] == ds]
    if not recs:
        continue
    r, p, f1, wf = eval_params(recs, TUNED_PARAMS)
    print(f"{ds:<25}  {r:>7.3f} {p:>7.3f} {f1:>7.3f}  {wf:>7.2%}  {len(recs)}")


# %% ── Cell 6: HELD-OUT evaluation on test datasets ───────────────────────────
# These 5 datasets were NOT used during parameter selection — this is the
# honest evaluation of how the filter generalises.
# Rebuild test cache first if needed (load_txt_score for .txt score files).

def load_txt_score(score_path):
    df    = pd.read_csv(score_path, delimiter=',')
    score = df.iloc[:, 3].to_numpy().astype(np.int32)
    score[score > 3] = -100
    score[score == 0] = 4
    return score

# Build test cache if missing
print("=== Checking held-out test cache ===")
for dataset_name, folder_path in TEST_DATASETS:
    edf_files = glob.glob(os.path.join(folder_path, "**", "*.edf"), recursive=True)
    for fp_edf in edf_files:
        base     = os.path.splitext(os.path.basename(fp_edf))[0]
        cache_fp = os.path.join(TEST_CACHE_DIR, f"{dataset_name}_{base}.npz")
        if os.path.exists(cache_fp):
            continue
        score_fp = os.path.join(os.path.dirname(fp_edf), base + ".txt")
        if not os.path.exists(score_fp):
            continue
        try:
            raw     = mne.io.read_raw_edf(fp_edf, preload=True, verbose=False)
            ch      = raw.info.ch_names
            eeg_idx = [i for i, n in enumerate(ch) if 'RF' in n and 'LP' not in n]
            if not eeg_idx:
                raw.close(); continue
            EEG = raw.get_data(eeg_idx)[0]
            raw.close()
            feats      = compute_features(EEG, fs=FS, epoch_seconds=EPOCH_SECONDS)
            true_score = load_txt_score(score_fp)
            n          = min(len(feats["ptp_raw"]), len(true_score))
            save_dict  = {k: v[:n] for k, v in feats.items()}
            save_dict["true_score"]   = true_score[:n]
            save_dict["dataset_name"] = np.array([dataset_name])
            save_dict["file_name"]    = np.array([base])
            np.savez_compressed(cache_fp, **save_dict)
        except Exception as e:
            print(f"  Error {base}: {e}")

test_records = _load_all_cache(TEST_CACHE_DIR)
print(f"Test cache: {len(test_records)} recordings\n")

print("=== HELD-OUT TEST EVALUATION (params tuned on training data) ===\n")
print(f"Params: {TUNED_PARAMS}\n")
print(f"{'Dataset':<10}  {'Recall':>7} {'Prec':>7} {'F1':>7}  {'W-flag%':>8}  Files")
print("-" * 60)
for ds_name in ["C57KA", "C57SA", "CD1", "DBAKA", "DBASA", "ALL"]:
    recs = test_records if ds_name == "ALL" else [d for d in test_records if d["dataset"] == ds_name]
    if not recs:
        continue
    r, p, f1, wf = eval_params(recs, TUNED_PARAMS)
    print(f"{ds_name:<10}  {r:>7.3f} {p:>7.3f} {f1:>7.3f}  {wf:>7.2%}  {len(recs)}")


# %% ── Cell 7: Sample flagged epochs for visual inspection ────────────────────
SAMPLE_N    = 8
SAMPLE_RULE = "ptp"   # "ptp" or "saturation"

print(f"=== Sample flagged epochs ({SAMPLE_RULE}) — test recordings ===\n")
for d in _load_all_cache(TEST_CACHE_DIR):
    ts    = d["true_score"]
    valid = ts != -100
    tv    = ts[valid]
    ptp   = d["ptp_raw"][valid]
    sat   = d["saturation"][valid]

    scores  = smooth_ptp_score(ptp,
                                ptp_percentile=TUNED_PARAMS["ptp_percentile"],
                                smooth_window=TUNED_PARAMS["smooth_window"])
    flagged = (scores >= TUNED_PARAMS["score_threshold"]) if SAMPLE_RULE == "ptp" \
              else (sat >= TUNED_PARAMS["saturation_threshold"])

    if flagged.sum() == 0:
        continue

    valid_idx  = np.where(valid)[0]
    flag_valid = valid_idx[flagged]

    print(f"  {d['dataset']}/{d['file']}  ({flagged.sum()} flagged)")
    for ep_raw in flag_valid[:SAMPLE_N]:
        t_sec = ep_raw * EPOCH_SECONDS
        t_hr, rem = divmod(t_sec, 3600)
        t_min, t_s = divmod(rem, 60)
        true_lbl = int(ts[ep_raw])
        sc = float(scores[flagged][0]) if SAMPLE_RULE == "ptp" else float(sat[flagged][0])
        print(f"    epoch={ep_raw:>6}  t={t_hr:02d}:{t_min:02d}:{t_s:02d}  "
              f"true={true_lbl}  score={sc:.3f}")
    print()


# %% ── Cell 8: Feature histograms ────────────────────────────────────────────
scores_art  = []
scores_norm = []
ptp_art_raw = []
ptp_norm_raw = []

for d in _load_all_cache(TEST_CACHE_DIR):
    ts    = d["true_score"]
    valid = ts != -100
    tv    = ts[valid]
    ptp   = d["ptp_raw"][valid]
    sc    = smooth_ptp_score(ptp,
                              ptp_percentile=TUNED_PARAMS["ptp_percentile"],
                              smooth_window=TUNED_PARAMS["smooth_window"])
    art = tv == 4
    scores_art.extend(sc[art].tolist())
    scores_norm.extend(sc[~art].tolist())
    ptp_art_raw.extend(ptp[art].tolist())
    ptp_norm_raw.extend(ptp[~art].tolist())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

ax = axes[0]
ax.hist(scores_norm, bins=100, range=(0, 2), density=True,
        alpha=0.5, color="steelblue", label="Non-artifact")
ax.hist(scores_art,  bins=100, range=(0, 2), density=True,
        alpha=0.7, color="tomato",    label="Artifact")
ax.axvline(TUNED_PARAMS["score_threshold"], color="k", linestyle="--",
           label=f"Threshold={TUNED_PARAMS['score_threshold']}")
ax.set_xlabel("Smooth-PTP score")
ax.set_title("Smooth-PTP score (test recordings)")
ax.legend()

ax = axes[1]
ptp_lim = np.percentile(ptp_norm_raw, 99.9) if ptp_norm_raw else 1000
ax.hist(ptp_norm_raw, bins=80, range=(0, ptp_lim), density=True,
        alpha=0.5, color="steelblue", label="Non-artifact")
ax.hist(ptp_art_raw,  bins=80, range=(0, ptp_lim), density=True,
        alpha=0.7, color="tomato",    label="Artifact")
ax.set_xlabel("PTP amplitude (µV)")
ax.set_title("PTP distribution (test recordings)")
ax.legend()

plt.tight_layout()
plt.show()


# %% ── Cell 9: Save tuned params ─────────────────────────────────────────────
SAVE_PATH = r"M:\Alex\Python\REST\artifact_params.json"
with open(SAVE_PATH, "w") as f:
    json.dump(TUNED_PARAMS, f, indent=2)
print(f"Params saved to {SAVE_PATH}")
print(json.dumps(TUNED_PARAMS, indent=2))
