"""
Check_training_data.py
======================
Sanity-checks every EDF+score pair found in TRAINING_DIRS without loading
full signal data.  Flags files with bad label distributions, epoch count
mismatches, unreadable score files, etc.

Saves a summary Excel report next to this script.

Usage:
    python Check_training_data.py
"""

import os
import sys
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm

# Re-use scanner and loaders from Data_compile without duplicating code.
# Data_compile has no top-level side-effects so this import is safe.
from Data_compile import (
    TRAINING_DIRS, FS,
    _scan_dirs, _load_rm_score, _load_tsv_score, _with_retry,
)

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds  —  adjust as needed
# ─────────────────────────────────────────────────────────────────────────────
MIN_VALID_EPOCHS  = 500    # < this → too short (~33 min at 4 s/epoch)
MAX_EPOCH_DELTA   = 0.10   # score vs EDF epoch count differ by more than 10 %
MAX_ART_PCT       = 30.0   # artifact fraction of valid epochs (%)
MAX_IGNORE_PCT    = 50.0   # fraction of all score epochs that are -100 (%)
MIN_WAKE_PCT      = 10.0   # wake fraction of valid epochs (%)
MAX_WAKE_PCT      = 97.0
MIN_NREM_PCT      =  2.0   # NREM fraction of valid epochs (%)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _probe_edf_epochs(edf_path):
    """Return expected epoch count from EDF header (no signal load)."""
    raw = _with_retry(
        f"probe {os.path.basename(edf_path)}",
        mne.io.read_raw_edf, edf_path, preload=False, verbose=False,
    )
    n_epochs = raw.n_times // (FS * 4)
    raw.close()
    return int(n_epochs)


def _check_score(score, edf_epochs, name):
    """
    Analyse a score array and return (stats_dict, issues_list).
    issues_list is empty when everything looks fine.
    """
    issues = []
    total  = len(score)
    valid  = score[score != -100]
    n_valid = len(valid)
    n_ignore = total - n_valid

    # Label sanity: only 1-4 and -100 allowed
    bad_labels = np.setdiff1d(np.unique(score), np.array([-100, 1, 2, 3, 4]))
    if len(bad_labels):
        issues.append(f"unexpected labels: {bad_labels.tolist()}")

    # Epoch count
    if total == 0:
        issues.append("score file is empty")
        return _empty_stats(total, edf_epochs), issues

    # Compare score length to EDF
    if edf_epochs > 0:
        delta = abs(total - edf_epochs) / max(total, edf_epochs)
        if delta > MAX_EPOCH_DELTA:
            issues.append(
                f"epoch count mismatch: score={total}, edf={edf_epochs} "
                f"({delta*100:.1f}% diff)"
            )

    ignore_pct = 100.0 * n_ignore / total
    if ignore_pct > MAX_IGNORE_PCT:
        issues.append(f"high ignored%: {ignore_pct:.1f}%")

    if n_valid == 0:
        issues.append("no valid (scored) epochs")
        return _empty_stats(total, edf_epochs), issues

    if n_valid < MIN_VALID_EPOCHS:
        issues.append(f"too few valid epochs: {n_valid} (< {MIN_VALID_EPOCHS})")

    # Per-class counts and percentages
    counts = {lbl: int(np.sum(valid == lbl)) for lbl in (1, 2, 3, 4)}
    pcts   = {lbl: 100.0 * counts[lbl] / n_valid for lbl in counts}

    if pcts[1] < MIN_WAKE_PCT:
        issues.append(f"low Wake%: {pcts[1]:.1f}%")
    if pcts[1] > MAX_WAKE_PCT:
        issues.append(f"high Wake%: {pcts[1]:.1f}%")
    if pcts[2] < MIN_NREM_PCT:
        issues.append(f"low NREM%: {pcts[2]:.1f}%")
    if counts[3] == 0:
        issues.append("no REM epochs")
    if pcts[4] > MAX_ART_PCT:
        issues.append(f"high Artifact%: {pcts[4]:.1f}%")

    stats = {
        'score_epochs': total,
        'edf_epochs':   edf_epochs,
        'valid_epochs': n_valid,
        'ignored_epochs': n_ignore,
        'ignored_%':    round(ignore_pct, 1),
        'Wake_%':       round(pcts[1], 1),
        'NREM_%':       round(pcts[2], 1),
        'REM_%':        round(pcts[3], 1),
        'Art_%':        round(pcts[4], 1),
        'Wake_n':       counts[1],
        'NREM_n':       counts[2],
        'REM_n':        counts[3],
        'Art_n':        counts[4],
    }
    return stats, issues


def _empty_stats(total, edf_epochs):
    return {
        'score_epochs': total, 'edf_epochs': edf_epochs,
        'valid_epochs': 0, 'ignored_epochs': total, 'ignored_%': 100.0,
        'Wake_%': 0.0, 'NREM_%': 0.0, 'REM_%': 0.0, 'Art_%': 0.0,
        'Wake_n': 0, 'NREM_n': 0, 'REM_n': 0, 'Art_n': 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not TRAINING_DIRS:
        print("TRAINING_DIRS is empty — nothing to check.")
        sys.exit(0)

    pairs = _scan_dirs(TRAINING_DIRS)
    print(f"Found {len(pairs)} EDF+score pair(s) across {len(TRAINING_DIRS)} folder(s).\n")

    rows = []

    for edf_stem, edf_path, score_path, fmt in tqdm(pairs, desc="Checking"):
        row = {
            'File':        edf_stem,
            'Format':      fmt,
            'Score file':  os.path.basename(score_path),
            'Folder':      os.path.dirname(edf_path),
        }

        # Load score
        try:
            if fmt == 'rm':
                score = _load_rm_score(score_path)
            else:
                score = _load_tsv_score(score_path)
        except Exception as e:
            row.update(_empty_stats(0, 0))
            row['Issues'] = f"score load error: {e}"
            row['Status'] = 'ERROR'
            rows.append(row)
            continue

        # Probe EDF
        try:
            edf_epochs = _probe_edf_epochs(edf_path)
        except Exception as e:
            edf_epochs = -1
            # Not fatal — continue with stats

        stats, issues = _check_score(score, edf_epochs, edf_stem)
        row.update(stats)
        row['Issues'] = ' | '.join(issues) if issues else ''
        row['Status'] = 'ERROR' if any(
            kw in i for i in issues for kw in ('empty', 'no valid', 'load error')
        ) else ('WARN' if issues else 'OK')
        rows.append(row)

    # ── Print summary ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    ok   = (df['Status'] == 'OK').sum()
    warn = (df['Status'] == 'WARN').sum()
    err  = (df['Status'] == 'ERROR').sum()

    print(f"\n{'═'*70}")
    print(f"  Total pairs : {len(df)}")
    print(f"  OK          : {ok}")
    print(f"  WARN        : {warn}")
    print(f"  ERROR       : {err}")
    print(f"{'═'*70}\n")

    flagged = df[df['Status'] != 'OK']
    if not flagged.empty:
        print("Flagged files:")
        print("-" * 70)
        for _, r in flagged.iterrows():
            print(f"  [{r['Status']:5s}]  {r['File']}")
            print(f"           {r['Issues']}")
        print()

    # ── Save Excel ────────────────────────────────────────────────────────────
    col_order = [
        'Status', 'File', 'Format', 'Folder', 'Score file',
        'score_epochs', 'edf_epochs', 'valid_epochs', 'ignored_epochs', 'ignored_%',
        'Wake_%', 'NREM_%', 'REM_%', 'Art_%',
        'Wake_n', 'NREM_n', 'REM_n', 'Art_n',
        'Issues',
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df.sort_values(['Status', 'File'], inplace=True)

    out_path = os.path.join(os.path.dirname(__file__), 'training_data_check.xlsx')
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='All')

        flagged_df = df[df['Status'] != 'OK']
        if not flagged_df.empty:
            flagged_df.to_excel(writer, index=False, sheet_name='Flagged')

    print(f"Report saved: {out_path}")


if __name__ == '__main__':
    main()
