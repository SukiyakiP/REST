"""
SleepDataset.py  —  REST V1.5
═══════════════════════════════════════════════════════════════════════════════
PyTorch Dataset for training REST V1.5.

Expects output from Data_compile.py in data_dir:
  <name>.npy        float32  [n_epochs, frames*feat]   (flattened STFT per epoch)
  all_scores.npy    int32    [total_epochs]
  manifest.npz      names, offsets, lengths

Each __getitem__ returns:
  X  : float32 tensor  [win_len, frames, feat]   ready to feed REST model
  Y  : int64   tensor  [win_len]                  labels (0=W, 1=N, 2=R, -100=ignore)
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    """
    Lazy-loading Dataset that reads pre-compiled per-recording .npy files
    via memory-mapped I/O.  Uses an LRU cache of open handles to avoid
    re-opening files repeatedly.

    Parameters
    ----------
    data_dir     : str   — folder with .npy, all_scores.npy, manifest.npz
    win_len      : int   — number of epochs per window (default 90)
    step         : int   — stride between window start positions (default 60)
    rem_repeat   : int   — how many extra copies of REM-containing windows (default 3)
    cache_size   : int   — max simultaneous open memmap handles (default 16)
    split        : str   — 'train' or 'val'
    val_split    : float — fraction of recordings held out for validation (default 0.2)
    frames       : int   — STFT time frames per epoch (default 5)
    feat         : int   — feature bins per frame after concatenation (default 130)
    """

    def __init__(self, data_dir, win_len=90, step=60,
                 rem_repeat=3, cache_size=16,
                 split='train', val_split=0.2,
                 frames=5, feat=130):
        super().__init__()
        self.data_dir  = data_dir
        self.win_len   = win_len
        self.split     = split
        self.cache_size = cache_size
        self.frames    = frames
        self.feat      = feat
        self._cache    = {}   # {rec_idx: memmap_array}

        # ── 1. Load manifest & scores ─────────────────────────────────────────
        manifest_path = os.path.join(data_dir, 'manifest.npz')
        scores_path   = os.path.join(data_dir, 'all_scores.npy')

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Missing {manifest_path}. Run Data_compile.py first.")
        if not os.path.exists(scores_path):
            raise FileNotFoundError(f"Missing {scores_path}. Run Data_compile.py first.")

        manifest = np.load(manifest_path, allow_pickle=True)
        self.scores = np.load(scores_path, mmap_mode='r')   # lazy scores

        all_names   = manifest['names']
        all_offsets = manifest['offsets']
        all_lengths = manifest['lengths']
        n_total = len(all_names)

        # ── 2. Deterministic train/val split (by recording) ───────────────────
        rng = np.random.RandomState(42)
        idx = np.arange(n_total)
        rng.shuffle(idx)
        n_val = int(n_total * val_split)

        if split == 'train':
            chosen = idx[n_val:]
        else:
            chosen = idx[:n_val]

        self.rec_names   = all_names[chosen]
        self.rec_offsets = all_offsets[chosen]
        self.rec_lengths = all_lengths[chosen]

        n_recs = len(self.rec_names)
        print(f"[SleepDataset] {split.upper()}: {n_recs}/{n_total} recordings")

        # ── 3. Build sliding-window index ─────────────────────────────────────
        self._index = []   # list of (rec_idx, local_start)
        for ri in range(n_recs):
            length = int(self.rec_lengths[ri])
            if length < win_len:
                continue
            for s in range(0, length - win_len + 1, step):
                self._index.append((ri, s))

        # ── 4. Oversample REM windows (train only) ────────────────────────────
        if split == 'train':
            rem_idx    = []
            normal_idx = []
            for i, (ri, s) in enumerate(self._index):
                g_start = int(self.rec_offsets[ri]) + s
                labels  = self.scores[g_start : g_start + win_len]
                if np.any(labels == 3):   # 3 = REM
                    rem_idx.append(i)
                else:
                    normal_idx.append(i)
            oversampled = normal_idx + rem_idx * rem_repeat
            rng2 = np.random.RandomState(0)
            rng2.shuffle(oversampled)
            self._sample_ids = oversampled
            print(f"  {len(normal_idx)} normal + {len(rem_idx)}×{rem_repeat} REM "
                  f"= {len(oversampled)} windows")
        else:
            self._sample_ids = list(range(len(self._index)))
            print(f"  {len(self._sample_ids)} windows (no oversampling)")

    # ── Pickle support for DataLoader workers ─────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = {}   # don't pickle open file handles
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_cache' not in self.__dict__:
            self._cache = {}

    def __len__(self):
        return len(self._sample_ids)

    # ── memmap LRU cache ──────────────────────────────────────────────────────
    def _open_memmap(self, rec_idx):
        name = str(self.rec_names[rec_idx])
        path = os.path.join(self.data_dir, f"{name}.npy")
        return np.load(path, mmap_mode='r')

    def _get_data(self, rec_idx):
        if rec_idx in self._cache:
            data = self._cache.pop(rec_idx)
            self._cache[rec_idx] = data   # move to end (LRU)
            return data
        data = self._open_memmap(rec_idx)
        if len(self._cache) >= self.cache_size:
            del self._cache[next(iter(self._cache))]
        self._cache[rec_idx] = data
        return data

    def _safe_slice(self, rec_idx, local_start):
        """Read one window with retry + stale-handle recovery."""
        max_retries = 6
        delay = 0.5
        for attempt in range(max_retries):
            try:
                data = self._get_data(rec_idx)
                X = data[local_start : local_start + self.win_len].copy()   # [W, flat]
                g_start = int(self.rec_offsets[rec_idx]) + local_start
                Y = self.scores[g_start : g_start + self.win_len].copy()    # [W]
                return X, Y
            except (OSError, IOError, ValueError):
                if rec_idx in self._cache:
                    del self._cache[rec_idx]
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 1.5

    def __getitem__(self, idx):
        rec_idx, local_start = self._index[self._sample_ids[idx]]
        X_flat, Y = self._safe_slice(rec_idx, local_start)

        # Reshape [win_len, frames*feat] → [win_len, frames, feat]
        X = X_flat.reshape(self.win_len, self.frames, self.feat)

        # Convert stored 1-based labels → 0-based (0=Wake, 1=NREM, 2=REM)
        # Stored: 1=Wake  2=NREM  3=REM  -100=ignore
        Y = Y.astype(np.int64).copy()
        valid = Y != -100
        Y[valid] = Y[valid] - 1   # → 0=Wake  1=NREM  2=REM

        return (torch.from_numpy(X.astype(np.float32)),
                torch.from_numpy(Y))
