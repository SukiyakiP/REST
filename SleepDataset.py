"""
SleepDataset.py  —  REST V2.0
═══════════════════════════════════════════════════════════════════════════════
PyTorch Dataset for training REST.

Expects output from Data_compile.py in data_dir:
  <name>.npy        float32  [n_epochs, frames*feat]   (flattened STFT per epoch)
  <name>_score.npy  int32    [n_epochs]                 (per-recording labels)

Recordings are discovered by scanning for *_score.npy files — no manifest needed.

Each __getitem__ returns:
  X  : float32 tensor  [win_len, frames, feat]   ready to feed REST model
  Y  : int64   tensor  [win_len]                  labels (0=W, 1=N, 2=R, 3=Art, -100=ignore)
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset

from ArtifactInjection import inject_artifacts


class SleepDataset(Dataset):
    """
    Lazy-loading Dataset that reads pre-compiled per-recording .npy files
    via memory-mapped I/O.  Uses an LRU cache of open handles to avoid
    re-opening files repeatedly.

    Parameters
    ----------
    data_dir     : str   — folder with <name>.npy and <name>_score.npy files
    win_len      : int   — number of epochs per window (default 90)
    step         : int   — stride between window start positions (default 60)
    rem_repeat   : int   — repeat factor for REM-containing windows (default 3)
    art_repeat   : int   — repeat factor for artifact-containing windows (default 1)
    cache_size   : int   — max simultaneous open memmap handles (default 16)
    split        : str   — 'train' or 'val'
    val_split    : float — fraction of recordings held out for validation (default 0.2)
    frames       : int   — STFT time frames per epoch (default 5)
    feat         : int   — feature bins per frame after concatenation (default 130)
    """

    def __init__(self, data_dir, win_len=90, step=60,
                 rem_repeat=3, art_repeat=1, cache_size=16,
                 split='train', val_split=0.2,
                 frames=5, feat=130,
                 inject_p=0.0, wake_share=0.8, inject_seed=0):
        super().__init__()
        self.data_dir    = data_dir
        self.win_len     = win_len
        self.split       = split
        self.cache_size  = cache_size
        self.frames      = frames
        self.feat        = feat
        self.art_repeat  = art_repeat
        self.inject_p    = float(inject_p)
        self.wake_share  = float(wake_share)
        self._inject_rng = (np.random.RandomState(inject_seed)
                            if self.inject_p > 0 else None)
        self._cache       = {}   # {rec_idx: feature memmap}
        self._score_cache = {}   # {rec_idx: score memmap}

        # ── 1. Discover recordings by scanning for *_score.npy files ──────────
        score_files = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith('_score.npy')
        )
        if not score_files:
            raise FileNotFoundError(
                f"No *_score.npy files found in {data_dir}. Run Data_compile.py first.")

        all_names = np.array([f[:-len('_score.npy')] for f in score_files], dtype=object)
        # Lengths from score files (reads only the numpy header, not the full array)
        all_lengths = np.array([
            np.load(os.path.join(data_dir, f), mmap_mode='r').shape[0]
            for f in score_files
        ], dtype=np.int64)
        n_total = len(all_names)

        # ── 2. Deterministic train/val split (by recording) ───────────────────
        rng = np.random.RandomState(42)
        idx = np.arange(n_total)
        rng.shuffle(idx)
        n_val = int(n_total * val_split)

        chosen = idx[n_val:] if split == 'train' else idx[:n_val]

        self.rec_names   = all_names[chosen]
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

        # ── 4. Oversample REM and artifact windows (train only) ───────────────
        if split == 'train':
            rem_idx   = []
            art_idx   = []
            plain_idx = []
            for i, (ri, s) in enumerate(self._index):
                labels  = self._get_score(ri)[s : s + win_len]
                has_rem = np.any(labels == 3)   # stored 1-based: 3=REM
                has_art = np.any(labels == 4)   # stored 1-based: 4=Artifact
                if has_rem:
                    rem_idx.append(i)
                if has_art:
                    art_idx.append(i)
                if not has_rem and not has_art:
                    plain_idx.append(i)
            oversampled = plain_idx + rem_idx * rem_repeat + art_idx * art_repeat
            rng2 = np.random.RandomState(0)
            rng2.shuffle(oversampled)
            self._sample_ids = oversampled
            print(f"  {len(plain_idx)} plain + {len(rem_idx)}×{rem_repeat} REM "
                  f"+ {len(art_idx)}×{art_repeat} Art = {len(oversampled)} windows")
        else:
            self._sample_ids = list(range(len(self._index)))
            print(f"  {len(self._sample_ids)} windows (no oversampling)")

    # ── Pickle support for DataLoader workers ─────────────────────────────────
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache']       = {}   # don't pickle open file handles
        state['_score_cache'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_cache' not in self.__dict__:
            self._cache = {}
        if '_score_cache' not in self.__dict__:
            self._score_cache = {}

    def __len__(self):
        return len(self._sample_ids)

    # ── memmap LRU caches ─────────────────────────────────────────────────────
    def _open_memmap(self, rec_idx):
        name = str(self.rec_names[rec_idx])
        return np.load(os.path.join(self.data_dir, f"{name}.npy"), mmap_mode='r')

    def _open_score_memmap(self, rec_idx):
        name = str(self.rec_names[rec_idx])
        return np.load(os.path.join(self.data_dir, f"{name}_score.npy"), mmap_mode='r')

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

    def _get_score(self, rec_idx):
        if rec_idx in self._score_cache:
            sc = self._score_cache.pop(rec_idx)
            self._score_cache[rec_idx] = sc   # move to end (LRU)
            return sc
        sc = self._open_score_memmap(rec_idx)
        if len(self._score_cache) >= self.cache_size:
            del self._score_cache[next(iter(self._score_cache))]
        self._score_cache[rec_idx] = sc
        return sc

    def _safe_slice(self, rec_idx, local_start):
        """Read one window with retry + stale-handle recovery."""
        max_retries = 6
        delay = 0.5
        for attempt in range(max_retries):
            try:
                X = self._get_data(rec_idx)[local_start : local_start + self.win_len].copy()
                Y = self._get_score(rec_idx)[local_start : local_start + self.win_len].copy()
                return X, Y
            except (OSError, IOError, ValueError):
                if rec_idx in self._cache:
                    del self._cache[rec_idx]
                if rec_idx in self._score_cache:
                    del self._score_cache[rec_idx]
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 1.5

    def __getitem__(self, idx):
        rec_idx, local_start = self._index[self._sample_ids[idx]]
        X_flat, Y = self._safe_slice(rec_idx, local_start)

        # Reshape [win_len, frames*feat] → [win_len, frames, feat]
        X = X_flat.reshape(self.win_len, self.frames, self.feat)

        # Convert stored 1-based labels → 0-based (0=Wake, 1=NREM, 2=REM, 3=Artifact)
        # Stored: 1=Wake  2=NREM  3=REM  4=Artifact  -100=ignore
        Y = Y.astype(np.int64).copy()
        valid = Y != -100
        Y[valid] = Y[valid] - 1   # → 0=Wake  1=NREM  2=REM  3=Artifact

        X = np.ascontiguousarray(X, dtype=np.float32)
        if self._inject_rng is not None:
            inject_artifacts(X, Y,
                             inject_p=self.inject_p,
                             wake_share=self.wake_share,
                             rng=self._inject_rng)

        return (torch.from_numpy(X),
                torch.from_numpy(Y))
