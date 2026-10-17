"""ZuCo 1.0 / 2.0 loading, preprocessing, caching, splitting."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset

FS_IN = 500  # ZuCo sampling rate
FNAME_RE = re.compile(r"results([A-Za-z0-9]+)_([A-Za-z]+)\.mat$")


# ----------------------------------------------------------------------------- loading
def _h5_text(f, ref):
    arr = np.array(f[ref]).reshape(-1)
    return "".join(chr(int(c)) for c in arr if int(c) > 0)


def iter_sentences(path):
    """Yield (sentence_text, eeg[C, T] float32 @500 Hz) from one ZuCo results*.mat file.
    Supports MATLAB v7.3 (HDF5, what ZuCo ships) and legacy .mat. Sentences with no EEG are skipped."""
    path = str(path)
    import h5py
    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            sd = f["sentenceData"]
            c_refs = sd["content"][()].reshape(-1)
            r_refs = sd["rawData"][()].reshape(-1)
            for cr, rr in zip(c_refs, r_refs):
                text = _h5_text(f, cr)
                raw = np.array(f[rr])
                if raw.ndim != 2 or min(raw.shape) < 8:
                    continue
                if raw.shape[0] > raw.shape[1]:  # h5py reverses MATLAB's (C,T) -> (T,C)
                    raw = raw.T
                yield text, raw.astype(np.float32)
    else:
        from scipy.io import loadmat
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
        for s in np.atleast_1d(m["sentenceData"]):
            raw = np.asarray(getattr(s, "rawData", np.nan))
            if raw.ndim != 2 or min(raw.shape) < 8:
                continue
            if raw.shape[0] > raw.shape[1]:
                raw = raw.T
            yield str(s.content), raw.astype(np.float32)


def inspect_mat(path):
    import h5py
    if h5py.is_hdf5(str(path)):
        with h5py.File(str(path), "r") as f:
            print("HDF5 file; top-level:", list(f.keys()))
            sd = f["sentenceData"]
            print("sentenceData fields:", list(sd.keys()))
            for k in ("content", "rawData"):
                print(k, "dataset shape:", sd[k].shape, sd[k].dtype)
            it = iter_sentences(path)
            t, x = next(it)
            print("first sentence:", repr(t[:80]), "| EEG shape (C,T):", x.shape)
    else:
        it = iter_sentences(path)
        t, x = next(it)
        print("legacy .mat; first sentence:", repr(t[:80]), x.shape)


def norm_text(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def sent_id(t):
    return hashlib.md5(norm_text(t).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- preprocessing
def preprocess(raw, fs_in=FS_IN, fs_out=128, band=(0.5, 45.0), notch=50.0):
    """raw (C,T)@fs_in -> filtered, resampled (C,T')@fs_out.
    NaN->0, demean, 50 Hz notch, 0.5-45 Hz zero-phase Butterworth(4), polyphase resample."""
    x = np.nan_to_num(raw.astype(np.float64))
    x = x - x.mean(1, keepdims=True)
    b, a = signal.iirnotch(notch, 30.0, fs_in)
    x = signal.filtfilt(b, a, x, axis=1)
    sos = signal.butter(4, band, btype="band", fs=fs_in, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=1)
    g = int(np.gcd(int(fs_in), int(fs_out)))
    x = signal.resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=1)
    return x.astype(np.float32)


def build_cache(src_root, cache_root, dataset, fs_out=128, max_sec=12.0, tasks=None, max_files=None):
    """Convert every results<SUBJ>_<TASK>.mat under src_root into one float16 .npy per file
    (N, C, Tmax) + a meta.csv.  Trials longer than max_sec are cropped, shorter are zero-padded
    (true length kept in column n)."""
    out = Path(cache_root) / dataset
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(src_root).rglob("results*.mat"))
    if max_files:
        files = files[:max_files]
    Tmax = int(fs_out * max_sec)
    rows, n_ch = [], None
    for fp in files:
        mt = FNAME_RE.search(fp.name)
        if not mt:
            continue
        subj, task = mt.group(1), mt.group(2).upper()
        if tasks and task not in tasks:
            continue
        xs, meta = [], []
        for text, raw in iter_sentences(fp):
            if raw.shape[1] < FS_IN:  # < 1 s
                continue
            if n_ch is None:
                n_ch = raw.shape[0]
            if raw.shape[0] != n_ch:
                print(f"  skip trial in {fp.name}: {raw.shape[0]} channels != {n_ch}")
                continue
            x = preprocess(raw, FS_IN, fs_out)
            n = min(x.shape[1], Tmax)
            buf = np.zeros((n_ch, Tmax), np.float16)
            buf[:, :n] = x[:, :n]
            xs.append(buf)
            meta.append((text, n))
        if not xs:
            continue
        name = f"{subj}_{task}.npy"
        np.save(out / name, np.stack(xs))
        for i, (text, n) in enumerate(meta):
            rows.append(dict(file=f"{dataset}/{name}", row=i, dataset=dataset, subject=subj, task=task,
                             text=text, n=n, sid=sent_id(text)))
        print(f"  {fp.name}: {len(xs)} trials")
    meta = pd.DataFrame(rows)
    # global amplitude scale (median over trials of median channel std) so that values are O(1)
    sample = meta.sample(min(len(meta), 1500), random_state=0)
    stds = []
    for _, r in sample.iterrows():
        a = np.load(Path(cache_root) / r.file, mmap_mode="r")[r.row, :, : r.n].astype(np.float32)
        stds.append(np.median(a.std(1)))
    scale = float(np.median(stds))
    meta["scale"] = scale
    meta.to_csv(out / "meta.csv", index=False)
    json.dump(dict(n_ch=int(n_ch), T=Tmax, fs=fs_out, scale=scale, n_trials=len(meta),
                   n_subjects=int(meta.subject.nunique()), n_sentences=int(meta.sid.nunique())),
              open(out / "info.json", "w"), indent=1)
    print(f"{dataset}: {len(meta)} trials, {meta.subject.nunique()} subjects, "
          f"{meta.sid.nunique()} unique sentences, scale={scale:.3g}")


def load_cache(cache_root, dataset, tasks=None):
    d = Path(cache_root) / dataset
    meta = pd.read_csv(d / "meta.csv")
    info = json.load(open(d / "info.json"))
    if tasks:
        meta = meta[meta.task.isin(tasks)].reset_index(drop=True)
    meta["subj_key"] = meta.dataset + "_" + meta.subject
    return meta, info


class ZucoEEG(Dataset):
    """Returns (x[n_ch,T] float32 scaled, n, sentence_code, subject_code, index)."""

    def __init__(self, df, cache_root, n_ch, codes=None):
        self.f, self.r = df.file.values, df.row.values
        self.n = df.n.values.astype(np.int64)
        self.scale = df.scale.values.astype(np.float32)
        self.codes = codes if codes is not None else pd.factorize(df.sid)[0]
        self.subj = pd.factorize(df.subj_key)[0]
        self.root, self.n_ch, self._mm = Path(cache_root), n_ch, {}

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        f = self.f[i]
        if f not in self._mm:
            self._mm[f] = np.load(self.root / f, mmap_mode="r")
        x = np.asarray(self._mm[f][self.r[i], : self.n_ch], dtype=np.float32) / self.scale[i]
        return torch.from_numpy(x), int(self.n[i]), int(self.codes[i]), int(self.subj[i]), i


# ----------------------------------------------------------------------------- splits
def make_splits(meta, seed=0, test_subj=0.2, val_subj=0.15, test_sent=0.2, val_sent=0.15):
    """Subject-held-out AND sentence-held-out splits.

    Every subject reads (nearly) the same sentences, so a subject-only split lets the decoder
    memorise the *text side* of the pairs. We therefore split subjects and sentences independently:
      train           : train subjects x train sentences
      val             : val   subjects x val   sentences
      test            : test  subjects x test  sentences   <- strict: unseen subject AND unseen sentence
      test_seen_sent  : test  subjects x train sentences   <- unseen subject only
      test_seen_subj  : train subjects x test  sentences   <- unseen sentence only
    """
    rng = np.random.RandomState(seed)

    def part(items, f_test, f_val):
        items = np.array(sorted(items))
        rng.shuffle(items)
        nt, nv = max(1, round(len(items) * f_test)), max(1, round(len(items) * f_val))
        return dict(test=set(items[:nt]), val=set(items[nt:nt + nv]), train=set(items[nt + nv:]))

    S = part(meta.subj_key.unique(), test_subj, val_subj)
    E = part(meta.sid.unique(), test_sent, val_sent)

    def sel(s, e):
        return meta[meta.subj_key.isin(S[s]) & meta.sid.isin(E[e])].reset_index(drop=True)

    sp = dict(train=sel("train", "train"), val=sel("val", "val"), test=sel("test", "test"),
              test_seen_sent=sel("test", "train"), test_seen_subj=sel("train", "test"))
    return sp, {k: sorted(v) for k, v in S.items()}
