"""Writes tiny synthetic files in the ZuCo v7.3 layout (sentenceData.content / .rawData as HDF5 object
references) to smoke-test the whole pipeline without the real data.  The EEG carries a weak
sentence-specific signal; trial durations are random and independent of the sentence."""
import sys, numpy as np, h5py
from pathlib import Path

root, n_subj, n_sent, C = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
rng = np.random.RandomState(seed)
vocab = [f"w{i}" for i in range(80)]
sents = [" ".join(rng.choice(vocab, rng.randint(5, 10))) + "." for _ in range(n_sent)]
code = {s: np.random.RandomState(abs(hash(s)) % 2**31) for s in sents}
root.mkdir(parents=True, exist_ok=True)
subs = ["".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 3)) for _ in range(n_subj)]
for subj in subs:
    with h5py.File(root / f"results{subj}_NR.mat", "w") as f:
        refs = f.create_group("#refs#")
        sd = f.create_group("sentenceData")
        rt = h5py.special_dtype(ref=h5py.Reference)
        cr = sd.create_dataset("content", (n_sent, 1), dtype=rt)
        rr = sd.create_dataset("rawData", (n_sent, 1), dtype=rt)
        for i, s in enumerate(sents):
            g = code[s]
            dur = rng.uniform(3, 6)
            T = int(dur * 500)
            t = np.arange(T) / 500
            x = rng.randn(C, T) * 10
            for k in range(3):
                fr, ph, ch = g.uniform(6, 30), g.uniform(0, 6.28), g.randint(0, C, 4)
                x[ch] += 8 * np.sin(2 * np.pi * fr * t + ph)
            c = refs.create_dataset(f"c{i}", data=np.array([ord(ch) for ch in s], np.uint16).reshape(-1, 1))
            r = refs.create_dataset(f"r{i}", data=x.T)  # (T, C) like h5py shows MATLAB (C, T)
            cr[i, 0], rr[i, 0] = c.ref, r.ref
print("wrote", len(subs), "subjects to", root)
