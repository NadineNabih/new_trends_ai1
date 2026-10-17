"""The three experiment families of the project:
  representations : baseline vs random vs SSL(ZuCo) vs fine-tuned vs (optional) pretrained CBraMod, + noise controls
  robustness      : signal-quality degradations, quality-aware vs plain models
  cross           : ZuCo1 <-> ZuCo2 transfer
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .data import load_cache, make_splits
from .text import text_vectors
from .train import (Config, build_encoder, evaluate, set_seed, ssl_pretrain, train_decoder)

# name: (encoder kind, SSL-initialised?, fine-tune encoder?)
RECIPES = {
    "bandpower": ("bandpower", False, False),        # hand-crafted baseline
    "random_frozen": ("patch", False, False),        # random-init Transformer, frozen (sanity floor)
    "ssl_frozen": ("patch", True, False),            # SSL on ZuCo, frozen
    "ssl_finetune": ("patch", True, True),           # SSL on ZuCo, then fine-tuned with the decoder loss
    "scratch_finetune": ("patch", False, True),      # supervised only, no SSL
}
GRID = [("gauss", [0.5, 1, 2, 4]), ("chdrop", [0.1, 0.2, 0.4]), ("line", [0.5, 1, 2]), ("artifact", [0.5, 1, 2])]
SPLITS = ["val", "test", "test_seen_sent", "test_seen_subj"]


def _setup(cfg, ds):
    meta, info = load_cache(cfg.cache_root, ds, cfg.tasks or None)
    sp, subj = make_splits(meta, cfg.seed)
    tv = text_vectors(meta, cfg.text_encoder, cfg.out_dir)
    print(f"[{ds}] {len(meta)} trials | subjects {meta.subj_key.nunique()} | sentences {meta.sid.nunique()} | "
          + " ".join(f"{k}={len(v)}" for k, v in sp.items()))
    return meta, info, sp, subj, tv


def _arch(cfg):
    return f"d{cfg.d_model}l{cfg.depth}p{cfg.patch}e{cfg.ssl_epochs}"


def _ssl_path(cfg, ds, qa, aug):
    return Path(cfg.out_dir) / "ckpt" / f"ssl_{ds}_q{int(qa)}_a{int(aug)}_{_arch(cfg)}_s{cfg.seed}.pt"


def _save(rows, cfg, name):
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    name = name.replace(".csv", f"_s{cfg.seed}.csv")
    pd.DataFrame(rows).assign(seed=cfg.seed).to_csv(Path(cfg.out_dir) / name, index=False)


# ----------------------------------------------------------------------------- 1. representations
def exp_representations(cfg: Config, ds, recipes=None):
    meta, info, sp, subj, tv = _setup(cfg, ds)
    recipes = list(recipes or RECIPES)
    if cfg.cbramod_weights:
        recipes.append("cbramod_frozen")
    rows = []
    for name in recipes:
        print(f"\n=== {ds} :: {name}")
        set_seed(cfg.seed)
        if name == "cbramod_frozen":
            kind, use_ssl, ft = "cbramod", False, False
        else:
            kind, use_ssl, ft = RECIPES[name]
        if use_ssl:
            enc = ssl_pretrain(cfg, sp["train"], info, path=_ssl_path(cfg, ds, False, False))
        else:
            enc = build_encoder(cfg, kind, info)
        enc, head = train_decoder(cfg, enc, ft, sp["train"], sp["val"], tv, info)
        for split in SPLITS:
            if len(sp[split]):
                mt, _ = evaluate(cfg, enc, head, sp[split], tv, info)
                rows.append(dict(dataset=ds, recipe=name, split=split, control="none", **mt))
        mt, _ = evaluate(cfg, enc, head, sp["test"], tv, info, noise_input=True)
        rows.append(dict(dataset=ds, recipe=name, split="test", control="noise_input_at_test", **mt))
        _save(rows, cfg, f"representations_{ds}.csv")
        print(pd.DataFrame(rows).query("recipe==@name and split=='test'")[["control", "top1", "top5", "mrr", "cos_gap"]])
    # noise-trained control: same architecture, trained only on length/amplitude-matched noise
    print(f"\n=== {ds} :: noise_train control")
    set_seed(cfg.seed)
    enc = build_encoder(cfg, "patch", info)
    enc, head = train_decoder(cfg, enc, True, sp["train"], sp["val"], tv, info, noise_input=True)
    for ctl, kw in (("noise_trained/noise_test", dict(noise_input=True)), ("noise_trained/real_test", {})):
        mt, _ = evaluate(cfg, enc, head, sp["test"], tv, info, **kw)
        rows.append(dict(dataset=ds, recipe="noise_train_control", split="test", control=ctl, **mt))
    _save(rows, cfg, f"representations_{ds}.csv")
    return rows


# ----------------------------------------------------------------------------- 2. robustness
VARIANTS = {"plain": (False, False), "aug": (True, False), "aug_qaware": (True, True)}  # (augment, quality-aware)


def exp_robustness(cfg: Config, ds):
    meta, info, sp, subj, tv = _setup(cfg, ds)
    rows, corr = [], []
    for name, (aug, qa) in VARIANTS.items():
        print(f"\n=== {ds} :: robustness variant {name}")
        set_seed(cfg.seed)
        enc = ssl_pretrain(cfg, sp["train"], info, qaware=qa, aug=aug, path=_ssl_path(cfg, ds, qa, aug))
        enc, head = train_decoder(cfg, enc, True, sp["train"], sp["val"], tv, info, aug=aug)
        te = sp["test"]
        mt, _ = evaluate(cfg, enc, head, te, tv, info)
        rows.append(dict(dataset=ds, variant=name, kind="clean", level=0.0, **mt))
        for kind, levels in GRID:
            for lv in levels:
                mt, _ = evaluate(cfg, enc, head, te, tv, info, degradation=(kind, lv))
                rows.append(dict(dataset=ds, variant=name, kind=kind, level=lv, **mt))
        mt, tr = evaluate(cfg, enc, head, te, tv, info, degradation="mixed", return_trials=True)
        rows.append(dict(dataset=ds, variant=name, kind="mixed", level=-1.0, **mt))
        # does the *estimated* signal quality predict decoding success?
        for col in ("bad", "hf", "logstd"):
            corr.append(dict(dataset=ds, variant=name, quality_feature=col,
                             spearman_with_reciprocal_rank=float(spearmanr(tr[col], tr.rr)[0])))
        _save(rows, cfg, f"robustness_{ds}.csv")
        _save(corr, cfg, f"quality_vs_performance_{ds}.csv")
    return rows


# ----------------------------------------------------------------------------- 3. cross-dataset
def exp_cross(cfg: Config, a, b):
    mA, iA, spA, _, tvA = _setup(cfg, a)
    mB, iB, spB, _, tvB = _setup(cfg, b)
    assert iA["T"] == iB["T"] and iA["fs"] == iB["fs"], "caches must share T and fs"
    info = dict(iA, n_ch=min(iA["n_ch"], iB["n_ch"]))
    tv = {**tvA, **tvB}
    rows = []
    # sentences of B that also occur in A are excluded from every cross-dataset test (no text leakage)
    novel = mB[~mB.sid.isin(set(mA.sid))].reset_index(drop=True)
    print(f"[{a}->{b}] {len(novel)}/{len(mB)} {b} trials have sentences unseen in {a}")
    set_seed(cfg.seed)
    ssl_enc = ssl_pretrain(cfg, spA["train"], info, path=Path(cfg.out_dir) / "ckpt" / f"ssl_{a}_for_{b}_{_arch(cfg)}_s{cfg.seed}.pt")
    # (i) zero-shot: SSL+decoder trained on A only, tested on B
    enc, head = train_decoder(cfg, ssl_enc, True, spA["train"], spA["val"], tv, info)
    for nm, df in (("in_domain_test", spA["test"]), (f"zero_shot_on_{b}", novel)):
        mt, _ = evaluate(cfg, enc, head, df, tv, info)
        rows.append(dict(train=a, test=b if "zero" in nm else a, protocol=nm, **mt))
    # (ii) representation transfer: A-pretrained SSL encoder, fine-tuned on B, tested on B (strict split)
    set_seed(cfg.seed)
    ssl_enc = ssl_pretrain(cfg, spA["train"], info, path=Path(cfg.out_dir) / "ckpt" / f"ssl_{a}_for_{b}_{_arch(cfg)}_s{cfg.seed}.pt")
    enc, head = train_decoder(cfg, ssl_enc, True, spB["train"], spB["val"], tv, info)
    mt, _ = evaluate(cfg, enc, head, spB["test"], tv, info)
    rows.append(dict(train=f"SSL@{a}+finetune@{b}", test=b, protocol="ssl_transfer", **mt))
    _save(rows, cfg, f"cross_{a}_to_{b}.csv")
    return rows


# ----------------------------------------------------------------------------- plots
def make_plots(cfg: Config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(cfg.out_dir)
    reps = [pd.read_csv(p) for p in sorted(out.glob("representations_*.csv"))]
    if reps:
        r = pd.concat(reps)
        r = r[(r.split == "test") & (r.control.isin(["none", "noise_input_at_test", "noise_trained/noise_test"]))]
        r = r.groupby(["dataset", "recipe", "control"], as_index=False).mean(numeric_only=True)
        fig, axes = plt.subplots(1, r.dataset.nunique(), figsize=(6 * r.dataset.nunique(), 4), squeeze=False)
        for ax, (ds, g) in zip(axes[0], r.groupby("dataset")):
            g = g.assign(label=g.recipe + np.where(g.control == "none", "", "\n(" + g.control + ")"))
            ax.bar(g.label, g.top5)
            ax.axhline(g.chance_top5.iloc[0], color="r", ls="--", label="chance")
            ax.set_title(f"{ds}: unseen subjects + unseen sentences"); ax.set_ylabel("top-5 retrieval")
            ax.tick_params(axis="x", rotation=75, labelsize=7); ax.legend()
        fig.tight_layout(); fig.savefig(out / "fig_representations.png", dpi=150)
    for p in sorted(out.glob("robustness_*.csv")):
        d = pd.read_csv(p)
        kinds = [k for k, _ in GRID]
        fig, axes = plt.subplots(1, len(kinds), figsize=(4 * len(kinds), 3.5), sharey=True)
        clean = d[d.kind == "clean"]
        for ax, k in zip(axes, kinds):
            for v, g in d[d.kind == k].groupby("variant"):
                c = clean[clean.variant == v]
                g = pd.concat([c.assign(level=0.0), g]).sort_values("level")
                ax.plot(g.level, g.top5, marker="o", label=v)
            ax.set_title(k); ax.set_xlabel("level")
        axes[0].set_ylabel("top-5 retrieval"); axes[0].legend()
        fig.tight_layout(); fig.savefig(out / f"fig_{p.stem}.png", dpi=150)
