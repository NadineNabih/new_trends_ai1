"""Classification tasks that reuse the same encoders, recipes and held-out splits:

  sentiment : 3-class sentence sentiment (ZuCo1 SR labels). Held-out subjects AND sentences.
  subject   : subject identity -- a POSITIVE CONTROL. EEG is highly subject-specific, so this should be
              decodable if the pipeline works. Split = seen subjects x unseen sentences (a classifier
              cannot recognise subjects it never saw).
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import load_cache, make_splits, norm_text
from .experiments import GRID, RECIPES, VARIANTS, _arch, _save
from .quality import degrade, noise_like, random_degrade, valid_mask
from .train import _loader, build_encoder, set_seed, ssl_pretrain


def load_sentiment(path):
    """sentence_id;sentence;control;sentiment_label  (lines starting with # are comments; -1/0/1)."""
    lines = [l for l in open(path, encoding="utf-8") if not l.lstrip().startswith("#")]
    df = pd.read_csv(io.StringIO("".join(lines)), sep=";", on_bad_lines="warn")
    return {norm_text(str(s)): int(l) for s, l in zip(df.sentence, df.sentiment_label) if pd.notna(l)}


def _prep(cfg, ds, task):
    meta, info = load_cache(cfg.cache_root, ds, cfg.tasks or None)
    if task == "sentiment":
        lab = load_sentiment(cfg.labels_csv)
        y = meta.text.map(lambda t: lab.get(norm_text(t)))
        print(f"[{ds}] sentiment labels matched for {y.notna().mean():.1%} of trials")
        assert y.notna().mean() > 0.8, "labels do not match the sentences (check --labels_csv)"
        meta = meta[y.notna()].assign(y=(y[y.notna()] + 1).astype(int)).reset_index(drop=True)
        K = 3
    sp, _ = make_splits(meta if task == "sentiment" else meta.assign(y=0), cfg.seed)
    if task == "subject":
        subs = sorted(sp["train"].subj_key.unique())
        mp = {s: i for i, s in enumerate(subs)}
        K = len(subs)
        val_sent = set(sp["val"].sid)
        sp["val"] = meta[meta.subj_key.isin(subs) & meta.sid.isin(val_sent)]
        for k in ("train", "val", "test_seen_subj"):
            sp[k] = sp[k][sp[k].subj_key.isin(mp)].assign(y=lambda d: d.subj_key.map(mp)).reset_index(drop=True)
        ev = {"test": sp["test_seen_subj"]}
    else:
        ev = {"test": sp["test"], "test_seen_sent": sp["test_seen_sent"]}
    print(f"[{ds}/{task}] classes={K} " + " ".join(f"{k}={len(v)}" for k, v in {**sp}.items() if k in ("train", "val")
          ) + " " + " ".join(f"{k}={len(v)}" for k, v in ev.items()))
    return sp, ev, K, info


class ClsHead(nn.Module):
    def __init__(self, d, K, hidden=256, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Dropout(drop), nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(hidden, K))

    def forward(self, f):
        return self.net(f)


def bal_acc(y, p, K):
    r = [(p[y == k] == k).mean() for k in range(K) if (y == k).any()]
    return float(np.mean(r))


@torch.no_grad()
def _logits(cfg, enc, head, df, info, degradation=None, noise_input=False, seed=123):
    dev = cfg.device
    enc.to(dev).eval()
    head.to(dev).eval()
    torch.manual_seed(seed)
    P = []
    for x, n, c, s, i in _loader(cfg, df, info, False):
        x, n = x.to(dev), n.to(dev)
        m = valid_mask(n, x.shape[-1])
        if noise_input:
            x = noise_like(x, m)
        elif degradation:
            x = degrade(x, m, degradation[0], degradation[1], info["fs"])
        P.append(F.softmax(head(enc(x, n)), -1).cpu())
    return torch.cat(P).numpy()


def evaluate_cls(cfg, enc, head, df, K, info, degradation=None, noise_input=False, n_perm=1000):
    P = _logits(cfg, enc, head, df, info, degradation, noise_input)
    y = df.y.values
    pred = P.argmax(1)
    out = dict(n_trials=len(y), n_classes=K, acc=float((pred == y).mean()), bal_acc=bal_acc(y, pred, K),
               macro_f1=float(f1_score(y, pred, average="macro")), chance_bal_acc=1.0 / K,
               const_pred_acc=float(np.bincount(y, minlength=K).max() / len(y)))
    rng = np.random.RandomState(0)
    null = [bal_acc(rng.permutation(y), pred, K) for _ in range(n_perm)]
    out["bal_acc_perm_p"] = (1 + sum(v >= out["bal_acc"] for v in null)) / (n_perm + 1)
    # sentence-averaged prediction (mean probability over all trials of the same sentence)
    codes = pd.factorize(df.sid)[0]
    Pa = np.zeros((codes.max() + 1, K))
    np.add.at(Pa, codes, P)
    ya = np.zeros(codes.max() + 1, int)
    ya[codes] = y
    out["avg_bal_acc"] = bal_acc(ya, Pa.argmax(1), K)
    return out


def train_classifier(cfg, enc, finetune, tr, va, K, info, aug=False, noise_input=False, log=print):
    dev = cfg.device
    enc.to(dev)
    for p in enc.parameters():
        p.requires_grad_(finetune)
    head = ClsHead(enc.out_dim, K).to(dev)
    Y = torch.tensor(tr.y.values, device=dev)
    cnt = np.bincount(tr.y.values, minlength=K)
    w = torch.tensor(len(tr) / (K * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    dl = _loader(cfg, tr, info, True)
    groups = [dict(params=head.parameters(), lr=cfg.lr_head)]
    ep_ = [p for p in enc.parameters() if p.requires_grad]
    if ep_:
        groups.append(dict(params=ep_, lr=cfg.lr_enc))
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, cfg.dec_epochs * len(dl)))
    best, best_state, bad = -1.0, None, 0
    for ep in range(cfg.dec_epochs):
        head.train()
        enc.train(finetune)
        tot, k = 0.0, 0
        for x, n, c, s, i in dl:
            x, n, i = x.to(dev), n.to(dev), i.to(dev)
            m = valid_mask(n, x.shape[-1])
            if noise_input:
                x = noise_like(x, m)
            elif aug:
                x = random_degrade(x, m)
            with torch.set_grad_enabled(finetune):
                f = enc(x, n)
            loss = F.cross_entropy(head(f), Y[i], weight=w, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, k = tot + loss.item(), k + 1
        msg = f"  [cls] epoch {ep + 1}/{cfg.dec_epochs} loss {tot / max(k, 1):.4f}"
        if len(va):
            v = evaluate_cls(cfg, enc, head, va, K, info, noise_input=noise_input, n_perm=1)["bal_acc"]
            msg += f" | val bal-acc {v:.3f}"
            bad = 0 if v > best else bad + 1
            if v > best:
                import copy
                best = v
                best_state = (copy.deepcopy(head.state_dict()), copy.deepcopy(enc.state_dict()) if finetune else None)
        log(msg)
        if bad >= cfg.patience:
            log("  early stop")
            break
    if best_state is not None:
        head.load_state_dict(best_state[0])
        if best_state[1] is not None:
            enc.load_state_dict(best_state[1])
    return enc, head


def _ssl(cfg, ds, task, sp, info, qa=False, aug=False):
    p = Path(cfg.out_dir) / "ckpt" / f"ssl_{ds}_{task}_q{int(qa)}_a{int(aug)}_{_arch(cfg)}_s{cfg.seed}.pt"
    return ssl_pretrain(cfg, sp["train"], info, qaware=qa, aug=aug, path=p)


def exp_classification(cfg, ds, task, recipes=None):
    sp, ev, K, info = _prep(cfg, ds, task)
    rows = []
    for name in (recipes or RECIPES):
        print(f"\n=== {ds}/{task} :: {name}")
        set_seed(cfg.seed)
        kind, use_ssl, ft = RECIPES[name]
        enc = _ssl(cfg, ds, task, sp, info) if use_ssl else build_encoder(cfg, kind, info)
        enc, head = train_classifier(cfg, enc, ft, sp["train"], sp["val"], K, info)
        for split, df in ev.items():
            rows.append(dict(dataset=ds, task=task, recipe=name, split=split, control="none",
                             **evaluate_cls(cfg, enc, head, df, K, info)))
        rows.append(dict(dataset=ds, task=task, recipe=name, split="test", control="noise_input_at_test",
                         **evaluate_cls(cfg, enc, head, ev["test"], K, info, noise_input=True)))
        _save(rows, cfg, f"cls_{task}_{ds}.csv")
        print(pd.DataFrame(rows).query("recipe==@name and split=='test'")[["control", "bal_acc", "chance_bal_acc", "bal_acc_perm_p"]])
    print(f"\n=== {ds}/{task} :: noise_train control")
    set_seed(cfg.seed)
    enc, head = train_classifier(cfg, build_encoder(cfg, "patch", info), True, sp["train"], sp["val"], K, info, noise_input=True)
    for ctl, kw in (("noise_trained/noise_test", dict(noise_input=True)), ("noise_trained/real_test", {})):
        rows.append(dict(dataset=ds, task=task, recipe="noise_train_control", split="test", control=ctl,
                         **evaluate_cls(cfg, enc, head, ev["test"], K, info, **kw)))
    _save(rows, cfg, f"cls_{task}_{ds}.csv")


def exp_cls_robust(cfg, ds, task):
    """Quality robustness on a task that has real signal (use task=subject, the positive control)."""
    sp, ev, K, info = _prep(cfg, ds, task)
    rows = []
    for name, (aug, qa) in VARIANTS.items():
        print(f"\n=== {ds}/{task} :: robustness variant {name}")
        set_seed(cfg.seed)
        enc = _ssl(cfg, ds, task, sp, info, qa, aug)
        enc, head = train_classifier(cfg, enc, True, sp["train"], sp["val"], K, info, aug=aug)
        te = ev["test"]
        rows.append(dict(dataset=ds, task=task, variant=name, kind="clean", level=0.0, **evaluate_cls(cfg, enc, head, te, K, info)))
        for kind, levels in GRID:
            for lv in levels:
                rows.append(dict(dataset=ds, task=task, variant=name, kind=kind, level=lv,
                                 **evaluate_cls(cfg, enc, head, te, K, info, degradation=(kind, lv), n_perm=200)))
        _save(rows, cfg, f"clsrob_{task}_{ds}.csv")
