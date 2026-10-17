"""SSL pre-training, contrastive EEG->sentence decoder training, evaluation + controls."""
import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from .data import ZucoEEG
from .models import BandPower, CBraModWrapper, PatchEncoder, Projector, info_nce
from .quality import degrade, noise_like, quality_features, random_degrade, valid_mask


@dataclass
class Config:
    cache_root: str = "cache"
    out_dir: str = "results"
    datasets: tuple = ("zuco1", "zuco2")
    tasks: tuple = ()               # () = all tasks found in the cache
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 2
    d_model: int = 256
    depth: int = 4
    heads: int = 4
    patch: int = 32
    ssl_epochs: int = 30
    ssl_lr: float = 3e-4
    mask_ratio: float = 0.5
    dec_epochs: int = 12
    patience: int = 4          # early stopping on validation MRR
    wd: float = 0.05
    head_hidden: int = 512
    head_drop: float = 0.3
    lr_head: float = 1e-3
    lr_enc: float = 1e-4
    text_encoder: str = "sbert"     # sbert | tfidf
    cbramod_repo: str = ""
    cbramod_weights: str = ""


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_encoder(cfg, kind, info, qaware=False):
    if kind == "bandpower":
        return BandPower(info["fs"], info["T"], info["n_ch"])
    if kind == "patch":
        return PatchEncoder(info["n_ch"], info["T"], cfg.patch, cfg.d_model, cfg.depth, cfg.heads, qaware=qaware)
    if kind == "cbramod":
        return CBraModWrapper(cfg.cbramod_repo, cfg.cbramod_weights, info["fs"])
    raise ValueError(kind)


def _loader(cfg, df, info, shuffle, codes=None):
    ds = ZucoEEG(df, cfg.cache_root, info["n_ch"], codes)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=shuffle and len(ds) >= cfg.batch_size,
                      num_workers=cfg.num_workers)


# ----------------------------------------------------------------------------- SSL
def ssl_pretrain(cfg, train_df, info, qaware=False, aug=False, path=None, log=print):
    """Masked-patch (denoising) reconstruction on ZuCo train trials only (no test subjects/sentences)."""
    enc = build_encoder(cfg, "patch", info, qaware)
    if path and Path(path).exists():
        enc.load_state_dict(torch.load(path, map_location="cpu"))
        log(f"  loaded SSL weights {path}")
        return enc
    dev = cfg.device
    enc.to(dev).train()
    dl = _loader(cfg, train_df, info, True)
    opt = torch.optim.AdamW(enc.parameters(), lr=cfg.ssl_lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, cfg.ssl_epochs * len(dl)))
    for ep in range(cfg.ssl_epochs):
        tot, k = 0.0, 0
        for x, n, *_ in dl:
            x, n = x.to(dev), n.to(dev)
            xi = random_degrade(x, valid_mask(n, x.shape[-1])) if aug else x
            loss = enc.ssl_loss(xi, x, n, cfg.mask_ratio)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, k = tot + loss.item(), k + 1
        log(f"  [ssl] epoch {ep + 1}/{cfg.ssl_epochs} recon-loss {tot / max(k, 1):.4f}")
    enc.cpu()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(enc.state_dict(), path)
    return enc


# ----------------------------------------------------------------------------- decoder
def _text_matrix(tv, uniq, dev):
    return torch.tensor(np.stack([tv[s] for s in uniq]), dtype=torch.float32, device=dev)


def train_decoder(cfg, enc, finetune, train_df, val_df, tv, info, aug=False, noise_input=False, log=print):
    """Contrastively align EEG features with sentence embeddings. Frozen encoder => only the head trains."""
    dev = cfg.device
    enc.to(dev)
    for p in enc.parameters():
        p.requires_grad_(finetune)
    codes, uniq = pd.factorize(train_df.sid)
    Tm = _text_matrix(tv, uniq, dev)
    dl = _loader(cfg, train_df, info, True, codes)
    head = Projector(enc.out_dim, Tm.shape[1], cfg.head_hidden, cfg.head_drop).to(dev)
    groups = [dict(params=head.parameters(), lr=cfg.lr_head)]
    enc_params = [p for p in enc.parameters() if p.requires_grad]
    if enc_params:
        groups.append(dict(params=enc_params, lr=cfg.lr_enc))
    opt = torch.optim.AdamW(groups, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, cfg.dec_epochs * len(dl)))
    best, best_state, bad_ep = -1.0, None, 0
    for ep in range(cfg.dec_epochs):
        head.train()
        enc.train(finetune)
        tot, k = 0.0, 0
        for x, n, c, s, i in dl:
            x, n, c = x.to(dev), n.to(dev), c.to(dev)
            m = valid_mask(n, x.shape[-1])
            if noise_input:
                x = noise_like(x, m)
            elif aug:
                x = random_degrade(x, m)
            with torch.set_grad_enabled(finetune):
                f = enc(x, n)
            loss = info_nce(head(f), Tm[c], c, head.logit_scale)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot, k = tot + loss.item(), k + 1
        msg = f"  [dec] epoch {ep + 1}/{cfg.dec_epochs} loss {tot / max(k, 1):.4f}"
        if len(val_df):
            mt, _ = evaluate(cfg, enc, head, val_df, tv, info, noise_input=noise_input)
            msg += f" | val top5 {mt['top5']:.3f} mrr {mt['mrr']:.3f}"
            bad_ep = 0 if mt["mrr"] > best else bad_ep + 1
            if mt["mrr"] > best:
                best = mt["mrr"]
                best_state = (copy.deepcopy(head.state_dict()), copy.deepcopy(enc.state_dict()) if finetune else None)
        log(msg)
        if bad_ep >= cfg.patience:
            log("  early stop")
            break
    if best_state is not None:
        head.load_state_dict(best_state[0])
        if best_state[1] is not None:
            enc.load_state_dict(best_state[1])
    return enc, head


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def embed_eeg(cfg, enc, head, df, info, degradation=None, noise_input=False, seed=123):
    """degradation: None | (kind, level) | 'mixed'. Returns Z (N,D), sentence codes, subj codes, q (N,3)."""
    dev = cfg.device
    enc.to(dev).eval()
    head.to(dev).eval()
    torch.manual_seed(seed)
    codes, uniq = pd.factorize(df.sid)
    dl = _loader(cfg, df, info, False, codes)
    Z, C, S, Q = [], [], [], []
    for x, n, c, s, i in dl:
        x, n = x.to(dev), n.to(dev)
        m = valid_mask(n, x.shape[-1])
        if noise_input:
            x = noise_like(x, m)
        elif degradation == "mixed":
            x = random_degrade(x, m, p_clean=0.0, fs=info["fs"])
        elif degradation:
            x = degrade(x, m, degradation[0], degradation[1], info["fs"])
        Q.append(quality_features(x, m).cpu())
        Z.append(head(enc(x, n)).cpu())
        C.append(c)
        S.append(s)
    return torch.cat(Z), torch.cat(C), torch.cat(S), torch.cat(Q), uniq


def retrieval_metrics(Z, codes, subj, Tc, n_perm=200, seed=0):
    N, Ns = len(Z), len(Tc)
    sim = Z @ Tc.T
    true = sim[torch.arange(N), codes]
    rank = (sim > true[:, None]).sum(1) + 1
    out = dict(n_trials=N, n_candidates=Ns,
               top1=(rank <= 1).float().mean().item(), top5=(rank <= 5).float().mean().item(),
               top10=(rank <= 10).float().mean().item(), mrr=(1.0 / rank.float()).mean().item(),
               median_rank=rank.float().median().item(),
               chance_top1=1.0 / Ns, chance_top5=min(5, Ns) / Ns)
    # semantic alignment: predicted-embedding similarity to true sentence vs to the other candidates
    out["cos_true"] = true.mean().item()
    out["cos_gap"] = (true.mean() - (sim.sum() - true.sum()) / (N * max(Ns - 1, 1))).item()
    # per-subject spread
    ps = [(rank[subj == s] <= 5).float().mean().item() for s in subj.unique()]
    out["top5_subj_mean"], out["top5_subj_std"], out["top5_subj_min"] = float(np.mean(ps)), float(np.std(ps)), float(np.min(ps))
    # sentence-averaged decoding: average the embeddings of all trials of the same sentence
    # (different subjects) -> higher signal-to-noise; one prediction per sentence
    if Ns >= 4:
        Ma = torch.nn.functional.normalize(torch.zeros(Ns, Z.shape[1]).index_add_(0, codes, Z), dim=-1)
        sa = Ma @ Tc.T
        ra = (sa > sa.diag()[:, None]).sum(1) + 1
        out.update(avg_top1=(ra <= 1).float().mean().item(), avg_top5=(ra <= 5).float().mean().item(),
                   avg_mrr=(1.0 / ra.float()).mean().item(), avg_chance_top5=min(5, Ns) / Ns)
    # permutation null for top-5
    g = torch.Generator().manual_seed(seed)
    null = []
    for _ in range(n_perm):
        idx = torch.randint(Ns, (N,), generator=g)
        null.append(((sim > sim[torch.arange(N), idx][:, None]).sum(1) + 1 <= 5).float().mean().item())
    out["top5_perm_p"] = (1 + sum(v >= out["top5"] for v in null)) / (n_perm + 1)
    # representational similarity (RSA): pairwise geometry of mean predicted vs true sentence embeddings
    if Ns >= 4:
        M = torch.zeros(Ns, Z.shape[1]).index_add_(0, codes, Z)
        M = torch.nn.functional.normalize(M, dim=-1)
        iu = np.triu_indices(Ns, 1)
        out["rsa_spearman"] = float(spearmanr((M @ M.T).numpy()[iu], (Tc @ Tc.T).numpy()[iu])[0])
    return out, rank.numpy()


def evaluate(cfg, enc, head, df, tv, info, degradation=None, noise_input=False, return_trials=False):
    Z, c, s, q, uniq = embed_eeg(cfg, enc, head, df, info, degradation, noise_input)
    Tc = torch.tensor(np.stack([tv[u] for u in uniq]), dtype=torch.float32)
    mt, rank = retrieval_metrics(Z, c, s, Tc)
    if return_trials:
        return mt, pd.DataFrame(dict(rank=rank, rr=1.0 / rank, bad=q[:, 0].numpy(), hf=q[:, 1].numpy(),
                                     logstd=q[:, 2].numpy()))
    return mt, None
