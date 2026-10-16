"""EEG signal-quality estimation + synthetic degradations (all torch, batched).

Shapes: x (B, C, T) float, n (B,) valid lengths, m (B, T) bool validity mask.
"""
import math
import torch

TWO_PI = 2 * math.pi


def valid_mask(n, T):
    return torch.arange(T, device=n.device)[None, :] < n[:, None]


def masked_std(x, m):
    mf = m[:, None, :].to(x.dtype)
    cnt = mf.sum(-1).clamp_min(1)
    mu = (x * mf).sum(-1) / cnt
    var = (((x - mu[..., None]) ** 2) * mf).sum(-1) / cnt
    return var.clamp_min(0).sqrt()


def standardize(x, m, clip=5.0):
    """Per-trial, per-channel z-score over the valid samples, clipped; padding stays 0."""
    mf = m[:, None, :].to(x.dtype)
    cnt = mf.sum(-1, keepdim=True).clamp_min(1)
    mu = (x * mf).sum(-1, keepdim=True) / cnt
    sd = masked_std(x, m)[..., None].clamp_min(1e-3)
    return ((x - mu) / sd).clamp(-clip, clip) * mf


def quality_features(x, m):
    """Cheap, differentiable-free quality descriptor computed on the *un-standardized* signal.
    Returns (B, 3): [fraction of bad channels, high-frequency ratio (median over channels),
    log median channel std].  Noise raises hf-ratio / amplitude, dead or exploding channels
    raise the bad-channel fraction."""
    sd = masked_std(x, m)
    med = sd.median(dim=1).values.clamp_min(1e-6)
    bad = ((sd > 3 * med[:, None]) | (sd < 0.1 * med[:, None]) | (sd < 1e-3)).float().mean(1)
    dx = x[..., 1:] - x[..., :-1]
    m2 = m[:, 1:] & m[:, :-1]
    hf = (masked_std(dx, m2) / sd.clamp_min(1e-6)).median(dim=1).values
    return torch.stack([bad, hf, torch.log(med)], dim=1)


def noise_like(x, m):
    """Noise-input control: Gaussian noise with the same per-channel scale and the same
    trial length as the real EEG (so duration/amplitude cues are preserved, EEG content is not)."""
    sd = masked_std(x, m)[..., None]
    return torch.randn_like(x) * sd * m[:, None, :].to(x.dtype)


def degrade(x, m, kind, level, fs=128):
    """Apply one degradation. `level` is a float or a (B,) tensor.
    gauss   : white noise, std = level * channel std   (SNR = -20 log10(level) dB)
    chdrop  : each channel zeroed with probability `level` (dead electrodes)
    line    : 50 Hz mains interference, amplitude = level * channel std
    artifact: 1-s high-amplitude burst (blink/muscle-like) on ~30% of channels, amp = 5*level*std
    """
    if kind in (None, "none", "clean"):
        return x
    B, C, T = x.shape
    dev = x.device
    lv = torch.as_tensor(level, dtype=x.dtype, device=dev).expand(B)
    lvv = lv[:, None, None]
    mf = m[:, None, :].to(x.dtype)
    sd = masked_std(x, m)[..., None]
    if kind == "gauss":
        return x + lvv * sd * torch.randn_like(x) * mf
    if kind == "chdrop":
        keep = (torch.rand(B, C, 1, device=dev) >= lvv).to(x.dtype)
        return x * keep
    if kind == "line":
        t = torch.arange(T, device=dev) / fs
        ph = torch.rand(B, 1, 1, device=dev) * TWO_PI
        return x + lvv * sd * torch.sin(TWO_PI * 50 * t[None, None, :] + ph) * mf
    if kind == "artifact":
        n = m.sum(1).to(x.dtype)
        start = (torch.rand(B, device=dev) * (n - fs).clamp_min(1)).long()
        t = torch.arange(T, device=dev)[None, :]
        w = ((t >= start[:, None]) & (t < start[:, None] + fs)).to(x.dtype)[:, None, :]
        chm = (torch.rand(B, C, 1, device=dev) < 0.3).to(x.dtype)
        return x + 5 * lvv * sd * torch.randn_like(x) * w * chm * mf
    raise ValueError(kind)


LEVEL_RANGE = {"gauss": (0.5, 3.0), "chdrop": (0.05, 0.4), "line": (0.5, 3.0), "artifact": (0.5, 2.0)}


def random_degrade(x, m, p_clean=0.3, fs=128):
    """Per-sample random degradation (mixed conditions) used for augmentation and mixed-quality evaluation."""
    B = x.shape[0]
    kinds = list(LEVEL_RANGE)
    choice = torch.randint(0, len(kinds), (B,), device=x.device)
    clean = torch.rand(B, device=x.device) < p_clean
    out = x
    for k, kind in enumerate(kinds):
        lo, hi = LEVEL_RANGE[kind]
        lvl = lo + (hi - lo) * torch.rand(B, device=x.device)
        xk = degrade(x, m, kind, lvl, fs)
        sel = ((choice == k) & ~clean)[:, None, None]
        out = torch.where(sel, xk, out)
    return out
