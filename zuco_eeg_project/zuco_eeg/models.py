"""EEG encoders: hand-crafted band power baseline, patch-Transformer (random / self-supervised),
optional pretrained CBraMod wrapper, and the contrastive projection head.

Every encoder has the interface  enc(x, n) -> (B, enc.out_dim)  with x (B,C,T) *scaled but not
standardised* EEG and n (B,) valid lengths; normalisation / quality estimation happen inside.
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quality import quality_features, standardize, valid_mask

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]  # delta theta alpha beta gamma


class BandPower(nn.Module):
    """Baseline: log band power per channel over the whole trial (no parameters)."""

    def __init__(self, fs, T, n_ch):
        super().__init__()
        self.T, self.out_dim = T, len(BANDS) * n_ch
        f = torch.fft.rfftfreq(T, 1.0 / fs)
        self.register_buffer("mk", torch.stack([((f >= lo) & (f < hi)).float() for lo, hi in BANDS]))

    def forward(self, x, n):
        m = valid_mask(n, self.T)
        X = torch.fft.rfft(x * m[:, None, :], dim=-1)
        P = X.abs() ** 2 / n.clamp_min(1)[:, None, None]
        bp = torch.einsum("bcf,kf->bck", P, self.mk) / self.mk.sum(1)
        return torch.log(bp + 1e-6).flatten(1)


class PatchEncoder(nn.Module):
    """Patch-Transformer over time. A token = all channels x `patch` samples (0.25 s at 128 Hz).
    Supports masked-patch reconstruction (self-supervised) and optional quality-aware FiLM
    conditioning on estimated signal quality."""

    def __init__(self, n_ch, T, patch=32, d=256, depth=4, heads=4, drop=0.1, qaware=False):
        super().__init__()
        assert T % patch == 0
        self.n_ch, self.T, self.patch, self.n_tok = n_ch, T, patch, T // patch
        self.out_dim, self.qaware = d, qaware
        self.embed = nn.Sequential(nn.Linear(n_ch * patch, d), nn.LayerNorm(d))
        self.pos = nn.Parameter(torch.randn(1, self.n_tok, d) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.recon = nn.Linear(d, n_ch * patch)  # SSL head
        if qaware:
            self.qmlp = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 2 * d))
            nn.init.zeros_(self.qmlp[-1].weight)
            nn.init.zeros_(self.qmlp[-1].bias)

    def patchify(self, xn):
        B, C, _ = xn.shape
        return xn.reshape(B, C, self.n_tok, self.patch).permute(0, 2, 1, 3).reshape(B, self.n_tok, C * self.patch)

    def patch_valid(self, n):
        return (torch.arange(self.n_tok, device=n.device) * self.patch)[None, :] < n[:, None]

    def tokens(self, x, n, patch_mask=None):
        m = valid_mask(n, self.T)
        xn = standardize(x, m)
        h = self.embed(self.patchify(xn))
        if patch_mask is not None:
            h = torch.where(patch_mask[..., None], self.mask_token.expand_as(h), h)
        if self.qaware:
            g, b = self.qmlp(quality_features(x, m)).chunk(2, -1)
            h = h * (1 + g[:, None]) + b[:, None]
        v = self.patch_valid(n)
        h = self.tr(h + self.pos, src_key_padding_mask=~v)
        return self.norm(h), v

    def forward(self, x, n):
        h, v = self.tokens(x, n)
        w = v[..., None].float()
        return (h * w).sum(1) / w.sum(1).clamp_min(1)

    def ssl_loss(self, x_in, x_clean, n, ratio=0.5):
        """Masked-patch reconstruction. x_in may be degraded (denoising); the target is the
        standardised *clean* signal."""
        v = self.patch_valid(n)
        pm = (torch.rand(v.shape, device=v.device) < ratio) & v
        if pm.sum() == 0:
            pm = v.clone()
        h, _ = self.tokens(x_in, n, pm)
        tgt = self.patchify(standardize(x_clean, valid_mask(n, self.T)))
        return ((self.recon(h) - tgt) ** 2).mean(-1)[pm].mean()


class CBraModWrapper(nn.Module):
    """Frozen pretrained CBraMod (github.com/wjq-learning/CBraMod). UNTESTED here (no network access
    in the authoring sandbox) -- check the repo's README if the constructor signature changed.
    CBraMod expects 200-sample (1 s @200 Hz) patches, signals in units of ~100 uV, channels in any number."""

    def __init__(self, repo, weights, fs, in_scale=0.3, max_sec=30):
        super().__init__()
        sys.path.insert(0, repo)
        from models.cbramod import CBraMod  # noqa
        self.m = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                         seq_len=30, n_layer=12, nhead=8)
        self.m.load_state_dict(torch.load(weights, map_location="cpu"))
        self.m.proj_out = nn.Identity()
        self.fs, self.in_scale, self.out_dim, self.max_sec = fs, in_scale, 200, max_sec

    def forward(self, x, n):
        m = valid_mask(n, x.shape[-1])
        xn = standardize(x, m) * self.in_scale
        r = 200.0 / self.fs
        xr = F.interpolate(xn, scale_factor=r, mode="linear", align_corners=False)
        k = min(xr.shape[-1] // 200, self.max_sec)
        B, C, _ = xr.shape
        z = self.m(xr[..., : k * 200].reshape(B, C, k, 200))  # (B,C,k,200)
        nsec = ((n.float() / self.fs).ceil().clamp(1, k)).long()
        v = (torch.arange(k, device=x.device)[None, :] < nsec[:, None]).float()[:, None, :, None]
        return (z * v).sum((1, 2)) / (v.sum((1, 2)) * C).clamp_min(1)


class Projector(nn.Module):
    """EEG feature -> sentence-embedding space (L2-normalised) + learnable temperature."""

    def __init__(self, in_dim, out_dim, hidden=1024, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(hidden, out_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, f):
        return F.normalize(self.net(f), dim=-1)


def info_nce(z, t, codes, logit_scale):
    """Symmetric InfoNCE. Trials of the same sentence in a batch (other subjects) are not negatives."""
    logits = logit_scale.exp().clamp(max=100) * z @ t.T
    same = codes[:, None] == codes[None, :]
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(same & ~eye, -1e4)
    y = torch.arange(len(z), device=z.device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
