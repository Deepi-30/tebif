"""
models.py - LSTM baseline, vanilla Transformer and TEBIF (time-aware Transformer).

All three read the same window dict produced by data.EventData.batch():
    tok  (B,L)    event token = type*2+role (10 = padding / unknown)
    cont (B,L,F)  normalised continuous features (log amount, balance ratios ...)
    step (B,L)    absolute step of every event (used ONLY to derive time features)
    mask (B,L)    True for real events (False = left padding or a deleted event)
Every forward() returns (logit, attention) - attention is None except for Transformers when
return_attn=True: the last layer's weights, averaged over heads, of the LAST event (the one
we predict at) onto every event in the window.
"""
import math
import torch
import torch.nn as nn


def time_features(step, mask):
    """From absolute steps and the mask derive, for each real event:
         pos   = its index among the real events of the window
         dt    = steps since the previous REAL event (0 for the first)
         ctime = steps since the first real event of the window
       Deleted events (mask False) are simply skipped, so the gaps are recomputed correctly."""
    pos = (mask.long().cumsum(1) - 1).clamp(min=0).float()
    neg = torch.full_like(step, -1e9)
    s = torch.where(mask, step, neg)
    latest = torch.cummax(s, dim=1).values                   # latest real step at or before j
    prev = torch.cat([neg[:, :1], latest[:, :-1]], 1)         # latest real step strictly before j
    dt = torch.where(prev > -1e8, step - prev, torch.zeros_like(step))
    first = torch.where(mask, step, torch.full_like(step, 1e9)).min(1, keepdim=True).values
    ctime = step - first
    z = torch.zeros_like(step)
    return pos, torch.where(mask, dt, z), torch.where(mask, ctime, z)


def sinusoid(x, dim, min_period, max_period):
    """Sinusoidal encoding of a scalar (position, time gap, elapsed time) into `dim` numbers.
    Periods are spaced geometrically, so both 1-hour and 300-hour differences are visible."""
    n = dim // 2
    periods = torch.exp(torch.linspace(math.log(min_period), math.log(max_period), n))
    ang = x.unsqueeze(-1) * (2 * math.pi / periods)
    return torch.cat([ang.sin(), ang.cos()], -1)


class Block(nn.Module):
    """Pre-norm Transformer encoder layer with simple multi-head self-attention written out by
    hand (so the attention weights are easy to return - and it is much faster on CPU than
    nn.MultiheadAttention)."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad, want_attn=False):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-1, -2) / math.sqrt(D // self.h)          # (B, heads, L, L)
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))   # ignore padding/deleted
        w = self.drop(scores.softmax(-1))
        x = x + self.drop(self.out((w @ v).transpose(1, 2).reshape(B, L, D)))
        x = x + self.drop(self.ff(self.n2(x)))
        return x, (w if want_attn else None)                                # w: (B, heads, L, L)


class SeqTransformer(nn.Module):
    """time_mode: 'none'  -> vanilla Transformer (positional encoding only)
                  'both'  -> TEBIF: + sinusoidal encoding of the gap dt AND of cumulative time
                  'dt' / 'ctime' -> ablations using only one of the two."""

    def __init__(self, n_cont, time_mode="none", d=64, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.d, self.time_mode = d, time_mode
        self.tok = nn.Embedding(11, d)                       # event type/role embedding
        self.cont = nn.Linear(n_cont, d)                     # projected continuous features
        self.t_dt = nn.Linear(d, d) if time_mode in ("both", "dt") else None
        self.t_ct = nn.Linear(d, d) if time_mode in ("both", "ctime") else None
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, b, return_attn=False):
        mask = b["mask"]
        pos, dt, ctime = time_features(b["step"], mask)
        x = self.tok(b["tok"]) + self.cont(b["cont"]) + sinusoid(pos, self.d, 2, 100)
        if self.t_dt is not None:                            # Δt encoding (how burst-like?)
            x = x + self.t_dt(sinusoid(dt, self.d, 1, 1500))
        if self.t_ct is not None:                            # cumulative-time encoding
            x = x + self.t_ct(sinusoid(ctime, self.d, 1, 1500))
        pad = ~mask
        attn = None
        for i, blk in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            x, w = blk(x, pad, want_attn=return_attn and last)
            if w is not None:
                attn = w[:, :, -1, :].mean(1)                # (B, L): last event -> all events
        x = self.norm(x)
        return self.head(x[:, -1]).squeeze(-1), attn         # pooled = the event we predict at


class SeqLSTM(nn.Module):
    """1-2 layer LSTM on the same inputs (no time features, like the vanilla Transformer)."""

    def __init__(self, n_cont, d=64, layers=2, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(11, 16)
        self.lstm = nn.LSTM(16 + n_cont, d, layers, batch_first=True, dropout=drop)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, b, return_attn=False):
        x = torch.cat([self.tok(b["tok"]), b["cont"]], -1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1), None


def make_model(name, n_cont):
    """Model zoo: lstm | vanilla | tebif | tebif_dt (Δt only) | tebif_ctime (cumulative time only)."""
    if name == "lstm":
        return SeqLSTM(n_cont)
    mode = {"vanilla": "none", "tebif": "both", "tebif_dt": "dt", "tebif_ctime": "ctime"}[name]
    return SeqTransformer(n_cont, mode)
