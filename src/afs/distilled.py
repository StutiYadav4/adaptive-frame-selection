"""Distilled selector: a tiny transformer (<1M params) trained to imitate the oracle's
frame choices. Frozen CLIP features in, per-frame keep-score out."""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, minmax, relevance
from .selectors import Selector


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(d_in=512, d=128, layers=2, heads=4, ff=256, dropout=0.1):
    torch, nn = _torch()

    class FrameScorer(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Linear(d_in, d)
            self.q = nn.Linear(d_in, d)
            self.s = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

        def forward(self, feats, scal, q, mask):
            """feats (B,N,D) scal (B,N,2) q (B,D) mask (B,N) True=valid -> logits (B,N)"""
            x = self.f(feats) + self.q(q)[:, None, :] + self.s(scal)
            x = self.enc(x, src_key_padding_mask=~mask)
            return self.out(x).squeeze(-1)

    return FrameScorer()


def make_inputs(ctx: Ctx, rel_mode="q"):
    """-> feats (N,D), scal (N,2)=[relevance, normalised time], q (D,)"""
    rel = minmax(relevance(ctx, rel_mode))
    t = ctx.times
    tn = (t - t.min()) / max(float(t.max() - t.min()), 1e-6)
    return ctx.feats, np.stack([rel, tn], -1).astype(np.float32), ctx.text[0]


class DistilledSelector(Selector):
    def __init__(self, name, ckpt, nms_thr=0.97, rel_mode="q", device="cpu"):
        torch, _ = _torch()
        blob = torch.load(ckpt, map_location="cpu")
        self.model = build_model(**blob["arch"])
        self.model.load_state_dict(blob["state"])
        self.model.eval().to(device)
        self.name, self.nms_thr, self.rel_mode, self.device = name, nms_thr, rel_mode, device

    def select(self, ctx, K):
        torch, _ = _torch()
        K = max(1, min(int(K), ctx.N))
        f, s, q = make_inputs(ctx, self.rel_mode)
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(f)[None].to(self.device),
                torch.from_numpy(s)[None].to(self.device),
                torch.from_numpy(q)[None].to(self.device),
                torch.ones(1, ctx.N, dtype=torch.bool, device=self.device),
            )[0].cpu().numpy()
        order = np.argsort(-logits)
        chosen = []
        for i in order:                              # light NMS to avoid near-duplicates
            if not chosen or float((ctx.feats[chosen] @ ctx.feats[i]).max()) < self.nms_thr:
                chosen.append(int(i))
            if len(chosen) == K:
                break
        for i in order:                              # back-fill if NMS was too strict
            if len(chosen) == K:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
        idx = np.array(sorted(chosen))
        return idx, len(idx)
