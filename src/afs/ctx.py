"""Per-question context shared by all selectors (pure numpy)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import seed_from


@dataclass
class Ctx:
    qid: str
    feats: np.ndarray        # (N, D) float32, L2-normalised
    times: np.ndarray        # (N,) seconds
    text: np.ndarray         # (1 + n_opts, D): [question, question+opt0, ...]
    rng: np.random.Generator

    @property
    def N(self) -> int:
        return self.feats.shape[0]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-8)


def relevance(ctx: Ctx, mode: str = "q") -> np.ndarray:
    """Raw cosine relevance of every frame to the question."""
    F, T = ctx.feats, ctx.text
    if mode == "q" or T.shape[0] < 2:
        return F @ T[0]
    sims = F @ T[1:].T
    if mode == "q_opt_mean":
        return sims.mean(1)
    if mode == "q_opt_max":
        return sims.max(1)
    raise ValueError(mode)


def build_ctx(q: dict, store) -> Ctx:
    d = store.frames(q["video_path"])
    return Ctx(
        qid=q["qid"],
        feats=d["feats"].astype(np.float32),
        times=d["times"].astype(np.float32),
        text=store.text_feats(q["qid"]).astype(np.float32),
        rng=np.random.default_rng(seed_from(q["qid"])),
    )
