"""Baselines + a name -> selector factory. Pure numpy."""
from __future__ import annotations

import numpy as np

from .ctx import Ctx, minmax, relevance
from .submodular import SubmodParams, select_submodular


def _k(N: int, K: int) -> int:
    return max(1, min(int(K), N))


def uniform_idx(N: int, K: int) -> np.ndarray:
    K = _k(N, K)
    if K >= N:
        return np.arange(N)
    return ((np.arange(K) + 0.5) * N / K).astype(int)


def stride_idx(N: int, K: int) -> np.ndarray:
    K = _k(N, K)
    if K >= N:
        return np.arange(N)
    return (np.arange(K) * (N / K)).astype(int)


def random_idx(ctx: Ctx, K: int) -> np.ndarray:
    K = _k(ctx.N, K)
    return np.sort(ctx.rng.choice(ctx.N, K, replace=False))


def topk_idx(ctx: Ctx, K: int, rel_mode: str = "q") -> np.ndarray:
    K = _k(ctx.N, K)
    return np.sort(np.argsort(-relevance(ctx, rel_mode))[:K])


def mmr_idx(ctx: Ctx, K: int, lam: float = 0.7, rel_mode: str = "q") -> np.ndarray:
    """Maximal Marginal Relevance (Carbonell & Goldstein)."""
    K = _k(ctx.N, K)
    r = minmax(relevance(ctx, rel_mode))
    max_sim = np.full(ctx.N, -1.0, np.float32)
    avail = np.ones(ctx.N, bool)
    chosen = []
    for _ in range(K):
        score = lam * r - (1 - lam) * np.maximum(max_sim, 0.0)
        score[~avail] = -np.inf
        i = int(np.argmax(score))
        chosen.append(i)
        avail[i] = False
        max_sim = np.maximum(max_sim, ctx.feats @ ctx.feats[i])
    return np.array(sorted(chosen))


class Selector:
    name = "base"
    adaptive = False

    def fit(self, ctx_iter_fn, target_k: float) -> None:  # only used by adaptive selectors
        pass

    def select(self, ctx: Ctx, K: int):
        """Returns (idx sorted ndarray, K_used)."""
        raise NotImplementedError


class FnSelector(Selector):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def select(self, ctx, K):
        idx = np.asarray(self.fn(ctx, K))
        return idx, len(idx)


class SubmodSelector(Selector):
    def __init__(self, name, params: SubmodParams):
        self.name, self.p = name, params

    def select(self, ctx, K):
        idx = select_submodular(ctx, K, self.p)
        return idx, len(idx)


ABLATIONS = {
    "ours": {},
    "ours_no_div": dict(w_div=0.0),
    "ours_no_time": dict(w_time=0.0),
    "ours_no_rel": dict(w_rel=0.0, cov_floor=1.0),   # question-agnostic coverage only
    "ours_unweighted": dict(cov_floor=1.0),           # coverage not relevance-weighted
}


def make_selector(name: str, cfg: dict, **overrides) -> Selector:
    sc = {**cfg["selector"], **overrides}
    rel_mode = sc.get("rel_mode", "q")
    if name == "uniform":
        return FnSelector(name, lambda c, K: uniform_idx(c.N, K))
    if name == "stride":
        return FnSelector(name, lambda c, K: stride_idx(c.N, K))
    if name == "random":
        return FnSelector(name, random_idx)
    if name == "topk":
        return FnSelector(name, lambda c, K: topk_idx(c, K, rel_mode))
    if name == "mmr":
        return FnSelector(name, lambda c, K: mmr_idx(c, K, sc.get("mmr_lambda", 0.7), rel_mode))
    if name in ABLATIONS:
        return SubmodSelector(name, SubmodParams.from_cfg(sc, **{**ABLATIONS[name], **overrides}))
    if name in ("ours_adaptive", "ours_adaptive_entropy"):
        from .adaptive import AdaptiveSelector
        rule = "entropy" if name.endswith("entropy") else sc.get("adaptive_rule", "gain")
        return AdaptiveSelector(name, SubmodParams.from_cfg(sc), rule, sc["k_min"], sc["k_max"])
    if name == "distilled":
        from .distilled import DistilledSelector
        return DistilledSelector(name, sc["distilled_ckpt"], sc.get("nms_thr", 0.97), rel_mode)
    raise ValueError(f"unknown selector '{name}'")
