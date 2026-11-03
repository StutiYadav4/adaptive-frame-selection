"""Oracle: best of T candidate K-subsets, chosen with ground-truth access.

True exhaustive search over C(N, K) subsets is infeasible, so the oracle is a *strong
approximation*: T diverse candidate subsets (baselines, our selector under several
weightings, shifted/jittered uniform grids, random draws from a high-relevance pool) are
each scored by the frozen reader; we keep the one with the largest correct-answer margin.
It is therefore a (tight-ish) lower bound on the true oracle and an upper bound on every
selector whose output is among the candidates. State this honestly in the report.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .ctx import Ctx, relevance
from .selectors import mmr_idx, stride_idx, topk_idx, uniform_idx
from .submodular import SubmodParams, select_submodular


def candidate_subsets(ctx: Ctx, K: int, base: SubmodParams, n_cand: int) -> list[list[int]]:
    N = ctx.N
    K = max(1, min(int(K), N))
    if K >= N:
        return [list(range(N))]
    rng = ctx.rng
    seen, out = set(), []

    def add(idx):
        t = tuple(sorted(set(int(i) for i in idx)))
        if len(t) == K and t not in seen:
            seen.add(t)
            out.append(list(t))

    add(uniform_idx(N, K))
    add(topk_idx(ctx, K, base.rel_mode))
    add(select_submodular(ctx, K, base))
    add(mmr_idx(ctx, K, 0.7, base.rel_mode))
    add(stride_idx(N, K))
    for kw in (dict(w_div=2.0), dict(w_div=0.3), dict(w_time=0.0), dict(w_time=1.5),
               dict(w_rel=0.0, cov_floor=1.0), dict(rel_gamma=2.0)):
        add(select_submodular(ctx, K, replace(base, **kw)))
    for off in (0.0, 0.25, 0.75):                    # shifted uniform grids
        add(((np.arange(K) + off) * N / K).astype(int).clip(0, N - 1))
    # random draws from a pool of high-relevance + ours + uniform frames
    rel = relevance(ctx, base.rel_mode)
    pool = set(np.argsort(-rel)[: 3 * K].tolist())
    pool |= set(select_submodular(ctx, min(3 * K, N), base).tolist())
    pool |= set(uniform_idx(N, K).tolist())
    pool = np.array(sorted(pool))
    tries = 0
    while len(out) < n_cand and tries < 20 * n_cand:
        tries += 1
        add(rng.choice(pool, K, replace=False))
        if tries % 4 == 0:                           # jittered uniform
            jit = uniform_idx(N, K) + rng.integers(-max(1, N // (2 * K)), max(1, N // (2 * K)) + 1, K)
            add(np.clip(jit, 0, N - 1))
    return out[:n_cand]
