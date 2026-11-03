"""Our selector: greedy maximisation of a monotone submodular objective.

    F(S) =  w_rel  * (1/k_ref) * sum_{i in S} r_i                     (relevance, modular)
          + w_div  * sum_j  w_j * max_{i in S} sim(i, j)               (visual facility location)
          + w_time * (1/N) * sum_j max_{i in S} exp(-|t_i - t_j|/tau)  (temporal coverage)

* r_i  : min-max normalised question-frame relevance in [0, 1]
* w_j  : coverage weights = cov_floor + (1 - cov_floor) * r_j, normalised to sum 1
         (so "coverage" is spent where the question says it matters)
* All three terms are non-negative, monotone, submodular, and a non-negative
  combination preserves this => greedy has the classic (1 - 1/e) guarantee.
* Because the relevance term is divided by the constant k_ref (not by K), greedy
  is NESTED: the first K picks for Kmax are exactly the greedy solution for K.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .ctx import Ctx, minmax, relevance


@dataclass
class SubmodParams:
    w_rel: float = 1.0
    w_div: float = 1.0
    w_time: float = 0.5
    rel_mode: str = "q"
    rel_gamma: float = 1.0
    sim_temp: float = 0.1
    time_tau: float = 0.1
    cov_floor: float = 0.1
    k_ref: int = 8

    @classmethod
    def from_cfg(cls, d: dict, **overrides) -> "SubmodParams":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        kw.update({k: v for k, v in overrides.items() if k in names})
        return cls(**kw)


def prepare(ctx: Ctx, p: SubmodParams) -> dict:
    r = minmax(relevance(ctx, p.rel_mode)) ** p.rel_gamma
    G = ctx.feats @ ctx.feats.T
    S = np.exp((G - 1.0) / p.sim_temp).astype(np.float32)
    t = ctx.times.astype(np.float32)
    span = max(float(t.max() - t.min()), 1e-6)
    D = np.abs(t[:, None] - t[None, :]) / span
    T = np.exp(-D / p.time_tau).astype(np.float32)
    w = p.cov_floor + (1.0 - p.cov_floor) * r
    w = (w / w.sum()).astype(np.float32)
    return dict(r=r.astype(np.float32), S=S, T=T, w=w)


def objective(prep: dict, p: SubmodParams, idx) -> float:
    idx = list(idx)
    if not idx:
        return 0.0
    rel = p.w_rel * float(prep["r"][idx].sum()) / p.k_ref
    div = p.w_div * float(prep["w"] @ prep["S"][idx].max(0))
    tim = p.w_time * float(prep["T"][idx].max(0).mean())
    return rel + div + tim


def greedy(prep: dict, p: SubmodParams, k_max: int):
    """Returns (order, gains, comps). comps[k] = (rel, div, time) gain of the k-th pick."""
    r, S, T, w = prep["r"], prep["S"], prep["T"], prep["w"]
    N = len(r)
    k_max = min(k_max, N)
    cur_v = np.zeros(N, np.float32)
    cur_t = np.zeros(N, np.float32)
    base_v, base_t = 0.0, 0.0
    avail = np.ones(N, bool)
    gr = p.w_rel * r / p.k_ref
    order, gains, comps = [], [], []
    for _ in range(k_max):
        gv = np.maximum(S, cur_v[None, :]) @ w - base_v
        gt = (np.maximum(T, cur_t[None, :]).sum(1) - cur_t.sum()) / N
        g = gr + p.w_div * gv + p.w_time * gt
        g = np.where(avail, g, -np.inf)
        i = int(np.argmax(g))
        order.append(i)
        gains.append(float(g[i]))
        comps.append((float(gr[i]), float(p.w_div * gv[i]), float(p.w_time * gt[i])))
        avail[i] = False
        cur_v = np.maximum(cur_v, S[i])
        cur_t = np.maximum(cur_t, T[i])
        base_v = float(w @ cur_v)
    return order, np.array(gains), np.array(comps)


def select_submodular(ctx: Ctx, K: int, p: SubmodParams, return_info: bool = False):
    K = max(1, min(int(K), ctx.N))
    prep = prepare(ctx, p)
    order, gains, comps = greedy(prep, p, K)
    idx = np.array(sorted(order[:K]))
    if return_info:
        return idx, dict(order=order, gains=gains, comps=comps, prep=prep)
    return idx
