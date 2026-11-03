"""Per-question adaptive budget.

Greedy gains of a monotone submodular function are non-increasing, so the gain curve
tells us when extra frames stop paying off:

  rule 'gain'    : stop at the first k whose marginal gain < tau * (gain of frame 1)
  rule 'entropy' : K = kappa * exp(H(softmax(relevance)))   ("effective # of relevant frames")

The single scalar (tau or kappa) is calibrated by bisection so that the AVERAGE K over
a set of questions equals the target budget. Calibration uses only features + question
text -- no answers -- so it is label-free. Compare against fixed-K at the same mean K.
"""
from __future__ import annotations

import numpy as np

from .selectors import Selector
from .submodular import SubmodParams, greedy, prepare


class AdaptiveSelector(Selector):
    adaptive = True

    def __init__(self, name, params: SubmodParams, rule: str, k_min: int, k_max: int):
        self.name, self.p, self.rule = name, params, rule
        self.k_min, self.k_max = k_min, k_max
        self.param = None  # tau (gain) or kappa (entropy)

    # ------------------------------------------------------------------ stats
    def stats(self, ctx):
        prep = prepare(ctx, self.p)
        order, gains, _ = greedy(prep, self.p, self.k_max)
        pr = np.exp(prep["r"] / 0.1)
        pr /= pr.sum()
        keff = float(np.exp(-(pr * np.log(pr + 1e-12)).sum()))
        return dict(order=order, gains=gains, keff=keff, N=ctx.N)

    def k_from(self, st, param) -> int:
        kmax = min(self.k_max, st["N"])
        if self.rule == "gain":
            g = st["gains"]
            ratio = g / max(g[0], 1e-12)
            below = np.where(ratio < param)[0]
            k = int(below[0]) if len(below) else len(g)
        else:
            k = int(round(param * st["keff"]))
        return int(np.clip(k, min(self.k_min, kmax), kmax))

    # -------------------------------------------------------------------- fit
    def fit(self, ctx_iter_fn, target_k: float, iters: int = 40) -> None:
        sts = [self.stats(c) for c in ctx_iter_fn()]
        lo, hi = (1e-4, 1.0) if self.rule == "gain" else (1e-3, 100.0)
        mean_k = lambda prm: float(np.mean([self.k_from(s, prm) for s in sts]))
        increasing = self.rule != "gain"        # mean K grows with kappa, shrinks with tau
        for _ in range(iters):
            mid = float(np.sqrt(lo * hi))
            too_small = mean_k(mid) < target_k
            if too_small == increasing:   # need a larger kappa / a larger tau
                lo = mid
            else:
                hi = mid
        cands = [lo, hi, float(np.sqrt(lo * hi))]
        self.param = min(cands, key=lambda c: abs(mean_k(c) - target_k))
        self.achieved_mean_k = mean_k(self.param)

    # ----------------------------------------------------------------- select
    def select(self, ctx, K=None):
        assert self.param is not None, "call fit() first"
        st = self.stats(ctx)
        k = self.k_from(st, self.param)
        return np.array(sorted(st["order"][:k])), k
