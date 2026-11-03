"""Runs with pytest OR plain `python tests/test_core.py`. No GPU / models needed."""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afs.adaptive import AdaptiveSelector
from afs.ctx import Ctx
from afs.oracle import candidate_subsets
from afs.selectors import make_selector, mmr_idx, random_idx, stride_idx, topk_idx, uniform_idx
from afs.submodular import SubmodParams, greedy, objective, prepare, select_submodular
from afs.utils import load_config

CFG = load_config(str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def make_ctx(N=40, D=16, seed=0, dup_block=None):
    rng = np.random.default_rng(seed)
    F = unit(rng.normal(size=(N, D))).astype(np.float32)
    if dup_block:                       # near-duplicate block, all highly relevant to the query
        a, b = dup_block
        base = unit(rng.normal(size=D))
        F[a:b] = unit(base + 0.02 * rng.normal(size=(b - a, D)))
    q = F[dup_block[0]] if dup_block else F[N // 2]
    text = np.stack([unit(q + 0.05 * rng.normal(size=D))]).astype(np.float32)
    return Ctx("t", F, np.arange(N, dtype=np.float32), text, np.random.default_rng(seed))


def test_baselines_shape():
    ctx = make_ctx(37)
    for K in (1, 4, 8, 37, 50):
        for idx in (uniform_idx(37, K), stride_idx(37, K), random_idx(ctx, K), topk_idx(ctx, K), mmr_idx(ctx, K)):
            assert len(set(idx.tolist())) == len(idx) == min(K, 37)
            assert idx.min() >= 0 and idx.max() < 37


def test_greedy_approximation_guarantee():
    """Greedy must achieve >= (1 - 1/e) * OPT (brute force on tiny instances)."""
    p = SubmodParams()
    for seed in range(5):
        ctx = make_ctx(N=14, D=8, seed=seed)
        prep = prepare(ctx, p)
        for K in (2, 3, 4):
            opt = max(objective(prep, p, c) for c in itertools.combinations(range(14), K))
            order, _, _ = greedy(prep, p, K)
            val = objective(prep, p, order)
            assert val >= (1 - 1 / np.e) * opt - 1e-6, (seed, K, val, opt)
            assert val >= 0.9 * opt, f"in practice greedy is near-optimal: {val/opt:.3f}"


def test_objective_monotone_and_submodular():
    p, ctx = SubmodParams(), make_ctx(30, seed=3)
    prep = prepare(ctx, p)
    rng = np.random.default_rng(0)
    for _ in range(200):
        A = list(rng.choice(30, 4, replace=False))
        B = A + list(rng.choice([i for i in range(30) if i not in A], 4, replace=False))
        e = int(rng.choice([i for i in range(30) if i not in B]))
        gA = objective(prep, p, A + [e]) - objective(prep, p, A)
        gB = objective(prep, p, B + [e]) - objective(prep, p, B)
        assert gA >= -1e-6 and gB >= -1e-6        # monotone
        assert gA >= gB - 1e-6                     # diminishing returns


def test_greedy_nested():
    ctx = make_ctx(60, seed=1)
    p = SubmodParams()
    full = set(select_submodular(ctx, 8, p).tolist())
    order, _, _ = greedy(prepare(ctx, p), p, 16)
    assert set(order[:8]) == full


def test_diversity_beats_topk_on_duplicates():
    ctx = make_ctx(N=80, D=16, seed=2, dup_block=(30, 40))
    K = 6
    top = topk_idx(ctx, K)
    ours = select_submodular(ctx, K, SubmodParams())
    # top-K collapses into the duplicate block; ours spreads out in time
    assert np.ptp(top) <= 12, top
    assert np.ptp(ours) > np.ptp(top), (top, ours)
    assert len({int(i) // 10 for i in ours}) > len({int(i) // 10 for i in top})
    # but ours still keeps at least one frame from the relevant block
    assert any(30 <= i < 40 for i in ours)


def test_adaptive_calibration_hits_target():
    ctxs = [make_ctx(N=int(n), seed=s) for s, n in enumerate(np.random.default_rng(0).integers(40, 200, 30))]
    for rule in ("gain", "entropy"):
        sel = AdaptiveSelector("a", SubmodParams(), rule, 2, 32)
        sel.fit(lambda: iter(ctxs), target_k=8)
        ks = [sel.select(c)[1] for c in ctxs]
        assert abs(np.mean(ks) - 8) < 1.0, (rule, np.mean(ks))
        assert min(ks) >= 2 and max(ks) <= 32
        assert len(set(ks)) > 1, f"{rule}: budget should vary across questions"


def test_oracle_candidates_valid():
    ctx = make_ctx(120, seed=4)
    cands = candidate_subsets(ctx, 8, SubmodParams(), 16)
    assert 8 <= len(cands) <= 16
    assert len({tuple(c) for c in cands}) == len(cands)
    assert all(len(c) == 8 and c == sorted(c) for c in cands)


def test_registry():
    ctx = make_ctx(50)
    for name in ["uniform", "random", "stride", "topk", "mmr", "ours", "ours_no_div", "ours_no_time",
                 "ours_no_rel", "ours_unweighted"]:
        idx, k = make_selector(name, CFG).select(ctx, 5)
        assert k == 5 and len(idx) == 5, name


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
