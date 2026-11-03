"""Oracle upper bound = best of T candidate K-subsets, chosen with ground-truth access.
Also writes the oracle's chosen frames, which 06_train_distilled.py imitates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np
from tqdm import tqdm
from afs.ctx import build_ctx
from afs.evaluate import Evaluator, ReaderCache, margin_and_pred
from afs.features import FeatureStore
from afs.oracle import candidate_subsets
from afs.reader import load_reader
from afs.submodular import SubmodParams
from afs.utils import Paths, load_config, read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
ap.add_argument("--ks", nargs="+", type=int, default=None)
ap.add_argument("--n_cand", type=int, default=None)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--control", action="store_true",
                help="CONTROL: best-of-T *random* subsets. Shows how much oracle gain is just selection on reader noise.")
args = ap.parse_args()

cfg = load_config(args.config)
P = Paths(cfg, args.subset)
qs = [q for q in read_jsonl(P.subset_file) if args.split == "all" or q["split"] == args.split][: args.limit]
store = FeatureStore(P.feat_dir, P.text_file)
ev = Evaluator(load_reader(cfg), ReaderCache(P.reader_cache), store, cfg["frames"]["reader_max_side"], P.frame_dir)
base = SubmodParams.from_cfg(cfg["selector"])
n_cand = args.n_cand or cfg["eval"]["oracle_candidates"]
out_dir = P.split_dir(args.split)

name = "oracle_random" if args.control else "oracle"
for K in args.ks or cfg["eval"]["oracle_ks"]:
    rows = []
    for q in tqdm(qs, desc=f"oracle K={K}"):
        ctx = build_ctx(q, store)
        if args.control:
            Kc = min(K, ctx.N)
            seen = set()
            cands = []
            for _ in range(20 * n_cand):
                c = sorted(int(i) for i in ctx.rng.choice(ctx.N, Kc, replace=False))
                if tuple(c) not in seen:
                    seen.add(tuple(c)); cands.append(c)
                if len(cands) >= n_cand:
                    break
        else:
            cands = candidate_subsets(ctx, K, base, n_cand)
        best = None
        for c in cands:
            rec = ev.score(q, c)
            m, pred = margin_and_pred(rec["logprobs"], q["answer"])
            if best is None or m > best[0]:
                best = (m, pred, rec, c)
        m, pred, rec, c = best
        rows.append(dict(qid=q["qid"], qtype=q["qtype"], selector=name, K=K, K_used=len(c), N=ctx.N,
                         gt=q["answer"], pred=pred, correct=int(m > 0), margin=m,
                         gflops_reader=rec["gflops"], gflops_enc=0.0, reader_latency_s=rec["latency_s"],
                         select_ms=0.0, n_img_tokens=rec["n_img_tokens"], idx=c, n_cand=len(cands)))
    write_jsonl(out_dir / f"{name}_K{K}.jsonl", rows)
    print(f"==> {name} K={K}: acc={np.mean([r['correct'] for r in rows]):.3f} (best of <= {n_cand} subsets)")
