"""Select frames with any selector and score them with the frozen reader.
Writes results/<subset>/<split>/<selector>_K<K>.jsonl (one row per question)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import time
import numpy as np
from tqdm import tqdm
from afs.ctx import build_ctx
from afs.evaluate import Evaluator, ReaderCache, margin_and_pred
from afs.features import FeatureStore
from afs.reader import load_reader
from afs.selectors import make_selector
from afs.utils import Paths, load_config, read_jsonl, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
ap.add_argument("--selectors", nargs="+", default=["uniform", "random", "stride", "topk", "mmr", "ours"])
ap.add_argument("--ks", nargs="+", type=float, default=None,
                help="budgets; for *_adaptive selectors this is the TARGET MEAN K")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config)
P = Paths(cfg, args.subset)
qs = [q for q in read_jsonl(P.subset_file) if args.split == "all" or q["split"] == args.split][: args.limit]
store = FeatureStore(P.feat_dir, P.text_file)
reader = load_reader(cfg)
ev = Evaluator(reader, ReaderCache(P.reader_cache), store, cfg["frames"]["reader_max_side"], P.frame_dir)
enc_gf = cfg["encoder"]["gflops_per_frame"]
out_dir = P.split_dir(args.split if args.split != "all" else "all")
ks = args.ks or cfg["eval"]["ks"]

for name in args.selectors:
    for K in ks:
        K = int(K) if float(K).is_integer() else K
        f = out_dir / f"{name}_K{K}.jsonl"
        if f.exists() and not args.force:
            print("skip", f); continue
        sel = make_selector(name, cfg)
        if sel.adaptive:
            sel.fit(lambda: (build_ctx(q, store) for q in qs), target_k=K)
            print(f"  [{name}] calibrated param={sel.param:.4g} -> mean K={sel.achieved_mean_k:.2f}")
        rows = []
        for q in tqdm(qs, desc=f"{name} K={K}"):
            ctx = build_ctx(q, store)
            t0 = time.perf_counter()
            idx, k_used = sel.select(ctx, int(round(K)))
            sel_ms = (time.perf_counter() - t0) * 1000
            rec = ev.score(q, idx)
            m, pred = margin_and_pred(rec["logprobs"], q["answer"])
            rows.append(dict(qid=q["qid"], qtype=q["qtype"], selector=name, K=K, K_used=int(k_used),
                             N=ctx.N, gt=q["answer"], pred=pred, correct=int(pred == q["answer"]), margin=m,
                             gflops_reader=rec["gflops"], gflops_enc=ctx.N * enc_gf,
                             reader_latency_s=rec["latency_s"], select_ms=sel_ms,
                             n_img_tokens=rec["n_img_tokens"], idx=[int(i) for i in idx]))
        write_jsonl(f, rows)
        print(f"==> {name:22s} K={K:<5} acc={np.mean([r['correct'] for r in rows]):.3f}  "
              f"meanK={np.mean([r['K_used'] for r in rows]):.1f}  "
              f"GFLOPs={np.mean([r['gflops_reader'] for r in rows]):.0f}")
