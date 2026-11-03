"""Coordinate sweep of the diversity and temporal weights on the DEV split only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from afs.ctx import build_ctx
from afs.evaluate import Evaluator, ReaderCache
from afs.features import FeatureStore
from afs.reader import load_reader
from afs.selectors import make_selector
from afs.utils import Paths, load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--w_div_grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
ap.add_argument("--w_time_grid", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
ap.add_argument("--apply", default=None, help="config file to update in place with the best weights")
args = ap.parse_args()
cfg = load_config(args.config)
P = Paths(cfg, args.subset)
qs = [q for q in read_jsonl(P.subset_file) if q["split"] == "dev"][: args.limit]
store = FeatureStore(P.feat_dir, P.text_file)
ev = Evaluator(load_reader(cfg), ReaderCache(P.reader_cache), store, cfg["frames"]["reader_max_side"], P.frame_dir)

def acc(**ov):
    sel = make_selector("ours", cfg, **ov)
    c = 0
    for q in tqdm(qs, desc=str(ov), leave=False):
        idx, _ = sel.select(build_ctx(q, store), args.K)
        rec = ev.score(q, idx)
        c += int(np.argmax(rec["logprobs"]) == q["answer"])
    return c / len(qs)

rows = []
best_div = cfg["selector"]["w_div"]
for wd in args.w_div_grid:
    rows.append(dict(w_div=wd, w_time=cfg["selector"]["w_time"], acc=acc(w_div=wd))); print(rows[-1])
best_div = max(rows, key=lambda r: r["acc"])["w_div"]
for wt in args.w_time_grid:
    rows.append(dict(w_div=best_div, w_time=wt, acc=acc(w_div=best_div, w_time=wt))); print(rows[-1])
df = pd.DataFrame(rows)
out = P.results / f"tune_dev_K{args.K}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
b = df.loc[df.acc.idxmax()]
print(f"\nBEST on dev: w_div={b.w_div} w_time={b.w_time} acc={b.acc:.3f}")
if args.apply:
    import re
    txt = open(args.apply).read()
    txt = re.sub(r"(?m)^(  w_div: ).*$", rf"\g<1>{b.w_div}", txt)
    txt = re.sub(r"(?m)^(  w_time: ).*$", rf"\g<1>{b.w_time}", txt)
    open(args.apply, "w").write(txt)
    print("-> wrote the best weights into", args.apply)
else:
    print("-> put these in the config under selector:")
