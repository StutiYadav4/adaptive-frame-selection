"""Fix the question subset once (dev/test split) so every later run is comparable."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
from afs.data import LOADERS, build_subset
from afs.utils import load_config, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--kind", required=True, choices=list(LOADERS))
ap.add_argument("--name", default=None, help="subset name (default = kind)")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--benchmark", default=None, help="tempcore only: longvideobench | egoschema | nextqa | ...")
ap.add_argument("--ann", default=None, help="override the annotation file (e.g. NExT-QA train.csv)")
args = ap.parse_args()

cfg = load_config(args.config)
dcfg = dict(cfg["datasets"]["longvideobench" if args.kind == "lvb" else args.kind])
if args.benchmark:
    dcfg["benchmark"] = args.benchmark
if args.ann:
    for k in ("ann_csv", "ann_json", "questions_json"):
        if k in dcfg:
            dcfg[k] = args.ann
rows = LOADERS[args.kind](**dcfg)
sc = cfg["subset"]
subset = build_subset(rows, args.n or sc["n_questions"], sc["dev_frac"], sc["seed"], sc["stratify"])
name = args.name or args.kind
out = Path(cfg["paths"]["data_dir"]) / "subsets" / f"{name}.jsonl"
write_jsonl(out, subset)
n_dev = sum(r["split"] == "dev" for r in subset)
print(f"wrote {len(subset)} questions ({n_dev} dev / {len(subset)-n_dev} test) -> {out}")
