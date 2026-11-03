"""Distil the oracle into a <1M-parameter transformer.
Input : frozen CLIP frame features + question embedding + (relevance, time) per frame
Target: which frames the oracle picked (soft label averaged over the oracle K's)
Train on DEV-split oracle labels; evaluate with 03_run_selector.py --selectors distilled on TEST."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from afs.ctx import build_ctx
from afs.distilled import build_model, make_inputs
from afs.features import FeatureStore
from afs.utils import Paths, load_config, read_jsonl, set_seed

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--label_split", default="dev")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--bs", type=int, default=16)
args = ap.parse_args()
cfg = load_config(args.config)
set_seed(0)
P = Paths(cfg, args.subset)
store = FeatureStore(P.feat_dir, P.text_file)
qs = {q["qid"]: q for q in read_jsonl(P.subset_file)}

# ---- gather oracle labels: soft target = fraction of oracle-K sets containing the frame
lab = {}
for f in sorted(P.split_dir(args.label_split).glob("oracle_K*.jsonl")):
    for r in read_jsonl(f):
        lab.setdefault(r["qid"], []).append(r["idx"])
assert lab, f"No oracle files in {P.split_dir(args.label_split)} -- run 04_run_oracle.py --split {args.label_split} first"
data = []
for qid, sets in lab.items():
    ctx = build_ctx(qs[qid], store)
    y = np.zeros(ctx.N, np.float32)
    for s in sets:
        y[s] += 1.0 / len(sets)
    f, s, q = make_inputs(ctx, cfg["selector"]["rel_mode"])
    data.append((f, s, q, y))
print(f"{len(data)} labelled questions")
rng = np.random.default_rng(0)
perm = rng.permutation(len(data))
n_val = max(1, int(0.15 * len(data)))
val, train = [data[i] for i in perm[:n_val]], [data[i] for i in perm[n_val:]]

dev = "cuda" if torch.cuda.is_available() else "cpu"
d_in = data[0][0].shape[1]
arch = dict(d_in=d_in, d=128, layers=2, heads=4, ff=256, dropout=0.1)
model = build_model(**arch).to(dev)
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)


def batchify(items):
    B, Nm = len(items), max(len(it[0]) for it in items)
    f = torch.zeros(B, Nm, d_in); s = torch.zeros(B, Nm, 2); y = torch.zeros(B, Nm)
    m = torch.zeros(B, Nm, dtype=torch.bool); q = torch.zeros(B, d_in)
    for i, (fi, si, qi, yi) in enumerate(items):
        n = len(fi)
        f[i, :n] = torch.from_numpy(fi); s[i, :n] = torch.from_numpy(si)
        y[i, :n] = torch.from_numpy(yi); m[i, :n] = True; q[i] = torch.from_numpy(qi)
    return [t.to(dev) for t in (f, s, q, m, y)]


def loss_fn(items):
    f, s, q, m, y = batchify(items)
    logits = model(f, s, q, m)
    pos_w = ((~(y > 0)) & m).sum() / (y > 0).sum().clamp(min=1)       # class imbalance
    l = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                           pos_weight=pos_w.clamp(max=20.0))
    return (l * m).sum() / m.sum()


best, out = 1e9, Path(cfg["selector"]["distilled_ckpt"])
out.parent.mkdir(parents=True, exist_ok=True)
for ep in range(args.epochs):
    model.train()
    order = rng.permutation(len(train))
    for i in range(0, len(order), args.bs):
        loss = loss_fn([train[j] for j in order[i:i + args.bs]])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        vl = float(np.mean([loss_fn(val[i:i + args.bs]).item() for i in range(0, len(val), args.bs)]))
    if vl < best:
        best = vl
        torch.save(dict(state=model.state_dict(), arch=arch), out)
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep:3d}  train {loss.item():.4f}  val {vl:.4f}  best {best:.4f}")
print("saved best checkpoint ->", out)
