"""Visual demo: which frames were selected, and why (relevance vs coverage vs temporal term)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from afs.ctx import build_ctx, minmax, relevance
from afs.features import FeatureStore, read_frames_at
from afs.selectors import topk_idx, uniform_idx
from afs.submodular import SubmodParams, select_submodular
from afs.utils import Paths, load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--qid", default=None, help="default: first test question")
ap.add_argument("--K", type=int, default=8)
ap.add_argument("--out", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
P = Paths(cfg, args.subset)
qs = read_jsonl(P.subset_file)
q = next((x for x in qs if x["qid"] == args.qid), None) if args.qid else next(x for x in qs if x["split"] == "test")
store = FeatureStore(P.feat_dir, P.text_file)
ctx = build_ctx(q, store)
p = SubmodParams.from_cfg(cfg["selector"])
idx, info = select_submodular(ctx, args.K, p, return_info=True)
comps = {i: info["comps"][k] for k, i in enumerate(info["order"][: args.K])}
rel = minmax(relevance(ctx, p.rel_mode))
sel = {"uniform": uniform_idx(ctx.N, args.K), "top-K relevance": topk_idx(ctx, args.K), "ours": idx}
col = {"uniform": "tab:gray", "top-K relevance": "tab:orange", "ours": "tab:blue"}

fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(3, args.K, height_ratios=[1.4, 1.5, 0.1])
ax = fig.add_subplot(gs[0, :])
ax.plot(ctx.times, rel, color="tab:red", lw=1.5, label="question-frame relevance")
for j, (name, ii) in enumerate(sel.items()):
    ax.scatter(ctx.times[ii], np.full(len(ii), -0.09 - 0.1 * j), marker="|", s=90, color=col[name], label=name)
ax.set_xlabel("time (s)"); ax.set_ylim(-0.4, 1.05); ax.grid(alpha=.3); ax.legend(ncol=4, fontsize=8, loc="upper right")
gt = "ABCDEFGH"[q["answer"]]
ax.set_title("\n".join(textwrap.wrap(f'Q: {q["question"]}   (answer {gt}: {q["options"][q["answer"]]})', 150)), fontsize=10)
imgs = read_frames_at(q["video_path"], store.frames(q["video_path"])["frame_idx"][idx], max_side=320)
for k, (i, im) in enumerate(zip(idx, imgs)):
    a = fig.add_subplot(gs[1, k])
    a.imshow(im); a.axis("off")
    r, d, t = comps[int(i)]
    a.set_title(f"t={ctx.times[i]:.0f}s\nrel {r:.2f} | cov {d:.2f} | time {t:.2f}", fontsize=7)
fig.suptitle(f"Frames chosen by our submodular selector (K={args.K}); per-frame marginal gain split into relevance / visual coverage / temporal coverage",
             fontsize=10)
out = Path(args.out or P.results / "report" / f"demo_{q['qid']}_K{args.K}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(out, dpi=170)
print("saved", out)
