"""Turn result files into the report artefacts: Pareto curves, oracle headroom, per-category
table, ablation table, paired significance, 'how many frames are really needed'."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from afs.utils import Paths, load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--split", default="test")
args = ap.parse_args()
cfg = load_config(args.config)
P = Paths(cfg, args.subset)
src, rep = P.split_dir(args.split), P.results / "report"
rep.mkdir(parents=True, exist_ok=True)

rows = []
for f in sorted(src.glob("*_K*.jsonl")):
    rows += read_jsonl(f)
df = pd.DataFrame(rows)
assert len(df), f"no result files in {src}"
df["latency_s"] = df.reader_latency_s + df.select_ms / 1000.0
df["gflops_total"] = df.gflops_reader + df.gflops_enc
rng = np.random.default_rng(0)


def md(t: pd.DataFrame, floatfmt="{:.3f}") -> str:
    t = t.reset_index()
    head = "| " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * len(t.columns) + "\n"
    body = "".join("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |\n"
                   for r in t.itertuples(index=False))
    return head + body


def boot_ci(x, n=1000):
    x = np.asarray(x, float)
    m = rng.choice(x, (n, len(x))).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


# ------------------------------------------------------------ main summary
g = df.groupby(["selector", "K"])
S = g.agg(n=("correct", "size"), acc=("correct", "mean"), mean_K=("K_used", "mean"),
          gflops=("gflops_reader", "mean"), gflops_total=("gflops_total", "mean"),
          latency_s=("latency_s", "mean"), select_ms=("select_ms", "mean")).reset_index()
ci = [boot_ci(df[(df.selector == s) & (df.K == k)].correct) for s, k in zip(S.selector, S.K)]
S["ci_lo"], S["ci_hi"] = zip(*ci)
S.to_csv(rep / "summary.csv", index=False)
(rep / "summary.md").write_text(md(S.set_index(["selector", "K"])))
print(S.round(3).to_string(index=False))

STYLE = {"oracle": dict(color="k", ls="--", marker="*", ms=10), "uniform": dict(color="tab:gray", marker="o"),
         "random": dict(color="tab:brown", marker="x"), "stride": dict(color="tab:olive", marker="s"),
         "topk": dict(color="tab:orange", marker="v"), "mmr": dict(color="tab:green", marker="^"),
         "ours": dict(color="tab:blue", marker="D", lw=2.5), "ours_adaptive": dict(color="tab:red", marker="P", lw=2.5),
         "ours_adaptive_entropy": dict(color="tab:pink", marker="P"), "distilled": dict(color="tab:purple", marker="h")}


def curve_plot(xcol, xlabel, fname, logx=False, only=None):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for s, t in S.groupby("selector"):
        if only and s not in only:
            continue
        if s.startswith("ours_") and s not in STYLE:      # ablations live in their own table
            continue
        t = t.sort_values(xcol)
        ax.errorbar(t[xcol], t.acc, yerr=[t.acc - t.ci_lo, t.ci_hi - t.acc], capsize=2,
                    label=s, **STYLE.get(s, dict(marker=".")))
    if "oracle" in S.selector.values and "uniform" in S.selector.values and xcol == "K":
        o, u = S[S.selector == "oracle"].set_index("K").acc, S[S.selector == "uniform"].set_index("K").acc
        common = sorted(set(o.index) & set(u.index))
        ax.fill_between(common, u[common], o[common], color="k", alpha=0.07, label="headroom")
    if logx or xcol == "K":
        ax.set_xscale("log", base=2)
    ax.set_xlabel(xlabel); ax.set_ylabel("accuracy"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(rep / fname, dpi=200); plt.close(fig)


curve_plot("K", "frames given to the reader (K; target mean K for adaptive)", "acc_vs_K.png")
curve_plot("gflops", "reader GFLOPs / question", "pareto_gflops.png", logx=True)
curve_plot("gflops_total", "reader + encoder GFLOPs / question (no caching amortisation)", "pareto_gflops_total.png", logx=True)
curve_plot("latency_s", "wall-clock latency / question (reader + selection), s", "pareto_latency.png", logx=True)

# --------------------------------------------------------- per-category table
ck = cfg["eval"]["cat_k"]
t = df[df.K == ck]
if len(t):
    cat = t.pivot_table(index="selector", columns="qtype", values="correct", aggfunc="mean")
    cat["ALL"] = t.groupby("selector").correct.mean()
    cat.to_csv(rep / f"per_category_K{ck}.csv")
    (rep / f"per_category_K{ck}.md").write_text(md(cat))
    print(f"\nPer-category accuracy @K={ck}\n", cat.round(3).to_string())

# --------------------------------------------------------------- ablations
abl = S[S.selector.isin(["uniform", "topk", "mmr", "ours", "ours_no_div", "ours_no_time", "ours_no_rel", "ours_unweighted"])]
if len(abl):
    A = abl.pivot(index="selector", columns="K", values="acc")
    A.to_csv(rep / "ablation.csv"); (rep / "ablation.md").write_text(md(A))
    print("\nAblation (accuracy)\n", A.round(3).to_string())

# ---------------------------------- paired significance: ours - uniform (per K)
sig = []
for K in sorted(df.K.unique(), key=float):
    a = df[(df.selector == "ours") & (df.K == K)].set_index("qid").correct
    b = df[(df.selector == "uniform") & (df.K == K)].set_index("qid").correct
    common = a.index.intersection(b.index)
    if len(common) < 10:
        continue
    d = (a[common] - b[common]).values.astype(float)
    bs = rng.choice(d, (2000, len(d))).mean(1)
    sig.append(dict(K=K, delta=d.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                    ours_only_right=int((d > 0).sum()), uniform_only_right=int((d < 0).sum())))
if sig:
    sg = pd.DataFrame(sig).set_index("K")
    sg.to_csv(rep / "paired_ours_vs_uniform.csv")
    print("\nPaired ours - uniform\n", sg.round(3).to_string())

# ---------------- "when does adaptive selection matter?": frames really needed
u = df[df.selector == "uniform"]
if u.K.nunique() > 1:
    Ks = sorted(u.K.unique(), key=float)
    piv = u.pivot(index="qid", columns="K", values="correct")[Ks]
    need = piv.apply(lambda r: next((float(k) for k in Ks if r[k] == 1), np.inf), axis=1)
    qt = u.drop_duplicates("qid").set_index("qid").qtype
    fig, ax = plt.subplots(figsize=(7, 4.4))
    tab = {}
    for name, ids in [("ALL", need.index)] + [(c, need.index[qt[need.index] == c]) for c in sorted(qt.unique())]:
        if len(ids) < 5:
            continue
        frac = [(need[ids] <= float(k)).mean() for k in Ks]
        tab[name] = frac
        ax.plot([float(k) for k in Ks], frac, marker="o", label=f"{name} (n={len(ids)})", lw=3 if name == "ALL" else 1)
    ax.set_xscale("log", base=2); ax.set_xlabel("K (uniform frames)"); ax.set_ylabel("fraction of questions answered correctly")
    ax.set_title("How many uniform frames does each question need?"); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(rep / "frames_needed.png", dpi=200); plt.close(fig)
    pd.DataFrame(tab, index=[f"K<={k}" for k in Ks]).to_csv(rep / "frames_needed.csv")

# ----------------------------------------------- adaptive: distribution of K
ad = df[df.selector.str.contains("adaptive")]
if len(ad):
    fig, ax = plt.subplots(figsize=(6, 4))
    for (s, K), t in ad.groupby(["selector", "K"]):
        ax.hist(t.K_used, bins=range(0, int(t.K_used.max()) + 2), alpha=.5, label=f"{s} target={K}")
    ax.set_xlabel("K chosen per question"); ax.set_ylabel("# questions"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(rep / "adaptive_K_hist.png", dpi=200); plt.close(fig)

# --------------------------------------------------------------- needle plot
nf = P.results / "needle" / "needle_results.csv"
if nf.exists():
    nd = pd.read_csv(nf)
    fig, ax = plt.subplots(figsize=(6, 4))
    for s, t in nd.groupby("selector"):
        m = t.groupby("K").hit.mean()
        ax.plot(m.index, m.values, marker="o", label=s)
    ax.set_xscale("log", base=2); ax.set_xlabel("K"); ax.set_ylabel("needle-frame hit rate"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(rep / "needle_hit_rate.png", dpi=200); plt.close(fig)
print("\nreport ->", rep)
