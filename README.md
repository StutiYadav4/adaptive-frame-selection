# Adaptive Frame Selection for Efficient Video Understanding

Question-conditioned frame selection for long-video QA with a **frozen** Qwen2.5-VL reader.
Selector = greedy submodular maximisation of *relevance + visual coverage + temporal coverage*
(training-free, ms-scale, (1-1/e) guarantee), plus a per-question adaptive budget and a distilled selector.

## Directory structure

```
afs-project/
├── README.md
├── requirements.txt · pyproject.toml · run_all.sh
├── configs/default.yaml          # ALL paths + hyper-parameters (edit dataset paths here)
├── src/afs/
│   ├── data.py                   # NExT-QA / EgoSchema / LongVideoBench loaders -> unified format, dev/test split
│   ├── features.py               # 1 fps decoding, CLIP/SigLIP encoder, on-disk FeatureStore
│   ├── ctx.py                    # per-question context + relevance scores (pure numpy)
│   ├── selectors.py              # uniform / random / stride / top-K / MMR + factory incl. ablations
│   ├── submodular.py             # OUR selector: objective, lazy-free vectorised greedy
│   ├── adaptive.py               # per-question adaptive budget K (gain-knee or entropy rule)
│   ├── distilled.py              # <1M-param transformer that imitates the oracle
│   ├── oracle.py                 # best-of-T candidate subsets (ground-truth access)
│   ├── reader.py                 # frozen Qwen2.5-VL: option log-probs, latency, GFLOPs
│   ├── evaluate.py               # cached (question, frame-subset) -> reader output
│   └── utils.py
├── scripts/
│   ├── 00_download_small_nextqa.py  # fetch only N NExT-QA videos (not the 24 GB zip)
│   ├── 01_build_subset.py        # fix question subset + dev/test split
│   ├── 02_cache_features.py      # decode + encode once, cache to disk
│   ├── 03_run_selector.py        # any selector x any K -> results/*.jsonl
│   ├── 04_run_oracle.py          # oracle upper bound (+ --control random best-of-T)
│   ├── 05_tune_diversity.py      # sweep w_div / w_time on DEV
│   ├── 06_train_distilled.py     # imitate the oracle
│   ├── 07_needle_probe.py        # synthetic needle-in-a-video test
│   ├── 08_analyze.py             # Pareto curves, tables, significance, report artefacts
│   └── 09_demo.py                # "which frames and why" figure for the viva
├── tests/test_core.py            # runs without GPU/models
├── data/subsets/                 # generated question subsets (jsonl)
├── cache/                        # features/ text/ reader/ distilled/   (generated)
└── results/<subset>/{dev,test,report,needle}/   (generated)
```

## Setup

```bash
conda create -n afs python=3.11 -y && conda activate afs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
pip install -e .                       # or: export PYTHONPATH=$PWD/src  (scripts also self-bootstrap)
python tests/test_core.py              # 8 tests, no GPU needed: checks the (1-1/e) guarantee etc.
```

## Datasets - small-download options

The full downloads are large: the NExT-QA video zip is ~24 GB and EgoSchema's videos ~105 GB (checked on the HF Hub).
You do **not** need them. The pipeline only touches videos that appear in your subset, and 100 videos is plenty.

**Short-form (NExT-QA slice, recommended).** The NExT-GQA mirror stores the NExT-QA val/test videos as individual files, so:

```bash
python scripts/00_download_small_nextqa.py --n_videos 100      # ~100 videos, all their questions (~500-600 QA)
python scripts/01_build_subset.py --kind nextqa                # config already points at data/raw/nextgqa
```
Individual NExT-QA videos are a few MB each (roughly 1-12 MB in the listings I saw), so this is well under a few GB.
Questions whose video is not on disk are skipped automatically, so any partial download works.
Sampling *videos* (not questions) is what keeps the download small; ~6 questions come with every video.

**Long-form (compact set).** `HJOK/TempCore` bundles small slices of several benchmarks
(EgoSchema 96 q, LongVideoBench 261 q, NExT-QA 629 q, MLVU, Video-MME, MVBench) with videos hosted per benchmark:

```bash
hf download HJOK/TempCore --repo-type dataset --local-dir data/raw/TempCore \
    --include "data/*" "scripts/*" "videos/longvideobench/*"
python scripts/01_build_subset.py --kind tempcore --benchmark longvideobench --name tempcore_lvb --n 200
```
(EgoSchema videos are not redistributed there; its `scripts/fetch_videos.py` pulls them from the upstream archives.)
**Unverified:** I could not download TempCore from here, so `load_tempcore` follows its dataset card
(`question, options, answer, video_path`); if a field name differs, the fix is in `src/afs/data.py`.
Use `--subset tempcore_lvb` in all later scripts.

**Full datasets** (NExT-QA, EgoSchema, LongVideoBench) still work: point `datasets:` at your local copies and use `--kind nextqa|egoschema|lvb`.
Loaders assume the official file formats.

**Even smaller while developing:** `--n_videos 30`, `--n 100`, and `--limit 20` on scripts 03/04/05.
Dev/test are now split **by video** (no video appears in both), so dev tuning and distillation cannot leak into test.
With very few videos, keep `dev_frac: 0.25` but expect noisy numbers; report bootstrap CIs (script 08 does).

## Run

Everything for one dataset: `bash run_all.sh nextqa`. Or step by step:

```bash
# Phase 1 - data + features (nothing else starts before this)
python scripts/00_download_small_nextqa.py --n_videos 100
python scripts/01_build_subset.py   --kind nextqa
python scripts/02_cache_features.py --subset nextqa

# Phase 2 - baselines + oracle (complete performance envelope BEFORE our method)
python scripts/03_run_selector.py --subset nextqa --split test --selectors uniform random stride topk mmr
python scripts/04_run_oracle.py   --subset nextqa --split test
python scripts/04_run_oracle.py   --subset nextqa --split test --control

# Phase 3 - our selector
python scripts/05_tune_diversity.py --subset nextqa --K 8       # DEV only; copy best weights into the config
python scripts/03_run_selector.py   --subset nextqa --split test --selectors ours

# Phase 4 - differentiators
python scripts/03_run_selector.py --subset nextqa --split test --selectors ours_adaptive ours_adaptive_entropy --ks 4 8 16
python scripts/04_run_oracle.py --subset nextqa --split dev           # labels to imitate
python scripts/06_train_distilled.py --subset nextqa
python scripts/03_run_selector.py --subset nextqa --split test --selectors distilled

# Phase 5 - ablations, needle, analysis
python scripts/03_run_selector.py --subset nextqa --split test --selectors ours_no_div ours_no_time ours_no_rel ours_unweighted --ks 4 8 16
python scripts/07_needle_probe.py --subset nextqa --trials 100 --use_reader
python scripts/08_analyze.py --subset nextqa --split test

# Phase 6 - demo
python scripts/09_demo.py --subset nextqa --K 8
```

Repeat with `--kind egoschema` / `--kind lvb` for the long-form dataset.
A quick smoke test before the real run: add `--limit 20` to scripts 03/04/05.

**Resumability.** Reader outputs are cached by `(qid, frame set)` in `cache/reader/...jsonl`. Re-running a script,
or running a different selector that picks the same frames, costs nothing. Delete the file to start clean.

## What you get (`results/<subset>/report/`)

| File | Rubric item |
|---|---|
| `acc_vs_K.png` | accuracy vs K, oracle dashed, shaded **headroom** between uniform and oracle |
| `pareto_gflops.png`, `pareto_gflops_total.png`, `pareto_latency.png` | Pareto curves vs GFLOPs (reader-only and incl. encoder) and wall-clock latency |
| `summary.csv/.md` | every number, with bootstrap 95% CIs, mean K, GFLOPs, latency, selection ms |
| `ablation.csv/.md` | per-component ablation (no-div / no-time / no-relevance / unweighted coverage / top-K / MMR) |
| `per_category_K8.csv/.md` | per question-type breakdown |
| `paired_ours_vs_uniform.csv` | paired bootstrap of the accuracy difference |
| `frames_needed.png/.csv` | *when does adaptive selection matter?* (fraction solvable with <=K uniform frames, per type) |
| `adaptive_K_hist.png` | how the adaptive budget distributes K across questions |
| `needle_hit_rate.png` | synthetic needle probe |
| `demo_*.png` | which frames were picked and why (relevance / coverage / time split of each pick) |

## Method in one paragraph

For question *q* and N cached frame features, with r_i the min-max normalised CLIP relevance:

`F(S) = w_rel/k_ref * sum_{i in S} r_i + w_div * sum_j w_j max_{i in S} sim(i,j) + w_time * (1/N) sum_j max_{i in S} k_t(i,j)`

with `w_j ∝ cov_floor + (1-cov_floor) r_j`. Each term is non-negative, monotone and submodular, so greedy gives a (1-1/e)
guarantee (verified by brute force in `tests/test_core.py`). Because the relevance term uses a constant `k_ref`,
greedy is **nested**: the K-subset is a prefix of the K_max-subset, which makes the adaptive budget free. The adaptive rule stops
when the marginal gain falls below `tau * gain_1`; `tau` is calibrated, label-free, so the *mean* K equals the fixed-K
budget you compare against.

## Honest caveats (put these in the report)

1. **Oracle = best of T candidate subsets**, not exhaustive search (infeasible). It is a lower bound on the true oracle.
   Because it selects using the reader's own noisy outputs, part of its gain is selection on noise; that is why
   `04_run_oracle.py --control` (best of T *random* subsets) exists. Report both.
2. **Reader answers are read from the option-letter logits** in one forward pass (no generation). Latency is therefore
   prefill latency, a fair and low-variance comparison across selectors.
3. **GFLOPs are analytic estimates**: `2 * params * tokens` for the LLM and `2 * ViT params * patches` for the vision tower,
   ignoring attention's quadratic term. Wall-clock latency is reported next to it for that reason. Encoder cost is
   reported separately (`gflops_enc`) because it is amortised by caching.
4. **CLIP relevance** works on the question text (`rel_mode: q`). `q_opt_mean / q_opt_max` also use the answer options;
   treat that as an ablation, since it injects option information into selection.
5. Tune only on the **dev** split; report only on **test**.
6. Needle probe: the question *describes* the cue (colour + shape), as in realistic needle tasks; the reader must still read the number.

## Rough compute (single 24 GB GPU, Qwen2.5-VL-3B, ~450 test questions)

Feature caching: tens of minutes (decode-bound). Each (selector, K) run: a few minutes at small K, ~10 min at K=32.
Oracle with 16 candidates: the dominant cost (~1-2 h per K). Use `--limit` and `--n_cand 8` while developing.
