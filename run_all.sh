#!/usr/bin/env bash
# End-to-end pipeline for ONE dataset. Usage: bash run_all.sh nextqa   (or egoschema | lvb)
# Every step is resumable (features and reader outputs are cached on disk).
set -euo pipefail
DS=${1:-nextqa}
CFG=configs/default.yaml
BASE="uniform random stride topk mmr"

# Phase 1 - data + cached features (nothing else starts before this)
# (for NExT-QA first run:  python scripts/00_download_small_nextqa.py --n_videos 100)
python scripts/01_build_subset.py   --config $CFG --kind $DS
python scripts/02_cache_features.py --config $CFG --subset $DS

# Phase 2 - baselines + oracle (the full performance envelope, before our method)
python scripts/03_run_selector.py   --config $CFG --subset $DS --split test --selectors $BASE
python scripts/04_run_oracle.py     --config $CFG --subset $DS --split test
python scripts/04_run_oracle.py     --config $CFG --subset $DS --split test --control   # random best-of-T control

# Phase 3 - our selector (tune weights on DEV only, then edit configs/default.yaml)
python scripts/05_tune_diversity.py --config $CFG --subset $DS --K 8
python scripts/03_run_selector.py   --config $CFG --subset $DS --split test --selectors ours

# Phase 4 - differentiators
python scripts/03_run_selector.py   --config $CFG --subset $DS --split test \
        --selectors ours_adaptive ours_adaptive_entropy --ks 4 8 16
python scripts/04_run_oracle.py     --config $CFG --subset $DS --split dev        # labels for distillation
python scripts/06_train_distilled.py --config $CFG --subset $DS
python scripts/03_run_selector.py   --config $CFG --subset $DS --split test --selectors distilled

# Phase 5 - ablations, needle probe, analysis
python scripts/03_run_selector.py   --config $CFG --subset $DS --split test \
        --selectors ours_no_div ours_no_time ours_no_rel ours_unweighted --ks 4 8 16
python scripts/07_needle_probe.py   --config $CFG --subset $DS --trials 100 --use_reader
python scripts/08_analyze.py        --config $CFG --subset $DS --split test

# Phase 6 - demo figure
python scripts/09_demo.py           --config $CFG --subset $DS --K 8
