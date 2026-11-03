"""Download a SMALL slice of NExT-QA instead of the 24 GB video zip.

Uses the NExT-GQA mirror on the HF Hub (val/test videos of NExT-QA stored as individual
<video_id>.mp4 files, so we can fetch just N of them). All questions of the chosen
videos are kept, so 100 videos ~ 500-600 questions.

    python scripts/00_download_small_nextqa.py --n_videos 100
then set in configs/default.yaml:
    nextqa: {ann_csv: data/raw/nextgqa/annotations/val.csv, video_dir: data/raw/nextgqa/videos}
"""
import argparse
import random

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="shuzhig/nextgqa")
ap.add_argument("--split", default="val", choices=["val", "test"])
ap.add_argument("--n_videos", type=int, default=100)
ap.add_argument("--out", default="data/raw/nextgqa")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

csv = hf_hub_download(args.repo, f"annotations/{args.split}.csv", repo_type="dataset", local_dir=args.out)
df = pd.read_csv(csv)
vcol = "video_id" if "video_id" in df.columns else "video"
all_vids = sorted(df[vcol].astype(str).unique())
chosen = sorted(random.Random(args.seed).sample(all_vids, min(args.n_videos, len(all_vids))))
n_q = int(df[vcol].astype(str).isin(chosen).sum())
print(f"{len(all_vids)} videos in {args.split}.csv -> downloading {len(chosen)} videos ({n_q} questions)")
snapshot_download(args.repo, repo_type="dataset", local_dir=args.out,
                  allow_patterns=[f"videos/{v}.mp4" for v in chosen])
print(f"""\nDone. In configs/default.yaml set:
  nextqa:
    ann_csv: {args.out}/annotations/{args.split}.csv
    video_dir: {args.out}/videos
Then: python scripts/01_build_subset.py --kind nextqa   (questions whose video is missing are skipped automatically)""")
