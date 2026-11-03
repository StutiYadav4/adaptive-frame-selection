"""Synthetic needle probe: splice a known visual cue (coloured shape + 3-digit number on a
white background) into ONE random frame of a real long video. The question describes the
shape/colour; answering requires reading the number. Fully controlled -> proves the mechanism.
  hit rate  : was the needle frame selected? (cheap, no reader needed)
  accuracy  : (--use_reader) can the frozen reader read the number from the selected frames?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from afs.ctx import Ctx
from afs.features import FeatureStore, FrameEncoder, read_frames_at
from afs.selectors import make_selector
from afs.utils import Paths, load_config, read_jsonl, seed_from

COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (40, 70, 220),
          "orange": (250, 140, 20), "purple": (140, 50, 170), "yellow": (240, 210, 20)}
SHAPES = ["circle", "square", "triangle"]


def font(size):
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)


def make_needle(rng):
    color, shape = rng.choice(list(COLORS)), rng.choice(SHAPES)
    number = str(rng.randint(100, 999))
    img = Image.new("RGB", (448, 336), "white")
    d = ImageDraw.Draw(img)
    box = (144, 20, 304, 170)
    if shape == "circle":
        d.ellipse(box, fill=COLORS[color])
    elif shape == "square":
        d.rectangle(box, fill=COLORS[color])
    else:
        d.polygon([(224, 20), (144, 170), (304, 170)], fill=COLORS[color])
    f = font(110)
    bb = d.textbbox((0, 0), number, font=f)
    d.text(((448 - (bb[2] - bb[0])) // 2, 185), number, fill="black", font=f)
    return img, color, shape, number


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
ap.add_argument("--trials", type=int, default=100)
ap.add_argument("--min_frames", type=int, default=60)
ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16])
ap.add_argument("--selectors", nargs="+", default=["uniform", "random", "topk", "ours"])
ap.add_argument("--use_reader", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config)
P = Paths(cfg, args.subset)
store = FeatureStore(P.feat_dir, P.text_file)
qs = read_jsonl(P.subset_file)
vids = sorted({q["video_path"] for q in qs})
rnd = random.Random(0)
enc = FrameEncoder(cfg["encoder"]["model_id"], "cuda" if torch.cuda.is_available() else "cpu",
                   cfg["encoder"]["dtype"], cfg["encoder"]["batch_size"])
reader = None
if args.use_reader:
    from afs.reader import load_reader
    reader = load_reader(cfg)
sels = {n: make_selector(n, cfg) for n in args.selectors}
rows = []
tried = 0
pbar = tqdm(total=args.trials)
while len(rows) // (len(sels) * len(args.ks)) < args.trials and tried < 20 * args.trials:
    tried += 1
    vp = rnd.choice(vids)
    d = store.frames(vp)
    N = len(d["times"])
    if N < args.min_frames:
        continue
    needle, color, shape, number = make_needle(rnd)
    pos = rnd.randrange(N)
    question = f"At some point in the video, a {color} {shape} on a white background is shown. What number is written below it?"
    wrong = rnd.sample([str(x) for x in range(100, 1000) if str(x) != number], 4)
    options = wrong + [number]
    rnd.shuffle(options)
    ans = options.index(number)
    feats = d["feats"].astype(np.float32).copy()
    feats[pos] = enc.encode_images([needle])[0]
    text = enc.encode_texts([question])
    ctx = Ctx("needle", feats, d["times"].astype(np.float32), text, np.random.default_rng(seed_from(tried)))
    for name, sel in sels.items():
        for K in args.ks:
            idx, _ = sel.select(ctx, K)
            row = dict(trial=tried, N=N, pos_frac=pos / N, selector=name, K=K, hit=int(pos in set(idx.tolist())))
            if reader is not None:
                others = [i for i in idx if i != pos]
                imgs = read_frames_at(vp, d["frame_idx"][others], max_side=cfg["frames"]["reader_max_side"]) if others else []
                imgs_all = dict(zip(others, imgs))
                if pos in set(idx.tolist()):
                    imgs_all[pos] = needle
                order = sorted(imgs_all)
                out = reader.answer([imgs_all[i] for i in order], d["times"][order], question, options)
                row["correct"] = int(int(np.argmax(out["logprobs"])) == ans)
            rows.append(row)
    pbar.update(1)
pbar.close()
df = pd.DataFrame(rows)
out = P.results / "needle"
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "needle_results.csv", index=False)
cols = ["hit"] + (["correct"] if reader is not None else [])
print(df.groupby(["selector", "K"])[cols].mean().unstack(0).round(3).to_string())
print("->", out / "needle_results.csv")
