"""THE most important engineering step: decode at 1 fps once, encode with the cheap
encoder, write everything to disk. Every later experiment then runs in seconds."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import numpy as np
import torch
from tqdm import tqdm
from afs.features import FrameEncoder, candidate_times, feat_path, find_feat, read_frames_at
from afs.utils import Paths, load_config, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--subset", required=True)
args = ap.parse_args()
cfg = load_config(args.config)
P = Paths(cfg, args.subset)
qs = read_jsonl(P.subset_file)
ec, fc = cfg["encoder"], cfg["frames"]
P.feat_dir.mkdir(parents=True, exist_ok=True)
enc = FrameEncoder(ec["model_id"], "cuda" if torch.cuda.is_available() else "cpu", ec["dtype"], ec["batch_size"])

# ---- frame features (one file per video) --------------------------------
videos = sorted({q["video_path"] for q in qs})
todo = [v for v in videos if find_feat(P.feat_dir, v) is None]
print(f"{len(videos)} videos, {len(todo)} to encode")


class DS(torch.utils.data.Dataset):          # decode in worker processes, encode on GPU
    def __len__(self): return len(todo)
    def __getitem__(self, i):
        v = todo[i]
        times, fidx = candidate_times(v, fc["sample_fps"], fc["max_candidates"])
        return v, times, fidx, read_frames_at(v, fidx, max_side=448)

def ident(x): return x

dl = torch.utils.data.DataLoader(DS(), batch_size=None, num_workers=ec["num_workers"], collate_fn=ident)
for v, times, fidx, imgs in tqdm(dl, total=len(todo)):
    feats = enc.encode_images(imgs)
    np.savez(feat_path(P.feat_dir, v), feats=feats.astype(np.float16), times=times, frame_idx=fidx)

# ---- text features: [question, question+opt_0, ..., question+opt_n] ------
P.text_file.parent.mkdir(parents=True, exist_ok=True)
texts, spans = [], {}
for q in qs:
    t = [q["question"]] + [f'{q["question"]} {o}' for o in q["options"]]
    spans[q["qid"]] = (len(texts), len(texts) + len(t))
    texts += t
emb = enc.encode_texts(texts)
np.savez(P.text_file, **{qid: emb[a:b].astype(np.float16) for qid, (a, b) in spans.items()})
print("done ->", P.feat_dir, P.text_file)
