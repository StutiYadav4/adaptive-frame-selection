"""Cached evaluation of (question, frame-subset) pairs with the frozen reader.
Every reader call is keyed by (qid, sorted frame indices) and appended to a jsonl, so
all experiments (baselines, ours, oracle, tuning) share work and runs are resumable."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from .features import _norm, read_frames_at
from .utils import read_jsonl, stable_hash


class ReaderCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.d = {r["key"]: r for r in read_jsonl(self.path)} if self.path.exists() else {}
        self.fh = open(self.path, "a")

    def get(self, key):
        return self.d.get(key)

    def put(self, rec):
        self.d[rec["key"]] = rec
        self.fh.write(json.dumps(rec) + "\n")
        self.fh.flush()


def margin_and_pred(logprobs, gt: int):
    lp = np.asarray(logprobs)
    others = np.delete(lp, gt)
    return float(lp[gt] - others.max()), int(lp.argmax())


class Evaluator:
    def __init__(self, reader, cache: ReaderCache, store, max_side: int = 448, frame_dir=None):
        self.reader, self.cache, self.store, self.max_side = reader, cache, store, max_side
        self.frame_dir = Path(frame_dir) if frame_dir else None

    def _frames(self, video_path, d, idx):
        """First use of a video decodes ALL its candidate frames once and stores JPEGs;
        every later (question, subset) call just reads a few small files (decoding was ~half the time)."""
        if self.frame_dir is None:
            return read_frames_at(video_path, d["frame_idx"][idx], max_side=self.max_side)
        vdir = self.frame_dir / f"{PurePosixPath(_norm(video_path)).stem}_{len(d['frame_idx'])}"
        done = vdir / "_done"
        if not done.exists():
            vdir.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(read_frames_at(video_path, d["frame_idx"], max_side=self.max_side)):
                im.save(vdir / f"{i:04d}.jpg", quality=92)
            done.touch()
        return [Image.open(vdir / f"{i:04d}.jpg").convert("RGB") for i in idx]

    def score(self, q: dict, idx) -> dict:
        idx = sorted({int(i) for i in idx})
        key = stable_hash(q["qid"], *idx)
        rec = self.cache.get(key)
        if rec is not None:
            return rec
        d = self.store.frames(q["video_path"])
        imgs = self._frames(q["video_path"], d, idx)
        out = self.reader.answer(imgs, d["times"][idx], q["question"], q["options"])
        rec = dict(key=key, qid=q["qid"], idx=idx, **out)
        self.cache.put(rec)
        return rec
