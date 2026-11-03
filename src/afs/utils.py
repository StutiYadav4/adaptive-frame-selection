from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def read_jsonl(path, skip_bad: bool = True) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_bad:
                    raise
    return rows


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stable_hash(*parts, n: int = 16) -> str:
    return hashlib.sha1("||".join(map(str, parts)).encode()).hexdigest()[:n]


def seed_from(*parts) -> int:
    return int(stable_hash(*parts, n=8), 16)


class Paths:
    """Resolves every on-disk location from the config."""

    def __init__(self, cfg: dict, subset: str):
        p = cfg["paths"]
        self.subset = subset
        self.subset_file = Path(p["data_dir"]) / "subsets" / f"{subset}.jsonl"
        self.feat_dir = Path(p["cache_dir"]) / "features" / cfg["encoder"]["tag"]
        self.text_file = Path(p["cache_dir"]) / "text" / cfg["encoder"]["tag"] / f"{subset}.npz"
        rtag = reader_tag(cfg)
        self.reader_cache = Path(p["cache_dir"]) / "reader" / rtag / f"{subset}.jsonl"
        self.results = Path(p["results_dir"]) / subset
        fd = p.get("frame_dir") or str(Path(p["cache_dir"]) / "frames")      # decoded-frame JPEG cache
        self.frame_dir = Path(fd) / f"side{cfg['frames']['reader_max_side']}"

    def split_dir(self, split: str) -> Path:
        return self.results / split


def reader_tag(cfg: dict) -> str:
    r = cfg["reader"]
    return f"{r['model_id'].split('/')[-1]}_px{r['max_pixels']}_side{cfg['frames']['reader_max_side']}"
