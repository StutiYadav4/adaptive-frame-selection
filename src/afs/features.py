"""Video decoding, cheap-encoder features, and the on-disk feature store."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image

from .utils import stable_hash


def _norm(p) -> str:
    """Windows-style backslashes -> '/', so subset files and caches made on Windows work on Linux."""
    return str(p).replace("\\", "/")


# ----------------------------------------------------------------- decoding
def video_meta(path) -> tuple[int, float]:
    cap = cv2.VideoCapture(_norm(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0
    return max(n, 1), float(fps)


def candidate_times(path, sample_fps: float = 1.0, max_candidates: int = 900):
    """Candidate-frame timestamps (seconds, mid-interval) and native frame indices."""
    n, fps = video_meta(path)
    duration = n / fps
    N = max(1, int(duration * sample_fps))
    if N > max_candidates:
        N = max_candidates
    times = (np.arange(N) + 0.5) * duration / N
    fidx = np.clip(np.round(times * fps).astype(int), 0, n - 1)
    return times.astype(np.float32), fidx


def _to_pil(bgr, max_side):
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if max_side and max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BICUBIC)
    return img


def read_frames_at(path, frame_indices, max_side: int | None = 448) -> list[Image.Image]:
    """Sequential-grab decoder (robust to bad seek tables). Order/duplicates preserved."""
    frame_indices = [int(i) for i in frame_indices]
    wanted = sorted(set(frame_indices))
    cap = cv2.VideoCapture(_norm(path))
    got, last, cur, wi = {}, None, -1, 0
    while wi < len(wanted):
        if not cap.grab():
            break
        cur += 1
        if cur == wanted[wi]:
            ok, frame = cap.retrieve()
            if ok:
                last = frame
            got[wanted[wi]] = last
            wi += 1
    cap.release()
    for i in wanted:                       # video ended early -> repeat last good frame
        if i not in got or got[i] is None:
            got[i] = last if last is not None else np.zeros((224, 224, 3), np.uint8)
    cache = {i: _to_pil(got[i], max_side) for i in wanted}
    return [cache[i] for i in frame_indices]


# ------------------------------------------------------------------ encoder
class FrameEncoder:
    """CLIP / SigLIP wrapper via HF transformers. Returns L2-normalised float32."""

    def __init__(self, model_id, device="cuda", dtype="float16", batch_size=64):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.device, self.bs = device, batch_size
        self.dtype = getattr(torch, dtype)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.siglip = "siglip" in model_id.lower()

    @staticmethod
    def _tensor(x):
        return x if hasattr(x, "shape") else x.pooler_output

    def encode_images(self, images) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for i in range(0, len(images), self.bs):
                px = self.processor(images=images[i:i + self.bs], return_tensors="pt")["pixel_values"]
                f = self._tensor(self.model.get_image_features(pixel_values=px.to(self.device, self.dtype)))
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def encode_texts(self, texts) -> np.ndarray:
        torch = self.torch
        out = []
        tok = self.processor.tokenizer
        with torch.inference_mode():
            for i in range(0, len(texts), 256):
                chunk = texts[i:i + 256]
                if self.siglip:
                    inp = tok(chunk, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
                    kw = dict(input_ids=inp["input_ids"].to(self.device))
                else:
                    inp = tok(chunk, padding=True, truncation=True, max_length=77, return_tensors="pt")
                    kw = {k: v.to(self.device) for k, v in inp.items() if k in ("input_ids", "attention_mask")}
                f = self._tensor(self.model.get_text_features(**kw))
                out.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu())
        return torch.cat(out).numpy()


# -------------------------------------------------------------------- store
def feat_path(feat_dir, video_path) -> Path:
    vp = _norm(video_path)
    return Path(feat_dir) / f"{PurePosixPath(vp).stem}_{stable_hash(vp, n=8)}.npz"


def find_feat(feat_dir, video_path):
    """Existing cache file for this video (or None). Falls back to matching by video name,
    so a cache built on Windows (different path string) is found on Colab/Linux."""
    p = feat_path(feat_dir, video_path)
    if p.exists():
        return p
    stem = PurePosixPath(_norm(video_path)).stem
    m = sorted(Path(feat_dir).glob(f"{stem}_*.npz"))
    return m[0] if m else None


class FeatureStore:
    def __init__(self, feat_dir, text_file):
        self.feat_dir = Path(feat_dir)
        self._text = dict(np.load(text_file)) if Path(text_file).exists() else {}

    def frames(self, video_path) -> dict:
        p = find_feat(self.feat_dir, video_path)
        if p is None:
            raise FileNotFoundError(f"No cached features for {video_path} in {self.feat_dir}")
        d = np.load(p)
        return {k: d[k] for k in d.files}

    def text_feats(self, qid) -> np.ndarray:
        return self._text[qid]
