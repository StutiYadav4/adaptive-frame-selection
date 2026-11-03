"""Frozen Qwen2.5-VL reader. One forward pass per question: we read the next-token
distribution over the option letters, so we get log-probs for ALL options (needed by the
oracle) without generation, and latency is a clean prefill measurement."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

LETTERS = "ABCDEFGH"


class QwenVLReader:
    def __init__(self, model_id, max_pixels, min_pixels, dtype="bfloat16", attn="sdpa", device="cuda"):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLM
        except ImportError as e:  # pragma: no cover
            raise ImportError("Need transformers>=4.49 for Qwen2.5-VL") from e
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("reader.device is 'cuda' but no NVIDIA GPU/driver was found. Run on a GPU machine "
                               "(e.g. Colab), or set reader.device: auto (CPU is only OK for a 2-3 question smoke test).")
        if device == "cpu" and dtype != "float32":
            print(f"[reader] CPU detected: using float32 instead of {dtype} (the 3B model needs ~13 GB RAM and is slow)")
            dtype = "float32"
        self.torch, self.device = torch, device
        self.model = VLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype),
                                         attn_implementation=attn).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        tok = self.processor.tokenizer
        self.letter_ids = [tok.encode(l, add_special_tokens=False)[0] for l in LETTERS]
        self.img_tok = getattr(self.model.config, "image_token_id", None) or tok.convert_tokens_to_ids("<|image_pad|>")
        total = sum(p.numel() for p in self.model.parameters())
        vis = sum(p.numel() for n, p in self.model.named_parameters() if "visual" in n)
        self.p_vit, self.p_llm = vis, total - vis
        print(f"[reader] {model_id}: LLM {self.p_llm/1e9:.2f}B params, vision tower {self.p_vit/1e9:.2f}B")

    def _sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def warmup(self):
        im = Image.new("RGB", (336, 336), "gray")
        for _ in range(2):
            self.answer([im] * 4, [0, 1, 2, 3], "What colour is this?", ["gray", "red"])

    def answer(self, images, times, question, options) -> dict:
        torch = self.torch
        content = []
        for t in times:
            content += [{"type": "text", "text": f"Frame at {float(t):.0f}s:"}, {"type": "image"}]
        opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
        content.append({"type": "text", "text":
                        f"Question: {question}\n{opts}\nAnswer with the option's letter from the given choices directly."})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=list(images), return_tensors="pt", padding=True).to(self.device)
        n_tok = int(inputs["input_ids"].shape[1])
        n_img = int((inputs["input_ids"] == self.img_tok).sum())
        with torch.inference_mode():
            self._sync()
            t0 = time.perf_counter()
            try:
                out = self.model(**inputs, logits_to_keep=1)
            except TypeError:
                out = self.model(**inputs)
            self._sync()
            lat = time.perf_counter() - t0
        logits = out.logits[0, -1].float()
        lp = torch.log_softmax(logits[self.letter_ids[: len(options)]], dim=-1).cpu().tolist()
        gflops = 2.0 * (self.p_llm * n_tok + self.p_vit * n_img * 4) / 1e9   # 4 ViT patches / merged token
        return dict(logprobs=lp, latency_s=lat, n_img_tokens=n_img, n_tokens=n_tok, gflops=gflops)


def load_reader(cfg) -> QwenVLReader:
    r = cfg["reader"]
    rd = QwenVLReader(r["model_id"], r["max_pixels"], r["min_pixels"], r["dtype"], r["attn"], r["device"])
    if rd.device.startswith("cuda"):
        rd.warmup()
    return rd
