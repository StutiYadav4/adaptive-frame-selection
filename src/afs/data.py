"""Dataset loaders -> one unified question format.

Every question is a dict:
  qid, dataset, video_path, question, options (list[str]), answer (int), qtype, split
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".avi", ".mov"}


def _index_videos(video_dir) -> dict[str, str]:
    idx = {}
    for p in Path(video_dir).rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            idx[p.stem] = p.as_posix()
    if not idx:
        raise FileNotFoundError(f"No videos found under {video_dir}")
    return idx


def _canon(x) -> str:
    return " ".join(str(x).split()).lower()          # strip, collapse spaces, lowercase


def parse_answer(ans, options) -> int:
    """answer text | int index | letter 'A'.. | numeric string  ->  option index.
    The answer TEXT is matched first (whitespace/case-insensitive), because some mirrors store the
    answer string rather than its index."""
    if isinstance(ans, int) or (hasattr(ans, "item") and float(ans) == int(ans)):
        return int(ans)
    canon = [_canon(o) for o in options]
    c = _canon(ans)
    if c in canon:
        return canon.index(c)
    letters = str(ans).strip().strip("().: ").upper()
    if len(letters) == 1 and letters in "ABCDEFGH" and "ABCDEFGH".index(letters) < len(options):
        return "ABCDEFGH".index(letters)
    if c.isdigit() and int(c) < len(options):
        return int(c)
    raise ValueError(f"cannot map answer {ans!r} onto options {options}")


def load_nextqa(ann_csv, video_dir) -> list[dict]:
    """NExT-QA / NExT-GQA csv: video (or video_id), question, answer, qid, type, a0..a4.
    Questions whose video is not on disk are skipped, so a partial video download is fine."""
    df = pd.read_csv(ann_csv)
    vcol = "video_id" if "video_id" in df.columns else "video"
    index_answer = pd.api.types.is_integer_dtype(df["answer"])
    vids = _index_videos(video_dir)
    rows, missing, bad = [], 0, 0
    for r in df.to_dict("records"):
        vid = str(r[vcol])
        if vid not in vids:
            missing += 1
            continue
        opts = [str(r[f"a{i}"]).strip() for i in range(5)]
        try:
            ans = int(r["answer"]) if index_answer else parse_answer(r["answer"], opts)
        except ValueError:
            bad += 1
            continue
        rows.append(dict(
            qid=f"nextqa_{vid}_{r['qid']}",
            dataset="nextqa",
            video_path=vids[vid],
            question=str(r["question"]).strip(),
            options=opts,
            answer=ans,
            qtype=str(r["type"]),
        ))
    print(f"[nextqa] {len(rows)} questions ({missing} skipped: video missing, {bad} skipped: answer not parseable)")
    return rows


def load_egoschema(questions_json, answers_json, video_dir) -> list[dict]:
    """Official EgoSchema: questions.json (q_uid, question, 'option 0'..'option 4'),
    subset_answers.json {q_uid: int}. Videos are named <q_uid>.mp4."""
    qs = json.load(open(questions_json))
    ans = json.load(open(answers_json))
    vids = _index_videos(video_dir)
    rows = []
    for q in qs:
        uid = q["q_uid"]
        if uid not in ans or uid not in vids:
            continue
        rows.append(dict(
            qid=f"ego_{uid}", dataset="egoschema", video_path=vids[uid],
            question=q["question"], options=[q[f"option {i}"] for i in range(5)],
            answer=int(ans[uid]), qtype="all",
        ))
    print(f"[egoschema] {len(rows)} questions")
    return rows


def load_lvb(ann_json, video_dir) -> list[dict]:
    """LongVideoBench lvb_val.json: id, video_id, question, candidates, correct_choice,
    video_path, question_category, duration_group."""
    data = json.load(open(ann_json))
    vids = _index_videos(video_dir)
    rows = []
    for d in data:
        vp = Path(video_dir) / d.get("video_path", "")
        path = str(vp) if vp.is_file() else vids.get(str(d["video_id"]))
        if path is None:
            continue
        ans = d["correct_choice"]
        if isinstance(ans, str):
            ans = "ABCDEFGH".index(ans.strip().upper()[0])
        rows.append(dict(
            qid=f"lvb_{d['id']}", dataset="longvideobench", video_path=path,
            question=d["question"], options=list(d["candidates"]), answer=int(ans),
            qtype=str(d.get("question_category", "all")),
        ))
    print(f"[longvideobench] {len(rows)} questions")
    return rows


def load_tempcore(benchmark, root, repo="HJOK/TempCore", split="test") -> list[dict]:
    """Compact multi-benchmark set (HF: HJOK/TempCore). `benchmark` is one of its configs, e.g.
    longvideobench | egoschema | nextqa | mlvu | videomme | mvbench. `root` = local download dir
    (video_path in each row is relative to it). Schema per the dataset card: question, options,
    answer, video_path."""
    from datasets import load_dataset
    ds = load_dataset(repo, benchmark, split=split)
    rows, missing = [], 0
    for i, r in enumerate(ds):
        path = Path(root) / r["video_path"]
        if not path.is_file():
            missing += 1
            continue
        opts = [str(o) for o in r["options"]]
        qtype = next((str(r[k]) for k in ("question_type", "task", "category", "type") if r.get(k)), "all")
        rows.append(dict(qid=f"tc_{benchmark}_{i}", dataset=f"tempcore_{benchmark}", video_path=str(path),
                         question=str(r["question"]), options=opts,
                         answer=parse_answer(r["answer"], opts), qtype=qtype))
    print(f"[tempcore/{benchmark}] {len(rows)} questions ({missing} skipped: video missing)")
    return rows


LOADERS = {"nextqa": load_nextqa, "egoschema": load_egoschema, "lvb": load_lvb, "tempcore": load_tempcore}


def build_subset(rows, n, dev_frac, seed=0, stratify=True) -> list[dict]:
    rng = random.Random(seed)
    n = min(n, len(rows))
    if stratify:
        groups = defaultdict(list)
        for r in rows:
            groups[r["qtype"]].append(r)
        total = len(rows)
        picked = []
        quotas = {k: len(v) * n / total for k, v in groups.items()}
        base = {k: int(q) for k, q in quotas.items()}
        left = n - sum(base.values())
        for k in sorted(quotas, key=lambda k: quotas[k] - base[k], reverse=True)[:left]:
            base[k] += 1
        for k, v in groups.items():
            picked += rng.sample(v, min(base[k], len(v)))
    else:
        picked = rng.sample(rows, n)
    rng.shuffle(picked)
    # split BY VIDEO so no video appears in both dev and test (avoids leakage for tuning/distillation)
    vids = sorted({r["video_path"] for r in picked})
    rng.shuffle(vids)
    n_dev_v = int(round(dev_frac * len(vids))) if len(vids) > 1 else 0
    dev_vids = set(vids[:n_dev_v])
    for r in picked:
        r["split"] = "dev" if r["video_path"] in dev_vids else "test"
    return picked
