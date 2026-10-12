#!/usr/bin/env python3
"""Generate alerts for one (model, arm, n_frames) combination.

One invocation = one evaluation cell, written to its own JSONL. Splitting
generation from reporting keeps a Kaggle session restartable: a finished cell is
never recomputed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dalerts.charades import load_classes
from dalerts.dataset import AlertDataset
from dalerts.infer import generate_alerts
from dalerts.model import MODEL_ID, load_for_inference, load_processor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/clips.csv")
    ap.add_argument("--frames", default="data/frames")
    ap.add_argument("--charades", default="data/raw/Charades")
    ap.add_argument("--split", default="test")
    ap.add_argument("--arm", default="formatted", choices=["open", "formatted", "fewshot", "choice"])
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit for the zero-shot baseline")
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-tokens-per-frame", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="output JSONL (auto-named if omitted)")
    ap.add_argument("--tag", default=None, help="label for this cell, e.g. 'baseline' / 'ft'")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    tag = args.tag or ("ft" if args.adapter else "baseline")
    out = Path(args.out or f"reports/preds/{tag}__{args.arm}__f{args.n_frames}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        print(f"[skip] {out} already exists (use --overwrite to redo)")
        return 0

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device; generation on CPU/MPS will be extremely slow.")

    processor = load_processor(args.model_id, max_tokens=args.max_tokens_per_frame)
    ds = AlertDataset(args.manifest, args.frames, args.split, args.n_frames)
    if args.limit:
        ds.rows = ds.rows[: args.limit]

    class_names = None
    if args.arm == "choice":
        class_names = list(load_classes(Path(args.charades) / "Charades_v1_classes.txt").values())

    print(f"[gen] tag={tag} arm={args.arm} n_frames={args.n_frames} clips={len(ds)} "
          f"adapter={args.adapter or 'none (zero-shot)'}")
    model = load_for_inference(args.adapter, args.model_id, args.precision, attn=args.attn)

    records = generate_alerts(
        model, processor, ds,
        arm=args.arm,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        class_names=class_names,
    )
    meta = {
        "tag": tag, "arm": args.arm, "n_frames": args.n_frames, "split": args.split,
        "adapter": args.adapter, "model_id": args.model_id, "precision": args.precision,
        "max_tokens_per_frame": args.max_tokens_per_frame, "n": len(records),
    }
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"__meta__": meta}) + "\n")
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"[gen] wrote {len(records)} predictions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
