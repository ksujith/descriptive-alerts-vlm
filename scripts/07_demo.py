#!/usr/bin/env python3
"""Run the alert model on an arbitrary video file.

    python scripts/07_demo.py --video clip.mp4 --adapter runs/t4_8frame/adapter

Uses the same uniform centre-of-interval frame sampling as training; feeding
clustered or single frames measurably degrades accuracy (see the ablation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from dalerts.extract import extract
from dalerts.model import MODEL_ID, load_for_inference, load_processor
from dalerts.prompts import build_messages


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--adapter", default=None, help="omit to run the base model zero-shot")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0, help="0 = to end of file")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--short-side", type=int, default=308)
    ap.add_argument("--arm", default="formatted", choices=["open", "formatted", "fewshot"])
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such file: {args.video}", file=sys.stderr)
        return 2

    end = args.end
    if end <= 0:
        import av

        with av.open(str(args.video)) as c:
            end = float(c.duration / 1_000_000) if c.duration else 0.0
        if end <= 0:
            print("could not determine duration; pass --end", file=sys.stderr)
            return 2

    print(f"[demo] {args.video.name}  window {args.start:.2f}-{end:.2f}s  frames={args.n_frames}")
    clip = extract(args.video, args.start, end, args.n_frames, short_side=args.short_side)
    print(f"[demo] sampled at t = {[round(t,2) for t in clip.actual_times]}")
    print(f"[demo] source {clip.source_size} fps {clip.source_fps:.2f} -> stored {clip.images[0].size}")

    if args.precision != "fp32" and not torch.cuda.is_available():
        print("[demo] no CUDA: falling back to fp32 on CPU (slow but works)")
        args.precision = "fp32"

    processor = load_processor(args.model_id)
    model = load_for_inference(args.adapter, args.model_id, args.precision,
                               device_map="auto" if torch.cuda.is_available() else None)

    messages = build_messages(args.arm, args.n_frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], videos=[clip.images], return_tensors="pt",
                      cap_pixels_per_frame=True)
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    print(f"[demo] {inputs['input_ids'].shape[1]} input tokens "
          f"(visual grid {inputs['video_grid_thw'].tolist()})")

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                             eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    answer = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print(f"ALERT: {answer.strip()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
