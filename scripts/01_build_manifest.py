#!/usr/bin/env python3
"""Build the clip manifest (CSV) from a local Charades annotation release.

This writes *metadata only* - no video or frame bytes - so the output is safe
to keep in the repo for reproducibility. See DATA_LICENSE.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalerts.charades import load_classes, load_videos
from dalerts.clips import Clip, build_clips, cap_per_class, split_by_subject, summarize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charades", default="data/raw/Charades", type=Path)
    ap.add_argument("--out", default="data/interim/clips.csv", type=Path)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--train-cap", type=int, default=100)
    ap.add_argument("--val-cap", type=int, default=4)
    ap.add_argument("--test-cap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    classes = load_classes(args.charades / "Charades_v1_classes.txt")
    train_videos = load_videos(args.charades / "Charades_v1_train.csv")
    test_videos = load_videos(args.charades / "Charades_v1_test.csv")

    shape = dict(min_dur=args.min_dur, max_dur=args.max_dur)
    trainval = build_clips(train_videos, classes, "train", **shape)
    trainval = split_by_subject(trainval, args.val_frac, args.seed)
    test = build_clips(test_videos, classes, "test", **shape)

    train = cap_per_class([c for c in trainval if c.split == "train"], args.train_cap, args.seed)
    val = cap_per_class([c for c in trainval if c.split == "val"], args.val_cap, args.seed)
    test = cap_per_class(test, args.test_cap, args.seed)

    allc = train + val + test
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(Clip)]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for c in allc:
            w.writerow(c.as_row())

    stats = {s: summarize([c for c in allc if c.split == s]) for s in ("train", "val", "test")}
    # Hard invariants: an actor or a video in two splits invalidates every number.
    by_split = {s: {c.subject for c in allc if c.split == s} for s in stats}
    vids = {s: {c.video_id for c in allc if c.split == s} for s in stats}
    stats["leakage"] = {
        "subject_train_val": len(by_split["train"] & by_split["val"]),
        "subject_train_test": len(by_split["train"] & by_split["test"]),
        "video_train_val": len(vids["train"] & vids["val"]),
        "video_train_test": len(vids["train"] & vids["test"]),
    }
    (args.out.parent / "manifest_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {args.out} ({len(allc)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
