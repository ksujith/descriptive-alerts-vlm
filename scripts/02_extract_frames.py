#!/usr/bin/env python3
"""Decode each manifest clip into a fixed number of JPEG frames.

Train/val clips are decoded at 8 frames and test clips at 16. Lower frame
budgets for the ablation are produced later by *striding* these caches
(:func:`dalerts.extract.stride_indices`), never by re-decoding, so every
ablation arm sees byte-identical pixels and differs only in temporal density.

Resumable: a clip whose frames all exist is skipped, so the job can be
interrupted and restarted. Per-clip failures are logged and do not abort the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalerts.extract import extract

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".webm", ".mov")


def find_videos(video_dir: Path) -> dict[str, Path]:
    """Map video id -> path, tolerating whatever directory the archive made."""
    index: dict[str, Path] = {}
    for p in video_dir.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            index.setdefault(p.stem, p)
    return index


def _expected_paths(out_root: Path, split: str, clip_id: str, n: int) -> list[Path]:
    d = out_root / split / clip_id
    return [d / f"f{i:02d}.jpg" for i in range(n)]


def _job(task: tuple) -> dict:
    clip_id, split, video_path, start, end, n_frames, short_side, quality, out_root = task
    out_root = Path(out_root)
    paths = _expected_paths(out_root, split, clip_id, n_frames)
    if all(p.exists() and p.stat().st_size > 0 for p in paths):
        return {"clip_id": clip_id, "status": "skipped"}
    try:
        frames = extract(video_path, start, end, n_frames, short_side=short_side)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        for img, p in zip(frames.images, paths):
            img.save(p, format="JPEG", quality=quality, optimize=True)
        return {
            "clip_id": clip_id,
            "split": split,
            "status": "ok",
            "n_frames": n_frames,
            "times": [round(t, 4) for t in frames.actual_times],
            # Charades fps ranges down to ~6; a high frame budget on a short,
            # low-fps window necessarily repeats source frames. Recording this
            # lets the report separate "model failed" from "clip had no more
            # temporal information to give".
            "unique_frames": len(set(round(t, 4) for t in frames.actual_times)),
            "source_fps": round(frames.source_fps, 4),
            "source_size": list(frames.source_size),
            "stored_size": list(frames.images[0].size),
        }
    except Exception as exc:  # noqa: BLE001 - one bad video must not kill the run
        return {"clip_id": clip_id, "split": split, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/interim/clips.csv"))
    ap.add_argument("--videos", type=Path, default=Path("data/raw/Charades_v1_480"))
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    ap.add_argument("--train-frames", type=int, default=8)
    ap.add_argument("--test-frames", type=int, default=16)
    ap.add_argument("--short-side", type=int, default=308, help="multiple of 28 suits Qwen2-VL's patch grid")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="process only N clips (smoke test)")
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    wanted = set(args.splits.split(","))
    rows = [r for r in csv.DictReader(open(args.manifest, encoding="utf-8")) if r["split"] in wanted]
    videos = find_videos(args.videos)
    if not videos:
        print(f"ERROR: no video files under {args.videos}", file=sys.stderr)
        return 2
    print(f"manifest: {len(rows)} clips | videos on disk: {len(videos)}")

    tasks, missing = [], []
    for r in rows:
        vp = videos.get(r["video_id"])
        if vp is None:
            missing.append(r["clip_id"])
            continue
        n = args.test_frames if r["split"] == "test" else args.train_frames
        tasks.append((r["clip_id"], r["split"], str(vp), float(r["start"]), float(r["end"]),
                      n, args.short_side, args.quality, str(args.out)))
    if missing:
        print(f"WARNING: {len(missing)} clips have no video on disk (e.g. {missing[:3]})")
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"to process: {len(tasks)} clips with {args.workers} workers")

    args.out.mkdir(parents=True, exist_ok=True)
    idx_path, err_path = args.out / "index.jsonl", args.out / "failures.csv"
    ok = skipped = failed = 0
    t0 = time.time()
    with open(idx_path, "a", encoding="utf-8") as idx, open(err_path, "a", newline="", encoding="utf-8") as errf:
        ew = csv.writer(errf)
        with Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_job, tasks, chunksize=8), 1):
                if res["status"] == "ok":
                    ok += 1
                    idx.write(json.dumps(res) + "\n")
                elif res["status"] == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    ew.writerow([res["clip_id"], res.get("error", "")])
                if i % 250 == 0 or i == len(tasks):
                    el = time.time() - t0
                    rate = i / max(el, 1e-6)
                    eta = (len(tasks) - i) / max(rate, 1e-6)
                    print(f"  {i}/{len(tasks)}  ok={ok} skip={skipped} fail={failed}  "
                          f"{rate:.1f} clips/s  eta {eta/60:.1f} min", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: ok={ok} skipped={skipped} failed={failed}")
    print(f"index  -> {idx_path}")
    if failed:
        print(f"failures -> {err_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
