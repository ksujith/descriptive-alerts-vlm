#!/usr/bin/env python3
"""Pack extracted frames into per-split tars for upload as a PRIVATE Kaggle dataset.

Why tars: the frame cache is ~140k small JPEGs, and uploading those file-by-file
is slow and fragile. Three uncompressed tars (JPEG is already compressed) upload
quickly and extract to Kaggle's local SSD in seconds.

LICENSING - read before running. Charades is non-commercial research data whose
licence does not grant redistribution rights. The dataset this produces must be
kept **private**. Training on any rented GPU necessarily places the frames on
third-party hardware, which is the same posture as Colab/RunPod/any cloud box;
what we avoid is *public* redistribution. This script never uploads anything -
it only writes local tars and a metadata stub with visibility forced to private.
See DATA_LICENSE.md.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, default=Path("data/frames"))
    ap.add_argument("--manifest", type=Path, default=Path("data/interim/clips.csv"))
    ap.add_argument("--charades", type=Path, default=Path("data/raw/Charades"))
    ap.add_argument("--out", type=Path, default=Path("data/kaggle_upload"))
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--username", default=None, help="Kaggle username for the metadata stub")
    ap.add_argument("--slug", default="charades-alert-frames")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in args.splits.split(","):
        src = args.frames / split
        if not src.is_dir():
            print(f"[skip] {src} missing")
            continue
        dest = args.out / f"frames_{split}.tar"
        n = 0
        with tarfile.open(dest, "w") as tf:
            for clip_dir in sorted(src.iterdir()):
                if not clip_dir.is_dir():
                    continue
                for jpg in sorted(clip_dir.glob("f*.jpg")):
                    tf.add(jpg, arcname=f"frames/{split}/{clip_dir.name}/{jpg.name}")
                    n += 1
        size = dest.stat().st_size
        total += size
        print(f"[pack] {dest.name}: {n} frames, {size/1e9:.2f} GB")

    # Metadata and annotations travel with the frames so the notebook is self-contained.
    meta_tar = args.out / "meta.tar"
    with tarfile.open(meta_tar, "w") as tf:
        tf.add(args.manifest, arcname=f"meta/{args.manifest.name}")
        for name in ("Charades_v1_classes.txt", "Charades_v1_mapping.txt",
                     "Charades_v1_objectclasses.txt", "Charades_v1_verbclasses.txt",
                     "license.txt"):
            p = args.charades / name
            if p.exists():
                tf.add(p, arcname=f"meta/Charades/{name}")
    print(f"[pack] {meta_tar.name}: {meta_tar.stat().st_size/1e6:.1f} MB")

    stub = {
        "title": "Charades descriptive-alert frames (private)",
        "id": f"{args.username or 'YOUR_KAGGLE_USERNAME'}/{args.slug}",
        "licenses": [{"name": "other"}],
        "isPrivate": True,
    }
    (args.out / "dataset-metadata.json").write_text(json.dumps(stub, indent=2))
    print(f"\ntotal payload: {total/1e9:.2f} GB -> {args.out}")
    print("\nUpload (keep it private):")
    print(f"  kaggle datasets create -p {args.out} --dir-mode skip")
    print("  # or, for an update:  kaggle datasets version -p "
          f"{args.out} -m 'refresh' --dir-mode skip")
    print("\nNOTE: dataset-metadata.json pins isPrivate=true. Do not publish it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
