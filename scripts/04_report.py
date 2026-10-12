#!/usr/bin/env python3
"""Score every prediction JSONL and emit the report tables.

Produces reports/results.json plus reports/RESULTS.md containing the
baseline-vs-fine-tuned table, the frame-count ablation, the failure taxonomy,
and sampled examples per failure bucket.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalerts.metrics import LabelSpace
from dalerts.report import markdown_table, score, taxonomy_table
from dalerts.taxonomy import BUCKET_DOC, BUCKETS


def load_cell(path: Path) -> tuple[dict, list[dict]]:
    meta, recs = {}, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if "__meta__" in obj:
                meta = obj["__meta__"]
            else:
                recs.append(obj)
    return meta, recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="reports/preds")
    ap.add_argument("--charades", default="data/raw/Charades")
    ap.add_argument("--out-json", default="reports/results.json")
    ap.add_argument("--out-md", default="reports/RESULTS.md")
    ap.add_argument("--examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    labels = LabelSpace.load(args.charades)
    files = sorted(Path(args.preds).glob("*.jsonl"))
    if not files:
        print(f"no prediction files under {args.preds}", file=sys.stderr)
        return 2

    cells: dict[str, dict] = {}
    all_recs: dict[str, list[dict]] = {}
    for f in files:
        meta, recs = load_cell(f)
        if not recs:
            continue
        name = f.stem
        # The naive `open` arm was never asked for a format, so don't charge it
        # a format violation.
        enforce = meta.get("arm") != "open"
        cells[name] = {"meta": meta, **score(recs, labels, enforce_format=enforce)}
        all_recs[name] = recs
        print(f"scored {name}: n={cells[name]['n']} strict={cells[name]['strict_top1']}")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(cells, indent=2))

    def subset(pred) -> dict:
        return {k: v for k, v in cells.items() if pred(v["meta"])}

    md: list[str] = ["# Results", "",
                     "All numbers are percentages on the held-out Charades test split",
                     "(actor- and video-disjoint from training). Greedy decoding.", ""]

    base = subset(lambda m: m.get("tag") == "baseline")
    ft = subset(lambda m: m.get("tag") == "ft")

    if base:
        md += ["## Zero-shot baseline, by prompt arm", "",
               "A stock VLM's apparent score depends heavily on how it is asked. "
               "Reporting only the naive `open` arm would overstate the fine-tune's gain.", "",
               markdown_table(base, "Baseline arm"), ""]
    if ft:
        md += ["## Fine-tuned (LoRA)", "", markdown_table(ft, "Fine-tuned run"), ""]

    main_pair = {k: v for k, v in cells.items() if v["meta"].get("n_frames") == 8
                 and v["meta"].get("arm") == "formatted"}
    if len(main_pair) >= 2:
        md += ["## Headline: baseline vs fine-tuned (same prompt, same 8 frames)", "",
               markdown_table(main_pair, "Model"), ""]

    abl = {k: v for k, v in cells.items() if v["meta"].get("tag") == "ft"
           and v["meta"].get("arm") == "formatted"}
    if len(abl) >= 2:
        ordered = dict(sorted(abl.items(), key=lambda kv: kv[1]["meta"].get("n_frames", 0)))
        md += ["## Ablation: number of frames at inference", "",
               "The model is trained once at 8 frames. Each arm strides the *same* cached "
               "frames, so only temporal density changes.", "",
               markdown_table(ordered, "Frames"), ""]

    md += ["## Failure taxonomy", ""]
    for b in BUCKETS:
        md.append(f"- **{b}** — {BUCKET_DOC[b]}")
    md += ["", taxonomy_table(cells), ""]

    rng = random.Random(args.seed)
    md += ["## Sampled failures", ""]
    for name, recs in all_recs.items():
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            by_bucket[r["bucket"]].append(r)
        interesting = [b for b in BUCKETS if b not in ("correct",) and by_bucket.get(b)]
        if not interesting:
            continue
        md += [f"### `{name}`", ""]
        for b in interesting:
            picks = rng.sample(by_bucket[b], min(args.examples, len(by_bucket[b])))
            md.append(f"**{b}** ({len(by_bucket[b])} cases)")
            md.append("")
            md.append("| gold | generated |")
            md.append("|---|---|")
            for r in picks:
                g = r["target"].replace("|", "/")
                p = (r["generated"] or "").replace("|", "/").replace("\n", " ")[:120]
                md.append(f"| {g} | {p} |")
            md.append("")
    Path(args.out_md).write_text("\n".join(md))
    print(f"\nwrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
