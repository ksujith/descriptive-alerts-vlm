#!/usr/bin/env python3
"""Push the LoRA adapter and a generated model card to the Hugging Face Hub.

Uploads the adapter, its config, the run config and the results tables.
Deliberately uploads **no Charades frames, clips or annotations**: the dataset
licence forbids redistribution (see DATA_LICENSE.md). The card is generated from
the real results JSON so the numbers it quotes cannot drift from the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dalerts.report import HEADLINE, markdown_table, taxonomy_table
from dalerts.taxonomy import BUCKET_DOC, BUCKETS

BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

CARD = """---
base_model: {base_model}
library_name: peft
pipeline_tag: video-text-to-text
tags:
- qwen2-vl
- lora
- peft
- video-understanding
- activity-recognition
- video-captioning
license: other
license_name: charades-non-commercial-derivative
license_link: https://prior.allenai.org/projects/charades
---

# Qwen2-VL-2B · Descriptive Alerts (LoRA)

A LoRA adapter for `{base_model}` that turns a short home-video clip into a
single-line activity alert:

> `A person is closing a door.`

This is the task behind consumer camera "descriptive alerts" — instead of
*motion detected*, say what actually happened. Trained on
[Charades](https://prior.allenai.org/projects/charades) clips, which are
crowd-acted everyday household activities shot in real homes.

## ⚠️ Licence and intended use

**Non-commercial research and evaluation only.** The adapter is a derivative of
the Charades dataset, whose licence restricts use to academic, non-profit or
government-sponsored research and evaluation, and does not grant commercial
rights. The base model is separately licensed by Qwen. No Charades video,
frames or annotations are distributed in this repository.

This is a portfolio/research artifact, not a safety or security product. Do not
use it for surveillance of people without consent, or for any decision that
affects someone's safety, access or liberty.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

BASE = "{base_model}"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    BASE, dtype=torch.float16, attn_implementation="sdpa", device_map="auto"
)
model = PeftModel.from_pretrained(model, "{repo_id}").merge_and_unload().eval()
processor = AutoProcessor.from_pretrained(BASE)

frames = [...]  # {n_frames} PIL RGB frames sampled uniformly across the clip
messages = [
    {{"role": "system", "content": [{{"type": "text", "text": "You are a home security camera assistant. You watch short clips and report the single main human activity in one brief sentence."}}]}},
    {{"role": "user", "content": [
        {{"type": "video"}},
        {{"type": "text", "text": "Describe the main activity happening in this video clip. Reply with exactly one short sentence beginning 'A person is', naming the single main activity. No preamble, no extra detail, no list."}},
    ]}},
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], videos=[frames], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
print(processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Frames must be sampled **uniformly across the clip** at the centres of N equal
sub-intervals — the same scheme used in training. Clustered or single-frame
input degrades accuracy measurably (see the ablation).

## Training

| | |
|---|---|
| Base model | `{base_model}` |
| Method | LoRA, r={lora_r}, α={lora_alpha}, dropout={lora_dropout} |
| Adapted modules | {lora_scope} ({n_modules} linear layers; vision tower frozen) |
| Precision | {precision} (Turing/T4 has no bf16) |
| Frames per clip | {n_frames}, uniform |
| Max tokens/frame | {max_tokens_per_frame} |
| Epochs | {epochs} |
| LR / schedule | {lr} / {lr_scheduler}, warmup {warmup_ratio} |
| Effective batch | {batch_size} × {grad_accum} grad-accum |
| Hardware | 1 × NVIDIA T4 (16 GB), Kaggle |
| Loss | response tokens only (prompt and visual tokens masked) |

### Data

Charades action segments windowed into short clips (2–10 s), one clip per
annotated segment, target = that segment's class rendered as a sentence.
Splits are **actor- and video-disjoint**: the official Charades test split
shares no subject with train, and validation is carved from train by subject.
Per-class capping reduces the native ~58× class imbalance to ~5×.

{data_table}

## Results

{results_section}

## Failure taxonomy

{taxonomy_doc}

{taxonomy_section}

## Limitations

- **Charades domain.** Crowd-acted household activities, mostly single-actor,
  handheld or tripod framing. Real fixed-mount camera footage differs in angle,
  lighting, compression and multi-person activity; expect a domain drop.
- **Closed label space.** Trained to emit phrasings drawn from 157 Charades
  classes. It will describe an unfamiliar activity using the nearest class it
  knows rather than saying it does not know.
- **Single action per clip.** Charades clips genuinely contain several
  simultaneous actions (~4.3 co-occurring labels per test clip), but the model
  emits one. The `lenient` metric accounts for this; the strict one does not.
- **No person re-identification, no audio, no temporal localisation.**

## Reproducing

Everything — data prep, training, evaluation and this card — is in
[{repo_url}]({repo_url}). Charades must be downloaded separately under its own
licence.

## Citation

```bibtex
@article{{sigurdsson2016hollywood,
  author  = {{Gunnar A. Sigurdsson and G{{\\"u}}l Varol and Xiaolong Wang and
             Ivan Laptev and Ali Farhadi and Abhinav Gupta}},
  title   = {{Hollywood in Homes: Crowdsourcing Data Collection for Activity Understanding}},
  journal = {{ArXiv e-prints}},
  eprint  = {{1604.01753}},
  year    = {{2016}}
}}
```
"""


def build_results_section(results: dict) -> tuple[str, str]:
    """Headline table + ablation + taxonomy, from the real results JSON."""
    def pick(**kw):
        out = {}
        for name, cell in results.items():
            m = cell.get("meta", {})
            if all(m.get(k) == v for k, v in kw.items()):
                out[name] = cell
        return out

    parts: list[str] = []
    head = pick(arm="formatted", n_frames=8)
    if len(head) >= 2:
        pretty = {}
        for name, cell in head.items():
            label = "Fine-tuned (this adapter)" if cell["meta"].get("tag") == "ft" else "Zero-shot base model"
            pretty[label] = cell
        parts += ["### Baseline vs fine-tuned", "",
                  "Identical prompt, identical 8 frames, greedy decoding, held-out test split.", "",
                  markdown_table(pretty, "Model"), ""]

    base_arms = pick(tag="baseline")
    if len(base_arms) >= 2:
        parts += ["### How much of the gap is output format?", "",
                  "A stock VLM asked to \"describe this video\" writes a paragraph, which scores "
                  "near zero against a one-line target. These arms give the base model a fair "
                  "chance before claiming a gain.", "",
                  markdown_table({n: c for n, c in sorted(base_arms.items())}, "Baseline prompt arm"), ""]

    abl = pick(tag="ft", arm="formatted")
    if len(abl) >= 2:
        ordered = dict(sorted(abl.items(), key=lambda kv: kv[1]["meta"].get("n_frames", 0)))
        pretty = {f"{c['meta']['n_frames']} frame" + ("" if c['meta']['n_frames'] == 1 else "s"): c for c in ordered.values()}
        parts += ["### Ablation: frames at inference", "",
                  "Trained once at 8 frames. Each arm strides the *same* cached frames, so only "
                  "temporal density varies.", "",
                  markdown_table(pretty, "Frames"), ""]

    tax = {}
    for name, cell in results.items():
        m = cell.get("meta", {})
        if m.get("arm") == "formatted" and m.get("n_frames") == 8:
            tax["Fine-tuned" if m.get("tag") == "ft" else "Zero-shot"] = cell
    tax_md = taxonomy_table(tax) if tax else ""
    return "\n".join(parts) if parts else "_Results pending._", tax_md


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--results", type=Path, default=Path("reports/results.json"))
    ap.add_argument("--run-config", type=Path, default=None)
    ap.add_argument("--repo-url", default="https://github.com/ksujith/descriptive-alerts-vlm")
    ap.add_argument("--manifest-stats", type=Path, default=Path("data/interim/manifest_stats.json"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, upload nothing")
    args = ap.parse_args()

    if not args.adapter.is_dir():
        print(f"adapter dir not found: {args.adapter}", file=sys.stderr)
        return 2

    cfg = json.loads(args.run_config.read_text()) if args.run_config and args.run_config.exists() else {}
    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    results_section, taxonomy_section = build_results_section(results)

    data_table = ""
    if args.manifest_stats.exists():
        st = json.loads(args.manifest_stats.read_text())
        rows = ["| Split | Clips | Videos | Subjects | Classes |", "|---|---|---|---|---|"]
        for s in ("train", "val", "test"):
            if s in st:
                d = st[s]
                rows.append(f"| {s} | {d['clips']} | {d['videos']} | {d['subjects']} | {d['classes_present']} |")
        data_table = "\n".join(rows)

    n_modules = 196 if cfg.get("lora_scope", "lm") == "lm" else 324
    card = CARD.format(
        base_model=cfg.get("model_id", BASE_MODEL),
        repo_id=args.repo_id,
        repo_url=args.repo_url,
        lora_r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        lora_scope="language model only" if cfg.get("lora_scope", "lm") == "lm" else "language model + vision tower",
        n_modules=n_modules,
        precision=cfg.get("precision", "fp16"),
        n_frames=cfg.get("n_frames", 8),
        max_tokens_per_frame=cfg.get("max_tokens_per_frame", 256),
        epochs=cfg.get("epochs", 2.0),
        lr=cfg.get("lr", 1e-4),
        lr_scheduler=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        batch_size=cfg.get("batch_size", 1),
        grad_accum=cfg.get("grad_accum", 8),
        data_table=data_table,
        results_section=results_section,
        taxonomy_doc="\n".join(f"- **{b}** — {BUCKET_DOC[b]}" for b in BUCKETS),
        taxonomy_section=taxonomy_section,
    )

    card_path = args.adapter / "README.md"
    card_path.write_text(card)
    print(f"model card -> {card_path} ({len(card)} chars)")

    if args.dry_run:
        print("--dry-run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    # Allowlist: adapter + card + result tables only. Never frames or annotations.
    allow = ["adapter_model.safetensors", "adapter_config.json", "README.md",
             "chat_template.*", "preprocessor_config.json", "tokenizer*", "vocab.json",
             "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    api.upload_folder(
        folder_path=str(args.adapter),
        repo_id=args.repo_id,
        repo_type="model",
        allow_patterns=allow,
        commit_message="Add LoRA adapter and model card",
    )
    for extra, dest in ((args.results, "results.json"), (args.run_config, "train_config.json")):
        if extra and Path(extra).exists():
            api.upload_file(path_or_fileobj=str(extra), path_in_repo=dest,
                            repo_id=args.repo_id, repo_type="model")
    print(f"pushed -> https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
