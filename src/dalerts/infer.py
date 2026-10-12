"""Batched greedy generation of descriptive alerts."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import AlertCollator, AlertDataset


@torch.no_grad()
def generate_alerts(
    model,
    processor,
    dataset: AlertDataset,
    *,
    arm: str = "formatted",
    batch_size: int = 4,
    max_new_tokens: int = 48,
    class_names: list[str] | None = None,
    workers: int = 2,
    log_every: int = 20,
) -> list[dict]:
    """Greedy-decode one alert per clip.

    Greedy (``do_sample=False``) keeps the baseline/fine-tuned comparison
    deterministic; sampling would make the reported deltas partly noise.
    """
    collator = AlertCollator(processor, arm=arm, train=False, class_names=class_names)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collator,
    )
    device = next(model.parameters()).device
    eos = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

    records: list[dict] = []
    t0 = time.time()
    for bi, batch in enumerate(loader, 1):
        meta = {k: batch.pop(k) for k in ("clip_ids", "gold_classes", "cooccurring", "targets")}
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos,
            pad_token_id=processor.tokenizer.pad_token_id or eos,
            use_cache=True,
        )
        # Strip the prompt: left padding means every row's prompt is the same length.
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        for i, text in enumerate(texts):
            records.append({
                "clip_id": meta["clip_ids"][i],
                "gold_class": meta["gold_classes"][i],
                "cooccurring": list(meta["cooccurring"][i]),
                "target": meta["targets"][i],
                "generated": text.strip(),
            })
        if log_every and (bi % log_every == 0 or bi == len(loader)):
            done = len(records)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  [gen] {done}/{len(dataset)}  {rate:.2f} clip/s  "
                  f"eta {(len(dataset)-done)/max(rate,1e-6)/60:.1f} min", flush=True)
    return records
