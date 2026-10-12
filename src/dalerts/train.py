"""LoRA SFT for descriptive alerts.

Run as ``python -m dalerts.train --config configs/t4_8frame.yaml``.

Kaggle-specific choices: checkpoints are written every ``save_steps`` and
``--resume auto`` picks up the newest one, because a free Kaggle GPU session is
capped (~12h) and the weekly quota is 30h. A run that cannot resume is a run
that cannot finish.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from .dataset import AlertCollator, AlertDataset
from .model import MODEL_ID, attach_lora, load_model, load_processor


@dataclass
class Config:
    # data
    manifest: str = "data/interim/clips.csv"
    frames: str = "data/frames"
    n_frames: int = 8
    arm: str = "formatted"
    max_train: int = 0            # 0 = all
    max_eval: int = 200
    # model
    model_id: str = MODEL_ID
    precision: str = "fp16"       # T4 has no bf16
    attn: str = "sdpa"            # FlashAttention-2 needs SM 8.0+
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_scope: str = "lm"
    max_tokens_per_frame: int = 256
    # optimisation
    output_dir: str = "runs/t4_8frame"
    epochs: float = 2.0
    lr: float = 1e-4
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    logging_steps: int = 25
    save_steps: int = 250
    eval_steps: int = 250
    save_total_limit: int = 2
    seed: int = 1234
    workers: int = 2
    gradient_checkpointing: bool = True

    @classmethod
    def load(cls, path: str | None, overrides: dict) -> "Config":
        data: dict = {}
        if path:
            data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--resume", default="auto", choices=["auto", "no"])
    for f in fields(Config):
        if f.type is bool or isinstance(f.default, bool):
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name, type=lambda s: s.lower() == "true", default=None)
        else:
            ap.add_argument(f"--{f.name.replace('_','-')}", dest=f.name, type=type(f.default), default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config, {f.name: getattr(args, f.name) for f in fields(Config)})

    set_seed_everywhere(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    print(json.dumps(cfg.__dict__, indent=2))

    if not torch.cuda.is_available():
        print("\n*** No CUDA device visible. This trainer targets a CUDA GPU "
              "(Kaggle T4); Apple MPS is not supported for Qwen2-VL training. ***")

    processor = load_processor(cfg.model_id, max_tokens=cfg.max_tokens_per_frame)
    train_ds = AlertDataset(cfg.manifest, cfg.frames, "train", cfg.n_frames)
    eval_ds = AlertDataset(cfg.manifest, cfg.frames, "val", cfg.n_frames)
    if cfg.max_train:
        train_ds.rows = train_ds.rows[: cfg.max_train]
    if cfg.max_eval:
        eval_ds.rows = eval_ds.rows[: cfg.max_eval]
    print(f"[data] train={len(train_ds)} val={len(eval_ds)} n_frames={cfg.n_frames} arm={cfg.arm}")

    model = load_model(cfg.model_id, cfg.precision, attn=cfg.attn)
    model = attach_lora(
        model,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        scope=cfg.lora_scope,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,
        fp16=(cfg.precision in {"fp16", "4bit"}),
        bf16=(cfg.precision == "bf16"),
        gradient_checkpointing=False,  # already enabled on the PEFT model
        dataloader_num_workers=cfg.workers,
        remove_unused_columns=False,   # our collator emits non-model keys
        label_names=["labels"],        # PEFT wrappers hide this from Trainer
        report_to=[],
        seed=cfg.seed,
        optim="adamw_torch",
        save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=AlertCollator(processor, arm=cfg.arm, train=True),
    )

    ckpt = None
    if args.resume == "auto" and out.exists():
        ckpt = get_last_checkpoint(str(out))
        if ckpt:
            print(f"[resume] continuing from {ckpt}")

    trainer.train(resume_from_checkpoint=ckpt)

    adapter_dir = out / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"\n[done] adapter -> {adapter_dir}")
    hist = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (out / "log_history.json").write_text(json.dumps(hist, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
