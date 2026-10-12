"""Dataset and collator turning extracted frames into Qwen2-VL training batches.

Two details carry most of the correctness risk:

**Response-only loss.** The target is a single short sentence; the prompt is
~690 tokens, most of them visual. Training on the full sequence would drown the
~12 tokens we actually care about. We locate the final
``<|im_start|>assistant\\n`` header in the tokenised sequence and mask
everything up to and including it, so loss is computed on the answer alone.
The header is found *after* the processor has expanded ``<|video_pad|>``, which
avoids having to predict the visual token count ourselves.

**Frame caches are strided, never re-decoded.** ``n_frames`` selects a subset of
each clip's cached frames via :func:`dalerts.extract.stride_indices`, so the
frame-count ablation varies only temporal density.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .extract import stride_indices
from .prompts import build_messages

IGNORE_INDEX = -100


def load_manifest(path: str | Path, split: str | None = None) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if split:
        rows = [r for r in rows if r["split"] == split]
    for r in rows:
        r["cooccurring"] = tuple(c for c in r["cooccurring"].split("|") if c)
    return rows


@dataclass
class ClipExample:
    clip_id: str
    frames: list[Image.Image]
    target: str
    gold_class: str
    cooccurring: tuple[str, ...]


class AlertDataset(Dataset):
    """Clips from the manifest whose frames exist on disk."""

    def __init__(
        self,
        manifest: str | Path,
        frames_root: str | Path,
        split: str,
        n_frames: int = 8,
        *,
        require_frames: bool = True,
    ):
        self.frames_root = Path(frames_root)
        self.split = split
        self.n_frames = n_frames
        rows = load_manifest(manifest, split)
        if require_frames:
            kept, missing = [], 0
            for r in rows:
                if len(self._frame_paths(r["clip_id"])) >= n_frames:
                    kept.append(r)
                else:
                    missing += 1
            if missing:
                print(f"[dataset] {split}: skipping {missing}/{len(rows)} clips with missing frames")
            rows = kept
        if not rows:
            raise RuntimeError(f"no usable clips for split={split!r} under {frames_root}")
        self.rows = rows

    def _frame_paths(self, clip_id: str) -> list[Path]:
        return sorted((self.frames_root / self.split / clip_id).glob("f*.jpg"))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> ClipExample:
        r = self.rows[i]
        paths = self._frame_paths(r["clip_id"])
        idx = stride_indices(len(paths), self.n_frames)
        frames = [Image.open(paths[j]).convert("RGB") for j in idx]
        return ClipExample(
            clip_id=r["clip_id"],
            frames=frames,
            target=r["alert"],
            gold_class=r["class_id"],
            cooccurring=r["cooccurring"],
        )


def _find_last_subsequence(haystack: list[int], needle: list[int]) -> int:
    """Index just past the last occurrence of ``needle``, or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start + len(needle)
    return -1


class AlertCollator:
    """Builds padded, label-masked batches for training or generation."""

    def __init__(self, processor, arm: str = "formatted", *, train: bool = True,
                 class_names: list[str] | None = None, cap_pixels_per_frame: bool = True):
        self.processor = processor
        self.tok = processor.tokenizer
        self.arm = arm
        self.train = train
        self.class_names = class_names
        self.cap = cap_pixels_per_frame
        # Batched generation requires LEFT padding: with right padding the pad
        # tokens sit between the prompt and the first generated token, so short
        # prompts in a batch decode garbage. Training uses right padding, which
        # is fine because the pad positions are masked out of the loss.
        self.tok.padding_side = "right" if train else "left"
        # Encoded once: the assistant header marking where the answer begins.
        self.assistant_header = self.tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _prompt_text(self, n_frames: int) -> str:
        msgs = build_messages(self.arm, n_frames, class_names=self.class_names)
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def __call__(self, batch: list[ClipExample]) -> dict:
        videos = [ex.frames for ex in batch]
        texts = []
        for ex in batch:
            prompt = self._prompt_text(len(ex.frames))
            texts.append(prompt + ex.target + "<|im_end|>" if self.train else prompt)

        inputs = self.processor(
            text=texts,
            videos=videos,
            return_tensors="pt",
            padding=True,
            cap_pixels_per_frame=self.cap,
        )
        # NOTE: `mm_token_type_ids` must be forwarded, not dropped. transformers
        # v5 raises if multimodal grids are passed without it, because M-RoPE
        # needs it to assign 3D positions to visual vs text tokens. It is also
        # consumed by `prepare_inputs_for_generation`, so eval needs it too.

        if self.train:
            labels = inputs["input_ids"].clone()
            labels[inputs["attention_mask"] == 0] = IGNORE_INDEX
            for row in range(labels.size(0)):
                ids = inputs["input_ids"][row].tolist()
                cut = _find_last_subsequence(ids, self.assistant_header)
                if cut < 0:
                    raise RuntimeError("assistant header not found; chat template changed?")
                labels[row, :cut] = IGNORE_INDEX
            inputs["labels"] = labels
        else:
            inputs["clip_ids"] = [ex.clip_id for ex in batch]
            inputs["gold_classes"] = [ex.gold_class for ex in batch]
            inputs["cooccurring"] = [ex.cooccurring for ex in batch]
            inputs["targets"] = [ex.target for ex in batch]
        return inputs
