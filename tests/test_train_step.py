"""End-to-end training-step test on a tiny randomly-initialised Qwen2-VL.

Validates the whole path the real run depends on - collator output ->
``forward(labels=...)`` -> loss -> backward -> optimizer step - without
downloading 4.4GB of weights or needing a GPU. A shape or label-masking
mismatch here would otherwise only surface after burning GPU quota.

The architecture is tiny but *structurally identical* where it matters: the
patch size, spatial merge and temporal patch size match the real model, so the
processor's ``pixel_values_videos``/``video_grid_thw`` line up exactly.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from dalerts.dataset import IGNORE_INDEX, AlertCollator, ClipExample
from dalerts.model import attach_lora, resolve_targets

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def _tiny_model():
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

    cfg = Qwen2VLConfig(
        text_config=dict(
            hidden_size=128,
            num_attention_heads=4,       # head_dim = 32
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=256,
            vocab_size=151936,           # real vocab: token ids must be valid
            # mrope sections must sum to head_dim/2 = 16
            rope_parameters=dict(type="mrope", mrope_section=[4, 6, 6],
                                 rope_theta=10000.0, rope_type="default"),
        ),
        vision_config=dict(
            depth=2,
            embed_dim=64,
            num_heads=4,
            hidden_size=128,             # must match text hidden_size (merger output)
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
            in_chans=3,
            mlp_ratio=4,
        ),
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen2VLForConditionalGeneration(cfg)


@pytest.fixture(scope="module")
def frames(tmp_path_factory):
    """Eight real JPEG frames at a Charades-like resolution."""
    from PIL import Image

    out = tmp_path_factory.mktemp("f")
    vid = out / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=410x308:rate=24:duration=4",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)],
        check=True,
    )
    from dalerts.extract import extract

    return extract(vid, 0.0, 4.0, 8, short_side=308).images


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_ID)


def _batch(processor, frames, n=2):
    examples = [
        ClipExample(f"c{i}", frames, "A person is closing a door.", "c006", ())
        for i in range(n)
    ]
    return AlertCollator(processor, arm="formatted", train=True)(examples)


@pytest.mark.slow
def test_forward_backward_and_step_updates_only_lora(processor, frames):
    model = _tiny_model()
    targets = resolve_targets(model, "lm")
    assert len(targets) == 2 * 7  # 2 layers x 7 projections
    model = attach_lora(model, r=4, alpha=8, scope="lm", gradient_checkpointing=True)

    batch = _batch(processor, frames, n=2)
    assert batch["labels"].shape == batch["input_ids"].shape
    supervised = (batch["labels"] != IGNORE_INDEX).sum().item()
    assert supervised == 2 * 8, supervised  # 8 target tokens per row

    model.train()
    out = model(**batch)
    assert out.loss is not None and torch.isfinite(out.loss), out.loss
    out.loss.backward()

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters"
    assert all("lora" in n.lower() for n, _ in trainable), \
        f"non-LoRA params trainable: {[n for n,_ in trainable if 'lora' not in n.lower()][:3]}"
    with_grad = [n for n, p in trainable if p.grad is not None and p.grad.abs().sum() > 0]
    assert with_grad, "no LoRA parameter received gradient - backward graph is severed"

    # Frozen base must stay frozen.
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    before = out.loss.item()
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-3)
    opt.step()
    opt.zero_grad()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p.detach(), frozen[n]), f"frozen param changed: {n}"

    after = model(**batch).loss.item()
    assert torch.isfinite(torch.tensor(after))
    assert after != before, "loss unchanged after an optimizer step"


@pytest.mark.slow
def test_loss_ignores_prompt_tokens(processor, frames):
    """Changing only the prompt's visual content must not change which tokens are supervised."""
    b1 = _batch(processor, frames, n=1)
    b2 = _batch(processor, list(reversed(frames)), n=1)
    assert (b1["labels"] != IGNORE_INDEX).sum() == (b2["labels"] != IGNORE_INDEX).sum()


@pytest.mark.slow
def test_generation_runs_and_respects_left_padding(processor, frames):
    model = _tiny_model().eval()
    examples = [ClipExample("a", frames, "x", "c006", ()),
                ClipExample("b", frames[:4], "y", "c006", ())]
    coll = AlertCollator(processor, arm="formatted", train=False)
    batch = coll(examples)
    meta = {k: batch.pop(k) for k in ("clip_ids", "gold_classes", "cooccurring", "targets")}
    assert batch["attention_mask"][:, -1].all(), "left padding: last column must be real tokens"
    with torch.no_grad():
        out = model.generate(**batch, max_new_tokens=4, do_sample=False)
    assert out.shape[0] == 2
    assert out.shape[1] == batch["input_ids"].shape[1] + 4
