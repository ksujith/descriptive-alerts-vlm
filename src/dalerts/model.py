"""Model/processor loading and LoRA attachment, targeted at a 16GB T4.

T4 (Turing, SM 7.5) constraints that shape the defaults:

* **No bf16.** Turing has no bfloat16 tensor cores, so training runs in fp16.
  The base weights are fp16 while the LoRA parameters are cast to fp32 -
  without that cast ``GradScaler`` raises "Attempting to unscale FP16
  gradients" and training dies on the first optimizer step.
* **No FlashAttention-2.** It requires SM 8.0+, so we use ``sdpa``.
* **4-bit is optional, not default.** Qwen2-VL-2B in fp16 is ~4.4GB and an
  8-frame clip is only ~750 tokens, so a plain fp16 LoRA fits a T4 with room to
  spare and avoids quantisation error. ``--precision 4bit`` exists as an escape
  hatch, and deliberately keeps the vision tower out of quantisation.
"""

from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Verified against the real module tree: the language model uses split
# q/k/v/o projections while the vision tower uses a *fused* `qkv` plus
# `proj`/`fc1`/`fc2`, so these leaf names cannot accidentally match the ViT.
LM_TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
VISION_TARGET_LEAVES = ("qkv", "proj", "fc1", "fc2")

PIXELS = 28 * 28  # one post-merge visual token


def load_processor(model_id: str = MODEL_ID, min_tokens: int = 4, max_tokens: int = 256):
    """Processor with an explicit pixel budget per frame.

    Bounds are expressed in post-merge visual tokens so the cost is legible:
    ``max_tokens=256`` caps one frame at 256 tokens before temporal pairing.
    """
    return AutoProcessor.from_pretrained(
        model_id,
        size={"shortest_edge": min_tokens * PIXELS, "longest_edge": max_tokens * PIXELS},
    )


def resolve_targets(model: torch.nn.Module, scope: str = "lm") -> list[str]:
    """Return the *full* module names LoRA should wrap.

    Resolving concrete names (rather than handing PEFT bare leaf names) makes
    the adapter surface explicit and auditable in the run log.
    """
    if scope not in {"lm", "lm+vision"}:
        raise ValueError(f"scope must be 'lm' or 'lm+vision', got {scope!r}")
    leaves = set(LM_TARGET_LEAVES)
    if scope == "lm+vision":
        leaves |= set(VISION_TARGET_LEAVES)

    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in leaves:
            continue
        if name.endswith("lm_head"):
            continue  # never adapt the output head
        if scope == "lm" and ".visual." in name:
            continue
        names.append(name)
    if not names:
        raise RuntimeError(f"no LoRA targets resolved for scope={scope!r}")
    return names


def load_model(
    model_id: str = MODEL_ID,
    precision: str = "fp16",
    *,
    attn: str = "sdpa",
    device_map: str | dict | None = "auto",
):
    """Load the base VLM at the requested precision."""
    kwargs: dict = {"attn_implementation": attn, "device_map": device_map}

    if precision == "4bit":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            # Quantising the ViT costs visual fidelity for little memory gain.
            llm_int8_skip_modules=["visual", "lm_head"],
        )
        kwargs["dtype"] = torch.float16
    elif precision == "fp16":
        kwargs["dtype"] = torch.float16
    elif precision == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif precision == "fp32":
        kwargs["dtype"] = torch.float32
    else:
        raise ValueError(f"unknown precision {precision!r}")

    return Qwen2VLForConditionalGeneration.from_pretrained(model_id, **kwargs)


def attach_lora(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    scope: str = "lm",
    gradient_checkpointing: bool = True,
):
    """Wrap ``model`` in a LoRA adapter and make it safely trainable in fp16."""
    targets = resolve_targets(model, scope)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, cfg)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required when checkpointing a frozen base: without it the inputs to
        # the first checkpointed block have no grad and the backward graph is
        # severed, producing "element 0 of tensors does not require grad".
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    # fp16 master weights would break GradScaler; keep trainables in fp32.
    n_trainable = 0
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
            n_trainable += p.numel()
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[lora] scope={scope} r={r} alpha={alpha} modules={len(targets)} "
        f"trainable={n_trainable/1e6:.2f}M / {total/1e6:.1f}M ({100*n_trainable/total:.3f}%)"
    )
    return model


def load_for_inference(
    adapter_path: str | None,
    model_id: str = MODEL_ID,
    precision: str = "fp16",
    *,
    attn: str = "sdpa",
    device_map: str | dict | None = "auto",
):
    """Base model, optionally with a trained adapter merged in for speed."""
    model = load_model(model_id, precision, attn=attn, device_map=device_map)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        if precision != "4bit":
            model = model.merge_and_unload()  # avoids per-layer adapter overhead
    model.eval()
    return model
