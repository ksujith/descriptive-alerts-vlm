"""Prompt construction for the descriptive-alert task.

An honest baseline needs more than one zero-shot prompt. A stock VLM asked to
"describe this video" answers with a verbose paragraph, which scores near zero
on CIDEr against a one-line target - that penalises *output format*, not
perception, and reporting it alone overstates the fine-tune's gain. We
therefore evaluate four arms:

``open``        plain captioning instruction (the naive baseline)
``formatted``   same, plus an explicit one-sentence format constraint
``fewshot``     ``formatted`` plus three worked text examples
``choice``      forced choice over the 157 Charades classes (fairest zero-shot
                classification comparison; the fine-tune is not given this)

The fine-tuned model is trained and evaluated on ``formatted`` so that the
baseline-vs-fine-tuned column compares like with like.
"""

from __future__ import annotations

SYSTEM = (
    "You are a home security camera assistant. You watch short clips and report "
    "the single main human activity in one brief sentence."
)

_FORMAT_RULE = (
    "Reply with exactly one short sentence beginning 'A person is', naming the "
    "single main activity. No preamble, no extra detail, no list."
)

FEWSHOT_EXAMPLES = (
    "A person is opening a door.",
    "A person is holding a cup.",
    "A person is sitting on a sofa.",
)

OPEN = "Describe the main activity happening in this video clip."

FORMATTED = f"Describe the main activity happening in this video clip. {_FORMAT_RULE}"

FEWSHOT = (
    "Describe the main activity happening in this video clip. "
    f"{_FORMAT_RULE}\n\nExamples of correctly formatted answers:\n"
    + "\n".join(f"- {e}" for e in FEWSHOT_EXAMPLES)
)


def choice_prompt(class_names: list[str]) -> str:
    """Forced choice over the full label set."""
    options = "\n".join(f"- {n}" for n in class_names)
    return (
        "Which single activity from the list below best describes this video clip?\n\n"
        f"{options}\n\nReply with exactly one option, copied verbatim from the list."
    )


ARMS = {"open": OPEN, "formatted": FORMATTED, "fewshot": FEWSHOT}


def build_messages(
    arm: str,
    n_frames: int,
    *,
    class_names: list[str] | None = None,
    system: str | None = SYSTEM,
) -> list[dict]:
    """Build a Qwen2-VL chat message list with ``n_frames`` image placeholders.

    Frames are passed as a list of images under a single ``video`` content item,
    which is how Qwen2-VL expects multi-frame input; the caller substitutes real
    PIL images for the ``None`` placeholders.
    """
    if arm == "choice":
        if not class_names:
            raise ValueError("arm 'choice' requires class_names")
        text = choice_prompt(class_names)
    elif arm in ARMS:
        text = ARMS[arm]
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS) + ['choice']}")

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": [{"type": "text", "text": system}]})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "video", "video": [None] * n_frames},
                {"type": "text", "text": text},
            ],
        }
    )
    return messages
