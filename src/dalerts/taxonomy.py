"""Failure taxonomy for generated descriptive alerts.

Buckets are assigned by a fixed-precedence cascade, so every prediction lands
in exactly one. The categories are derived from Charades' own label structure
(157 classes = 33 verbs x 38 objects) plus the clip's co-occurring actions,
rather than invented after eyeballing outputs.

Why ``cooccurring_action`` is separated from the error buckets: a Charades test
clip carries ~4.3 other annotated actions, so a model that says "drinking from
a cup" on a clip labelled "holding a cup" is usually *right about the video* and
wrong only about which label the annotator happened to attach. Folding that
into the error rate would overstate real perception failure, so it is reported
as a distinct, non-punitive bucket.

Precedence is **content-first**: a prediction that names the right action is
``correct`` even if it broke the sentence format, because shape errors and
perception errors are different defects and must not be summed. Format
compliance is reported separately as an orthogonal rate (see
:func:`dalerts.metrics.is_well_formed`), so nothing is lost by this ordering.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .metrics import STOP, LabelSpace, caption_tokens, is_well_formed, stem

BUCKETS = (
    "correct",
    "cooccurring_action",
    "right_verb_wrong_object",
    "right_object_wrong_verb",
    "related_both_wrong",
    "unmatched_prose",
    "format_violation",
    "degenerate",
)

BUCKET_DOC = {
    "correct": "Mapped to the clip's own annotated class.",
    "cooccurring_action": "Mapped to a different action that genuinely co-occurs in the clip window. Not counted as a perception failure.",
    "right_verb_wrong_object": "Correct action verb, wrong object (e.g. 'putting a book' for 'putting a cup').",
    "right_object_wrong_verb": "Correct object, wrong action verb (e.g. 'drinking from a cup' for 'holding a cup').",
    "related_both_wrong": "Mapped to a real class, but both verb and object are wrong.",
    "unmatched_prose": "Fluent, plausible description that could not be mapped onto any of the 157 classes.",
    "format_violation": "Broke the one-sentence 'A person is ...' output contract.",
    "degenerate": "Empty, truncated, repetition loop, or an explicit refusal.",
}

_REFUSAL = re.compile(
    r"\b(i\s+(?:can(?:not|'t)|am\s+unable|don't\s+know)|unable\s+to|sorry|as\s+an\s+ai|no\s+(?:activity|person))\b",
    re.I,
)


def _is_repetitive(text: str) -> bool:
    """Detect decoder repetition loops on *content* words.

    Operates on stemmed content tokens of the full surface string. Using the
    label-matching ``normalize`` here was a bug: it strips the "A person is"
    lead, so "A person is holding holding holding holding clothes" fell under a
    length guard and scored as a clean answer.
    """
    toks = [stem(w) for w in caption_tokens(text) if w not in STOP]
    if len(toks) < 3:
        return False
    if Counter(toks).most_common(1)[0][1] >= 3:
        return True
    if len(toks) >= 4:
        bigrams = Counter(zip(toks, toks[1:]))
        if bigrams.most_common(1)[0][1] >= 2:
            return True
    return False


def is_degenerate(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 3:
        return True
    if _REFUSAL.search(t):
        return True
    return _is_repetitive(t)


@dataclass(frozen=True)
class Verdict:
    bucket: str
    pred_class: str | None
    match_f1: float
    verb_ok: bool
    object_ok: bool


def classify(
    generated: str,
    gold_class: str,
    cooccurring: tuple[str, ...] | list[str],
    labels: LabelSpace,
    *,
    enforce_format: bool = True,
) -> Verdict:
    """Assign one bucket to one prediction.

    ``enforce_format`` is False for the naive ``open`` baseline arm, which was
    never asked for a specific format - penalising it there would be unfair.
    """
    pred, f1 = labels.match(generated)
    verb_ok = pred is not None and labels.verbs.get(pred) == labels.verbs.get(gold_class)
    obj_ok = pred is not None and labels.objects.get(pred) == labels.objects.get(gold_class)

    def v(bucket: str) -> Verdict:
        return Verdict(bucket, pred, round(f1, 4), verb_ok, obj_ok)

    if is_degenerate(generated):
        return v("degenerate")
    if pred == gold_class:
        return v("correct")
    if pred is not None and pred in set(cooccurring):
        return v("cooccurring_action")
    if enforce_format and not is_well_formed(generated):
        return v("format_violation")
    if pred is None:
        return v("unmatched_prose")
    if verb_ok and not obj_ok:
        return v("right_verb_wrong_object")
    if obj_ok and not verb_ok:
        return v("right_object_wrong_verb")
    return v("related_both_wrong")


def tabulate(verdicts: list[Verdict]) -> dict:
    n = max(len(verdicts), 1)
    counts = Counter(v.bucket for v in verdicts)
    return {
        "n": len(verdicts),
        "counts": {b: counts.get(b, 0) for b in BUCKETS},
        "pct": {b: round(100 * counts.get(b, 0) / n, 2) for b in BUCKETS},
    }
