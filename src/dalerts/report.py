"""Aggregate generated alerts into the reported metric tables."""

from __future__ import annotations

from collections import Counter, defaultdict

from .metrics import LabelSpace, corpus_bleu4, first_sentence, is_well_formed, rouge_l
from .taxonomy import BUCKETS, classify, tabulate


def macro_f1(pairs: list[tuple[str | None, str]], classes: list[str] | None = None) -> float:
    """Macro-averaged F1 (unmatched counts as no prediction).

    Averages over the classes present in gold or prediction - sklearn's default
    convention. Averaging over all 157 regardless would dilute the score on any
    subset, while still penalising hallucinated labels (they enter via the
    prediction side with zero recall).
    """
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for pred, gold in pairs:
        if pred == gold:
            tp[gold] += 1
        else:
            fn[gold] += 1
            if pred is not None:
                fp[pred] += 1
    if classes is None:
        classes = sorted({g for _, g in pairs} | {p for p, _ in pairs if p is not None})
    total = 0.0
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        total += (2 * p * r / (p + r)) if (p + r) else 0.0
    return total / max(len(classes), 1)


def score(records: list[dict], labels: LabelSpace, *, enforce_format: bool = True) -> dict:
    """Compute every reported number for one arm."""
    n = len(records)
    pairs: list[tuple[str | None, str]] = []
    strict = lenient = verb_ok = obj_ok = well = matched = 0
    verdicts = []
    per_scene: dict[str, list[int]] = defaultdict(list)

    hyps, refs = [], []
    for r in records:
        gen = r["generated"]
        # Normalising a verbose answer to its first sentence is a fair
        # concession to the zero-shot arms; the raw string is kept for audit.
        norm = first_sentence(gen)
        r["normalized"] = norm

        pred, f1 = labels.match(norm)
        r["pred_class"] = pred
        r["match_f1"] = round(f1, 4)
        pairs.append((pred, r["gold_class"]))
        if pred is not None:
            matched += 1

        hit = pred == r["gold_class"]
        strict += hit
        lenient += hit or (pred is not None and pred in set(r["cooccurring"]))
        if pred is not None:
            verb_ok += labels.verbs.get(pred) == labels.verbs.get(r["gold_class"])
            obj_ok += labels.objects.get(pred) == labels.objects.get(r["gold_class"])
        well += is_well_formed(gen)

        v = classify(norm, r["gold_class"], r["cooccurring"], labels, enforce_format=enforce_format)
        r["bucket"] = v.bucket
        verdicts.append(v)

        hyps.append(norm)
        refs.append([r["target"]])

    d = max(n, 1)
    out = {
        "n": n,
        "strict_top1": round(100 * strict / d, 2),
        "lenient_top1": round(100 * lenient / d, 2),
        "verb_acc": round(100 * verb_ok / d, 2),
        "object_acc": round(100 * obj_ok / d, 2),
        "macro_f1": round(100 * macro_f1(pairs), 2),
        "format_valid": round(100 * well / d, 2),
        "mapped_to_a_class": round(100 * matched / d, 2),
        "bleu4": round(100 * corpus_bleu4(hyps, refs), 2),
        "rouge_l": round(100 * rouge_l(hyps, refs), 2),
        "taxonomy": tabulate(verdicts),
    }
    try:  # optional, pure-Python CIDEr; no Java needed
        from pycocoevalcap.cider.cider import Cider

        gts = {str(i): [refs[i][0].lower()] for i in range(n)}
        res = {str(i): [hyps[i].lower()] for i in range(n)}
        out["cider"] = round(float(Cider().compute_score(gts, res)[0]), 4)
    except Exception:
        out["cider"] = None
    return out


HEADLINE = [
    ("strict_top1", "Strict top-1"),
    ("lenient_top1", "Lenient top-1"),
    ("macro_f1", "Macro-F1"),
    ("verb_acc", "Verb acc"),
    ("object_acc", "Object acc"),
    ("format_valid", "Format valid"),
    ("bleu4", "BLEU-4"),
    ("rouge_l", "ROUGE-L"),
]


def markdown_table(results: dict[str, dict], row_label: str = "Arm") -> str:
    cols = [lbl for _, lbl in HEADLINE]
    lines = [f"| {row_label} | n | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 2)]
    for name, r in results.items():
        vals = [f"{r[k]:.2f}" for k, _ in HEADLINE]
        lines.append(f"| {name} | {r['n']} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def taxonomy_table(results: dict[str, dict]) -> str:
    lines = ["| Arm | " + " | ".join(b.replace("_", " ") for b in BUCKETS) + " |",
             "|" + "---|" * (len(BUCKETS) + 1)]
    for name, r in results.items():
        pct = r["taxonomy"]["pct"]
        lines.append(f"| {name} | " + " | ".join(f"{pct[b]:.1f}%" for b in BUCKETS) + " |")
    return "\n".join(lines)
