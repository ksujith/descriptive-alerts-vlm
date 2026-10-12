"""Scoring for generated descriptive alerts.

Charades is multi-label (a test clip carries ~4.3 other annotated actions on
average), so a single accuracy number is misleading. We report:

``strict``    generated alert maps to the clip's own annotated class
``lenient``   maps to that class *or* any action genuinely co-occurring in the
              window - credit for being right about the scene, not lucky
``verb`` /    partial credit via Charades' 157 -> (33 verbs x 38 objects)
``object``    decomposition; also drives the failure taxonomy
``macro_f1``  over the 157 classes, because the set is ~5x imbalanced
``format``    fraction obeying the one-sentence contract (a stock VLM fails
              this constantly, which is why we also report format-free numbers)

Caption metrics (BLEU-4 / ROUGE-L / CIDEr) come from pycocoevalcap's pure-Python
scorers with our own tokenizer, deliberately avoiding PTBTokenizer/METEOR/SPICE
so the repo needs no Java toolchain.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_PUNCT = re.compile(r"[^a-z0-9/ ]+")
_WS = re.compile(r"\s+")
_LEAD = re.compile(r"^\s*(?:the\s+|a\s+|an\s+)?person(?:\s+is|\s+appears\s+to\s+be)?\s+", re.I)
_SOMEONE_LEAD = re.compile(r"^\s*someone(?:\s+is)?\s+", re.I)

# Deliberately minimal. Prepositions and vague placeholders are *load-bearing*
# in the Charades label space: dropping them as ordinary stopwords collapses
# "Sitting on a table"/"Sitting at a table" and "Taking a box from somewhere"/
# "Taking something from a box" into identical token sets, which silently
# corrupts strict accuracy. With the list below all 157 classes stay distinct
# (asserted by tests/test_core.py::test_label_space_matches_templated_alerts_exactly).
STOP = {
    "a", "an", "the", "is", "are", "be", "being", "it", "its", "this", "that",
    "there", "and", "as", "by", "while", "then", "someone", "person", "they",
    "their", "appears", "seems", "video", "clip",
}


def stem(word: str) -> str:
    """Crude consistent stemmer applied to both sides of every comparison.

    Correctness matters less than determinism here: as long as hypothesis and
    reference go through the identical transform, "holding"/"holds"/"held" all
    collapse to the same key.
    """
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "lszfo":
                base = base[:-1]
            return base
    return word


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = _SOMEONE_LEAD.sub("", text)
    text = _LEAD.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def content_tokens(text: str) -> set[str]:
    return {stem(t) for t in normalize(text).replace("/", " ").split() if t and t not in STOP}


def caption_tokens(text: str) -> list[str]:
    """Tokenizer for n-gram caption metrics.

    Deliberately *not* :func:`normalize`: label matching strips the "A person
    is" lead and stopwords, which leaves 3-token strings where no 4-gram exists
    and BLEU-4 collapses to zero even for an exact match. Caption metrics score
    the sentence as written.
    """
    t = _PUNCT.sub(" ", text.strip().lower())
    return [w for w in _WS.sub(" ", t).strip().replace("/", " ").split() if w]


def first_sentence(text: str) -> str:
    """Take the first sentence; stock VLMs emit paragraphs and bullet lists."""
    t = text.strip()
    t = re.split(r"\n\s*[-*•]|\n\n", t)[0].strip()
    m = re.search(r"[.!?]", t)
    return (t[: m.end()] if m else t).strip()


def is_well_formed(text: str) -> bool:
    """Obeys the one-sentence 'A person is ...' contract."""
    t = text.strip()
    if not re.match(r"^(a|the)\s+person\s+is\s+\S", t, re.I):
        return False
    body = re.sub(r"\s+", " ", t)
    # Exactly one terminal period and no list/newline structure.
    return "\n" not in t and len(re.findall(r"[.!?]", body)) <= 1 + body.count("...") and len(body) <= 140


@dataclass(frozen=True)
class LabelSpace:
    """The 157 classes plus their verb/object decomposition."""

    names: dict[str, str]                  # c000 -> "Holding some clothes"
    verbs: dict[str, str]                  # c000 -> "hold"
    objects: dict[str, str]                # c000 -> "clothes"
    _tokens: dict[str, frozenset[str]]

    @staticmethod
    def load(charades_dir: str | Path) -> "LabelSpace":
        d = Path(charades_dir)
        names: dict[str, str] = {}
        for line in (d / "Charades_v1_classes.txt").read_text().splitlines():
            if line.strip():
                cid, nm = line.strip().split(None, 1)
                names[cid] = nm.strip()

        def _vocab(fn: str) -> dict[str, str]:
            out = {}
            for line in (d / fn).read_text().splitlines():
                if line.strip():
                    k, v = line.strip().split(None, 1)
                    out[k] = v.strip()
            return out

        obj_v, verb_v = _vocab("Charades_v1_objectclasses.txt"), _vocab("Charades_v1_verbclasses.txt")
        verbs, objects = {}, {}
        for line in (d / "Charades_v1_mapping.txt").read_text().splitlines():
            if not line.strip():
                continue
            cid, oid, vid = line.strip().split()
            objects[cid] = obj_v.get(oid, "none")
            verbs[cid] = verb_v.get(vid, "none")
        toks = {cid: frozenset(content_tokens(nm)) for cid, nm in names.items()}
        return LabelSpace(names, verbs, objects, toks)

    def match(self, text: str, min_f1: float = 0.34) -> tuple[str | None, float]:
        """Map free-form generated text onto a class id.

        A fine-tuned model emits the template verbatim (exact hit); a zero-shot
        model emits prose, so we fall back to best content-token F1 and return
        ``None`` below a floor rather than forcing a wrong label.
        """
        hyp = content_tokens(text)
        if not hyp:
            return None, 0.0
        best, best_f1 = None, 0.0
        for cid in sorted(self._tokens):
            ref = self._tokens[cid]
            if not ref:
                continue
            inter = len(hyp & ref)
            if not inter:
                continue
            f1 = 2 * inter / (len(hyp) + len(ref))
            if f1 > best_f1:
                best, best_f1 = cid, f1
        return (best, best_f1) if best_f1 >= min_f1 else (None, best_f1)


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu4(hyps: list[str], refs: list[list[str]]) -> float:
    """Corpus-level BLEU-4 with add-nothing smoothing and brevity penalty."""
    import math

    num = [0] * 4
    den = [0] * 4
    hyp_len = ref_len = 0
    for h, rs in zip(hyps, refs):
        ht = caption_tokens(h)
        rts = [caption_tokens(r) for r in rs]
        hyp_len += len(ht)
        ref_len += min((abs(len(rt) - len(ht)), len(rt)) for rt in rts)[1]
        for n in range(1, 5):
            hc = _ngrams(ht, n)
            if not hc:
                continue
            maxref: Counter = Counter()
            for rt in rts:
                rc = _ngrams(rt, n)
                for g, c in rc.items():
                    maxref[g] = max(maxref[g], c)
            num[n - 1] += sum(min(c, maxref[g]) for g, c in hc.items())
            den[n - 1] += sum(hc.values())
    orders = [i for i in range(4) if den[i] > 0]
    if not orders:
        return 0.0
    # Short references (our targets are ~6 tokens) can lack high-order n-grams
    # entirely; average over the orders that exist rather than zeroing out.
    precs = [(num[i] / den[i]) if num[i] else 1e-9 for i in orders]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(sum(math.log(p) for p in precs) / len(precs))


def rouge_l(hyps: list[str], refs: list[list[str]], beta: float = 1.2) -> float:
    """Mean sentence-level ROUGE-L F-measure (best over references)."""

    def lcs(a: list[str], b: list[str]) -> int:
        if not a or not b:
            return 0
        prev = [0] * (len(b) + 1)
        for x in a:
            cur = [0]
            for j, y in enumerate(b):
                cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
            prev = cur
        return prev[-1]

    total = 0.0
    for h, rs in zip(hyps, refs):
        ht = caption_tokens(h)
        best = 0.0
        for r in rs:
            rt = caption_tokens(r)
            l = lcs(ht, rt)
            if not l:
                continue
            p, rc = l / len(ht), l / len(rt)
            best = max(best, ((1 + beta**2) * p * rc) / (rc + beta**2 * p))
        total += best
    return total / max(len(hyps), 1)
