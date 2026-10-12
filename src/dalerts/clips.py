"""Turn Charades action segments into short, trainable clip records.

One clip == one annotated action segment, windowed to a short span. Three
things here matter for result validity:

* **Window shaping.** Segments run 0.5s-151s (median 9s). Long ones are
  center-cropped and very short ones symmetrically padded, so every clip lands
  in a bounded duration band and frame sampling stays comparable across the set.
* **Co-occurring labels.** Charades is multi-label: a clip showing "holding a
  cup" very often also shows "drinking from a cup". Scoring only the segment's
  own class punishes answers that are actually correct, so every clip carries
  the other classes overlapping its window and the evaluator reports both a
  strict and a lenient number.
* **Actor-disjoint splits.** The official train/test split shares no subject
  and no video. The val split is carved out of train by *subject* to preserve
  that property, so val never leaks an actor into training.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .charades import Segment, Video, to_alert


@dataclass(frozen=True)
class Clip:
    clip_id: str
    video_id: str
    split: str
    class_id: str
    label: str
    alert: str
    start: float
    end: float
    seg_start: float
    seg_end: float
    duration: float
    scene: str
    subject: str
    video_length: float
    cooccurring: tuple[str, ...]
    script: str

    def as_row(self) -> dict:
        row = asdict(self)
        row["cooccurring"] = "|".join(self.cooccurring)
        return row


def _window(seg: Segment, length: float, min_dur: float, max_dur: float) -> tuple[float, float] | None:
    """Shape a segment into a bounded window inside [0, length]."""
    start, end = seg.start, seg.end
    dur = end - start
    if dur <= 0:
        return None

    if dur > max_dur:  # center-crop long segments
        mid = (start + end) / 2.0
        start, end = mid - max_dur / 2.0, mid + max_dur / 2.0
    elif dur < min_dur:  # symmetrically pad short ones
        mid = (start + end) / 2.0
        start, end = mid - min_dur / 2.0, mid + min_dur / 2.0

    # Slide (don't just clip) back inside the video so duration is preserved.
    want = end - start
    if start < 0.0:
        start, end = 0.0, min(length, want)
    if end > length:
        end, start = length, max(0.0, length - want)

    if end - start < min_dur:
        return None  # video itself is too short
    return round(start, 3), round(end, 3)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_clips(
    videos: list[Video],
    classes: dict[str, str],
    split: str,
    *,
    min_dur: float = 2.0,
    max_dur: float = 10.0,
    cooc_frac: float = 0.5,
    require_verified: bool = True,
    min_quality: float | None = 3.0,
) -> list[Clip]:
    """Expand videos into clips. Quality gates are applied at video level."""
    clips: list[Clip] = []
    for v in videos:
        if require_verified and not v.verified:
            continue
        if min_quality is not None and v.quality is not None and v.quality < min_quality:
            continue
        for i, seg in enumerate(v.segments):
            if seg.class_id not in classes:
                continue
            win = _window(seg, v.length, min_dur, max_dur)
            if win is None:
                continue
            start, end = win
            span = end - start
            cooc = tuple(
                sorted(
                    {
                        other.class_id
                        for j, other in enumerate(v.segments)
                        if j != i
                        and other.class_id in classes
                        and _overlap(start, end, other.start, other.end) / span >= cooc_frac
                    }
                )
            )
            name = classes[seg.class_id]
            clips.append(
                Clip(
                    clip_id=f"{v.id}_{seg.class_id}_{int(start * 1000):07d}",
                    video_id=v.id,
                    split=split,
                    class_id=seg.class_id,
                    label=name,
                    alert=to_alert(name),
                    start=start,
                    end=end,
                    seg_start=round(seg.start, 3),
                    seg_end=round(seg.end, 3),
                    duration=round(span, 3),
                    scene=v.scene,
                    subject=v.subject,
                    video_length=v.length,
                    cooccurring=cooc,
                    script=v.script,
                )
            )

    # Identical (video, class, window) can appear twice in the raw annotations.
    seen: set[str] = set()
    deduped = []
    for c in clips:
        if c.clip_id in seen:
            continue
        seen.add(c.clip_id)
        deduped.append(c)
    return deduped


def split_by_subject(clips: list[Clip], val_frac: float, seed: int) -> list[Clip]:
    """Reassign a subject-disjoint slice of ``clips`` to the ``val`` split."""
    subjects = sorted({c.subject for c in clips})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n_val = max(1, round(len(subjects) * val_frac))
    val = set(subjects[:n_val])
    out = []
    for c in clips:
        if c.subject in val:
            out.append(Clip(**{**asdict(c), "cooccurring": c.cooccurring, "split": "val"}))
        else:
            out.append(c)
    return out


def cap_per_class(clips: list[Clip], cap: int, seed: int) -> list[Clip]:
    """Downsample over-represented classes. Charades is 58x imbalanced."""
    by_class: dict[str, list[Clip]] = defaultdict(list)
    for c in clips:
        by_class[c.class_id].append(c)
    rng = random.Random(seed)
    kept: list[Clip] = []
    for class_id in sorted(by_class):
        group = sorted(by_class[class_id], key=lambda c: c.clip_id)
        rng.shuffle(group)
        kept.extend(group[:cap])
    kept.sort(key=lambda c: c.clip_id)
    return kept


def summarize(clips: list[Clip]) -> dict:
    counts = Counter(c.class_id for c in clips)
    durs = sorted(c.duration for c in clips)
    cooc = [len(c.cooccurring) for c in clips]
    return {
        "clips": len(clips),
        "videos": len({c.video_id for c in clips}),
        "subjects": len({c.subject for c in clips}),
        "classes_present": len(counts),
        "per_class_min": min(counts.values()) if counts else 0,
        "per_class_max": max(counts.values()) if counts else 0,
        "dur_median": round(durs[len(durs) // 2], 2) if durs else 0,
        "mean_cooccurring": round(sum(cooc) / len(cooc), 2) if cooc else 0,
    }
