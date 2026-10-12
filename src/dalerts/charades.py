"""Parsing for the Charades annotation release.

Charades ships two CSVs (train/test) plus a class vocabulary. The CSVs are
mostly clean but contain real hazards that silently corrupt a training set if
you trust them: segment timestamps can be negative, can run past the end of the
video, and a handful are degenerate (end <= start). Everything here clamps and
validates rather than assuming.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

_CLASS_LINE = re.compile(r"^(c\d{3})\s+(.+)$")
_SOMEONE = re.compile(r"^someone\s+is\s+", re.IGNORECASE)


@dataclass(frozen=True)
class Segment:
    """One annotated action instance inside a video."""

    video_id: str
    class_id: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Video:
    """One Charades video with its metadata and action segments."""

    id: str
    subject: str
    scene: str
    quality: float | None
    relevance: float | None
    verified: bool
    script: str
    objects: tuple[str, ...]
    descriptions: str
    length: float
    segments: tuple[Segment, ...]


def load_classes(path: str | Path) -> dict[str, str]:
    """Map ``c000`` -> ``"Holding some clothes"``."""
    classes: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _CLASS_LINE.match(raw)
        if not m:
            raise ValueError(f"unparseable class line: {raw!r}")
        classes[m.group(1)] = m.group(2).strip()
    if len(classes) != 157:
        raise ValueError(f"expected 157 Charades classes, parsed {len(classes)}")
    return classes


def to_alert(class_name: str) -> str:
    """Turn a class name into a Wyze-style descriptive-alert sentence.

    144 of the 157 names are gerund phrases ("Holding some clothes"); the other
    13 are already sentences ("Someone is standing up from somewhere"). Both
    normalise to "A person is <verb>ing ...".
    """
    name = class_name.strip()
    if _SOMEONE.match(name):
        body = _SOMEONE.sub("", name)
    elif name:
        # Lowercase the whole leading token, not just its first character:
        # three classes are slash-compounds ("Watching/Looking outside of a
        # window") where only the first alternative would otherwise be lowered.
        head, _, tail = name.partition(" ")
        body = head.lower() + (" " + tail if tail else "")
    else:
        body = name
    body = body.rstrip(". ")
    return f"A person is {body}."


def _opt_float(value: str) -> float | None:
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_segments(video_id: str, field: str, length: float) -> tuple[Segment, ...]:
    """Parse the ``actions`` field, clamping to [0, length] and dropping junk."""
    out: list[Segment] = []
    for chunk in field.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if len(parts) != 3:
            continue
        class_id, raw_start, raw_end = parts
        try:
            start, end = float(raw_start), float(raw_end)
        except ValueError:
            continue
        # Clamp into the real video extent before validating.
        start = max(0.0, min(start, length))
        end = max(0.0, min(end, length))
        if end - start <= 0.0:
            continue
        out.append(Segment(video_id, class_id, start, end))
    return tuple(out)


def load_videos(csv_path: str | Path) -> list[Video]:
    """Load one Charades CSV into validated :class:`Video` records."""
    videos: list[Video] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            length = _opt_float(row["length"]) or 0.0
            if length <= 0.0:
                continue
            videos.append(
                Video(
                    id=row["id"],
                    subject=row["subject"],
                    scene=row["scene"],
                    quality=_opt_float(row["quality"]),
                    relevance=_opt_float(row["relevance"]),
                    verified=row["verified"].strip().lower() == "yes",
                    script=row["script"].strip(),
                    objects=tuple(o for o in row["objects"].split(";") if o.strip()),
                    descriptions=row["descriptions"].strip(),
                    length=length,
                    segments=_parse_segments(row["id"], row["actions"], length),
                )
            )
    return videos
