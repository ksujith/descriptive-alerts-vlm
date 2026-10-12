"""Frame sampling from video clips via PyAV.

Design note on the frame-count ablation: every clip is decoded **once** at the
maximum frame budget (16) and lower budgets are produced by striding that same
cache (16 -> 8 -> 4 -> 1). The ablation arms therefore share identical decoded
pixels and differ only in temporal density, which is what we want to measure.
Decoding each arm separately would also vary JPEG quantisation and seek
behaviour, confounding the comparison.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image

MAX_FRAMES = 16


def sample_timestamps(start: float, end: float, n: int) -> list[float]:
    """Centres of ``n`` equal sub-intervals of [start, end].

    Centre sampling (rather than endpoint sampling) avoids the first frame
    landing exactly on a shot boundary and keeps n=1 at the clip midpoint.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    span = end - start
    if span <= 0:
        raise ValueError(f"empty window: {start}..{end}")
    step = span / n
    return [start + (i + 0.5) * step for i in range(n)]


def stride_indices(n_have: int, n_want: int) -> list[int]:
    """Pick ``n_want`` indices spread across ``n_have`` cached frames."""
    if n_want > n_have:
        raise ValueError(f"cannot take {n_want} from {n_have}")
    if n_want == 1:
        return [n_have // 2]
    step = n_have / n_want
    return [min(n_have - 1, int((i + 0.5) * step)) for i in range(n_want)]


@dataclass
class ClipFrames:
    images: list[Image.Image]
    actual_times: list[float]
    source_fps: float
    source_size: tuple[int, int]


def _resize(img: Image.Image, short_side: int | None) -> Image.Image:
    if short_side is None:
        return img
    w, h = img.size
    if min(w, h) <= short_side:
        return img
    scale = short_side / min(w, h)
    # Keep both dims even; some video pipelines dislike odd dimensions.
    nw, nh = (max(2, int(round(w * scale / 2)) * 2), max(2, int(round(h * scale / 2)) * 2))
    return img.resize((nw, nh), Image.BICUBIC)


def extract(
    video_path: str | Path,
    start: float,
    end: float,
    n_frames: int = MAX_FRAMES,
    short_side: int | None = 336,
) -> ClipFrames:
    """Decode ``n_frames`` frames evenly covering [start, end].

    Performs one forward decode pass from a keyframe before ``start`` and picks
    the first decoded frame at or after each target timestamp.
    """
    targets = sample_timestamps(start, end, n_frames)
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise ValueError(f"no video stream in {video_path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        tb = stream.time_base or Fraction(1, 1000)
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        size = (stream.codec_context.width, stream.codec_context.height)

        # Seek to a keyframe slightly before the window so the first target is
        # reachable by forward decoding.
        seek_t = max(0.0, start - 1.0)
        try:
            container.seek(int(seek_t / tb), stream=stream)
        except (av.AVError, ValueError):
            container.seek(0)

        picked: list[Image.Image] = []
        times: list[float] = []
        ti = 0
        last = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * tb)
            last = frame
            while ti < len(targets) and t >= targets[ti] - 1e-9:
                picked.append(_resize(frame.to_image(), short_side))
                times.append(t)
                ti += 1
            if ti >= len(targets):
                break

        # Window can extend past the last decodable frame (Charades timestamps
        # occasionally exceed the real stream duration); pad with the last frame.
        while ti < len(targets) and last is not None:
            picked.append(_resize(last.to_image(), short_side))
            times.append(float(last.pts * tb))
            ti += 1

    if len(picked) != n_frames:
        raise RuntimeError(f"{video_path}: got {len(picked)}/{n_frames} frames")
    return ClipFrames(picked, times, fps, size)
