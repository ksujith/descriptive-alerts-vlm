"""Regression tests for the data, metric and taxonomy layers.

These cover the bugs actually hit while building the pipeline:
slash-compound class names keeping a stray capital, BLEU-4 collapsing to zero
on exact matches because the label-matching tokenizer stripped the sentence
lead, and repetition loops scoring as clean answers.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from dalerts.charades import load_classes, load_videos, to_alert
from dalerts.clips import build_clips, cap_per_class, split_by_subject
from dalerts.extract import extract, sample_timestamps, stride_indices
from dalerts.metrics import (LabelSpace, caption_tokens, corpus_bleu4, first_sentence,
                             is_well_formed, rouge_l, stem)
from dalerts.taxonomy import classify, is_degenerate

CHARADES = Path(__file__).resolve().parents[1] / "data" / "raw" / "Charades"
needs_data = pytest.mark.skipif(not CHARADES.exists(), reason="Charades annotations not downloaded")


# ---------------------------------------------------------------- annotations
@needs_data
def test_class_vocab_is_complete_and_alerts_unique():
    classes = load_classes(CHARADES / "Charades_v1_classes.txt")
    assert len(classes) == 157
    alerts = {c: to_alert(n) for c, n in classes.items()}
    assert len(set(alerts.values())) == 157
    for a in alerts.values():
        assert a.startswith("A person is ") and a.endswith(".")
        # No stray capitals: three classes are slash-compounds such as
        # "Watching/Looking outside of a window".
        assert not re.search(r"(?<=.)[A-Z]", a), a


def test_to_alert_handles_both_phrasings_and_slash_compounds():
    assert to_alert("Holding some clothes") == "A person is holding some clothes."
    assert to_alert("Someone is standing up from somewhere") == "A person is standing up from somewhere."
    assert to_alert("Watching/Looking outside of a window") == "A person is watching/looking outside of a window."


@needs_data
def test_segment_timestamps_are_clamped_into_the_video():
    """Raw CSVs contain negative starts and ends past the video duration."""
    for name in ("Charades_v1_train.csv", "Charades_v1_test.csv"):
        for v in load_videos(CHARADES / name):
            for s in v.segments:
                assert 0.0 <= s.start < s.end <= v.length + 1e-6, (v.id, s)


# ---------------------------------------------------------------- clip splits
@needs_data
def test_splits_are_subject_and_video_disjoint():
    classes = load_classes(CHARADES / "Charades_v1_classes.txt")
    tv = split_by_subject(build_clips(load_videos(CHARADES / "Charades_v1_train.csv"), classes, "train"), 0.1, 1234)
    te = build_clips(load_videos(CHARADES / "Charades_v1_test.csv"), classes, "test")
    tr = [c for c in tv if c.split == "train"]
    va = [c for c in tv if c.split == "val"]
    assert tr and va and te
    for a, b in ((tr, va), (tr, te), (va, te)):
        assert not ({c.subject for c in a} & {c.subject for c in b})
        assert not ({c.video_id for c in a} & {c.video_id for c in b})


@needs_data
def test_clip_windows_respect_duration_bounds():
    classes = load_classes(CHARADES / "Charades_v1_classes.txt")
    clips = build_clips(load_videos(CHARADES / "Charades_v1_test.csv"), classes, "test", min_dur=2.0, max_dur=10.0)
    assert clips
    for c in clips:
        assert 2.0 - 1e-6 <= c.duration <= 10.0 + 1e-6
        assert 0.0 <= c.start < c.end <= c.video_length + 1e-6


@needs_data
def test_cap_per_class_is_deterministic():
    classes = load_classes(CHARADES / "Charades_v1_classes.txt")
    clips = build_clips(load_videos(CHARADES / "Charades_v1_test.csv"), classes, "test")
    a = [c.clip_id for c in cap_per_class(clips, 5, 7)]
    b = [c.clip_id for c in cap_per_class(clips, 5, 7)]
    assert a == b
    assert all(v <= 5 for v in __import__("collections").Counter(
        c.class_id for c in cap_per_class(clips, 5, 7)).values())


# ------------------------------------------------------------------ extractor
@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    out = tmp_path_factory.mktemp("vid") / "tv.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc2=size=320x240:rate=24:duration=12",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "48", str(out)],
        check=True,
    )
    return out


def test_sample_timestamps_are_interval_centres():
    assert sample_timestamps(0.0, 4.0, 4) == [0.5, 1.5, 2.5, 3.5]
    assert sample_timestamps(2.0, 10.0, 1) == [6.0]
    with pytest.raises(ValueError):
        sample_timestamps(5.0, 5.0, 4)


def test_stride_indices_are_uniform_subsets():
    assert stride_indices(16, 16) == list(range(16))
    assert stride_indices(16, 1) == [8]
    for want in (2, 4, 8):
        idx = stride_indices(16, want)
        assert len(set(idx)) == want
        assert len({idx[i + 1] - idx[i] for i in range(want - 1)}) == 1  # even spacing


def test_extract_hits_targets_within_one_frame(tiny_video):
    frames = extract(tiny_video, 2.0, 10.0, 8, short_side=112)
    targets = sample_timestamps(2.0, 10.0, 8)
    assert len(frames.images) == 8
    assert max(abs(a - b) for a, b in zip(frames.actual_times, targets)) <= 1 / 24 + 1e-6
    assert frames.actual_times == sorted(frames.actual_times)


def test_extract_pads_when_window_exceeds_stream(tiny_video):
    """Charades windows can run past the real stream duration."""
    frames = extract(tiny_video, 10.0, 13.0, 8, short_side=112)
    assert len(frames.images) == 8


def test_resize_preserves_aspect_and_even_dims(tiny_video):
    img = extract(tiny_video, 1.0, 3.0, 1, short_side=112).images[0]
    w, h = img.size
    assert h == 112 and w % 2 == 0 and abs(w / h - 320 / 240) < 0.02


# -------------------------------------------------------------------- metrics
def test_bleu4_is_one_for_identical_short_captions():
    """Regression: label-matching tokenizer left 3 tokens, so BLEU-4 was 0."""
    h = ["A person is holding some clothes."]
    r = [["A person is holding some clothes."]]
    assert corpus_bleu4(h, r) == pytest.approx(1.0, abs=1e-6)
    assert rouge_l(h, r) == pytest.approx(1.0, abs=1e-6)


def test_bleu4_penalises_verbose_output():
    r = [["A person is holding some clothes."]]
    verbose = ["The video shows a person who is holding some clothes in a bedroom."]
    assert corpus_bleu4(verbose, r) < 0.5


def test_caption_tokens_keeps_the_sentence_lead():
    assert caption_tokens("A person is opening a door.") == ["a", "person", "is", "opening", "a", "door"]


def test_stem_is_consistent_across_inflections():
    assert stem("holding") == stem("holds") == "hold"
    assert stem("sitting") == "sit"


def test_format_gate():
    assert is_well_formed("A person is opening a door.")
    assert not is_well_formed("In this video, a person opens a door.")
    assert not is_well_formed("- A person is opening a door\n- A person is walking")


def test_first_sentence_truncates_paragraphs():
    assert first_sentence("The video shows a person. They open the fridge.") == "The video shows a person."


@needs_data
def test_label_space_matches_templated_alerts_exactly():
    ls = LabelSpace.load(CHARADES)
    assert len(ls.names) == 157
    assert len(set(ls.verbs.values())) == 33 and len(set(ls.objects.values())) == 38
    for cid, name in ls.names.items():
        got, f1 = ls.match(to_alert(name))
        assert got == cid and f1 == pytest.approx(1.0), (cid, name, got, f1)


@needs_data
def test_label_space_declines_unmappable_text():
    ls = LabelSpace.load(CHARADES)
    assert ls.match("I cannot determine the activity.")[0] is None


# ------------------------------------------------------------------- taxonomy
def test_repetition_loops_are_degenerate():
    """Regression: these previously scored as clean answers."""
    assert is_degenerate("A person is holding holding holding holding clothes.")
    assert is_degenerate("A person is opening a door opening a door.")
    assert is_degenerate("")
    assert is_degenerate("I cannot determine what is happening.")


def test_legitimate_captions_are_not_degenerate():
    assert not is_degenerate("A person is going from standing to sitting.")
    assert not is_degenerate("A person is holding a cup and holding a plate.")
    assert not is_degenerate("A person is watching/looking outside of a window.")


@needs_data
@pytest.mark.parametrize(
    "text,bucket",
    [
        ("A person is holding some clothes.", "correct"),
        # content-first precedence: right action, wrong shape, still `correct`
        ("In this video a person holds some clothes.", "correct"),
        ("A person is putting clothes somewhere.", "cooccurring_action"),
        ("A person is holding a blanket.", "right_verb_wrong_object"),
        ("A person is washing some clothes.", "right_object_wrong_verb"),
        ("A person is washing a window.", "related_both_wrong"),
        ("In this clip, a figure moves gracefully near the light.", "format_violation"),
        ("A person is moving gracefully near the light.", "unmatched_prose"),
        ("", "degenerate"),
    ],
)
def test_taxonomy_buckets(text, bucket):
    ls = LabelSpace.load(CHARADES)
    assert classify(text, "c000", ("c001", "c006"), ls).bucket == bucket
