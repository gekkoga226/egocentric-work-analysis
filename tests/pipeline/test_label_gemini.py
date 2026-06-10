# tests/pipeline/test_label_gemini.py
import pytest
from src.schemas import Segment
from src.pipeline.label_gemini import (
    _build_coarse_segments,
    _group_into_windows,
    _merge_adjacent_enrich,
    _stitch,
)


def test_build_coarse_segments_basic():
    segs = _build_coarse_segments([10.0, 20.0], 30.0)
    assert segs == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_build_coarse_segments_empty_boundaries():
    segs = _build_coarse_segments([], 15.0)
    assert segs == [(0.0, 15.0)]


def test_build_coarse_segments_deduplicates():
    segs = _build_coarse_segments([10.0, 10.0, 20.0], 30.0)
    assert len(segs) == 3


def test_group_into_windows_small():
    coarse = [(float(i), float(i+1)) for i in range(5)]
    windows = _group_into_windows(coarse, target_size=10)
    assert windows == [(0, 5)]


def test_group_into_windows_overlap():
    coarse = [(float(i), float(i+1)) for i in range(25)]
    windows = _group_into_windows(coarse, target_size=10)
    # First window: [0..10), second: [9..19), etc. — 1-segment overlap
    assert windows[0] == (0, 10)
    assert windows[1][0] == 9  # starts at index 9 (overlap)
    assert windows[1][1] == 19


def test_group_into_windows_covers_all():
    coarse = [(float(i), float(i+1)) for i in range(25)]
    windows = _group_into_windows(coarse, target_size=10)
    # Last window must reach index 25
    assert windows[-1][1] == 25


def test_merge_adjacent_enrich_same_label_and_category():
    segs = [
        Segment(0.0, 10.0, "A", 0.9, "seimi", "desc1", None),
        Segment(10.0, 20.0, "A", 0.8, "seimi", "desc2", None),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 1
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 20.0
    # longer segment (desc1, 10s) wins over shorter (desc2, 10s) → first wins on tie
    assert merged[0].description == "desc1"


def test_merge_adjacent_enrich_different_category_no_merge():
    segs = [
        Segment(0.0, 10.0, "A", 0.9, "seimi", "d1", None),
        Segment(10.0, 20.0, "A", 0.8, "muda", "d2", "改善"),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 2


def test_merge_adjacent_enrich_core_priority_longer_wins():
    # longer segment's description should be kept
    segs = [
        Segment(0.0, 5.0,  "A", 0.9, "seimi", "short-desc", None),
        Segment(5.0, 20.0, "A", 0.9, "seimi", "long-desc",  None),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 1
    assert merged[0].description == "long-desc"  # 15s > 5s


def test_stitch_invariant_continuous_no_gap():
    coarse = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    windows = [(0, 3)]
    window_results = [[
        Segment(0.0, 10.0, "A", 0.9, "seimi"),
        Segment(10.0, 20.0, "B", 0.9, "fuzui"),
        Segment(20.0, 30.0, "A", 0.9, "seimi"),
    ]]
    vocab = ["A", "B"]
    result = _stitch(window_results, windows, coarse, 30.0, vocab)
    assert result[0].start_sec == 0.0
    assert result[-1].end_sec == 30.0
    for i in range(len(result) - 1):
        assert abs(result[i].end_sec - result[i+1].start_sec) < 0.01


def test_stitch_no_duplicate_intervals():
    coarse = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    windows = [(0, 2), (1, 3)]  # overlap at index 1
    window_results = [
        [Segment(0.0, 10.0, "A", 0.9), Segment(10.0, 20.0, "B", 0.9)],
        [Segment(10.0, 20.0, "B", 0.8), Segment(20.0, 30.0, "C", 0.9)],
    ]
    vocab = ["A", "B", "C"]
    result = _stitch(window_results, windows, coarse, 30.0, vocab)
    # No time range should appear twice
    for i in range(len(result) - 1):
        assert result[i].end_sec <= result[i+1].start_sec + 0.01
