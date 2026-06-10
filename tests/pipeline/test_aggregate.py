# tests/pipeline/test_aggregate.py
import pytest
from src.schemas import Segment, SegmentList
from src.pipeline.aggregate import aggregate


def _make_sl(segs):
    return SegmentList(
        video_id="x", fps_sampled=1.0,
        label_vocabulary=["A", "B"],
        segments=segs, source="track_std",
    )


def test_aggregate_total_sec():
    sl = _make_sl([
        Segment(0.0, 10.0, "A", 1.0, "seimi"),
        Segment(10.0, 25.0, "B", 1.0, "muda"),
    ])
    result = aggregate(sl)
    assert abs(result["total_sec"] - 25.0) < 0.01


def test_aggregate_by_category_sums():
    sl = _make_sl([
        Segment(0.0, 10.0, "A", 1.0, "seimi"),
        Segment(10.0, 20.0, "B", 1.0, "seimi"),
        Segment(20.0, 30.0, "C", 1.0, "muda"),
    ])
    result = aggregate(sl)
    assert abs(result["by_category"]["seimi"]["total_sec"] - 20.0) < 0.01
    assert result["by_category"]["seimi"]["count"] == 2
    assert abs(result["by_category"]["seimi"]["ratio"] - 20/30) < 0.01


def test_aggregate_by_label_mean():
    sl = _make_sl([
        Segment(0.0, 10.0, "ネジ締め", 1.0, "seimi"),
        Segment(10.0, 14.0, "ネジ締め", 1.0, "seimi"),
    ])
    result = aggregate(sl)
    # two segments: 10s and 4s → mean = 7.0
    assert abs(result["by_label"]["ネジ締め"]["mean_sec"] - 7.0) < 0.01
    assert result["by_label"]["ネジ締め"]["count"] == 2


def test_aggregate_none_category_bucketed_as_unknown():
    sl = _make_sl([Segment(0.0, 5.0, "A", 1.0, None)])
    result = aggregate(sl)
    assert "unknown" in result["by_category"]
    assert result["by_category"]["unknown"]["total_sec"] == 5.0


def test_aggregate_empty_segments():
    sl = _make_sl([])
    result = aggregate(sl)
    assert result["total_sec"] == 0.0
    assert result["by_category"] == {}
    assert result["by_label"] == {}
