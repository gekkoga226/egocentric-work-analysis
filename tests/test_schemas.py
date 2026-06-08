# tests/test_schemas.py
import json
import pytest
from src.schemas import Segment, SegmentList


def test_segment_fields():
    s = Segment(start_sec=0.0, end_sec=10.0, label="ネジ締め", confidence=0.9)
    assert s.start_sec == 0.0
    assert s.end_sec == 10.0
    assert s.label == "ネジ締め"
    assert s.confidence == 0.9


def test_segment_list_roundtrip():
    sl = SegmentList(
        video_id="test_video",
        fps_sampled=1.0,
        label_vocabulary=["作業A", "作業B"],
        segments=[
            Segment(0.0, 10.0, "作業A", 0.9),
            Segment(10.0, 20.0, "作業B", 0.8),
        ],
        source="track_b",
    )
    serialized = sl.to_json()
    restored = SegmentList.from_json(serialized)
    assert restored.video_id == "test_video"
    assert len(restored.segments) == 2
    assert restored.segments[0].label == "作業A"
    assert restored.segments[1].start_sec == 10.0


def test_to_frame_labels_basic():
    sl = SegmentList(
        video_id="v",
        fps_sampled=1.0,
        label_vocabulary=["A", "B"],
        segments=[Segment(0.0, 5.0, "A", 1.0), Segment(5.0, 10.0, "B", 1.0)],
        source="track_b",
    )
    labels = sl.to_frame_labels(total_duration=10.0, fps=1.0)
    assert labels[:5] == ["A", "A", "A", "A", "A"]
    assert labels[5:] == ["B", "B", "B", "B", "B"]


def test_to_frame_labels_gap_is_background():
    sl = SegmentList(
        video_id="v",
        fps_sampled=1.0,
        label_vocabulary=["A"],
        segments=[Segment(2.0, 5.0, "A", 1.0)],
        source="track_b",
    )
    labels = sl.to_frame_labels(total_duration=7.0, fps=1.0)
    assert labels[0] == "background"
    assert labels[2] == "A"
    assert labels[5] == "background"
