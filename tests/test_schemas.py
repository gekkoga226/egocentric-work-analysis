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


def test_segment_has_enrich_fields():
    seg = Segment(0.0, 10.0, "ネジ締め", 0.9,
                  category="seimi",
                  description="フランジを締結",
                  improvement=None)
    assert seg.category == "seimi"
    assert seg.description == "フランジを締結"
    assert seg.improvement is None


def test_segment_enrich_defaults_to_none():
    seg = Segment(0.0, 10.0, "A", 1.0)
    assert seg.category is None
    assert seg.description is None
    assert seg.improvement is None


def test_segmentlist_roundtrip_with_enrich():
    sl = SegmentList(
        video_id="test", fps_sampled=1.0,
        label_vocabulary=["ネジ締め"],
        segments=[Segment(0.0, 10.0, "ネジ締め", 0.9, "seimi", "締結作業", None)],
        source="track_std",
    )
    sl2 = SegmentList.from_json(sl.to_json())
    assert sl2.segments[0].category == "seimi"
    assert sl2.segments[0].description == "締結作業"
    assert sl2.segments[0].improvement is None


def test_from_json_backward_compat_old_json():
    old_json = json.dumps({
        "video_id": "old", "fps_sampled": 1.0,
        "label_vocabulary": ["A"],
        "segments": [{"start_sec": 0.0, "end_sec": 5.0, "label": "A", "confidence": 0.8}],
        "source": "track_b",
    })
    sl = SegmentList.from_json(old_json)
    assert sl.segments[0].category is None
    assert sl.segments[0].description is None
    assert sl.segments[0].improvement is None


def test_hint_dataclass():
    from src.schemas import Hint
    h = Hint(label="ドライバー", frame_sec=12.3)
    assert h.bbox is None
    assert h.note is None
    h2 = Hint(label="手", frame_sec=5.0, bbox=(0.1, 0.2, 0.3, 0.4), note="右手")
    assert h2.bbox == (0.1, 0.2, 0.3, 0.4)
