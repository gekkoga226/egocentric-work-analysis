import json
import pytest
from unittest.mock import patch, MagicMock
from src.schemas import SegmentList, Segment
from src.pipeline.label_vlm_single import label_vlm_single, _merge_adjacent

LABELS = ["部品取り出し", "ネジ締め", "検査"]


def _make_gemini_response(segments: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.text = json.dumps(segments)
    return mock


@patch("src.pipeline.label_vlm_single.genai")
def test_label_vlm_returns_segment_list(mock_genai, synthetic_video_path):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _make_gemini_response([
        {"start_sec": 0.0, "end_sec": 10.0, "label": "部品取り出し", "confidence": 0.9},
        {"start_sec": 10.0, "end_sec": 30.0, "label": "ネジ締め", "confidence": 0.8},
    ])

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_vlm_single(synthetic_video_path, LABELS)

    assert isinstance(result, SegmentList)
    assert result.source == "track_a"
    assert len(result.segments) >= 1


@patch("src.pipeline.label_vlm_single.genai")
def test_label_vlm_handles_empty_response(mock_genai, synthetic_video_path):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _make_gemini_response([])

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_vlm_single(synthetic_video_path, LABELS)

    assert result.source == "track_a"
    assert isinstance(result.segments, list)


def test_merge_adjacent_same_label():
    segs = [
        Segment(0.0, 10.0, "A", 0.9),
        Segment(10.0, 20.0, "A", 0.8),
        Segment(20.0, 30.0, "B", 0.7),
    ]
    merged = _merge_adjacent(segs)
    assert len(merged) == 2
    assert merged[0].label == "A"
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 20.0
    assert merged[1].label == "B"


def test_merge_adjacent_no_adjacent_same():
    segs = [Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)]
    merged = _merge_adjacent(segs)
    assert len(merged) == 2
