"""End-to-end pipeline test using synthetic video and mocked Gemini API."""
import tempfile
import pytest
import numpy as np
import json
from unittest.mock import patch, MagicMock
from src.schemas import SegmentList, Segment
from src.pipeline.label_zeroshot import label_zeroshot
from src.pipeline.report import save_segments, to_timeline_markdown
from src.evaluate.metrics import compute_all

LABELS = ["作業A", "作業B", "作業C"]


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_track_b_end_to_end(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    D = 64
    n = 30
    mock_ingest.return_value = [(float(i), None) for i in range(n)]
    emb = np.zeros((n, D))
    emb[:10, 0] = 1.0
    emb[10:20, 1] = 1.0
    emb[20:, 2] = 1.0
    mock_embed_f.return_value = ([float(i) for i in range(n)], emb)
    mock_embed_t.return_value = np.eye(3, D)

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )

    assert isinstance(result, SegmentList)
    assert result.source == "track_b"
    assert len(result.segments) == 3
    assert [s.label for s in result.segments] == LABELS

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(result, tmpdir)
        reloaded = SegmentList.from_json(open(path, encoding="utf-8").read())
        assert len(reloaded.segments) == 3

    gt = SegmentList("test", 1.0, LABELS, [
        Segment(0.0, 10.0, "作業A", 1.0),
        Segment(10.0, 20.0, "作業B", 1.0),
        Segment(20.0, 30.0, "作業C", 1.0),
    ], "ground_truth")

    metrics = compute_all(result, gt, fps=1.0)
    assert metrics["f1@50"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["edit"] == pytest.approx(100.0)

    md = to_timeline_markdown(result)
    assert "作業A" in md and "作業B" in md and "作業C" in md


def _gemini_resp(segs):
    m = MagicMock()
    m.text = json.dumps(segs)
    return m


@patch("src.pipeline.label_gemini.genai")
def test_track_std_pipeline_produces_valid_segment_list(mock_genai, synthetic_video_path, tmp_path):
    from src.pipeline.ingest import extract_frames
    from src.pipeline.embed import embed_frames
    from src.pipeline.presegment import detect_boundaries
    from src.pipeline.label_gemini import label_gemini
    from src.pipeline.aggregate import aggregate
    from src.schemas import SegmentList

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_resp([
        {"start_sec": 0.0, "end_sec": 10.0, "label": "部品取り出し",
         "category": "fuzui", "description": "棚から取る", "improvement": None, "confidence": 0.9},
        {"start_sec": 10.0, "end_sec": 20.0, "label": "ネジ締め",
         "category": "seimi", "description": "4本締結", "improvement": None, "confidence": 0.95},
        {"start_sec": 20.0, "end_sec": 30.0, "label": "手待ち",
         "category": "muda", "description": "次工程待ち", "improvement": "同期化で削減可", "confidence": 0.8},
    ])

    import os
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        frames = extract_frames(synthetic_video_path, fps=1.0)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=10.0)
        seg_list = label_gemini(synthetic_video_path, ["部品取り出し", "ネジ締め", "手待ち"], boundaries)

    # Invariant: continuous, non-overlapping, covers [0, total_duration]
    assert seg_list.segments[0].start_sec == pytest.approx(0.0, abs=0.1)
    for i in range(len(seg_list.segments) - 1):
        a, b = seg_list.segments[i], seg_list.segments[i + 1]
        assert a.end_sec <= b.start_sec + 0.01, f"Gap/overlap between seg {i} and {i+1}"

    # Aggregate works without error
    stats = aggregate(seg_list)
    assert stats["total_sec"] > 0
    assert "by_category" in stats

    # Enrich fields present
    cats = {s.category for s in seg_list.segments if s.category}
    assert cats <= {"seimi", "fuzui", "muda"}
