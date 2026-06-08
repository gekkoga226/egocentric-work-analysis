"""End-to-end pipeline test using synthetic video and mocked Gemini API."""
import tempfile
import pytest
import numpy as np
from unittest.mock import patch
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
