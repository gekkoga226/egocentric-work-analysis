import json
import os
import tempfile
import pytest
from src.schemas import SegmentList, Segment
from src.pipeline.report import to_timeline_markdown, to_procedure_markdown, save_segments


@pytest.fixture
def sample_seglist():
    return SegmentList(
        video_id="line01",
        fps_sampled=1.0,
        label_vocabulary=["部品取り出し", "ネジ締め"],
        segments=[
            Segment(0.0, 12.4, "部品取り出し", 0.87),
            Segment(12.4, 30.1, "ネジ締め", 0.79),
        ],
        source="track_b",
    )


def test_timeline_contains_video_id(sample_seglist):
    md = to_timeline_markdown(sample_seglist)
    assert "line01" in md


def test_timeline_contains_all_labels(sample_seglist):
    md = to_timeline_markdown(sample_seglist)
    assert "部品取り出し" in md
    assert "ネジ締め" in md


def test_procedure_contains_step_numbers(sample_seglist):
    md = to_procedure_markdown(sample_seglist)
    assert "Step 1" in md
    assert "Step 2" in md


def test_save_segments_creates_file(sample_seglist):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(sample_seglist, tmpdir)
        assert os.path.isfile(path)
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["video_id"] == "line01"
        assert len(data["segments"]) == 2


def test_save_segments_filename_includes_source(sample_seglist):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(sample_seglist, tmpdir)
        assert "track_b" in os.path.basename(path)
