import numpy as np
import pytest
from unittest.mock import patch
from src.schemas import SegmentList
from src.pipeline.label_zeroshot import label_zeroshot

LABELS = ["作業A", "作業B", "作業C"]


def _make_mock_embed_frames(clusters):
    D = 64
    cluster_vecs = np.eye(len(clusters), D)

    def _mock(frames):
        timestamps = [ts for ts, _ in frames]
        emb = np.zeros((len(frames), D))
        for i, (ts, _) in enumerate(frames):
            t = int(ts)
            if t < 10:
                emb[i] = cluster_vecs[0]
            elif t < 20:
                emb[i] = cluster_vecs[1]
            else:
                emb[i] = cluster_vecs[2]
        return timestamps, emb
    return _mock


def _make_mock_embed_texts(labels):
    D = 64
    vecs = np.eye(len(labels), D)
    return lambda _: vecs


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_assigns_correct_labels(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(30)]
    mock_embed_f.side_effect = _make_mock_embed_frames(LABELS)
    mock_embed_t.side_effect = _make_mock_embed_texts(LABELS)

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )
    labels = [s.label for s in result.segments]
    assert labels == ["作業A", "作業B", "作業C"]


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_source_is_track_b(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(10)]
    mock_embed_f.return_value = ([float(i) for i in range(10)], np.eye(10, 64)[:, :64])
    mock_embed_t.return_value = np.eye(3, 64)

    result = label_zeroshot(synthetic_video_path, LABELS, boundary_timestamps=[])
    assert result.source == "track_b"


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_segments_cover_full_duration(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(30)]
    mock_embed_f.side_effect = _make_mock_embed_frames(LABELS)
    mock_embed_t.side_effect = _make_mock_embed_texts(LABELS)

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )
    assert result.segments[0].start_sec == 0.0
    assert result.segments[-1].end_sec > 20.0
