# tests/conftest.py
import numpy as np
import cv2
import pytest
import tempfile
import os
from src.schemas import Segment, SegmentList


@pytest.fixture
def synthetic_video_path():
    """30-second video with 3 color-distinct visual segments (10s each)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    fps = 10
    duration = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (64, 64))

    for i in range(duration * fps):
        t = i / fps
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if t < 10:
            frame[:, :] = (200, 50, 50)    # segment A: blue-ish
        elif t < 20:
            frame[:, :] = (50, 200, 50)    # segment B: green-ish
        else:
            frame[:, :] = (50, 50, 200)    # segment C: red-ish
        out.write(frame)
    out.release()

    yield path
    os.unlink(path)


@pytest.fixture
def ground_truth_segments():
    return SegmentList(
        video_id="test",
        fps_sampled=1.0,
        label_vocabulary=["作業A", "作業B", "作業C"],
        segments=[
            Segment(0.0, 10.0, "作業A", 1.0),
            Segment(10.0, 20.0, "作業B", 1.0),
            Segment(20.0, 30.0, "作業C", 1.0),
        ],
        source="ground_truth",
    )


@pytest.fixture
def perfect_prediction(ground_truth_segments):
    import copy
    sl = copy.deepcopy(ground_truth_segments)
    sl.source = "track_b"
    return sl


@pytest.fixture
def mock_embeddings():
    """Synthetic embeddings with 3 clear visual clusters (matches synthetic_video_path)."""
    np.random.seed(42)
    n = 30  # 30 frames at 1fps
    D = 512
    emb = np.zeros((n, D))
    # cluster A: frames 0-9
    emb[:10] = np.random.randn(10, D) * 0.1 + np.array([1.0] + [0.0] * (D - 1))
    # cluster B: frames 10-19
    emb[10:20] = np.random.randn(10, D) * 0.1 + np.array([0.0, 1.0] + [0.0] * (D - 2))
    # cluster C: frames 20-30
    emb[20:] = np.random.randn(10, D) * 0.1 + np.array([0.0, 0.0, 1.0] + [0.0] * (D - 3))
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms
