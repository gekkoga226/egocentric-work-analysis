import numpy as np
import pytest
from src.pipeline.presegment import detect_boundaries


def test_detect_clear_boundaries(mock_embeddings):
    timestamps = list(range(30))
    boundaries = detect_boundaries(timestamps, mock_embeddings, penalty=1.0)
    assert len(boundaries) >= 1
    assert any(8 <= b <= 12 for b in boundaries), f"Expected boundary near 10, got {boundaries}"
    assert any(18 <= b <= 22 for b in boundaries), f"Expected boundary near 20, got {boundaries}"


def test_detect_no_change_flat_signal():
    np.random.seed(0)
    emb = np.random.randn(20, 64) * 0.01 + np.array([1.0] + [0.0] * 63)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    timestamps = list(range(20))
    boundaries = detect_boundaries(timestamps, emb, penalty=50.0)
    assert len(boundaries) == 0


def test_detect_min_segment_enforced(mock_embeddings):
    timestamps = list(range(30))
    boundaries = detect_boundaries(
        timestamps, mock_embeddings, penalty=0.5, min_segment_sec=15.0
    )
    if len(boundaries) > 1:
        diffs = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        assert all(d >= 15.0 for d in diffs)


def test_detect_returns_float_list(mock_embeddings):
    timestamps = [float(i) for i in range(30)]
    boundaries = detect_boundaries(timestamps, mock_embeddings)
    assert isinstance(boundaries, list)
    assert all(isinstance(b, float) for b in boundaries)


def test_detect_empty_returns_empty():
    boundaries = detect_boundaries([], np.zeros((0, 64)))
    assert boundaries == []
