import numpy as np
import pytest
from src.pipeline.embed import embed_frames, embed_texts


def _make_bgr_frames(n: int = 3) -> list[tuple[float, np.ndarray]]:
    frames = []
    for i in range(n):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 80, 100, 150)
        frames.append((float(i), frame))
    return frames


def test_embed_frames_shape():
    frames = _make_bgr_frames(3)
    timestamps, embeddings = embed_frames(frames)
    assert len(timestamps) == 3
    assert embeddings.shape == (3, 512)  # ViT-B-32 → 512


def test_embed_frames_l2_normalized():
    frames = _make_bgr_frames(4)
    _, embeddings = embed_frames(frames)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)


def test_embed_texts_shape():
    labels = ["部品取り出し", "ネジ締め", "検査"]
    embeddings = embed_texts(labels)
    assert embeddings.shape[0] == 3
    assert embeddings.ndim == 2


def test_embed_texts_l2_normalized():
    labels = ["A", "B"]
    embeddings = embed_texts(labels)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


def test_embed_frames_timestamps_match_input():
    frames = [(0.5, np.zeros((64, 64, 3), dtype=np.uint8)),
              (1.5, np.zeros((64, 64, 3), dtype=np.uint8))]
    timestamps, _ = embed_frames(frames)
    assert timestamps == [0.5, 1.5]
