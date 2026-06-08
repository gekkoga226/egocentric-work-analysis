# tests/pipeline/test_ingest.py
import numpy as np
import pytest
from src.pipeline.ingest import extract_frames


def test_extract_returns_tuples(synthetic_video_path):
    frames = extract_frames(synthetic_video_path, fps=1.0)
    assert len(frames) > 0
    ts, frame = frames[0]
    assert isinstance(ts, float)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3  # H, W, C


def test_extract_fps_controls_count(synthetic_video_path):
    frames_1fps = extract_frames(synthetic_video_path, fps=1.0)
    frames_2fps = extract_frames(synthetic_video_path, fps=2.0)
    assert len(frames_2fps) > len(frames_1fps)


def test_extract_timestamps_monotonic(synthetic_video_path):
    frames = extract_frames(synthetic_video_path, fps=1.0)
    timestamps = [ts for ts, _ in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= 0.0


def test_extract_invalid_path_raises():
    with pytest.raises(ValueError, match="Cannot open video"):
        extract_frames("nonexistent_video.mp4", fps=1.0)


def test_blur_faces_returns_same_shape(synthetic_video_path):
    frames_no_blur = extract_frames(synthetic_video_path, fps=1.0, blur_faces=False)
    frames_blur = extract_frames(synthetic_video_path, fps=1.0, blur_faces=True)
    assert len(frames_no_blur) == len(frames_blur)
    _, f1 = frames_no_blur[0]
    _, f2 = frames_blur[0]
    assert f1.shape == f2.shape
