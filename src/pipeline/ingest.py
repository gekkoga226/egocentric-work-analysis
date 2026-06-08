# src/pipeline/ingest.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    blur_faces: bool = False,
) -> list[tuple[float, np.ndarray]]:
    """Extract frames from video at target fps.

    Returns: list of (timestamp_sec, BGR_frame_ndarray)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            if blur_faces:
                frame = _blur_faces(frame)
            frames.append((timestamp, frame))
        frame_idx += 1
    cap.release()
    return frames


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """Blur detected faces using OpenCV Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    result = frame.copy()
    for (x, y, w, h) in faces:
        roi = result[y : y + h, x : x + w]
        result[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (51, 51), 0)
    return result
