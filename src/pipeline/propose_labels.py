# src/pipeline/propose_labels.py
import base64
import json
import logging
import os
from typing import Optional

import cv2
from google import genai  # module-level: tests patch src.pipeline.propose_labels.genai

from src.pipeline.ingest import _blur_faces as _do_blur_faces

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-pro"
_MAX_FRAMES = 20


def propose_labels(
    video_path: str,
    *,
    reference_context: Optional[str] = None,
    blur_faces: bool = False,
    model: str = _GEMINI_MODEL,
    max_labels: int = 12,
) -> list[str]:
    """Sample frames from video and ask Gemini to suggest work operation labels.
    Returns empty list on any failure (non-blocking)."""
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        frames = _sample_frames(video_path, blur_faces, _do_blur_faces)
        if not frames:
            return []

        ref_section = f"\n\nWork Standard Reference:\n{reference_context[:1500]}" if reference_context else ""
        prompt = (
            f"You are an industrial engineer. Analyze this factory work video and propose "
            f"up to {max_labels} distinct work operation labels in Japanese. "
            f"Labels should be specific action names (e.g. 部品取り出し, ネジ締め). "
            f"Output ONLY a JSON array of strings.{ref_section}"
        )

        parts: list = [prompt]
        for ts, b64 in frames:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
            parts.append(f"[t={ts:.0f}s]")

        resp = client.models.generate_content(
            model=model, contents=parts,
            config={"temperature": 0},
        )
        text = resp.text.strip()
        s, e = text.find("["), text.rfind("]") + 1
        if s == -1 or e == 0:
            return []
        labels = json.loads(text[s:e])
        return [str(l).strip() for l in labels if str(l).strip()][:max_labels]
    except Exception as exc:
        logger.warning("propose_labels failed: %s", exc)
        return []


def _sample_frames(
    video_path: str,
    blur_faces: bool,
    blur_fn,
) -> list[tuple[float, str]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    interval = max(total / _MAX_FRAMES, 5.0)
    frames = []
    ts = 0.0
    while ts < total and len(frames) < _MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if blur_faces:
            frame = blur_fn(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frames.append((ts, base64.b64encode(buf.tobytes()).decode()))
        ts += interval
    cap.release()
    return frames
