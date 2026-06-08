import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from google import genai
from src.schemas import Segment, SegmentList
from src.pipeline.ingest import _blur_faces
import logging

logger = logging.getLogger(__name__)

_WINDOW_SEC = 300
_WINDOW_FPS = 0.2
_GEMINI_MODEL = "gemini-2.5-pro"


def label_vlm_single(
    video_path: str,
    label_vocabulary: list[str],
    blur_faces: bool = False,
) -> SegmentList:
    video_id = Path(video_path).stem
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    all_segments: list[Segment] = []
    win_starts = list(range(0, int(total_duration), _WINDOW_SEC))

    for win_start in win_starts:
        win_end = min(win_start + _WINDOW_SEC, total_duration)
        segs = _process_window(
            video_path, label_vocabulary, float(win_start), win_end,
            client, blur_faces,
        )
        all_segments.extend(segs)

    return SegmentList(
        video_id=video_id,
        fps_sampled=_WINDOW_FPS,
        label_vocabulary=label_vocabulary,
        segments=_merge_adjacent(all_segments),
        source="track_a",
    )


def _process_window(
    video_path: str,
    labels: list[str],
    start_sec: float,
    end_sec: float,
    client,
    blur_faces: bool,
) -> list[Segment]:
    frames = _extract_window_frames(video_path, start_sec, end_sec, blur_faces)
    if not frames:
        return []

    label_list = "\n".join(f"- {l}" for l in labels)
    prompt = (
        f"You are analyzing a factory line work video.\n"
        f"The video segment covers {start_sec:.1f}s to {end_sec:.1f}s from the start.\n"
        f"Available action labels:\n{label_list}\n\n"
        f"Analyze the frames (timestamps shown as [t=Xs]) and output a JSON array.\n"
        f"Each element: {{\"start_sec\": float, \"end_sec\": float, "
        f"\"label\": \"exact label from above\", \"confidence\": float 0-1}}\n"
        f"Output ONLY the JSON array, no other text."
    )

    parts: list = [prompt]
    for ts, frame in frames:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append(f"[t={ts:.1f}s]")

    try:
        resp = client.models.generate_content(model=_GEMINI_MODEL, contents=parts)
        text = resp.text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("Gemini returned no JSON array for window %s-%s", start_sec, end_sec)
            return []
        raw = json.loads(text[start:end])
        return [Segment(**s) for s in raw]
    except Exception as exc:
        logger.warning("Gemini call failed for window %s-%s: %s", start_sec, end_sec, exc)
        return []


def _extract_window_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    blur_faces: bool,
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    interval = 1.0 / _WINDOW_FPS
    frames = []
    ts = start_sec
    while ts < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if blur_faces:
            frame = _blur_faces(frame)
        frames.append((ts, frame))
        ts += interval
    cap.release()
    return frames


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    merged = [Segment(segments[0].start_sec, segments[0].end_sec,
                      segments[0].label, segments[0].confidence)]
    for seg in segments[1:]:
        if seg.label == merged[-1].label:
            merged[-1] = Segment(
                merged[-1].start_sec, seg.end_sec, merged[-1].label,
                (merged[-1].confidence + seg.confidence) / 2,
            )
        else:
            merged.append(Segment(seg.start_sec, seg.end_sec, seg.label, seg.confidence))
    return merged
