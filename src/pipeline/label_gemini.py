# src/pipeline/label_gemini.py
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from google import genai  # module-level: tests patch src.pipeline.label_gemini.genai

from src.schemas import Hint, Segment, SegmentList
from src.pipeline.ingest import _blur_faces as _do_blur_faces

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-pro"
MAX_FRAMES_PER_WINDOW = 30
FRAMES_PER_SEGMENT = 3

_CATEGORY_ALIASES: dict[str, str] = {
    "正味": "seimi", "正味作業": "seimi",
    "付随": "fuzui", "付随作業": "fuzui",
    "ムダ": "muda", "ムダ作業": "muda",
    "value-adding": "seimi", "value adding": "seimi", "productive": "seimi",
    "ancillary": "fuzui", "supporting": "fuzui", "auxiliary": "fuzui",
    "waste": "muda", "non-value-adding": "muda",
    "seimi": "seimi", "fuzui": "fuzui", "muda": "muda",
}

_LABEL_SYNONYMS: dict[str, str] = {
    "手待ち時間": "手待ち",
    "待機": "手待ち",
    "待ち": "手待ち",
    "手待": "手待ち",
    "探す": "モノ探し",
    "探している": "モノ探し",
    "もの探し": "モノ探し",
    "物探し": "モノ探し",
    "無駄歩行": "歩行（ムダ）",
    "歩いている": "歩行（ムダ）",
    "余分な歩行": "歩行（ムダ）",
    "不要な移動": "歩行（ムダ）",
    "手戻り": "やり直し",
    "やりなおし": "やり直し",
    "ミスの修正": "やり直し",
}


# ── Public entry point ────────────────────────────────────────────────────────

def label_gemini(
    video_path: str,
    label_vocabulary: list[str],
    boundary_timestamps: list[float],
    *,
    blur_faces: bool = False,
    hints: list[Hint] | None = None,
    reference_context: str | None = None,
    model: str = _GEMINI_MODEL,
    source: str = "track_std",
    raw_output_dir: str = "results",
) -> SegmentList:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    coarse_segs = _build_coarse_segments(boundary_timestamps, total_duration)
    windows = _group_into_windows(coarse_segs)

    video_id = Path(video_path).stem
    effective_vocab = list(label_vocabulary)
    window_results: list[list[Segment]] = []
    all_raw: list[dict] = []

    for win_idx, (start_idx, end_idx) in enumerate(windows):
        win_coarse = coarse_segs[start_idx:end_idx]
        win_start = win_coarse[0][0]
        win_end = win_coarse[-1][1]

        frames = _extract_window_frames(video_path, win_coarse, blur_faces)
        segments, raw_text = _call_gemini(
            client=client, model=model, frames=frames,
            label_vocabulary=effective_vocab,
            win_start=win_start, win_end=win_end,
            reference_context=reference_context, hints=hints,
        )

        clamped = []
        for seg in segments:
            s = max(win_start, min(seg.start_sec, win_end))
            e = max(win_start, min(seg.end_sec, win_end))
            if e > s:
                clamped.append(Segment(s, e, seg.label, seg.confidence,
                                       seg.category, seg.description, seg.improvement))
        window_results.append(clamped)
        all_raw.append({"window": win_idx, "win_start": win_start, "win_end": win_end, "raw": raw_text})

    # Persist raw responses for reproducibility
    results_dir = Path(raw_output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{video_id}_{source}_raw.json").write_text(
        json.dumps(all_raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_segments = _stitch(window_results, windows, coarse_segs, total_duration, effective_vocab)

    return SegmentList(
        video_id=video_id,
        fps_sampled=float(FRAMES_PER_SEGMENT),
        label_vocabulary=effective_vocab,
        segments=all_segments,
        source=source,
    )


# ── Pure helpers (tested in isolation) ───────────────────────────────────────

def _build_coarse_segments(
    boundary_timestamps: list[float],
    total_duration: float,
) -> list[tuple[float, float]]:
    points = sorted({0.0} | set(boundary_timestamps) | {total_duration})
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)
            if points[i + 1] - points[i] > 0.01]


def _group_into_windows(
    coarse_segs: list[tuple[float, float]],
    target_size: int = 10,
) -> list[tuple[int, int]]:
    n = len(coarse_segs)
    if n == 0:
        return []
    windows: list[tuple[int, int]] = []
    i = 0
    while i < n:
        end = min(i + target_size, n)
        windows.append((i, end))
        if end == n:
            break
        i = end - 1  # 1-segment overlap
    return windows


def _merge_adjacent_enrich(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent segments with same (label, category). Core-priority: longer segment's
    description/improvement is kept (approximates keeping the window-core side)."""
    if not segments:
        return []
    merged = [Segment(
        segments[0].start_sec, segments[0].end_sec, segments[0].label,
        segments[0].confidence, segments[0].category,
        segments[0].description, segments[0].improvement,
    )]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.label == last.label and seg.category == last.category:
            last_dur = last.end_sec - last.start_sec
            seg_dur = seg.end_sec - seg.start_sec
            keep = last if last_dur >= seg_dur else seg
            merged[-1] = Segment(
                last.start_sec, seg.end_sec, last.label,
                (last.confidence + seg.confidence) / 2,
                last.category, keep.description, keep.improvement,
            )
        else:
            merged.append(Segment(
                seg.start_sec, seg.end_sec, seg.label, seg.confidence,
                seg.category, seg.description, seg.improvement,
            ))
    return merged


def _stitch(
    window_results: list[list[Segment]],
    windows: list[tuple[int, int]],
    coarse_segs: list[tuple[float, float]],
    total_duration: float,
    vocabulary: list[str],
) -> list[Segment]:
    if not window_results:
        return []

    # Determine core range for each window
    def core_range(win_idx: int) -> tuple[float, float]:
        start_idx, end_idx = windows[win_idx]
        segs = coarse_segs[start_idx:end_idx]
        c_start = segs[1][0] if win_idx > 0 and len(segs) > 1 else segs[0][0]
        c_end = segs[-2][1] if win_idx < len(windows) - 1 and len(segs) > 1 else segs[-1][1]
        return c_start, c_end

    # Tag each candidate segment with is_core
    tagged: list[tuple[Segment, bool, int]] = []
    for win_idx, result in enumerate(window_results):
        c_start, c_end = core_range(win_idx)
        for seg in result:
            center = (seg.start_sec + seg.end_sec) / 2
            is_core = c_start <= center <= c_end
            tagged.append((seg, is_core, win_idx))

    # Sort: by start_sec asc, core first (is_core=True sorts before False)
    tagged.sort(key=lambda x: (x[0].start_sec, 0 if x[1] else 1))

    # Greedy de-overlap: keep first (core-preferred) non-overlapping segments
    kept: list[Segment] = []
    current_end = 0.0
    for seg, is_core, _ in tagged:
        if seg.end_sec <= current_end:
            continue
        if seg.start_sec < current_end:
            seg = Segment(current_end, seg.end_sec, seg.label, seg.confidence,
                         seg.category, seg.description, seg.improvement)
        if seg.end_sec > seg.start_sec:
            kept.append(seg)
            current_end = seg.end_sec

    # Normalize labels (synonym + out-of-vocab append)
    for seg in kept:
        seg.label = _normalize_label(seg.label, vocabulary)

    # Fill leading gap
    filled: list[Segment] = []
    if kept and kept[0].start_sec > 0.01:
        filled.append(Segment(0.0, kept[0].start_sec, kept[0].label, 0.5,
                              kept[0].category, None, None))

    for seg in kept:
        if filled and filled[-1].end_sec < seg.start_sec - 0.01:
            filled.append(Segment(filled[-1].end_sec, seg.start_sec,
                                  filled[-1].label, 0.5, filled[-1].category, None, None))
        filled.append(seg)

    # Fill trailing gap
    if filled and filled[-1].end_sec < total_duration - 0.01:
        filled.append(Segment(filled[-1].end_sec, total_duration,
                              filled[-1].label, 0.5, filled[-1].category, None, None))

    return _merge_adjacent_enrich(filled)


def _normalize_label(label: str, vocabulary: list[str]) -> str:
    label = label.strip()
    if label in vocabulary:
        return label
    normalized = _LABEL_SYNONYMS.get(label)
    if normalized:
        return normalized
    if label not in vocabulary:
        vocabulary.append(label)
    return label


# ── Gemini I/O (not tested directly; covered via label_gemini integration) ───

def _extract_window_frames(
    video_path: str,
    coarse_segs: list[tuple[float, float]],
    blur_faces: bool,
) -> list[tuple[float, np.ndarray]]:
    frames: list[tuple[float, np.ndarray]] = []
    cap = cv2.VideoCapture(video_path)
    for seg_start, seg_end in coarse_segs:
        n = min(FRAMES_PER_SEGMENT, max(1, int(seg_end - seg_start)))
        for ts in np.linspace(seg_start, seg_end, n, endpoint=False):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            if blur_faces:
                frame = _do_blur_faces(frame)
            frames.append((float(ts), frame))
    cap.release()
    if len(frames) > MAX_FRAMES_PER_WINDOW:
        step = len(frames) / MAX_FRAMES_PER_WINDOW
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES_PER_WINDOW)]
    return frames


def _call_gemini(
    *,
    client,
    model: str,
    frames: list[tuple[float, np.ndarray]],
    label_vocabulary: list[str],
    win_start: float,
    win_end: float,
    reference_context: str | None,
    hints: list[Hint] | None,
) -> tuple[list[Segment], str]:
    label_list = "\n".join(f"- {l}" for l in label_vocabulary)
    ref_section = f"\n\nWork Standard Reference:\n{reference_context[:2000]}" if reference_context else ""
    hints_section = ""
    if hints:
        hints_section = "\n\nKnown objects/operations:\n" + "\n".join(
            f"- {h.label}" + (f": {h.note}" if h.note else "") for h in hints
        )

    prompt = (
        f"You are an industrial engineer analyzing a factory work video.\n"
        f"Segment: {win_start:.1f}s to {win_end:.1f}s.\n"
        f"Labels (add new label only if none fits):\n{label_list}\n"
        f"Categories: seimi (正味作業=value-adding), fuzui (付随作業=ancillary), muda (ムダ作業=waste)\n"
        f"{ref_section}{hints_section}\n\n"
        f"Frames shown as [t=Xs]. Output a JSON array covering [{win_start:.1f}, {win_end:.1f}] "
        f"with no gaps. Each element: "
        f"{{\"start_sec\":float, \"end_sec\":float, \"label\":string, "
        f"\"category\":\"seimi\"|\"fuzui\"|\"muda\", "
        f"\"description\":string (observed facts only), "
        f"\"improvement\":string|null (only for muda/fuzui), "
        f"\"confidence\":float}}\n"
        f"Output ONLY the JSON array."
    )

    parts: list = [prompt]
    for ts, frame in frames:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                       "data": base64.b64encode(buf.tobytes()).decode()}})
        parts.append(f"[t={ts:.1f}s]")

    try:
        resp = client.models.generate_content(
            model=model, contents=parts,
            config={"temperature": 0},
        )
        raw = resp.text.strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e == 0:
            logger.warning("No JSON array from Gemini for window %s-%s", win_start, win_end)
            return [], raw
        items = json.loads(raw[s:e])
        segs = []
        for it in items:
            cat_key = str(it.get("category", "")).strip()
            cat = _CATEGORY_ALIASES.get(cat_key) or _CATEGORY_ALIASES.get(cat_key.lower())
            segs.append(Segment(
                start_sec=float(it["start_sec"]),
                end_sec=float(it["end_sec"]),
                label=str(it["label"]),
                confidence=float(it.get("confidence", 1.0)),
                category=cat,
                description=it.get("description") or None,
                improvement=it.get("improvement") or None,
            ))
        return segs, raw
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return [], ""
