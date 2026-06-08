import numpy as np
import cv2
from pathlib import Path
from src.schemas import Segment, SegmentList
from src.pipeline.ingest import extract_frames
from src.pipeline.embed import embed_frames, embed_texts


def label_zeroshot(
    video_path: str,
    label_vocabulary: list[str],
    fps: float = 1.0,
    boundary_timestamps: list[float] | None = None,
    blur_faces: bool = False,
) -> SegmentList:
    video_id = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not boundary_timestamps:
        ranges = [(0.0, total_duration)]
    else:
        starts = [0.0] + list(boundary_timestamps)
        ends = list(boundary_timestamps) + [total_duration]
        ranges = list(zip(starts, ends))

    frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
    timestamps, img_emb = embed_frames(frames)
    text_emb = embed_texts(label_vocabulary)

    segments: list[Segment] = []
    for start_sec, end_sec in ranges:
        mask = [start_sec <= ts < end_sec for ts in timestamps]
        if not any(mask):
            continue
        seg_img_emb = img_emb[mask]
        sims = seg_img_emb @ text_emb.T
        avg_sims = sims.mean(axis=0)
        best_idx = int(np.argmax(avg_sims))
        segments.append(Segment(
            start_sec=start_sec,
            end_sec=end_sec,
            label=label_vocabulary[best_idx],
            confidence=float(avg_sims[best_idx]),
        ))

    return SegmentList(
        video_id=video_id,
        fps_sampled=fps,
        label_vocabulary=label_vocabulary,
        segments=segments,
        source="track_b",
    )
