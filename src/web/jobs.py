"""ジョブ登録・状態遷移・別スレッド実行管理。"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=1)

# Registry: track_key → callable(video_path, label_vocabulary, **opts) -> SegmentList
# Populated lazily to avoid circular imports at module load time.
TRACK_RUNNERS: dict[str, Any] = {}


def _init_runners() -> None:
    if TRACK_RUNNERS:
        return

    from src.pipeline.label_zeroshot import label_zeroshot
    from src.pipeline.label_vlm_single import label_vlm_single
    from src.pipeline.label_gemini import label_gemini

    def _run_b(video_path, label_vocabulary, *, fps=1.0, penalty=10.0, blur_faces=False, **_):
        from src.pipeline.ingest import extract_frames
        from src.pipeline.embed import embed_frames
        from src.pipeline.presegment import detect_boundaries
        frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        return label_zeroshot(video_path, label_vocabulary,
                              fps=fps, boundary_timestamps=boundaries, blur_faces=blur_faces)

    def _run_a(video_path, label_vocabulary, *, blur_faces=False, **_):
        return label_vlm_single(video_path, label_vocabulary, blur_faces=blur_faces)

    def _run_std(video_path, label_vocabulary, *,
                 fps=1.0, penalty=10.0, blur_faces=False,
                 reference_context=None, hints=None, **_):
        from src.pipeline.ingest import extract_frames
        from src.pipeline.embed import embed_frames
        from src.pipeline.presegment import detect_boundaries
        frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        return label_gemini(
            video_path, label_vocabulary, boundaries,
            blur_faces=blur_faces,
            reference_context=reference_context,
            hints=hints,
        )

    TRACK_RUNNERS["b"] = _run_b
    TRACK_RUNNERS["a"] = _run_a
    TRACK_RUNNERS["std"] = _run_std


def register(job_id: str, track: str) -> None:
    _jobs[job_id] = {"status": "registered", "stage": "", "track": track, "error": None}


def get_status(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def _run_pipeline(
    job_id: str,
    video_path: Path,
    label_list: list[str],
    track: str,
    output_dir: Path,
    blur_faces: bool,
    fps: float,
    penalty: float,
    reference_context: str | None = None,
    hints: list | None = None,
) -> None:
    try:
        _jobs[job_id]["status"] = "running"
        _init_runners()

        tracks = ["b", "a"] if track == "both" else [track]
        for t in tracks:
            runner = TRACK_RUNNERS.get(t)
            if runner is None:
                raise ValueError(f"Unknown track: {t!r}. Available: {list(TRACK_RUNNERS)}")
            _jobs[job_id]["stage"] = f"{t}: analyzing"
            seg_list = runner(
                str(video_path), label_list,
                fps=fps, penalty=penalty, blur_faces=blur_faces,
                reference_context=reference_context, hints=hints,
            )
            from src.pipeline.report import save_segments
            save_segments(seg_list, str(output_dir))

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["stage"] = "complete"

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


async def start_pipeline(
    job_id: str,
    video_path: Path,
    label_list: list[str],
    track: str,
    output_dir: Path,
    blur_faces: bool = False,
    fps: float = 1.0,
    penalty: float = 10.0,
    reference_context: str | None = None,
    hints: list | None = None,
) -> None:
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        _executor, _run_pipeline,
        job_id, video_path, label_list, track, output_dir,
        blur_faces, fps, penalty, reference_context, hints,
    )
