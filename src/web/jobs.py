"""ジョブ登録・状態遷移・別スレッド実行管理。

分析ロジックは持たず、既存パイプライン関数を呼ぶだけ。
状態: registered → running → done | error
"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# インメモリジョブストア（再起動で消える — 将来の継ぎ目）
_jobs: dict[str, dict[str, Any]] = {}

# 重いジョブを直列化して OOM/スラッシングを防ぐ
_executor = ThreadPoolExecutor(max_workers=1)


def register(job_id: str, track: str) -> None:
    """ジョブを登録する。"""
    _jobs[job_id] = {
        "status": "registered",
        "stage": "",
        "track": track,
        "error": None,
    }


def get_status(job_id: str) -> dict[str, Any] | None:
    """ジョブ状態を返す。未登録なら None。"""
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
) -> None:
    """別スレッドで実行されるパイプライン処理。"""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["stage"] = "starting"

        if track in ("b", "both"):
            _jobs[job_id]["stage"] = "track_b: extracting frames"
            from src.pipeline.ingest import extract_frames
            from src.pipeline.embed import embed_frames
            from src.pipeline.presegment import detect_boundaries
            from src.pipeline.label_zeroshot import label_zeroshot
            from src.pipeline.report import save_segments

            frames = extract_frames(str(video_path), fps=fps, blur_faces=blur_faces)
            _jobs[job_id]["stage"] = "track_b: embedding"
            timestamps, embeddings = embed_frames(frames)
            _jobs[job_id]["stage"] = "track_b: detecting boundaries"
            boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
            _jobs[job_id]["stage"] = "track_b: labeling"
            seg_list = label_zeroshot(
                str(video_path), label_list,
                fps=fps, boundary_timestamps=boundaries, blur_faces=blur_faces,
            )
            save_segments(seg_list, str(output_dir))

        if track in ("a", "both"):
            _jobs[job_id]["stage"] = "track_a: gemini inference"
            from src.pipeline.label_vlm_single import label_vlm_single
            from src.pipeline.report import save_segments

            seg_list = label_vlm_single(str(video_path), label_list, blur_faces=blur_faces)
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
) -> None:
    """パイプラインを別スレッドで非同期起動する（イベントループをブロックしない）。"""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline,
        job_id, video_path, label_list, track, output_dir, blur_faces, fps, penalty,
    )
