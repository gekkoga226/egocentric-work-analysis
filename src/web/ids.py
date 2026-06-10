"""video_id の一意化・サニタイズ・ジョブレジストリへの登録チェック。

§5.2: 同名ファイル衝突対策 — {sanitized}_{timestamp} で一意化
§5.3: パストラバーサル対策 — 登録済みIDのみ受理
"""
import re
import time
from pathlib import Path

# 登録済み job_id → 動画ファイルパス のマッピング
# { job_id: Path(video_path) }
_registry: dict[str, Path] = {}

# 登録済み job_id → reference_context のマッピング
# { job_id: context_string or None }
_ref_contexts: dict[str, str | None] = {}


def _sanitize(name: str) -> str:
    """ファイル名から拡張子を除き、安全な文字のみに絞る。"""
    stem = Path(name).stem
    # 英数字・アンダースコア・ハイフン以外をアンダースコアに置換
    return re.sub(r"[^\w\-]", "_", stem)[:64]


def register_video(filename: str, video_path: Path) -> str:
    """ファイル名を一意化した job_id を生成して登録し、job_id を返す。"""
    base = _sanitize(filename)
    ts = int(time.time() * 1000) % 1_000_000  # ミリ秒下6桁
    job_id = f"{base}_{ts}"
    _registry[job_id] = video_path
    return job_id


def get_video_path(job_id: str) -> Path | None:
    """登録済み job_id に対応する動画パスを返す。未登録なら None。"""
    return _registry.get(job_id)


def is_registered(job_id: str) -> bool:
    """job_id が登録済みかどうかを返す（パストラバーサル防止用）。"""
    return job_id in _registry


def all_job_ids() -> list[str]:
    """登録済みの全 job_id を返す。"""
    return list(_registry.keys())


def store_ref_context(job_id: str, context: str | None) -> None:
    """job_id に対応する reference_context を保存する。"""
    _ref_contexts[job_id] = context


def get_ref_context(job_id: str) -> str | None:
    """job_id に対応する reference_context を取得する。未登録なら None。"""
    return _ref_contexts.get(job_id)
