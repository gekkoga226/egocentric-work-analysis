"""src/web/ids.py のテスト。"""
import time
from pathlib import Path

import pytest

import src.web.ids as ids_module


@pytest.fixture(autouse=True)
def clear_registry():
    """各テスト前後でレジストリをクリア。"""
    ids_module._registry.clear()
    yield
    ids_module._registry.clear()


def test_register_returns_job_id():
    job_id = ids_module.register_video("video.mp4", Path("/tmp/video.mp4"))
    assert isinstance(job_id, str)
    assert len(job_id) > 0


def test_register_unique_ids_for_same_filename(monkeypatch):
    """同名ファイルを登録しても衝突しないこと（タイムスタンプ差を強制）。"""
    call_count = [0]
    original = time.time

    def fake_time():
        call_count[0] += 1
        return original() + call_count[0] * 0.01  # 10msずつずらす

    monkeypatch.setattr(time, "time", fake_time)

    id1 = ids_module.register_video("video.mp4", Path("/tmp/a.mp4"))
    id2 = ids_module.register_video("video.mp4", Path("/tmp/b.mp4"))
    assert id1 != id2


def test_is_registered_true():
    job_id = ids_module.register_video("test.mp4", Path("/tmp/test.mp4"))
    assert ids_module.is_registered(job_id) is True


def test_is_registered_false():
    assert ids_module.is_registered("nonexistent_id") is False


def test_get_video_path_returns_path():
    p = Path("/tmp/myvideo.mp4")
    job_id = ids_module.register_video("myvideo.mp4", p)
    assert ids_module.get_video_path(job_id) == p


def test_get_video_path_unregistered_returns_none():
    assert ids_module.get_video_path("ghost_id") is None


def test_sanitize_removes_special_chars():
    job_id = ids_module.register_video("my video (1).mp4", Path("/tmp/x.mp4"))
    # job_id の先頭部分は sanitize 済みのはず（スペース・括弧がない）
    prefix = job_id.rsplit("_", 1)[0]
    assert " " not in prefix
    assert "(" not in prefix
    assert ")" not in prefix


def test_sanitize_handles_path_traversal():
    """../../../etc/passwd のような入力でもパストラバーサルが起きないこと。"""
    job_id = ids_module.register_video("../../etc/passwd", Path("/tmp/x.mp4"))
    assert ".." not in job_id
    assert "/" not in job_id


def test_all_job_ids():
    ids_module.register_video("a.mp4", Path("/a.mp4"))
    ids_module.register_video("b.mp4", Path("/b.mp4"))
    assert len(ids_module.all_job_ids()) == 2
