"""src/web/routes.py のエンドポイントテスト（分析関数はモック）。"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import src.web.ids as ids_module
import src.web.jobs as jobs_module
from src.web.app import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clear_state():
    ids_module._registry.clear()
    jobs_module._jobs.clear()
    yield
    ids_module._registry.clear()
    jobs_module._jobs.clear()


# ── GET / ──
def test_index_returns_200():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_index_contains_upload_form():
    resp = client.get("/")
    # HTML に upload フォームが含まれること（大文字小文字不問）
    assert b"upload" in resp.content.lower() or "アップロード".encode() in resp.content


# ── POST /upload ──
def test_upload_mp4_returns_200(tmp_path):
    fake_video = b"fake mp4 content"
    resp = client.post(
        "/upload",
        files={"file": ("test_video.mp4", fake_video, "video/mp4")},
    )
    assert resp.status_code == 200
    assert b"job_id" in resp.content or b"label" in resp.content.lower()


def test_upload_invalid_extension_returns_400():
    resp = client.post(
        "/upload",
        files={"file": ("bad.txt", b"text content", "text/plain")},
    )
    assert resp.status_code == 400


def test_upload_registers_job_id():
    before = set(ids_module.all_job_ids())
    client.post(
        "/upload",
        files={"file": ("myvideo.mp4", b"data", "video/mp4")},
    )
    after = set(ids_module.all_job_ids())
    assert len(after) > len(before)


# ── POST /analyze ──
def test_analyze_unregistered_job_returns_404():
    resp = client.post(
        "/analyze",
        data={"job_id": "ghost_job", "labels": "A,B", "track": "b"},
    )
    assert resp.status_code == 404


def test_analyze_track_a_without_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # 先にアップロードして job_id を取得
    upload_resp = client.post(
        "/upload",
        files={"file": ("v.mp4", b"data", "video/mp4")},
    )
    # アップロード自体は成功するはず
    assert upload_resp.status_code == 200, upload_resp.text
    job_id = list(ids_module.all_job_ids())[-1]

    resp = client.post(
        "/analyze",
        data={"job_id": job_id, "labels": "A,B", "track": "a"},
    )
    # GEMINI_API_KEY なしでは案内断片が返る（200 で本文にメッセージ）
    assert resp.status_code == 200
    assert b"GEMINI_API_KEY" in resp.content


def test_analyze_empty_labels_returns_400():
    upload_resp = client.post(
        "/upload",
        files={"file": ("v.mp4", b"data", "video/mp4")},
    )
    job_id = list(ids_module.all_job_ids())[-1]
    resp = client.post(
        "/analyze",
        data={"job_id": job_id, "labels": "   ", "track": "b"},
    )
    assert resp.status_code == 400


# ── GET /status/{job_id} ──
def test_status_registered_job_returns_running_template():
    jobs_module.register("test_job", "b")
    jobs_module._jobs["test_job"]["status"] = "running"
    ids_module._registry["test_job"] = Path("/tmp/v.mp4")

    resp = client.get("/status/test_job")
    assert resp.status_code == 200
    assert b"test_job" in resp.content


def test_status_done_returns_done_template():
    jobs_module.register("done_job", "b")
    jobs_module._jobs["done_job"]["status"] = "done"
    ids_module._registry["done_job"] = Path("/tmp/v.mp4")

    resp = client.get("/status/done_job")
    assert resp.status_code == 200
    # done テンプレートは hx-get="/results/..." を含む
    assert b"results" in resp.content


def test_status_unknown_job_returns_404():
    resp = client.get("/status/ghost_job")
    assert resp.status_code == 404


# ── GET /video/{job_id} ──
def test_video_unregistered_returns_404():
    resp = client.get("/video/ghost_job")
    assert resp.status_code == 404


def test_video_registered_but_missing_file_returns_404(tmp_path):
    ids_module._registry["vid_job"] = tmp_path / "nonexistent.mp4"
    resp = client.get("/video/vid_job")
    assert resp.status_code == 404


def test_video_returns_206_for_range_request(tmp_path):
    video_file = tmp_path / "test.mp4"
    video_file.write_bytes(b"Y" * 200)
    ids_module._registry["vid_ok"] = video_file

    resp = client.get("/video/vid_ok", headers={"Range": "bytes=0-99"})
    assert resp.status_code == 206
    assert len(resp.content) == 100
