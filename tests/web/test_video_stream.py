"""src/web/video_stream.py のテスト（Range 206 レスポンス）。"""
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.web.video_stream import stream_video


@pytest.fixture
def video_file(tmp_path):
    """ダミー動画ファイル（100バイト）を作成する。"""
    p = tmp_path / "test.mp4"
    p.write_bytes(b"X" * 100)
    return p


@pytest.fixture
def stream_app(video_file):
    """video_file を指す TestClient を返す。"""
    _app = FastAPI()

    @_app.get("/video")
    async def _ep(request: Request):
        return await stream_video(request, video_file)

    return TestClient(_app, raise_server_exceptions=True)


def test_no_range_returns_200(stream_app):
    """Range ヘッダなしで 200 が返ること。"""
    resp = stream_app.get("/video")
    assert resp.status_code == 200
    assert resp.content == b"X" * 100


def test_range_request_returns_206(stream_app):
    """Range ヘッダ付きで 206 が返ること。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=0-49"})
    assert resp.status_code == 206


def test_range_response_has_correct_content_range(stream_app):
    """Content-Range ヘッダが正しいこと。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=10-29"})
    assert resp.status_code == 206
    assert resp.headers["content-range"] == "bytes 10-29/100"


def test_range_response_correct_bytes(stream_app):
    """Range で指定した範囲のバイト数が返ること。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=0-9"})
    assert resp.status_code == 206
    assert len(resp.content) == 10
    assert resp.content == b"X" * 10


def test_range_end_clamped_to_file_size(stream_app):
    """end が file_size を超える Range でも正しく返ること。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=90-999"})
    assert resp.status_code == 206
    assert len(resp.content) == 10  # 100 - 90


def test_missing_file_returns_404(tmp_path):
    """存在しないファイルは 404。"""
    _app = FastAPI()

    @_app.get("/video")
    async def _ep(request: Request):
        return await stream_video(request, tmp_path / "nonexistent.mp4")

    client = TestClient(_app, raise_server_exceptions=False)
    resp = client.get("/video", headers={"Range": "bytes=0-9"})
    assert resp.status_code == 404


def test_accept_ranges_header_present(stream_app):
    """Accept-Ranges: bytes ヘッダが存在すること。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=0-9"})
    assert resp.headers.get("accept-ranges") == "bytes"


def test_content_length_correct(stream_app):
    """Content-Length が Range サイズと一致すること。"""
    resp = stream_app.get("/video", headers={"Range": "bytes=20-39"})
    assert resp.status_code == 206
    assert resp.headers.get("content-length") == "20"
