"""Range(206) 対応の動画ストリーム配信。

HTML5 video タグはシーク時に Range リクエストを送る。
206 Partial Content を正しく返さないとシークが動かない。
"""
from pathlib import Path

from fastapi import Request
from fastapi.responses import StreamingResponse, Response

_CHUNK = 1024 * 256  # 256 KB


def _parse_range(range_header: str | None, file_size: int) -> tuple[int, int]:
    """Range ヘッダを解析して (start, end) バイト位置を返す。"""
    if not range_header or not range_header.startswith("bytes="):
        return 0, file_size - 1
    parts = range_header[6:].split("-")
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    end = min(end, file_size - 1)
    return start, end


async def stream_video(request: Request, video_path: Path) -> Response:
    """動画ファイルを Range 対応でストリーミング配信する。"""
    if not video_path.exists():
        return Response(status_code=404, content="Video not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("range")
    start, end = _parse_range(range_header, file_size)
    content_length = end - start + 1

    # Content-Type を拡張子から推定
    suffix = video_path.suffix.lower()
    content_type_map = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }
    content_type = content_type_map.get(suffix, "video/mp4")

    async def _iter():
        with video_path.open("rb") as f:
            f.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk_size = min(_CHUNK, remaining)
                data = f.read(chunk_size)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": content_type,
    }
    status_code = 206 if range_header else 200
    return StreamingResponse(_iter(), status_code=status_code, headers=headers)
