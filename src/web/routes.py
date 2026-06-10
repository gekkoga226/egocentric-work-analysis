# src/web/routes.py
"""FastAPI エンドポイント定義。HTTP入出力とテンプレート描画のみ担当。"""
import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from src.web import ids, jobs
from src.web.video_stream import stream_video

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_VIDEOS_DIR = Path("videos_upload")
_RESULTS_DIR = Path("results")
_ANNOTATIONS_DIR = Path("annotations")

_ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}
_GEMINI_TRACKS = {"a", "std", "both"}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
) -> HTMLResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_VIDEO_EXT:
        return HTMLResponse(
            content=f'<p class="error">非対応の形式: {suffix}。MP4/MOV/AVI/MKVを使用してください。</p>',
            status_code=400,
        )

    _VIDEOS_DIR.mkdir(exist_ok=True)
    job_id = ids.register_video(file.filename, Path(""))

    dest = _VIDEOS_DIR / f"{job_id}{suffix}"
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    ids._registry[job_id] = dest
    ids.store_ref_context(job_id, None)

    return templates.TemplateResponse(
        request=request,
        name="_label_form.html",
        context={"job_id": job_id, "filename": file.filename},
    )


@router.post("/upload-pdf", response_class=HTMLResponse)
async def upload_pdf(
    request: Request,
    job_id: str = Form(...),
    pdf: UploadFile = File(...),
) -> HTMLResponse:
    """Attach a work-standard PDF to an already-uploaded video job."""
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>未登録の job_id です。</p>", status_code=404)

    if not (pdf.filename and pdf.filename.lower().endswith(".pdf")):
        return HTMLResponse(content="<p class='error'>PDFファイルを選択してください。</p>", status_code=400)

    _VIDEOS_DIR.mkdir(exist_ok=True)
    pdf_dest = _VIDEOS_DIR / f"{job_id}_ref.pdf"
    with pdf_dest.open("wb") as f:
        while chunk := await pdf.read(1024 * 1024):
            f.write(chunk)

    # Parse reference context in thread (non-blocking for the event loop)
    loop = asyncio.get_event_loop()
    ref_ctx = await loop.run_in_executor(None, _parse_pdf_safe, str(pdf_dest))
    ids.store_ref_context(job_id, ref_ctx)

    status = "✓ 参照文脈を抽出しました" if ref_ctx else "⚠ テキスト抽出に失敗（PDFなしで分析を続行できます）"
    return HTMLResponse(content=f'<span class="pdf-parse-status">{status}</span>')


def _parse_pdf_safe(pdf_path: str) -> Optional[str]:
    try:
        from src.pipeline.parse_reference import parse_reference
        return parse_reference(pdf_path)
    except Exception:
        return None


@router.post("/propose-labels", response_class=HTMLResponse)
async def propose_labels_endpoint(
    request: Request,
    job_id: str = Form(...),
) -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>未登録の job_id です。</p>", status_code=404)

    if not os.environ.get("GEMINI_API_KEY"):
        return HTMLResponse(
            content=(
                "<div class='lf-warn'>"
                "<strong>GEMINI_API_KEY が設定されていません。</strong>"
                "<p>語彙提案をスキップします。手入力してください。</p>"
                "</div>"
            ),
        )

    video_path = ids.get_video_path(job_id)
    ref_ctx = ids.get_ref_context(job_id)

    loop = asyncio.get_event_loop()
    labels = await loop.run_in_executor(None, _propose_safe, str(video_path), ref_ctx)
    labels_csv = ", ".join(labels) if labels else ""
    return HTMLResponse(content=f'<span id="proposedLabels" data-labels="{labels_csv}">{labels_csv}</span>')


def _propose_safe(video_path: str, reference_context: Optional[str]) -> list[str]:
    try:
        from src.pipeline.propose_labels import propose_labels
        return propose_labels(video_path, reference_context=reference_context)
    except Exception:
        return []


@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_id: str = Form(...),
    labels: str = Form(...),
    track: str = Form("std"),
    fps: float = Form(1.0),
    penalty: float = Form(10.0),
    blur_faces: bool = Form(False),
    hints: Optional[str] = Form(None),
) -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>不明な job_id です。</p>", status_code=404)

    hint_list = None
    if hints:
        try:
            import json as _json
            from src.schemas import Hint
            raw = _json.loads(hints)
            hint_list = [Hint(**{k: v for k, v in h.items()
                                 if k in Hint.__dataclass_fields__}) for h in raw]
        except Exception:
            hint_list = None

    needs_gemini = track in _GEMINI_TRACKS
    if needs_gemini and not os.environ.get("GEMINI_API_KEY"):
        return HTMLResponse(
            content=(
                "<div class='status-error'>"
                "<strong>Track STD/A には GEMINI_API_KEY が必要です。</strong>"
                "<p>PowerShell: <code>$env:GEMINI_API_KEY = \"AIza...\"</code></p>"
                "</div>"
            ),
        )

    label_list = [lb.strip() for lb in labels.split(",") if lb.strip()]
    if not label_list:
        return HTMLResponse(content="<p class='error'>ラベルを1つ以上入力してください。</p>", status_code=400)

    video_path = ids.get_video_path(job_id)
    reference_context = ids.get_ref_context(job_id)

    jobs.register(job_id, track)
    _RESULTS_DIR.mkdir(exist_ok=True)

    await jobs.start_pipeline(
        job_id=job_id,
        video_path=video_path,
        label_list=label_list,
        track=track,
        output_dir=_RESULTS_DIR,
        blur_faces=blur_faces,
        fps=fps,
        penalty=penalty,
        reference_context=reference_context,
        hints=hint_list,
    )

    return templates.TemplateResponse(
        request=request,
        name="_status_running.html",
        context={"job_id": job_id, "track": track, "stage": ""},
    )


@router.get("/status/{job_id}", response_class=HTMLResponse)
async def status(request: Request, job_id: str) -> HTMLResponse:
    job = jobs.get_status(job_id)
    if job is None:
        return HTMLResponse(content="<p class='error'>ジョブが見つかりません。</p>", status_code=404)

    if job["status"] == "done":
        return templates.TemplateResponse(
            request=request, name="_status_done.html",
            context={"job_id": job_id, "track": job["track"]},
        )
    elif job["status"] == "error":
        return templates.TemplateResponse(
            request=request, name="_status_error.html",
            context={"job_id": job_id, "error": job["error"]},
        )
    else:
        return templates.TemplateResponse(
            request=request, name="_status_running.html",
            context={"job_id": job_id, "stage": job.get("stage", ""), "track": job["track"]},
        )


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: str, track: str = "std") -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>未登録の job_id です。</p>", status_code=404)

    if track == "both":
        track = "b"

    result_path = _RESULTS_DIR / f"{job_id}_track_{track}.json"
    if not result_path.exists():
        return HTMLResponse(
            content=f"<p class='error'>結果ファイルが見つかりません: track={track}</p>",
            status_code=404,
        )

    from dataclasses import asdict
    from src.schemas import SegmentList
    from src.evaluate.compare import compare_systems

    seg_list = SegmentList.from_json(result_path.read_text(encoding="utf-8"))
    segments_dicts = [asdict(s) for s in seg_list.segments]

    metrics = None
    ann_path = _ANNOTATIONS_DIR / f"{job_id}.json"
    if ann_path.exists():
        gt = SegmentList.from_json(ann_path.read_text(encoding="utf-8"))
        metrics = compare_systems(gt, {seg_list.source: seg_list}).get(seg_list.source)

    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={
            "job_id": job_id,
            "seg_list": seg_list,
            "segments_dicts": segments_dicts,
            "metrics": metrics,
            "track": track,
        },
    )


@router.get("/video/{job_id}")
async def video(request: Request, job_id: str) -> Response:
    if not ids.is_registered(job_id):
        return Response(status_code=404, content="未登録の job_id です。")
    video_path = ids.get_video_path(job_id)
    if not video_path or not video_path.exists():
        return Response(status_code=404, content="動画ファイルが見つかりません。")
    return await stream_video(request, video_path)
