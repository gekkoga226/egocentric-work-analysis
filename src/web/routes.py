"""FastAPI エンドポイント定義。HTTP入出力とテンプレート描画のみ担当。"""
import os
from pathlib import Path

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

_ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@router.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)) -> HTMLResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        return HTMLResponse(
            content=f'<p class="error">非対応の形式です: {suffix}。MP4/MOV/AVI/MKVを使用してください。</p>',
            status_code=400,
        )

    _VIDEOS_DIR.mkdir(exist_ok=True)
    job_id = ids.register_video(file.filename, Path(""))  # 仮登録

    dest = _VIDEOS_DIR / f"{job_id}{suffix}"
    with dest.open("wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB チャンク
            f.write(chunk)

    # パスを正式登録
    ids._registry[job_id] = dest

    return templates.TemplateResponse(
        request=request,
        name="_label_form.html",
        context={"job_id": job_id, "filename": file.filename},
    )


@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    job_id: str = Form(...),
    labels: str = Form(...),
    track: str = Form("b"),
    fps: float = Form(1.0),
    penalty: float = Form(10.0),
    blur_faces: bool = Form(False),
) -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>不明な job_id です。</p>", status_code=404)

    if track in ("a", "both") and not os.environ.get("GEMINI_API_KEY"):
        return HTMLResponse(
            content=(
                "<div class='status-error'>"
                "<strong>Track A を使用するには GEMINI_API_KEY が必要です。</strong>"
                "<p>PowerShell: <code>$env:GEMINI_API_KEY = \"AIza...\"</code></p>"
                "</div>"
            ),
            status_code=200,
        )

    label_list = [lb.strip() for lb in labels.split(",") if lb.strip()]
    if not label_list:
        return HTMLResponse(content="<p class='error'>ラベルを1つ以上入力してください。</p>", status_code=400)

    video_path = ids.get_video_path(job_id)
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
            request=request,
            name="_status_done.html",
            context={"job_id": job_id, "track": job["track"]},
        )
    elif job["status"] == "error":
        return templates.TemplateResponse(
            request=request,
            name="_status_error.html",
            context={"job_id": job_id, "error": job["error"]},
        )
    else:
        return templates.TemplateResponse(
            request=request,
            name="_status_running.html",
            context={"job_id": job_id, "stage": job.get("stage", ""), "track": job["track"]},
        )


@router.get("/results/{job_id}", response_class=HTMLResponse)
async def results(request: Request, job_id: str, track: str = "b") -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>未登録の job_id です。</p>", status_code=404)

    video_id = job_id
    result_path = _RESULTS_DIR / f"{video_id}_track_{track}.json"
    if not result_path.exists():
        return HTMLResponse(
            content=f"<p class='error'>結果ファイルが見つかりません: {result_path.name}</p>",
            status_code=404,
        )

    from src.schemas import SegmentList
    from src.evaluate.compare import compare_systems

    seg_list = SegmentList.from_json(result_path.read_text(encoding="utf-8"))

    metrics = None
    ann_path = _ANNOTATIONS_DIR / f"{video_id}.json"
    if ann_path.exists():
        gt = SegmentList.from_json(ann_path.read_text(encoding="utf-8"))
        results_dict = compare_systems(gt, {seg_list.source: seg_list})
        metrics = results_dict.get(seg_list.source)

    return templates.TemplateResponse(
        request=request,
        name="_timeline.html",
        context={"job_id": job_id, "seg_list": seg_list, "metrics": metrics, "track": track},
    )


@router.get("/video/{job_id}")
async def video(request: Request, job_id: str) -> Response:
    if not ids.is_registered(job_id):
        return Response(status_code=404, content="未登録の job_id です。")

    video_path = ids.get_video_path(job_id)
    if not video_path or not video_path.exists():
        return Response(status_code=404, content="動画ファイルが見つかりません。")

    return await stream_video(request, video_path)
