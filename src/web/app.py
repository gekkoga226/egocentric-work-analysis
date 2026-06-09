"""FastAPI アプリケーション生成・ルーター登録・静的ファイルマウント・起動時プリウォーム。"""
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.web.routes import router

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


def _prewarm_clip() -> None:
    """起動時に CLIP モデルをロードしておく（初回 /analyze で固まるのを防ぐ）。"""
    try:
        from src.pipeline.embed import _get_model
        _get_model()
        logger.info("CLIP model prewarmed.")
    except Exception as exc:
        logger.warning("CLIP prewarm skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリ起動時に CLIP をプリウォームする。"""
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _prewarm_clip)
    yield


app = FastAPI(
    title="Egocentric Work Analysis",
    description="一人称視点映像の作業分析システム（Track A/B）",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)

app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static",
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=True)
