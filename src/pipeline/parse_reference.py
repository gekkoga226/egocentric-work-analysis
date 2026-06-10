# src/pipeline/parse_reference.py
import base64
import io
import logging
import os
import threading
from pathlib import Path
from typing import Optional

# Module-level imports: tests patch src.pipeline.parse_reference.{pdfplumber,fitz,Image,genai}
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from google import genai

logger = logging.getLogger(__name__)

MAX_PDF_IMAGE_PAGES = 10
PDF_IMAGE_DPI = 72
PDF_PARSE_TIMEOUT = 60
_GEMINI_MODEL = "gemini-2.5-pro"


def parse_reference(pdf_path: str, *, model: str = _GEMINI_MODEL) -> Optional[str]:
    """Parse PDF into reference context string. Returns None on any failure."""
    result: list[Optional[str]] = [None]
    exc_holder: list[Optional[Exception]] = [None]

    def _run():
        try:
            result[0] = _parse_sync(pdf_path, model=model)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=PDF_PARSE_TIMEOUT)

    if t.is_alive():
        logger.warning("PDF parse timed out for %s (limit=%ds)", pdf_path, PDF_PARSE_TIMEOUT)
        return None
    if exc_holder[0]:
        logger.warning("PDF parse failed: %s", exc_holder[0])
        return None
    return result[0]


def _parse_sync(pdf_path: str, *, model: str) -> Optional[str]:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            texts = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(texts).strip()
        if len(text) > 100:
            return text[:4000]
    except Exception as exc:
        logger.debug("pdfplumber text extraction failed: %s", exc)

    return _image_fallback(pdf_path, model=model)


def _image_fallback(pdf_path: str, *, model: str) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        n_pages = min(len(doc), MAX_PDF_IMAGE_PAGES)
        if n_pages < len(doc):
            logger.info("PDF has %d pages; using first %d for image fallback", len(doc), n_pages)

        parts: list = [
            "Extract the work procedure and key operations from these work standard document pages. "
            "Output as structured text in Japanese: step names, durations, quality checkpoints."
        ]
        mat = fitz.Matrix(PDF_IMAGE_DPI / 72, PDF_IMAGE_DPI / 72)
        for i in range(n_pages):
            pix = doc[i].get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(buf.getvalue()).decode()}})
            parts.append(f"[Page {i + 1}]")

        if n_pages < len(doc):
            parts.append(f"(Note: only first {n_pages} of {len(doc)} pages shown)")

        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(model=model, contents=parts)
        return resp.text.strip()[:4000]

    except Exception as exc:
        logger.warning("PDF image fallback failed: %s", exc)
        return None
