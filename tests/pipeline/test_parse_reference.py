# tests/pipeline/test_parse_reference.py
import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_parse_reference_returns_none_for_missing_file():
    from src.pipeline.parse_reference import parse_reference
    result = parse_reference("/nonexistent/file.pdf")
    assert result is None


def test_parse_reference_text_path(tmp_path):
    from src.pipeline.parse_reference import _parse_sync
    with patch("src.pipeline.parse_reference.pdfplumber") as mock_plumber:
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Step 1: 部品取り出し\n" * 20
        mock_pdf.pages = [mock_page]
        mock_plumber.open.return_value = mock_pdf

        result = _parse_sync("/fake/file.pdf", model="gemini-2.5-pro")
    assert result is not None
    assert "部品取り出し" in result


def test_parse_reference_sparse_text_triggers_image_fallback(tmp_path):
    from src.pipeline.parse_reference import _parse_sync
    with patch("src.pipeline.parse_reference.pdfplumber") as mock_plumber, \
         patch("src.pipeline.parse_reference._image_fallback") as mock_fallback:
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = lambda s: s
        mock_pdf.__exit__ = MagicMock(return_value=False)
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "AB"  # too short
        mock_pdf.pages = [mock_page]
        mock_plumber.open.return_value = mock_pdf
        mock_fallback.return_value = "fallback text"

        result = _parse_sync("/fake/file.pdf", model="gemini-2.5-pro")
    mock_fallback.assert_called_once()
    assert result == "fallback text"


def test_parse_reference_page_limit_respected():
    from src.pipeline.parse_reference import MAX_PDF_IMAGE_PAGES, _image_fallback
    with patch("src.pipeline.parse_reference.fitz") as mock_fitz, \
         patch("src.pipeline.parse_reference.genai") as mock_genai, \
         patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        mock_doc = MagicMock()
        mock_doc.__len__ = lambda s: MAX_PDF_IMAGE_PAGES + 5  # 15 pages
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"fake_jpeg"
        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix
        mock_getitem = MagicMock(side_effect=lambda i: mock_page)
        mock_doc.__getitem__ = mock_getitem
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix.return_value = MagicMock()

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = "context text"
        mock_client.models.generate_content.return_value = mock_resp

        with patch("src.pipeline.parse_reference.Image") as mock_image:
            mock_img = MagicMock()
            mock_image.open.return_value = mock_img
            import io
            mock_img.save = lambda buf, **kw: buf.write(b"compressed")

            _image_fallback("/fake.pdf", model="gemini-2.5-pro")

        # Only MAX_PDF_IMAGE_PAGES pages should be fetched
        assert mock_getitem.call_count <= MAX_PDF_IMAGE_PAGES


def test_parse_reference_returns_none_on_timeout():
    from src.pipeline.parse_reference import parse_reference
    with patch("src.pipeline.parse_reference._parse_sync", side_effect=Exception("fail")):
        result = parse_reference("/fake.pdf")
    assert result is None
