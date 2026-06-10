# tests/pipeline/test_propose_labels.py
import json
from unittest.mock import MagicMock, patch
import pytest


def _mock_gemini_labels(labels: list[str]) -> MagicMock:
    m = MagicMock()
    m.text = json.dumps(labels)
    return m


@patch("src.pipeline.propose_labels.genai")
def test_propose_labels_returns_list(mock_genai, synthetic_video_path):
    from src.pipeline.propose_labels import propose_labels
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_labels(
        ["ネジ締め", "部品取り出し", "検査"]
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        result = propose_labels(synthetic_video_path)
    assert isinstance(result, list)
    assert "ネジ締め" in result


@patch("src.pipeline.propose_labels.genai")
def test_propose_labels_respects_max_labels(mock_genai, synthetic_video_path):
    from src.pipeline.propose_labels import propose_labels
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _mock_gemini_labels(
        [f"label{i}" for i in range(20)]
    )
    with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        result = propose_labels(synthetic_video_path, max_labels=5)
    assert len(result) <= 5


@patch("src.pipeline.propose_labels.genai")
def test_propose_labels_returns_empty_on_failure(mock_genai, synthetic_video_path):
    from src.pipeline.propose_labels import propose_labels
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.side_effect = Exception("API error")
    with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        result = propose_labels(synthetic_video_path)
    assert result == []


@patch("src.pipeline.propose_labels.genai")
def test_propose_labels_no_json_returns_empty(mock_genai, synthetic_video_path):
    from src.pipeline.propose_labels import propose_labels
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    m = MagicMock()
    m.text = "Sorry, I cannot analyze this."
    mock_client.models.generate_content.return_value = m
    with patch.dict("os.environ", {"GEMINI_API_KEY": "k"}):
        result = propose_labels(synthetic_video_path)
    assert result == []
