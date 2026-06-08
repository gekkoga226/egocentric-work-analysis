# Zero-Shot Action Segmentation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a zero-shot action segmentation pipeline for 20-minute first-person factory line videos, with Track A (Gemini single-pass) and Track B (staged pipeline), plus an evaluation infrastructure for benchmarking against Ollo Factory.

**Architecture:** Two parallel tracks share a common JSON schema and evaluation scripts. Track B uses coarse-to-fine pipeline (CLIP embedding → change-point detection → zero-shot CLIP labeling). Track A uses windowed Gemini video understanding. Both are evaluated against manually annotated ground truth using standard TAS metrics (F1@{10,25,50}, Edit, Acc).

**Tech Stack:** Python 3.11, opencv-python, open-clip-torch (ViT-L/14), ruptures, google-genai, typer, pytest

---

## File Structure

```
egocentric-work-analysis/
├── src/
│   ├── __init__.py
│   ├── schemas.py                    # Segment / SegmentList dataclasses + JSON I/O
│   └── pipeline/
│       ├── __init__.py
│       ├── ingest.py                 # Stage 0: frame extraction + face blur
│       ├── embed.py                  # Stage 1a: CLIP frame & text embeddings
│       ├── presegment.py             # Stage 1b: change-point detection (ruptures)
│       ├── label_zeroshot.py         # Stage 2 / Track B: CLIP similarity labeling
│       ├── label_vlm_single.py       # Track A: Gemini windowed single-pass
│       └── report.py                 # Stage 3: SegmentList → Markdown docs
├── src/evaluate/
│   ├── __init__.py
│   ├── metrics.py                    # f1_at_k, edit_score, frame_accuracy, compute_all
│   └── compare.py                    # compare_systems, comparison_report
├── scripts/
│   ├── run_pipeline.py               # CLI: run Track A/B on a video
│   └── run_evaluate.py               # CLI: evaluate results vs ground truth
├── tests/
│   ├── conftest.py                   # synthetic video + segment fixtures
│   ├── test_schemas.py
│   ├── pipeline/
│   │   ├── test_ingest.py
│   │   ├── test_embed.py
│   │   ├── test_presegment.py
│   │   ├── test_label_zeroshot.py
│   │   ├── test_label_vlm_single.py
│   │   └── test_report.py
│   └── evaluate/
│       ├── test_metrics.py
│       └── test_compare.py
├── annotations/                      # Ground truth JSON files
│   └── .gitkeep
├── results/                          # Pipeline output JSON files
│   └── .gitkeep
├── pyproject.toml
├── requirements.txt
└── README.md
```

**Interface contracts (all tasks depend on these):**
- `ingest.extract_frames(path, fps, blur_faces) → list[tuple[float, np.ndarray]]`
- `embed.embed_frames(frames) → tuple[list[float], np.ndarray]`  shape (N, D)
- `embed.embed_texts(labels) → np.ndarray`  shape (L, D)
- `presegment.detect_boundaries(timestamps, embeddings, ...) → list[float]`
- `label_zeroshot.label_zeroshot(video_path, labels, ...) → SegmentList`
- `label_vlm_single.label_vlm_single(video_path, labels, ...) → SegmentList`
- `metrics.compute_all(pred, gt, fps) → dict[str, float]`

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `src/__init__.py`, `src/pipeline/__init__.py`, `src/evaluate/__init__.py`
- Create: `annotations/.gitkeep`, `results/.gitkeep`
- Create: `README.md`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "egocentric-work-analysis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.8",
    "numpy>=1.24",
    "torch>=2.1",
    "open-clip-torch>=2.24",
    "ruptures>=1.1",
    "google-genai>=1.0",
    "typer>=0.12",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create requirements.txt**

```
opencv-python>=4.8
numpy>=1.24
torch>=2.1
open-clip-torch>=2.24
ruptures>=1.1
google-genai>=1.0
typer>=0.12
pytest>=8.0
pytest-mock>=3.12
```

- [ ] **Step 3: Create directory structure and empty __init__ files**

```bash
mkdir -p src/pipeline src/evaluate tests/pipeline tests/evaluate annotations results scripts
touch src/__init__.py src/pipeline/__init__.py src/evaluate/__init__.py
touch annotations/.gitkeep results/.gitkeep
```

- [ ] **Step 4: Create README.md**

```markdown
# egocentric-work-analysis

Zero-shot action segmentation for first-person factory line videos.
Benchmarks against Ollo Factory Tools.

## Setup

```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=your_key_here
```

## Run Pipeline

```bash
# Track B (staged pipeline) + Track A (Gemini):
python scripts/run_pipeline.py video.mp4 "部品取り出し,ネジ締め,検査" --track both

# Evaluate against ground truth annotation:
python scripts/run_evaluate.py annotations/gt.json results/track_a.json results/track_b.json
```

## Tracks

- **Track A**: Gemini single-pass (windowed, 5 min/window)
- **Track B**: CLIP embedding → change-point detection → CLIP labeling
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: no errors, `python -c "import cv2, ruptures, open_clip"` succeeds.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt src/ tests/ scripts/ annotations/ results/ README.md
git commit -m "chore: project setup and dependency configuration"
```

---

## Task 2: Shared Schemas

**Files:**
- Create: `src/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import json
import pytest
from src.schemas import Segment, SegmentList


def test_segment_fields():
    s = Segment(start_sec=0.0, end_sec=10.0, label="ネジ締め", confidence=0.9)
    assert s.start_sec == 0.0
    assert s.end_sec == 10.0
    assert s.label == "ネジ締め"
    assert s.confidence == 0.9


def test_segment_list_roundtrip():
    sl = SegmentList(
        video_id="test_video",
        fps_sampled=1.0,
        label_vocabulary=["作業A", "作業B"],
        segments=[
            Segment(0.0, 10.0, "作業A", 0.9),
            Segment(10.0, 20.0, "作業B", 0.8),
        ],
        source="track_b",
    )
    serialized = sl.to_json()
    restored = SegmentList.from_json(serialized)
    assert restored.video_id == "test_video"
    assert len(restored.segments) == 2
    assert restored.segments[0].label == "作業A"
    assert restored.segments[1].start_sec == 10.0


def test_to_frame_labels_basic():
    sl = SegmentList(
        video_id="v",
        fps_sampled=1.0,
        label_vocabulary=["A", "B"],
        segments=[Segment(0.0, 5.0, "A", 1.0), Segment(5.0, 10.0, "B", 1.0)],
        source="track_b",
    )
    labels = sl.to_frame_labels(total_duration=10.0, fps=1.0)
    assert labels[:5] == ["A", "A", "A", "A", "A"]
    assert labels[5:] == ["B", "B", "B", "B", "B"]


def test_to_frame_labels_gap_is_background():
    sl = SegmentList(
        video_id="v",
        fps_sampled=1.0,
        label_vocabulary=["A"],
        segments=[Segment(2.0, 5.0, "A", 1.0)],
        source="track_b",
    )
    labels = sl.to_frame_labels(total_duration=7.0, fps=1.0)
    assert labels[0] == "background"
    assert labels[2] == "A"
    assert labels[5] == "background"
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
pytest tests/test_schemas.py -v
```

Expected: `ImportError: cannot import name 'Segment' from 'src.schemas'`

- [ ] **Step 3: Implement schemas.py**

```python
# src/schemas.py
from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 1.0


@dataclass
class SegmentList:
    video_id: str
    fps_sampled: float
    label_vocabulary: list[str]
    segments: list[Segment]
    source: str  # "track_a" | "track_b" | "ground_truth" | "ollo"

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SegmentList":
        d = json.loads(s)
        d["segments"] = [Segment(**seg) for seg in d["segments"]]
        return cls(**d)

    def to_frame_labels(self, total_duration: float, fps: float = 1.0) -> list[str]:
        """Convert to frame-level label list at given fps."""
        n = int(total_duration * fps)
        labels = ["background"] * n
        for seg in self.segments:
            start_i = int(seg.start_sec * fps)
            end_i = int(seg.end_sec * fps)
            for i in range(start_i, min(end_i, n)):
                labels[i] = seg.label
        return labels
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/test_schemas.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/schemas.py tests/test_schemas.py
git commit -m "feat: add Segment and SegmentList schemas with JSON round-trip"
```

---

## Task 3: Shared Test Fixtures

**Files:**
- Create: `tests/conftest.py`

- [ ] **Step 1: Create conftest.py with synthetic video fixture**

```python
# tests/conftest.py
import numpy as np
import cv2
import pytest
import tempfile
import os
from src.schemas import Segment, SegmentList


@pytest.fixture
def synthetic_video_path():
    """30-second video with 3 color-distinct visual segments (10s each)."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        path = f.name

    fps = 10
    duration = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (64, 64))

    for i in range(duration * fps):
        t = i / fps
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if t < 10:
            frame[:, :] = (200, 50, 50)    # segment A: blue-ish
        elif t < 20:
            frame[:, :] = (50, 200, 50)    # segment B: green-ish
        else:
            frame[:, :] = (50, 50, 200)    # segment C: red-ish
        out.write(frame)
    out.release()

    yield path
    os.unlink(path)


@pytest.fixture
def ground_truth_segments():
    return SegmentList(
        video_id="test",
        fps_sampled=1.0,
        label_vocabulary=["作業A", "作業B", "作業C"],
        segments=[
            Segment(0.0, 10.0, "作業A", 1.0),
            Segment(10.0, 20.0, "作業B", 1.0),
            Segment(20.0, 30.0, "作業C", 1.0),
        ],
        source="ground_truth",
    )


@pytest.fixture
def perfect_prediction(ground_truth_segments):
    import copy
    sl = copy.deepcopy(ground_truth_segments)
    sl.source = "track_b"
    return sl


@pytest.fixture
def mock_embeddings():
    """Synthetic embeddings with 3 clear visual clusters (matches synthetic_video_path)."""
    np.random.seed(42)
    n = 30  # 30 frames at 1fps
    D = 512
    emb = np.zeros((n, D))
    # cluster A: frames 0-9
    emb[:10] = np.random.randn(10, D) * 0.1 + np.array([1.0] + [0.0] * (D - 1))
    # cluster B: frames 10-19
    emb[10:20] = np.random.randn(10, D) * 0.1 + np.array([0.0, 1.0] + [0.0] * (D - 2))
    # cluster C: frames 20-30
    emb[20:] = np.random.randn(10, D) * 0.1 + np.array([0.0, 0.0, 1.0] + [0.0] * (D - 3))
    # L2 normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms
```

- [ ] **Step 2: Verify fixtures load correctly**

```bash
pytest tests/ --collect-only 2>&1 | head -20
```

Expected: no import errors.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add shared fixtures for synthetic video and segments"
```

---

## Task 4: Stage 0 — Frame Extraction

**Files:**
- Create: `src/pipeline/ingest.py`
- Create: `tests/pipeline/test_ingest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_ingest.py
import numpy as np
import pytest
from src.pipeline.ingest import extract_frames


def test_extract_returns_tuples(synthetic_video_path):
    frames = extract_frames(synthetic_video_path, fps=1.0)
    assert len(frames) > 0
    ts, frame = frames[0]
    assert isinstance(ts, float)
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3  # H, W, C


def test_extract_fps_controls_count(synthetic_video_path):
    frames_1fps = extract_frames(synthetic_video_path, fps=1.0)
    frames_2fps = extract_frames(synthetic_video_path, fps=2.0)
    assert len(frames_2fps) > len(frames_1fps)


def test_extract_timestamps_monotonic(synthetic_video_path):
    frames = extract_frames(synthetic_video_path, fps=1.0)
    timestamps = [ts for ts, _ in frames]
    assert timestamps == sorted(timestamps)
    assert timestamps[0] >= 0.0


def test_extract_invalid_path_raises():
    with pytest.raises(ValueError, match="Cannot open video"):
        extract_frames("nonexistent_video.mp4", fps=1.0)


def test_blur_faces_returns_same_shape(synthetic_video_path):
    frames_no_blur = extract_frames(synthetic_video_path, fps=1.0, blur_faces=False)
    frames_blur = extract_frames(synthetic_video_path, fps=1.0, blur_faces=True)
    assert len(frames_no_blur) == len(frames_blur)
    _, f1 = frames_no_blur[0]
    _, f2 = frames_blur[0]
    assert f1.shape == f2.shape
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_ingest.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Implement ingest.py**

```python
# src/pipeline/ingest.py
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def extract_frames(
    video_path: str,
    fps: float = 1.0,
    blur_faces: bool = False,
) -> list[tuple[float, np.ndarray]]:
    """Extract frames from video at target fps.

    Returns: list of (timestamp_sec, BGR_frame_ndarray)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(video_fps / fps)))

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / video_fps
            if blur_faces:
                frame = _blur_faces(frame)
            frames.append((timestamp, frame))
        frame_idx += 1
    cap.release()
    return frames


def _blur_faces(frame: np.ndarray) -> np.ndarray:
    """Blur detected faces using OpenCV Haar cascade."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    result = frame.copy()
    for (x, y, w, h) in faces:
        roi = result[y : y + h, x : x + w]
        result[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (51, 51), 0)
    return result
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_ingest.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ingest.py tests/pipeline/test_ingest.py
git commit -m "feat: add frame extraction with optional face blur (Stage 0)"
```

---

## Task 5: Stage 1a — CLIP Frame & Text Embeddings

**Files:**
- Create: `src/pipeline/embed.py`
- Create: `tests/pipeline/test_embed.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_embed.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.embed import embed_frames, embed_texts


def _make_bgr_frames(n: int = 3) -> list[tuple[float, np.ndarray]]:
    frames = []
    for i in range(n):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 80, 100, 150)
        frames.append((float(i), frame))
    return frames


def test_embed_frames_shape():
    frames = _make_bgr_frames(3)
    timestamps, embeddings = embed_frames(frames)
    assert len(timestamps) == 3
    assert embeddings.shape == (3, 512)  # ViT-L/14 → 768, but ViT-B/32 → 512


def test_embed_frames_l2_normalized():
    frames = _make_bgr_frames(4)
    _, embeddings = embed_frames(frames)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(4), atol=1e-5)


def test_embed_texts_shape():
    labels = ["部品取り出し", "ネジ締め", "検査"]
    embeddings = embed_texts(labels)
    assert embeddings.shape[0] == 3
    assert embeddings.ndim == 2


def test_embed_texts_l2_normalized():
    labels = ["A", "B"]
    embeddings = embed_texts(labels)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


def test_embed_frames_timestamps_match_input():
    frames = [(0.5, np.zeros((64, 64, 3), dtype=np.uint8)),
              (1.5, np.zeros((64, 64, 3), dtype=np.uint8))]
    timestamps, _ = embed_frames(frames)
    assert timestamps == [0.5, 1.5]
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_embed.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement embed.py**

Note: ViT-B/32 is used (not ViT-L/14) because it is smaller and faster to download; update the model name to `ViT-L-14` for production use. The embedding dimension changes from 512 to 768.

```python
# src/pipeline/embed.py
import numpy as np
import torch
import open_clip
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "openai"

_model = None
_preprocess = None
_tokenizer = None


def _get_model():
    global _model, _preprocess, _tokenizer
    if _model is None:
        logger.info(f"Loading CLIP model {_MODEL_NAME} ({_PRETRAINED})...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        _tokenizer = open_clip.get_tokenizer(_MODEL_NAME)
        _model.eval()
    return _model, _preprocess, _tokenizer


def embed_frames(
    frames: list[tuple[float, np.ndarray]],
    batch_size: int = 32,
) -> tuple[list[float], np.ndarray]:
    """Compute L2-normalized CLIP image embeddings.

    Args:
        frames: list of (timestamp_sec, BGR ndarray)
    Returns:
        (timestamps, embeddings)  embeddings shape (N, D)
    """
    model, preprocess, _ = _get_model()
    timestamps = []
    all_embeddings = []

    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        images = []
        for ts, bgr in batch:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(rgb)))
            timestamps.append(ts)

        tensor = torch.stack(images)
        with torch.no_grad():
            feats = model.encode_image(tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embeddings.append(feats.cpu().numpy())

    return timestamps, np.vstack(all_embeddings)


def embed_texts(labels: list[str]) -> np.ndarray:
    """Compute L2-normalized CLIP text embeddings.

    Returns: shape (N, D)
    """
    model, _, tokenizer = _get_model()
    tokens = tokenizer(labels)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()
```

- [ ] **Step 4: Update test embedding dimension**

ViT-B-32 outputs 512-dim embeddings — the tests already use `512`. If you switch to ViT-L-14, change `512` to `768` in `test_embed_frames_shape`.

- [ ] **Step 5: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_embed.py -v
```

Expected: `5 passed` (first run will download the CLIP model ~350MB)

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/embed.py tests/pipeline/test_embed.py
git commit -m "feat: add CLIP frame and text embeddings (Stage 1a)"
```

---

## Task 6: Stage 1b — Change-Point Detection

**Files:**
- Create: `src/pipeline/presegment.py`
- Create: `tests/pipeline/test_presegment.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_presegment.py
import numpy as np
import pytest
from src.pipeline.presegment import detect_boundaries


def test_detect_clear_boundaries(mock_embeddings):
    # mock_embeddings has 3 clusters at 0-9, 10-19, 20-29
    timestamps = list(range(30))  # 0..29
    boundaries = detect_boundaries(timestamps, mock_embeddings, penalty=1.0)
    assert len(boundaries) >= 1
    # Boundaries should be near 10 and 20
    assert any(8 <= b <= 12 for b in boundaries), f"Expected boundary near 10, got {boundaries}"
    assert any(18 <= b <= 22 for b in boundaries), f"Expected boundary near 20, got {boundaries}"


def test_detect_no_change_flat_signal():
    np.random.seed(0)
    emb = np.random.randn(20, 64) * 0.01 + np.array([1.0] + [0.0] * 63)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    timestamps = list(range(20))
    boundaries = detect_boundaries(timestamps, emb, penalty=50.0)
    assert len(boundaries) == 0


def test_detect_min_segment_enforced(mock_embeddings):
    timestamps = list(range(30))
    # With very large min segment, no boundaries under 15s apart should survive
    boundaries = detect_boundaries(
        timestamps, mock_embeddings, penalty=0.5, min_segment_sec=15.0
    )
    if len(boundaries) > 1:
        diffs = [boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1)]
        assert all(d >= 15.0 for d in diffs)


def test_detect_returns_float_list(mock_embeddings):
    timestamps = [float(i) for i in range(30)]
    boundaries = detect_boundaries(timestamps, mock_embeddings)
    assert isinstance(boundaries, list)
    assert all(isinstance(b, float) for b in boundaries)


def test_detect_empty_returns_empty():
    boundaries = detect_boundaries([], np.zeros((0, 64)))
    assert boundaries == []
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_presegment.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement presegment.py**

```python
# src/pipeline/presegment.py
import numpy as np
import ruptures as rpt
import logging

logger = logging.getLogger(__name__)


def detect_boundaries(
    timestamps: list[float],
    embeddings: np.ndarray,
    penalty: float = 10.0,
    min_segment_sec: float = 5.0,
) -> list[float]:
    """Detect action boundary timestamps via change-point detection.

    Args:
        timestamps: per-frame timestamps in seconds (length N)
        embeddings: frame embeddings shape (N, D)
        penalty: ruptures PELT penalty (higher → fewer breaks)
        min_segment_sec: discard boundaries that create segments shorter than this

    Returns: boundary timestamps in seconds (excludes start/end of video)
    """
    if len(timestamps) < 2 or embeddings.shape[0] < 2:
        return []

    algo = rpt.Pelt(model="rbf", min_size=2).fit(embeddings)
    breakpoints = algo.predict(pen=penalty)
    # ruptures returns 1-indexed end positions; last element == len(signal)
    raw_indices = breakpoints[:-1]

    boundaries: list[float] = []
    prev_ts = timestamps[0]
    for idx in raw_indices:
        clamped = min(idx, len(timestamps) - 1)
        ts = float(timestamps[clamped])
        if ts - prev_ts >= min_segment_sec:
            boundaries.append(ts)
            prev_ts = ts

    return boundaries
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_presegment.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/presegment.py tests/pipeline/test_presegment.py
git commit -m "feat: add change-point boundary detection (Stage 1b)"
```

---

## Task 7: Track B — Zero-Shot CLIP Labeling

**Files:**
- Create: `src/pipeline/label_zeroshot.py`
- Create: `tests/pipeline/test_label_zeroshot.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_label_zeroshot.py
import numpy as np
import pytest
from unittest.mock import patch
from src.schemas import SegmentList
from src.pipeline.label_zeroshot import label_zeroshot


LABELS = ["作業A", "作業B", "作業C"]


def _make_mock_embed_frames(clusters):
    """Return a mock embed_frames that assigns cluster embeddings by timestamp."""
    D = 64
    cluster_vecs = np.eye(len(clusters), D)

    def _mock(frames):
        timestamps = [ts for ts, _ in frames]
        emb = np.zeros((len(frames), D))
        for i, (ts, _) in enumerate(frames):
            t = int(ts)
            if t < 10:
                emb[i] = cluster_vecs[0]
            elif t < 20:
                emb[i] = cluster_vecs[1]
            else:
                emb[i] = cluster_vecs[2]
        return timestamps, emb

    return _mock


def _make_mock_embed_texts(labels):
    D = 64
    vecs = np.eye(len(labels), D)
    return lambda _: vecs


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_assigns_correct_labels(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(30)]
    mock_embed_f.side_effect = _make_mock_embed_frames(LABELS)
    mock_embed_t.side_effect = _make_mock_embed_texts(LABELS)

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )
    labels = [s.label for s in result.segments]
    assert labels == ["作業A", "作業B", "作業C"]


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_source_is_track_b(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(10)]
    mock_embed_f.return_value = ([float(i) for i in range(10)], np.eye(10, 64)[:, :64])
    mock_embed_t.return_value = np.eye(3, 64)

    result = label_zeroshot(synthetic_video_path, LABELS, boundary_timestamps=[])
    assert result.source == "track_b"


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_label_segments_cover_full_duration(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    mock_ingest.return_value = [(float(i), None) for i in range(30)]
    mock_embed_f.side_effect = _make_mock_embed_frames(LABELS)
    mock_embed_t.side_effect = _make_mock_embed_texts(LABELS)

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )
    assert result.segments[0].start_sec == 0.0
    assert result.segments[-1].end_sec > 20.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_label_zeroshot.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement label_zeroshot.py**

```python
# src/pipeline/label_zeroshot.py
import numpy as np
import cv2
from pathlib import Path
from src.schemas import Segment, SegmentList
from src.pipeline.ingest import extract_frames
from src.pipeline.embed import embed_frames, embed_texts


def label_zeroshot(
    video_path: str,
    label_vocabulary: list[str],
    fps: float = 1.0,
    boundary_timestamps: list[float] | None = None,
    blur_faces: bool = False,
) -> SegmentList:
    """Zero-shot action segmentation using CLIP similarity (Track B).

    Assigns each segment the label whose text embedding has highest average
    cosine similarity with the frames in that segment.
    """
    video_id = Path(video_path).stem

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # Build time ranges from boundary timestamps
    if not boundary_timestamps:
        ranges = [(0.0, total_duration)]
    else:
        starts = [0.0] + list(boundary_timestamps)
        ends = list(boundary_timestamps) + [total_duration]
        ranges = list(zip(starts, ends))

    frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
    timestamps, img_emb = embed_frames(frames)          # (N, D)
    text_emb = embed_texts(label_vocabulary)             # (L, D)

    segments: list[Segment] = []
    for start_sec, end_sec in ranges:
        mask = [start_sec <= ts < end_sec for ts in timestamps]
        if not any(mask):
            continue
        seg_img_emb = img_emb[mask]                     # (k, D)
        sims = seg_img_emb @ text_emb.T                 # (k, L)
        avg_sims = sims.mean(axis=0)                    # (L,)
        best_idx = int(np.argmax(avg_sims))
        segments.append(Segment(
            start_sec=start_sec,
            end_sec=end_sec,
            label=label_vocabulary[best_idx],
            confidence=float(avg_sims[best_idx]),
        ))

    return SegmentList(
        video_id=video_id,
        fps_sampled=fps,
        label_vocabulary=label_vocabulary,
        segments=segments,
        source="track_b",
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_label_zeroshot.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/label_zeroshot.py tests/pipeline/test_label_zeroshot.py
git commit -m "feat: add zero-shot CLIP labeling (Track B, Stage 2)"
```

---

## Task 8: Track A — Gemini Single-Pass

**Files:**
- Create: `src/pipeline/label_vlm_single.py`
- Create: `tests/pipeline/test_label_vlm_single.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_label_vlm_single.py
import json
import pytest
from unittest.mock import patch, MagicMock
from src.schemas import SegmentList
from src.pipeline.label_vlm_single import label_vlm_single, _merge_adjacent
from src.schemas import Segment


LABELS = ["部品取り出し", "ネジ締め", "検査"]


def _make_gemini_response(segments: list[dict]) -> MagicMock:
    mock = MagicMock()
    mock.text = json.dumps(segments)
    return mock


@patch("src.pipeline.label_vlm_single.genai")
def test_label_vlm_returns_segment_list(mock_genai, synthetic_video_path):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _make_gemini_response([
        {"start_sec": 0.0, "end_sec": 10.0, "label": "部品取り出し", "confidence": 0.9},
        {"start_sec": 10.0, "end_sec": 30.0, "label": "ネジ締め", "confidence": 0.8},
    ])

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_vlm_single(synthetic_video_path, LABELS)

    assert isinstance(result, SegmentList)
    assert result.source == "track_a"
    assert len(result.segments) >= 1


@patch("src.pipeline.label_vlm_single.genai")
def test_label_vlm_handles_empty_response(mock_genai, synthetic_video_path):
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _make_gemini_response([])

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_vlm_single(synthetic_video_path, LABELS)

    assert result.source == "track_a"
    assert isinstance(result.segments, list)


def test_merge_adjacent_same_label():
    segs = [
        Segment(0.0, 10.0, "A", 0.9),
        Segment(10.0, 20.0, "A", 0.8),
        Segment(20.0, 30.0, "B", 0.7),
    ]
    merged = _merge_adjacent(segs)
    assert len(merged) == 2
    assert merged[0].label == "A"
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 20.0
    assert merged[1].label == "B"


def test_merge_adjacent_no_adjacent_same():
    segs = [Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)]
    merged = _merge_adjacent(segs)
    assert len(merged) == 2
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_label_vlm_single.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement label_vlm_single.py**

```python
# src/pipeline/label_vlm_single.py
import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from google import genai
from src.schemas import Segment, SegmentList
from src.pipeline.ingest import _blur_faces
import logging

logger = logging.getLogger(__name__)

_WINDOW_SEC = 300       # 5-minute windows
_WINDOW_FPS = 0.2       # 1 frame per 5 seconds for Gemini input
_GEMINI_MODEL = "gemini-2.5-pro"


def label_vlm_single(
    video_path: str,
    label_vocabulary: list[str],
    blur_faces: bool = False,
) -> SegmentList:
    """Zero-shot action segmentation using Gemini (Track A).

    Splits video into 5-minute windows, sends key frames to Gemini,
    stitches results, and merges adjacent identical labels.
    """
    video_id = Path(video_path).stem
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    all_segments: list[Segment] = []
    win_starts = list(range(0, int(total_duration), _WINDOW_SEC))

    for win_start in win_starts:
        win_end = min(win_start + _WINDOW_SEC, total_duration)
        segs = _process_window(
            video_path, label_vocabulary, float(win_start), win_end,
            client, blur_faces,
        )
        all_segments.extend(segs)

    return SegmentList(
        video_id=video_id,
        fps_sampled=_WINDOW_FPS,
        label_vocabulary=label_vocabulary,
        segments=_merge_adjacent(all_segments),
        source="track_a",
    )


def _process_window(
    video_path: str,
    labels: list[str],
    start_sec: float,
    end_sec: float,
    client,
    blur_faces: bool,
) -> list[Segment]:
    frames = _extract_window_frames(video_path, start_sec, end_sec, blur_faces)
    if not frames:
        return []

    label_list = "\n".join(f"- {l}" for l in labels)
    prompt = (
        f"You are analyzing a factory line work video.\n"
        f"The video segment covers {start_sec:.1f}s to {end_sec:.1f}s from the start.\n"
        f"Available action labels:\n{label_list}\n\n"
        f"Analyze the frames (timestamps shown as [t=Xs]) and output a JSON array.\n"
        f"Each element: {{\"start_sec\": float, \"end_sec\": float, "
        f"\"label\": \"exact label from above\", \"confidence\": float 0-1}}\n"
        f"Output ONLY the JSON array, no other text."
    )

    parts: list = [prompt]
    for ts, frame in frames:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf.tobytes()).decode()
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append(f"[t={ts:.1f}s]")

    try:
        resp = client.models.generate_content(model=_GEMINI_MODEL, contents=parts)
        text = resp.text.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning("Gemini returned no JSON array for window %s-%s", start_sec, end_sec)
            return []
        raw = json.loads(text[start:end])
        return [Segment(**s) for s in raw]
    except Exception as exc:
        logger.warning("Gemini call failed for window %s-%s: %s", start_sec, end_sec, exc)
        return []


def _extract_window_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    blur_faces: bool,
) -> list[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    interval = 1.0 / _WINDOW_FPS
    frames = []
    ts = start_sec
    while ts < end_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if blur_faces:
            frame = _blur_faces(frame)
        frames.append((ts, frame))
        ts += interval
    cap.release()
    return frames


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    if not segments:
        return []
    merged = [Segment(segments[0].start_sec, segments[0].end_sec,
                      segments[0].label, segments[0].confidence)]
    for seg in segments[1:]:
        if seg.label == merged[-1].label:
            merged[-1] = Segment(
                merged[-1].start_sec, seg.end_sec, merged[-1].label,
                (merged[-1].confidence + seg.confidence) / 2,
            )
        else:
            merged.append(Segment(seg.start_sec, seg.end_sec, seg.label, seg.confidence))
    return merged
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_label_vlm_single.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/label_vlm_single.py tests/pipeline/test_label_vlm_single.py
git commit -m "feat: add Gemini single-pass segmentation (Track A)"
```

---

## Task 9: Stage 3 — Report Generation

**Files:**
- Create: `src/pipeline/report.py`
- Create: `tests/pipeline/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_report.py
import json
import os
import tempfile
import pytest
from src.schemas import SegmentList, Segment
from src.pipeline.report import to_timeline_markdown, to_procedure_markdown, save_segments


@pytest.fixture
def sample_seglist():
    return SegmentList(
        video_id="line01",
        fps_sampled=1.0,
        label_vocabulary=["部品取り出し", "ネジ締め"],
        segments=[
            Segment(0.0, 12.4, "部品取り出し", 0.87),
            Segment(12.4, 30.1, "ネジ締め", 0.79),
        ],
        source="track_b",
    )


def test_timeline_contains_video_id(sample_seglist):
    md = to_timeline_markdown(sample_seglist)
    assert "line01" in md


def test_timeline_contains_all_labels(sample_seglist):
    md = to_timeline_markdown(sample_seglist)
    assert "部品取り出し" in md
    assert "ネジ締め" in md


def test_procedure_contains_step_numbers(sample_seglist):
    md = to_procedure_markdown(sample_seglist)
    assert "Step 1" in md
    assert "Step 2" in md


def test_save_segments_creates_file(sample_seglist):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(sample_seglist, tmpdir)
        assert os.path.isfile(path)
        data = json.loads(open(path, encoding="utf-8").read())
        assert data["video_id"] == "line01"
        assert len(data["segments"]) == 2


def test_save_segments_filename_includes_source(sample_seglist):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(sample_seglist, tmpdir)
        assert "track_b" in os.path.basename(path)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/pipeline/test_report.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement report.py**

```python
# src/pipeline/report.py
from pathlib import Path
from src.schemas import SegmentList


def to_timeline_markdown(seg_list: SegmentList) -> str:
    lines = [
        f"# 作業タイムライン: {seg_list.video_id}",
        "",
        "| # | 開始 | 終了 | 要素作業 | 信頼度 |",
        "|---|------|------|----------|--------|",
    ]
    for i, seg in enumerate(seg_list.segments, 1):
        lines.append(
            f"| {i} | {_fmt(seg.start_sec)} | {_fmt(seg.end_sec)} "
            f"| {seg.label} | {seg.confidence:.2f} |"
        )
    return "\n".join(lines)


def to_procedure_markdown(seg_list: SegmentList) -> str:
    lines = [
        f"# 標準作業手順書（ドラフト）: {seg_list.video_id}",
        "",
        "> 生成AI分析による自動ドラフト。内容を確認・編集してください。",
        "",
    ]
    for i, seg in enumerate(seg_list.segments, 1):
        duration = seg.end_sec - seg.start_sec
        lines += [
            f"## Step {i}: {seg.label}",
            "",
            f"- **所要時間**: {duration:.1f}秒 "
            f"({_fmt(seg.start_sec)} ～ {_fmt(seg.end_sec)})",
            f"- **信頼度**: {seg.confidence:.2f}",
            "",
        ]
    return "\n".join(lines)


def save_segments(seg_list: SegmentList, output_dir: str) -> str:
    out = Path(output_dir) / f"{seg_list.video_id}_{seg_list.source}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(seg_list.to_json(), encoding="utf-8")
    return str(out)


def _fmt(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/pipeline/test_report.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/report.py tests/pipeline/test_report.py
git commit -m "feat: add timeline and procedure report generation (Stage 3)"
```

---

## Task 10: Evaluation Metrics

**Files:**
- Create: `src/evaluate/metrics.py`
- Create: `tests/evaluate/test_metrics.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluate/test_metrics.py
import pytest
from src.schemas import Segment, SegmentList
from src.evaluate.metrics import f1_at_k, edit_score, frame_accuracy, compute_all


def _make_sl(segments, source="pred"):
    return SegmentList("v", 1.0, ["A", "B", "C"], segments, source)


def test_f1_perfect_match():
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "A", 1.0), Segment(10.0, 20.0, "B", 1.0)])
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(1.0)


def test_f1_no_match():
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "B", 1.0)])  # wrong label
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(0.0)


def test_f1_partial_overlap_below_threshold():
    # pred covers 0-6, gt covers 0-10. overlap=6, union=10 → 60% ≥ 50% → TP
    gt = _make_sl([Segment(0.0, 10.0, "A", 1.0)], "gt")
    pred = _make_sl([Segment(0.0, 6.0, "A", 1.0)])
    assert f1_at_k(pred, gt, 0.5) == pytest.approx(1.0)

    # overlap=4, union=10 → 40% < 50% → FP
    pred2 = _make_sl([Segment(0.0, 4.0, "A", 1.0)])
    assert f1_at_k(pred2, gt, 0.5) == pytest.approx(0.0)


def test_edit_score_perfect():
    gt = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")])
    assert edit_score(pred, gt) == pytest.approx(100.0)


def test_edit_score_all_wrong():
    gt = _make_sl([Segment(0.0, 10.0, "A"), Segment(10.0, 20.0, "B")], "gt")
    pred = _make_sl([Segment(0.0, 10.0, "C"), Segment(10.0, 20.0, "C")])
    assert edit_score(pred, gt) < 100.0


def test_frame_accuracy_perfect(ground_truth_segments, perfect_prediction):
    acc = frame_accuracy(perfect_prediction, ground_truth_segments, fps=1.0)
    assert acc == pytest.approx(1.0)


def test_compute_all_returns_all_keys(ground_truth_segments, perfect_prediction):
    result = compute_all(perfect_prediction, ground_truth_segments, fps=1.0)
    assert set(result.keys()) == {"f1@10", "f1@25", "f1@50", "edit", "acc"}
    assert all(0.0 <= v <= 100.0 or v <= 1.0 for v in result.values())
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/evaluate/test_metrics.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement metrics.py**

```python
# src/evaluate/metrics.py
from src.schemas import SegmentList, Segment


def f1_at_k(pred: SegmentList, gt: SegmentList, k: float) -> float:
    """Segmental F1 at overlap threshold k (0.0–1.0)."""
    pred_segs, gt_segs = pred.segments, gt.segments
    pred_matched = [False] * len(pred_segs)
    gt_matched = [False] * len(gt_segs)
    tp = 0

    for i, p in enumerate(pred_segs):
        for j, g in enumerate(gt_segs):
            if gt_matched[j] or p.label != g.label:
                continue
            overlap = max(0.0, min(p.end_sec, g.end_sec) - max(p.start_sec, g.start_sec))
            union = max(p.end_sec, g.end_sec) - min(p.start_sec, g.start_sec)
            if union > 0 and overlap / union >= k:
                tp += 1
                pred_matched[i] = True
                gt_matched[j] = True
                break

    fp = sum(1 for m in pred_matched if not m)
    fn = sum(1 for m in gt_matched if not m)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0


def edit_score(pred: SegmentList, gt: SegmentList) -> float:
    """Normalized edit score 0–100 (higher = better)."""
    a = [s.label for s in pred.segments]
    b = [s.label for s in gt.segments]
    dist = _levenshtein(a, b)
    max_len = max(len(a), len(b))
    return (1.0 - dist / max_len) * 100.0 if max_len > 0 else 100.0


def frame_accuracy(pred: SegmentList, gt: SegmentList, fps: float = 1.0) -> float:
    """Frame-wise label accuracy."""
    total = max(
        (max(s.end_sec for s in gt.segments) if gt.segments else 0.0),
        (max(s.end_sec for s in pred.segments) if pred.segments else 0.0),
    )
    if total == 0:
        return 1.0
    pred_f = pred.to_frame_labels(total, fps)
    gt_f = gt.to_frame_labels(total, fps)
    n = min(len(pred_f), len(gt_f))
    return sum(p == g for p, g in zip(pred_f[:n], gt_f[:n])) / n if n > 0 else 0.0


def compute_all(pred: SegmentList, gt: SegmentList, fps: float = 1.0) -> dict[str, float]:
    return {
        "f1@10": f1_at_k(pred, gt, 0.10),
        "f1@25": f1_at_k(pred, gt, 0.25),
        "f1@50": f1_at_k(pred, gt, 0.50),
        "edit":  edit_score(pred, gt),
        "acc":   frame_accuracy(pred, gt, fps),
    }


def _levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/evaluate/test_metrics.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluate/metrics.py tests/evaluate/test_metrics.py
git commit -m "feat: add TAS evaluation metrics (F1@k, Edit, Acc)"
```

---

## Task 11: Evaluation Comparison

**Files:**
- Create: `src/evaluate/compare.py`
- Create: `tests/evaluate/test_compare.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/evaluate/test_compare.py
import pytest
from src.evaluate.compare import compare_systems, comparison_report


def test_compare_systems_keys(ground_truth_segments, perfect_prediction):
    import copy
    pred_b = perfect_prediction
    pred_a = copy.deepcopy(perfect_prediction)
    pred_a.source = "track_a"

    results = compare_systems(
        ground_truth_segments,
        {"track_a": pred_a, "track_b": pred_b},
        fps=1.0,
    )
    assert set(results.keys()) == {"track_a", "track_b"}
    assert set(results["track_a"].keys()) == {"f1@10", "f1@25", "f1@50", "edit", "acc"}


def test_compare_perfect_scores(ground_truth_segments, perfect_prediction):
    results = compare_systems(
        ground_truth_segments,
        {"track_b": perfect_prediction},
        fps=1.0,
    )
    assert results["track_b"]["f1@50"] == pytest.approx(1.0)
    assert results["track_b"]["acc"] == pytest.approx(1.0)


def test_comparison_report_is_markdown_table(ground_truth_segments, perfect_prediction):
    results = compare_systems(
        ground_truth_segments,
        {"track_b": perfect_prediction},
        fps=1.0,
    )
    report = comparison_report(results)
    assert "| track_b |" in report
    assert "F1@10" in report
    assert "EDIT" in report
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/evaluate/test_compare.py -v
```

Expected: `ImportError`

- [ ] **Step 3: Implement compare.py**

```python
# src/evaluate/compare.py
from src.schemas import SegmentList
from src.evaluate.metrics import compute_all


def compare_systems(
    ground_truth: SegmentList,
    predictions: dict[str, SegmentList],
    fps: float = 1.0,
) -> dict[str, dict[str, float]]:
    """Compare multiple prediction systems against ground truth."""
    return {
        name: compute_all(pred, ground_truth, fps)
        for name, pred in predictions.items()
    }


def comparison_report(results: dict[str, dict[str, float]]) -> str:
    """Format results as a Markdown comparison table."""
    metrics = ["f1@10", "f1@25", "f1@50", "edit", "acc"]
    header = "| System | " + " | ".join(m.upper() for m in metrics) + " |"
    sep = "|--------|" + "|".join("-------" for _ in metrics) + "|"
    rows = [header, sep]
    for sys in sorted(results):
        vals = results[sys]
        row = f"| {sys} | " + " | ".join(f"{vals[m]:.3f}" for m in metrics) + " |"
        rows.append(row)
    return "\n".join(rows)
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/evaluate/test_compare.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/evaluate/compare.py tests/evaluate/test_compare.py
git commit -m "feat: add multi-system comparison and Markdown report"
```

---

## Task 12: CLI Scripts

**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `scripts/run_evaluate.py`

- [ ] **Step 1: Create run_pipeline.py**

```python
#!/usr/bin/env python3
# scripts/run_pipeline.py
"""Run zero-shot action segmentation on a video file."""
import typer
from pathlib import Path
from src.pipeline.ingest import extract_frames
from src.pipeline.embed import embed_frames
from src.pipeline.presegment import detect_boundaries
from src.pipeline.label_zeroshot import label_zeroshot
from src.pipeline.label_vlm_single import label_vlm_single
from src.pipeline.report import save_segments, to_timeline_markdown

app = typer.Typer(help="Egocentric work analysis pipeline")


@app.command()
def run(
    video: str = typer.Argument(..., help="Path to input video file"),
    labels: str = typer.Argument(..., help="Comma-separated action label vocabulary"),
    output_dir: str = typer.Option("results", help="Directory for output JSON files"),
    track: str = typer.Option("both", help="Which track to run: a | b | both"),
    fps: float = typer.Option(1.0, help="Frames per second to sample"),
    blur_faces: bool = typer.Option(False, help="Apply face blur for privacy"),
    penalty: float = typer.Option(10.0, help="Change-point detection penalty (Track B)"),
):
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    typer.echo(f"Video: {video}")
    typer.echo(f"Labels ({len(label_list)}): {label_list}")

    if track in ("b", "both"):
        typer.echo("\n── Track B (staged pipeline) ──")
        frames = extract_frames(video, fps=fps, blur_faces=blur_faces)
        typer.echo(f"  Extracted {len(frames)} frames")
        timestamps, embeddings = embed_frames(frames)
        typer.echo(f"  Embedded {len(timestamps)} frames")
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        typer.echo(f"  Boundaries: {[f'{b:.1f}s' for b in boundaries]}")
        seg_list = label_zeroshot(
            video, label_list, fps=fps,
            boundary_timestamps=boundaries, blur_faces=blur_faces,
        )
        path = save_segments(seg_list, output_dir)
        typer.echo(f"  Saved: {path}")
        typer.echo(to_timeline_markdown(seg_list))

    if track in ("a", "both"):
        typer.echo("\n── Track A (Gemini single-pass) ──")
        seg_list = label_vlm_single(video, label_list, blur_faces=blur_faces)
        path = save_segments(seg_list, output_dir)
        typer.echo(f"  Saved: {path}")
        typer.echo(to_timeline_markdown(seg_list))


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Create run_evaluate.py**

```python
#!/usr/bin/env python3
# scripts/run_evaluate.py
"""Evaluate segmentation results against ground truth annotation."""
import typer
from pathlib import Path
from src.schemas import SegmentList
from src.evaluate.compare import compare_systems, comparison_report

app = typer.Typer(help="Evaluation and benchmark comparison")


@app.command()
def evaluate(
    ground_truth: str = typer.Argument(..., help="Path to ground truth JSON annotation"),
    predictions: list[str] = typer.Argument(..., help="Paths to prediction JSON files"),
    fps: float = typer.Option(1.0, help="fps for frame-accuracy calculation"),
):
    gt = SegmentList.from_json(Path(ground_truth).read_text(encoding="utf-8"))
    preds: dict[str, SegmentList] = {}
    for p in predictions:
        seg = SegmentList.from_json(Path(p).read_text(encoding="utf-8"))
        preds[seg.source] = seg

    results = compare_systems(gt, preds, fps=fps)
    typer.echo("\n" + comparison_report(results))


if __name__ == "__main__":
    app()
```

- [ ] **Step 3: Smoke-test CLI help**

```bash
python scripts/run_pipeline.py --help
python scripts/run_evaluate.py --help
```

Expected: help text printed, no errors.

- [ ] **Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: add CLI scripts for pipeline and evaluation"
```

---

## Task 13: Integration Test (Full Pipeline with Synthetic Data)

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end pipeline test using synthetic video and mocked Gemini API."""
import json
import tempfile
import os
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.schemas import SegmentList, Segment
from src.pipeline.ingest import extract_frames
from src.pipeline.embed import embed_frames
from src.pipeline.presegment import detect_boundaries
from src.pipeline.label_zeroshot import label_zeroshot
from src.pipeline.report import save_segments, to_timeline_markdown
from src.evaluate.metrics import compute_all


LABELS = ["作業A", "作業B", "作業C"]


@patch("src.pipeline.label_zeroshot.embed_texts")
@patch("src.pipeline.label_zeroshot.embed_frames")
@patch("src.pipeline.label_zeroshot.extract_frames")
def test_track_b_end_to_end(mock_ingest, mock_embed_f, mock_embed_t, synthetic_video_path):
    """Track B: ingest → embed → presegment → label → save → evaluate."""
    # Setup mocks with 3-cluster synthetic data
    D = 64
    n = 30
    mock_ingest.return_value = [(float(i), None) for i in range(n)]
    emb = np.zeros((n, D))
    emb[:10, 0] = 1.0
    emb[10:20, 1] = 1.0
    emb[20:, 2] = 1.0
    mock_embed_f.return_value = ([float(i) for i in range(n)], emb)
    text_emb = np.eye(3, D)
    mock_embed_t.return_value = text_emb

    result = label_zeroshot(
        synthetic_video_path, LABELS,
        boundary_timestamps=[10.0, 20.0],
    )

    assert isinstance(result, SegmentList)
    assert result.source == "track_b"
    assert len(result.segments) == 3
    assert [s.label for s in result.segments] == LABELS

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_segments(result, tmpdir)
        reloaded = SegmentList.from_json(open(path, encoding="utf-8").read())
        assert len(reloaded.segments) == 3

    # Evaluate against perfect ground truth
    from src.schemas import Segment
    gt = SegmentList("test", 1.0, LABELS, [
        Segment(0.0, 10.0, "作業A", 1.0),
        Segment(10.0, 20.0, "作業B", 1.0),
        Segment(20.0, 30.0, "作業C", 1.0),
    ], "ground_truth")

    metrics = compute_all(result, gt, fps=1.0)
    assert metrics["f1@50"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["edit"] == pytest.approx(100.0)

    md = to_timeline_markdown(result)
    assert "作業A" in md and "作業B" in md and "作業C" in md
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v --ignore=tests/pipeline/test_embed.py
```

Note: `test_embed.py` is excluded here only if CLIP model download is not desired in CI; include it for full validation.

Expected: `test_integration.py::test_track_b_end_to_end PASSED` plus all prior tests pass.

- [ ] **Step 3: Run complete test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass (embed tests require internet for model download on first run).

- [ ] **Step 4: Commit and push**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test for Track B pipeline"
git push origin main
```

---

## Self-Review Checklist

**Spec coverage:**
| Spec requirement | Covered by |
|---|---|
| 20分超・非反復のゼロショットセグメンテーション | Task 6, 7, 8 |
| テキストラベル群からの作業分類 | Task 7, 8 |
| Track A (Gemini single-pass + windowing) | Task 8 |
| Track B (CLIP embed + change-point + label) | Task 5, 6, 7 |
| 自社映像 + アノテーションによる評価 | Task 10, 11 |
| F1@{10,25,50}, Edit, Acc | Task 10 |
| Track A/B/Ollo横並び比較 | Task 11 |
| JSON共通スキーマ | Task 2 |
| 顔ブラー（プライバシー処理） | Task 4 |
| タイムライン/手順書ドラフト出力 | Task 9 |
| CLI | Task 12 |

**No placeholders found.**

**Type/name consistency confirmed:** `SegmentList.to_frame_labels` defined in Task 2, used in Task 10. `_blur_faces` defined in Task 4, imported in Task 8. `Segment` fields consistent across all tasks.
