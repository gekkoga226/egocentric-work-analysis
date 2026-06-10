# Ollo Tools Gemini 作業分析 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLIP境界検出 + Gemini精緻化（正味/付随/ムダ分類・作業説明・改善提案）で Ollo Factory Tools の3本柱を再現する。

**Architecture:** Stage 1（CLIP境界）→ Stage 2（`label_gemini`：窓ごとGemini1回で境界見直し＋enrich）→ Stage 3（stitch・aggregate・report）。`TRACK_RUNNERS` レジストリにより将来モデル（TASOT等）への差し替えは関数1本＋1行。

**Tech Stack:** Python 3.11+, FastAPI, htmx, Alpine.js, google-genai, pdfplumber, PyMuPDF (fitz), Pillow, OpenCV, pytest

---

## File Map

| Action | Path | Purpose |
|---|---|---|
| Modify | `pyproject.toml` | add pdfplumber / PyMuPDF / Pillow deps |
| Modify | `src/schemas.py` | Segment += category/description/improvement; Hint dataclass |
| Create | `src/pipeline/aggregate.py` | by_category + by_label stats |
| Modify | `src/pipeline/report.py` | enrich fields in procedure markdown |
| Create | `src/pipeline/label_gemini.py` | Stage 2 CLIP+Gemini track_std (module-level genai import for mockability) |
| Create | `src/pipeline/parse_reference.py` | PDF → reference_context (module-level imports for mockability) |
| Create | `src/pipeline/propose_labels.py` | Vocabulary auto-suggestion |
| Modify | `src/web/ids.py` | _ref_contexts store |
| Modify | `src/web/jobs.py` | TRACK_RUNNERS registry + track_std branch |
| Modify | `src/web/routes.py` | /upload-pdf, /propose-labels, track std/both, hints receiver |
| Modify | `src/web/templates/_label_form.html` | track std option (default), PDF zone → /upload-pdf, vocab prefill |
| Modify | `src/web/templates/_timeline.html` | segments-loaded fix + enrich fields + track_std tab |
| Modify | `src/web/templates/index.html` | category colors, description/improvement in Gantt tooltip + sidebar, category stats |
| Modify | `src/web/static/app.css` | category color vars |
| Modify | `src/evaluate/metrics.py` | boundary deviation log |
| Modify | `scripts/run_pipeline.py` | --track std support |
| Modify | `tests/test_schemas.py` | new field round-trip + backward compat |
| Create | `tests/pipeline/test_aggregate.py` | aggregate unit tests |
| Modify | `tests/pipeline/test_report.py` | enrich fields tests |
| Create | `tests/pipeline/test_label_gemini.py` | label_gemini unit tests (Gemini mocked) |
| Create | `tests/pipeline/test_parse_reference.py` | parse_reference tests |
| Create | `tests/pipeline/test_propose_labels.py` | propose_labels tests |
| Modify | `tests/web/test_routes.py` | PDF upload, propose-labels, track std tests |
| Modify | `tests/evaluate/test_metrics.py` | boundary_deviation_log tests |
| Modify | `tests/test_integration.py` | track_std with mocked Gemini |

**Mocking convention (critical):** All external libraries that tests patch (`genai`, `pdfplumber`, `fitz`, `Image`) MUST be imported at **module level** in the source files — function-local imports bypass `unittest.mock.patch("module.attr")` and the mocks will not take effect. Follow the existing pattern in `src/pipeline/label_vlm_single.py` (line 7: `from google import genai` at top).

---

## Task 0: Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add PDF/image dependencies**

In `pyproject.toml`, add to the `dependencies` list after `"python-multipart>=0.0.9",`:

```toml
    "pdfplumber>=0.11",
    "PyMuPDF>=1.24",
    "Pillow>=10.0",
```

- [ ] **Step 2: Install and verify imports**

```
pip install -e ".[dev]"
python -c "import pdfplumber, fitz, PIL; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add pdfplumber, PyMuPDF, Pillow for PDF reference parsing"
```

---

## Task 1: Schema Extension

**Files:**
- Modify: `src/schemas.py`
- Modify: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests for new Segment fields**

```python
# tests/test_schemas.py — add these tests
def test_segment_has_enrich_fields():
    seg = Segment(0.0, 10.0, "ネジ締め", 0.9,
                  category="seimi",
                  description="フランジを締結",
                  improvement=None)
    assert seg.category == "seimi"
    assert seg.description == "フランジを締結"
    assert seg.improvement is None


def test_segment_enrich_defaults_to_none():
    seg = Segment(0.0, 10.0, "A", 1.0)
    assert seg.category is None
    assert seg.description is None
    assert seg.improvement is None


def test_segmentlist_roundtrip_with_enrich():
    sl = SegmentList(
        video_id="test", fps_sampled=1.0,
        label_vocabulary=["ネジ締め"],
        segments=[Segment(0.0, 10.0, "ネジ締め", 0.9, "seimi", "締結作業", None)],
        source="track_std",
    )
    sl2 = SegmentList.from_json(sl.to_json())
    assert sl2.segments[0].category == "seimi"
    assert sl2.segments[0].description == "締結作業"
    assert sl2.segments[0].improvement is None


def test_from_json_backward_compat_old_json():
    old_json = json.dumps({
        "video_id": "old", "fps_sampled": 1.0,
        "label_vocabulary": ["A"],
        "segments": [{"start_sec": 0.0, "end_sec": 5.0, "label": "A", "confidence": 0.8}],
        "source": "track_b",
    })
    sl = SegmentList.from_json(old_json)
    assert sl.segments[0].category is None
    assert sl.segments[0].description is None
    assert sl.segments[0].improvement is None


def test_hint_dataclass():
    from src.schemas import Hint
    h = Hint(label="ドライバー", frame_sec=12.3)
    assert h.bbox is None
    assert h.note is None
    h2 = Hint(label="手", frame_sec=5.0, bbox=(0.1, 0.2, 0.3, 0.4), note="右手")
    assert h2.bbox == (0.1, 0.2, 0.3, 0.4)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_schemas.py::test_segment_has_enrich_fields -v
```
Expected: `FAILED — TypeError: Segment() takes...`

- [ ] **Step 3: Update `src/schemas.py`**

Replace the entire file content:

```python
# src/schemas.py
from dataclasses import dataclass, asdict
import json
from typing import Optional


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    label: str
    confidence: float = 1.0
    category: Optional[str] = None      # "seimi" | "fuzui" | "muda"
    description: Optional[str] = None   # what is observed (factual)
    improvement: Optional[str] = None   # kaizen hint (muda/fuzui only)


@dataclass
class Hint:
    label: str
    frame_sec: float
    bbox: Optional[tuple] = None        # normalized (x, y, w, h); None = whole frame
    note: Optional[str] = None


@dataclass
class SegmentList:
    video_id: str
    fps_sampled: float
    label_vocabulary: list[str]
    segments: list[Segment]
    source: str  # "track_std" | "track_a" | "track_b" | "ground_truth" | "ollo"

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "SegmentList":
        d = json.loads(s)
        known = set(Segment.__dataclass_fields__)
        d["segments"] = [
            Segment(**{k: v for k, v in seg.items() if k in known})
            for seg in d["segments"]
        ]
        return cls(**d)

    def to_frame_labels(self, total_duration: float, fps: float = 1.0) -> list[str]:
        n = int(total_duration * fps)
        labels = ["background"] * n
        for seg in self.segments:
            start_i = int(seg.start_sec * fps)
            end_i = int(seg.end_sec * fps)
            for i in range(start_i, min(end_i, n)):
                labels[i] = seg.label
        return labels
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_schemas.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/schemas.py tests/test_schemas.py
git commit -m "feat: extend Segment with category/description/improvement; add Hint dataclass"
```

---

## Task 2: Aggregate Module

**Files:**
- Create: `src/pipeline/aggregate.py`
- Create: `tests/pipeline/test_aggregate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/pipeline/test_aggregate.py
import pytest
from src.schemas import Segment, SegmentList
from src.pipeline.aggregate import aggregate


def _make_sl(segs):
    return SegmentList(
        video_id="x", fps_sampled=1.0,
        label_vocabulary=["A", "B"],
        segments=segs, source="track_std",
    )


def test_aggregate_total_sec():
    sl = _make_sl([
        Segment(0.0, 10.0, "A", 1.0, "seimi"),
        Segment(10.0, 25.0, "B", 1.0, "muda"),
    ])
    result = aggregate(sl)
    assert abs(result["total_sec"] - 25.0) < 0.01


def test_aggregate_by_category_sums():
    sl = _make_sl([
        Segment(0.0, 10.0, "A", 1.0, "seimi"),
        Segment(10.0, 20.0, "B", 1.0, "seimi"),
        Segment(20.0, 30.0, "C", 1.0, "muda"),
    ])
    result = aggregate(sl)
    assert abs(result["by_category"]["seimi"]["total_sec"] - 20.0) < 0.01
    assert result["by_category"]["seimi"]["count"] == 2
    assert abs(result["by_category"]["seimi"]["ratio"] - 20/30) < 0.01


def test_aggregate_by_label_mean():
    sl = _make_sl([
        Segment(0.0, 10.0, "ネジ締め", 1.0, "seimi"),
        Segment(10.0, 14.0, "ネジ締め", 1.0, "seimi"),
    ])
    result = aggregate(sl)
    # two segments: 10s and 4s → mean = 7.0
    assert abs(result["by_label"]["ネジ締め"]["mean_sec"] - 7.0) < 0.01
    assert result["by_label"]["ネジ締め"]["count"] == 2


def test_aggregate_none_category_bucketed_as_unknown():
    sl = _make_sl([Segment(0.0, 5.0, "A", 1.0, None)])
    result = aggregate(sl)
    assert "unknown" in result["by_category"]
    assert result["by_category"]["unknown"]["total_sec"] == 5.0


def test_aggregate_empty_segments():
    sl = _make_sl([])
    result = aggregate(sl)
    assert result["total_sec"] == 0.0
    assert result["by_category"] == {}
    assert result["by_label"] == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/pipeline/test_aggregate.py -v
```
Expected: `FAILED — ModuleNotFoundError: No module named 'src.pipeline.aggregate'`

- [ ] **Step 3: Create `src/pipeline/aggregate.py`**

```python
# src/pipeline/aggregate.py
from src.schemas import SegmentList


def aggregate(seg_list: SegmentList) -> dict:
    """Compute category-level and label-level statistics.

    Returns:
        {
            "by_category": {cat: {"total_sec", "count", "ratio"}},
            "by_label":    {lbl: {"total_sec", "count", "mean_sec"}},
            "total_sec":   float,
        }
    """
    by_category: dict = {}
    by_label: dict = {}
    total_sec = 0.0

    for seg in seg_list.segments:
        dur = seg.end_sec - seg.start_sec
        total_sec += dur

        cat = seg.category or "unknown"
        entry = by_category.setdefault(cat, {"total_sec": 0.0, "count": 0, "ratio": 0.0})
        entry["total_sec"] += dur
        entry["count"] += 1

        lentry = by_label.setdefault(seg.label, {"total_sec": 0.0, "count": 0, "mean_sec": 0.0})
        lentry["total_sec"] += dur
        lentry["count"] += 1

    for data in by_category.values():
        data["ratio"] = data["total_sec"] / total_sec if total_sec > 0 else 0.0

    for data in by_label.values():
        data["mean_sec"] = data["total_sec"] / data["count"] if data["count"] > 0 else 0.0

    return {"by_category": by_category, "by_label": by_label, "total_sec": total_sec}
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/pipeline/test_aggregate.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/aggregate.py tests/pipeline/test_aggregate.py
git commit -m "feat: add aggregate module for category/label statistics"
```

---

## Task 3: Report Update

**Files:**
- Modify: `src/pipeline/report.py`
- Modify: `tests/pipeline/test_report.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/pipeline/test_report.py`:

```python
def test_procedure_includes_category():
    sl = SegmentList(
        video_id="v", fps_sampled=1.0,
        label_vocabulary=["ネジ締め"],
        segments=[Segment(0.0, 10.0, "ネジ締め", 0.9, "seimi", "電動ドライバーで締結", None)],
        source="track_std",
    )
    md = to_procedure_markdown(sl)
    assert "正味作業" in md


def test_procedure_includes_description():
    sl = SegmentList(
        video_id="v", fps_sampled=1.0,
        label_vocabulary=["A"],
        segments=[Segment(0.0, 5.0, "A", 1.0, "fuzui", "部品を棚から取り出す", None)],
        source="track_std",
    )
    md = to_procedure_markdown(sl)
    assert "部品を棚から取り出す" in md


def test_procedure_includes_improvement():
    sl = SegmentList(
        video_id="v", fps_sampled=1.0,
        label_vocabulary=["手待ち"],
        segments=[Segment(0.0, 30.0, "手待ち", 0.8, "muda", "次工程待ち", "前工程との同期化")],
        source="track_std",
    )
    md = to_procedure_markdown(sl)
    assert "前工程との同期化" in md


def test_procedure_graceful_without_enrich():
    # Segments with no enrich (track_b style) should still render
    sl = SegmentList(
        video_id="v", fps_sampled=1.0,
        label_vocabulary=["A"],
        segments=[Segment(0.0, 5.0, "A", 1.0)],  # no category/desc/improvement
        source="track_b",
    )
    md = to_procedure_markdown(sl)
    assert "Step 1" in md
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/pipeline/test_report.py::test_procedure_includes_category -v
```
Expected: `FAILED — AssertionError: assert '正味作業' in ...`

- [ ] **Step 3: Update `src/pipeline/report.py`**

```python
# src/pipeline/report.py
from pathlib import Path
from src.schemas import SegmentList

_CAT_NAMES = {
    "seimi": "正味作業",
    "fuzui": "付随作業",
    "muda":  "ムダ作業",
}


def to_timeline_markdown(seg_list: SegmentList) -> str:
    lines = [
        f"# 作業タイムライン: {seg_list.video_id}",
        "",
        "| # | 開始 | 終了 | 要素作業 | 分類 | 信頼度 |",
        "|---|------|------|----------|------|--------|",
    ]
    for i, seg in enumerate(seg_list.segments, 1):
        cat = _CAT_NAMES.get(seg.category or "", "—")
        lines.append(
            f"| {i} | {_fmt(seg.start_sec)} | {_fmt(seg.end_sec)} "
            f"| {seg.label} | {cat} | {seg.confidence:.2f} |"
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
            f"- **所要時間**: {duration:.1f}秒 ({_fmt(seg.start_sec)} ～ {_fmt(seg.end_sec)})",
            f"- **信頼度**: {seg.confidence:.2f}",
        ]
        if seg.category and seg.category in _CAT_NAMES:
            lines.append(f"- **分類**: {_CAT_NAMES[seg.category]}")
        if seg.description:
            lines.append(f"- **内容**: {seg.description}")
        if seg.improvement:
            lines.append(f"- **改善ヒント**: {seg.improvement}")
        lines.append("")
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

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/pipeline/test_report.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/report.py tests/pipeline/test_report.py
git commit -m "feat: add category/description/improvement to procedure markdown"
```

---

## Task 4: label_gemini — Pure Helpers (Window + Stitch)

**Files:**
- Create: `src/pipeline/label_gemini.py` (pure helper functions only)
- Create: `tests/pipeline/test_label_gemini.py` (pure function tests)

- [ ] **Step 1: Write failing tests for pure helpers**

```python
# tests/pipeline/test_label_gemini.py
import pytest
from src.schemas import Segment
from src.pipeline.label_gemini import (
    _build_coarse_segments,
    _group_into_windows,
    _merge_adjacent_enrich,
    _stitch,
)


def test_build_coarse_segments_basic():
    segs = _build_coarse_segments([10.0, 20.0], 30.0)
    assert segs == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_build_coarse_segments_empty_boundaries():
    segs = _build_coarse_segments([], 15.0)
    assert segs == [(0.0, 15.0)]


def test_build_coarse_segments_deduplicates():
    segs = _build_coarse_segments([10.0, 10.0, 20.0], 30.0)
    assert len(segs) == 3


def test_group_into_windows_small():
    coarse = [(float(i), float(i+1)) for i in range(5)]
    windows = _group_into_windows(coarse, target_size=10)
    assert windows == [(0, 5)]


def test_group_into_windows_overlap():
    coarse = [(float(i), float(i+1)) for i in range(25)]
    windows = _group_into_windows(coarse, target_size=10)
    # First window: [0..10), second: [9..19), etc. — 1-segment overlap
    assert windows[0] == (0, 10)
    assert windows[1][0] == 9  # starts at index 9 (overlap)
    assert windows[1][1] == 19


def test_group_into_windows_covers_all():
    coarse = [(float(i), float(i+1)) for i in range(25)]
    windows = _group_into_windows(coarse, target_size=10)
    # Last window must reach index 25
    assert windows[-1][1] == 25


def test_merge_adjacent_enrich_same_label_and_category():
    segs = [
        Segment(0.0, 10.0, "A", 0.9, "seimi", "desc1", None),
        Segment(10.0, 20.0, "A", 0.8, "seimi", "desc2", None),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 1
    assert merged[0].start_sec == 0.0
    assert merged[0].end_sec == 20.0
    # longer segment (desc1, 10s) wins over shorter (desc2, 10s) → first wins on tie
    assert merged[0].description == "desc1"


def test_merge_adjacent_enrich_different_category_no_merge():
    segs = [
        Segment(0.0, 10.0, "A", 0.9, "seimi", "d1", None),
        Segment(10.0, 20.0, "A", 0.8, "muda", "d2", "改善"),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 2


def test_merge_adjacent_enrich_core_priority_longer_wins():
    # longer segment's description should be kept
    segs = [
        Segment(0.0, 5.0,  "A", 0.9, "seimi", "short-desc", None),
        Segment(5.0, 20.0, "A", 0.9, "seimi", "long-desc",  None),
    ]
    merged = _merge_adjacent_enrich(segs)
    assert len(merged) == 1
    assert merged[0].description == "long-desc"  # 15s > 5s


def test_stitch_invariant_continuous_no_gap():
    coarse = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    windows = [(0, 3)]
    window_results = [[
        Segment(0.0, 10.0, "A", 0.9, "seimi"),
        Segment(10.0, 20.0, "B", 0.9, "fuzui"),
        Segment(20.0, 30.0, "A", 0.9, "seimi"),
    ]]
    vocab = ["A", "B"]
    result = _stitch(window_results, windows, coarse, 30.0, vocab)
    assert result[0].start_sec == 0.0
    assert result[-1].end_sec == 30.0
    for i in range(len(result) - 1):
        assert abs(result[i].end_sec - result[i+1].start_sec) < 0.01


def test_stitch_no_duplicate_intervals():
    coarse = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    windows = [(0, 2), (1, 3)]  # overlap at index 1
    window_results = [
        [Segment(0.0, 10.0, "A", 0.9), Segment(10.0, 20.0, "B", 0.9)],
        [Segment(10.0, 20.0, "B", 0.8), Segment(20.0, 30.0, "C", 0.9)],
    ]
    vocab = ["A", "B", "C"]
    result = _stitch(window_results, windows, coarse, 30.0, vocab)
    # No time range should appear twice
    for i in range(len(result) - 1):
        assert result[i].end_sec <= result[i+1].start_sec + 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/pipeline/test_label_gemini.py -v
```
Expected: `FAILED — ModuleNotFoundError: No module named 'src.pipeline.label_gemini'`

- [ ] **Step 3: Create `src/pipeline/label_gemini.py` with pure helpers**

```python
# src/pipeline/label_gemini.py
import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from google import genai  # module-level: tests patch src.pipeline.label_gemini.genai

from src.schemas import Hint, Segment, SegmentList
from src.pipeline.ingest import _blur_faces as _do_blur_faces

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-pro"
MAX_FRAMES_PER_WINDOW = 30
FRAMES_PER_SEGMENT = 3

_CATEGORY_ALIASES: dict[str, str] = {
    "正味": "seimi", "正味作業": "seimi",
    "付随": "fuzui", "付随作業": "fuzui",
    "ムダ": "muda", "ムダ作業": "muda",
    "value-adding": "seimi", "value adding": "seimi", "productive": "seimi",
    "ancillary": "fuzui", "supporting": "fuzui", "auxiliary": "fuzui",
    "waste": "muda", "non-value-adding": "muda",
    "seimi": "seimi", "fuzui": "fuzui", "muda": "muda",
}

_LABEL_SYNONYMS: dict[str, str] = {
    "手待ち時間": "手待ち",
    "待機": "手待ち",
    "待ち": "手待ち",
    "手待": "手待ち",
    "探す": "モノ探し",
    "探している": "モノ探し",
    "もの探し": "モノ探し",
    "物探し": "モノ探し",
    "無駄歩行": "歩行（ムダ）",
    "歩いている": "歩行（ムダ）",
    "余分な歩行": "歩行（ムダ）",
    "不要な移動": "歩行（ムダ）",
    "手戻り": "やり直し",
    "やりなおし": "やり直し",
    "ミスの修正": "やり直し",
}


# ── Public entry point ────────────────────────────────────────────────────────

def label_gemini(
    video_path: str,
    label_vocabulary: list[str],
    boundary_timestamps: list[float],
    *,
    blur_faces: bool = False,
    hints: list[Hint] | None = None,
    reference_context: str | None = None,
    model: str = _GEMINI_MODEL,
    source: str = "track_std",
    raw_output_dir: str = "results",
) -> SegmentList:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    cap = cv2.VideoCapture(video_path)
    total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    coarse_segs = _build_coarse_segments(boundary_timestamps, total_duration)
    windows = _group_into_windows(coarse_segs)

    video_id = Path(video_path).stem
    effective_vocab = list(label_vocabulary)
    window_results: list[list[Segment]] = []
    all_raw: list[dict] = []

    for win_idx, (start_idx, end_idx) in enumerate(windows):
        win_coarse = coarse_segs[start_idx:end_idx]
        win_start = win_coarse[0][0]
        win_end = win_coarse[-1][1]

        frames = _extract_window_frames(video_path, win_coarse, blur_faces)
        segments, raw_text = _call_gemini(
            client=client, model=model, frames=frames,
            label_vocabulary=effective_vocab,
            win_start=win_start, win_end=win_end,
            reference_context=reference_context, hints=hints,
        )

        clamped = []
        for seg in segments:
            s = max(win_start, min(seg.start_sec, win_end))
            e = max(win_start, min(seg.end_sec, win_end))
            if e > s:
                clamped.append(Segment(s, e, seg.label, seg.confidence,
                                       seg.category, seg.description, seg.improvement))
        window_results.append(clamped)
        all_raw.append({"window": win_idx, "win_start": win_start, "win_end": win_end, "raw": raw_text})

    # Persist raw responses for reproducibility
    results_dir = Path(raw_output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{video_id}_{source}_raw.json").write_text(
        json.dumps(all_raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    all_segments = _stitch(window_results, windows, coarse_segs, total_duration, effective_vocab)

    return SegmentList(
        video_id=video_id,
        fps_sampled=float(FRAMES_PER_SEGMENT),
        label_vocabulary=effective_vocab,
        segments=all_segments,
        source=source,
    )


# ── Pure helpers (tested in isolation) ───────────────────────────────────────

def _build_coarse_segments(
    boundary_timestamps: list[float],
    total_duration: float,
) -> list[tuple[float, float]]:
    points = sorted({0.0} | set(boundary_timestamps) | {total_duration})
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)
            if points[i + 1] - points[i] > 0.01]


def _group_into_windows(
    coarse_segs: list[tuple[float, float]],
    target_size: int = 10,
) -> list[tuple[int, int]]:
    n = len(coarse_segs)
    if n == 0:
        return []
    windows: list[tuple[int, int]] = []
    i = 0
    while i < n:
        end = min(i + target_size, n)
        windows.append((i, end))
        if end == n:
            break
        i = end - 1  # 1-segment overlap
    return windows


def _merge_adjacent_enrich(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent segments with same (label, category). Core-priority: longer segment's
    description/improvement is kept (approximates keeping the window-core side)."""
    if not segments:
        return []
    merged = [Segment(
        segments[0].start_sec, segments[0].end_sec, segments[0].label,
        segments[0].confidence, segments[0].category,
        segments[0].description, segments[0].improvement,
    )]
    for seg in segments[1:]:
        last = merged[-1]
        if seg.label == last.label and seg.category == last.category:
            last_dur = last.end_sec - last.start_sec
            seg_dur = seg.end_sec - seg.start_sec
            keep = last if last_dur >= seg_dur else seg
            merged[-1] = Segment(
                last.start_sec, seg.end_sec, last.label,
                (last.confidence + seg.confidence) / 2,
                last.category, keep.description, keep.improvement,
            )
        else:
            merged.append(Segment(
                seg.start_sec, seg.end_sec, seg.label, seg.confidence,
                seg.category, seg.description, seg.improvement,
            ))
    return merged


def _stitch(
    window_results: list[list[Segment]],
    windows: list[tuple[int, int]],
    coarse_segs: list[tuple[float, float]],
    total_duration: float,
    vocabulary: list[str],
) -> list[Segment]:
    if not window_results:
        return []

    # Determine core range for each window
    def core_range(win_idx: int) -> tuple[float, float]:
        start_idx, end_idx = windows[win_idx]
        segs = coarse_segs[start_idx:end_idx]
        c_start = segs[1][0] if win_idx > 0 and len(segs) > 1 else segs[0][0]
        c_end = segs[-2][1] if win_idx < len(windows) - 1 and len(segs) > 1 else segs[-1][1]
        return c_start, c_end

    # Tag each candidate segment with is_core
    tagged: list[tuple[Segment, bool, int]] = []
    for win_idx, result in enumerate(window_results):
        c_start, c_end = core_range(win_idx)
        for seg in result:
            center = (seg.start_sec + seg.end_sec) / 2
            is_core = c_start <= center <= c_end
            tagged.append((seg, is_core, win_idx))

    # Sort: by start_sec asc, core first (is_core=True sorts before False)
    tagged.sort(key=lambda x: (x[0].start_sec, 0 if x[1] else 1))

    # Greedy de-overlap: keep first (core-preferred) non-overlapping segments
    kept: list[Segment] = []
    current_end = 0.0
    for seg, is_core, _ in tagged:
        if seg.end_sec <= current_end:
            continue
        if seg.start_sec < current_end:
            seg = Segment(current_end, seg.end_sec, seg.label, seg.confidence,
                         seg.category, seg.description, seg.improvement)
        if seg.end_sec > seg.start_sec:
            kept.append(seg)
            current_end = seg.end_sec

    # Normalize labels (synonym + out-of-vocab append)
    for seg in kept:
        seg.label = _normalize_label(seg.label, vocabulary)

    # Fill leading gap
    filled: list[Segment] = []
    if kept and kept[0].start_sec > 0.01:
        filled.append(Segment(0.0, kept[0].start_sec, kept[0].label, 0.5,
                              kept[0].category, None, None))

    for seg in kept:
        if filled and filled[-1].end_sec < seg.start_sec - 0.01:
            filled.append(Segment(filled[-1].end_sec, seg.start_sec,
                                  filled[-1].label, 0.5, filled[-1].category, None, None))
        filled.append(seg)

    # Fill trailing gap
    if filled and filled[-1].end_sec < total_duration - 0.01:
        filled.append(Segment(filled[-1].end_sec, total_duration,
                              filled[-1].label, 0.5, filled[-1].category, None, None))

    return _merge_adjacent_enrich(filled)


def _normalize_label(label: str, vocabulary: list[str]) -> str:
    label = label.strip()
    if label in vocabulary:
        return label
    normalized = _LABEL_SYNONYMS.get(label)
    if normalized:
        return normalized
    if label not in vocabulary:
        vocabulary.append(label)
    return label


# ── Gemini I/O (not tested directly; covered via label_gemini integration) ───

def _extract_window_frames(
    video_path: str,
    coarse_segs: list[tuple[float, float]],
    blur_faces: bool,
) -> list[tuple[float, np.ndarray]]:
    frames: list[tuple[float, np.ndarray]] = []
    cap = cv2.VideoCapture(video_path)
    for seg_start, seg_end in coarse_segs:
        n = min(FRAMES_PER_SEGMENT, max(1, int(seg_end - seg_start)))
        for ts in np.linspace(seg_start, seg_end, n, endpoint=False):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000)
            ret, frame = cap.read()
            if not ret:
                continue
            if blur_faces:
                frame = _do_blur_faces(frame)
            frames.append((float(ts), frame))
    cap.release()
    if len(frames) > MAX_FRAMES_PER_WINDOW:
        step = len(frames) / MAX_FRAMES_PER_WINDOW
        frames = [frames[int(i * step)] for i in range(MAX_FRAMES_PER_WINDOW)]
    return frames


def _call_gemini(
    *,
    client,
    model: str,
    frames: list[tuple[float, np.ndarray]],
    label_vocabulary: list[str],
    win_start: float,
    win_end: float,
    reference_context: str | None,
    hints: list[Hint] | None,
) -> tuple[list[Segment], str]:
    label_list = "\n".join(f"- {l}" for l in label_vocabulary)
    ref_section = f"\n\nWork Standard Reference:\n{reference_context[:2000]}" if reference_context else ""
    hints_section = ""
    if hints:
        hints_section = "\n\nKnown objects/operations:\n" + "\n".join(
            f"- {h.label}" + (f": {h.note}" if h.note else "") for h in hints
        )

    prompt = (
        f"You are an industrial engineer analyzing a factory work video.\n"
        f"Segment: {win_start:.1f}s to {win_end:.1f}s.\n"
        f"Labels (add new label only if none fits):\n{label_list}\n"
        f"Categories: seimi (正味作業=value-adding), fuzui (付随作業=ancillary), muda (ムダ作業=waste)\n"
        f"{ref_section}{hints_section}\n\n"
        f"Frames shown as [t=Xs]. Output a JSON array covering [{win_start:.1f}, {win_end:.1f}] "
        f"with no gaps. Each element: "
        f"{{\"start_sec\":float, \"end_sec\":float, \"label\":string, "
        f"\"category\":\"seimi\"|\"fuzui\"|\"muda\", "
        f"\"description\":string (observed facts only), "
        f"\"improvement\":string|null (only for muda/fuzui), "
        f"\"confidence\":float}}\n"
        f"Output ONLY the JSON array."
    )

    parts: list = [prompt]
    for ts, frame in frames:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                       "data": base64.b64encode(buf.tobytes()).decode()}})
        parts.append(f"[t={ts:.1f}s]")

    try:
        resp = client.models.generate_content(
            model=model, contents=parts,
            config={"temperature": 0},
        )
        raw = resp.text.strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s == -1 or e == 0:
            logger.warning("No JSON array from Gemini for window %s-%s", win_start, win_end)
            return [], raw
        items = json.loads(raw[s:e])
        segs = []
        for it in items:
            cat_key = str(it.get("category", "")).strip()
            cat = _CATEGORY_ALIASES.get(cat_key) or _CATEGORY_ALIASES.get(cat_key.lower())
            segs.append(Segment(
                start_sec=float(it["start_sec"]),
                end_sec=float(it["end_sec"]),
                label=str(it["label"]),
                confidence=float(it.get("confidence", 1.0)),
                category=cat,
                description=it.get("description") or None,
                improvement=it.get("improvement") or None,
            ))
        return segs, raw
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return [], ""
```

- [ ] **Step 4: Run pure helper tests**

```
pytest tests/pipeline/test_label_gemini.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/label_gemini.py tests/pipeline/test_label_gemini.py
git commit -m "feat: add label_gemini — window grouping, stitch, normalization"
```

---

## Task 5: label_gemini — Gemini Integration Tests

**Files:**
- Modify: `tests/pipeline/test_label_gemini.py` (add integration-style tests with mock)

- [ ] **Step 1: Add Gemini-mocked tests**

Add to `tests/pipeline/test_label_gemini.py`:

```python
import json
from unittest.mock import MagicMock, patch


def _gemini_response(segments: list[dict]) -> MagicMock:
    m = MagicMock()
    m.text = json.dumps(segments)
    return m


@patch("src.pipeline.label_gemini.genai")
def test_label_gemini_returns_segment_list(mock_genai, synthetic_video_path):
    from src.pipeline.label_gemini import label_gemini
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_response([
        {"start_sec": 0.0, "end_sec": 15.0, "label": "ネジ締め",
         "category": "seimi", "description": "締結中", "improvement": None, "confidence": 0.9},
        {"start_sec": 15.0, "end_sec": 30.0, "label": "部品取り出し",
         "category": "fuzui", "description": "棚から取る", "improvement": None, "confidence": 0.85},
    ])
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_gemini(synthetic_video_path, ["ネジ締め", "部品取り出し"], [10.0, 20.0])
    from src.schemas import SegmentList
    assert isinstance(result, SegmentList)
    assert result.source == "track_std"
    assert len(result.segments) >= 1
    assert result.segments[0].category in ("seimi", "fuzui", "muda", None)


@patch("src.pipeline.label_gemini.genai")
def test_label_gemini_category_normalization(mock_genai, synthetic_video_path):
    from src.pipeline.label_gemini import label_gemini
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_response([
        {"start_sec": 0.0, "end_sec": 30.0, "label": "A",
         "category": "value-adding", "description": "d", "improvement": None, "confidence": 0.9},
    ])
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_gemini(synthetic_video_path, ["A"], [])
    assert result.segments[0].category == "seimi"


@patch("src.pipeline.label_gemini.genai")
def test_label_gemini_out_of_vocab_label_appended(mock_genai, synthetic_video_path):
    from src.pipeline.label_gemini import label_gemini
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_response([
        {"start_sec": 0.0, "end_sec": 30.0, "label": "新ラベル",
         "category": "muda", "description": "d", "improvement": "改善", "confidence": 0.7},
    ])
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_gemini(synthetic_video_path, ["既存ラベル"], [])
    assert "新ラベル" in result.label_vocabulary


@patch("src.pipeline.label_gemini.genai")
def test_label_gemini_synonym_normalization(mock_genai, synthetic_video_path):
    from src.pipeline.label_gemini import label_gemini
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_response([
        {"start_sec": 0.0, "end_sec": 30.0, "label": "待機",
         "category": "muda", "description": "d", "improvement": None, "confidence": 0.8},
    ])
    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        result = label_gemini(synthetic_video_path, ["手待ち"], [])
    # "待機" → normalized to "手待ち" via _LABEL_SYNONYMS
    assert result.segments[0].label == "手待ち"


def test_normalize_label_synonym():
    from src.pipeline.label_gemini import _normalize_label
    vocab = ["手待ち"]
    assert _normalize_label("待機", vocab) == "手待ち"
    assert _normalize_label("手待ち時間", vocab) == "手待ち"


def test_normalize_label_out_of_vocab_appended():
    from src.pipeline.label_gemini import _normalize_label
    vocab = ["A"]
    result = _normalize_label("新作業", vocab)
    assert result == "新作業"
    assert "新作業" in vocab
```

- [ ] **Step 2: Run new tests**

```
pytest tests/pipeline/test_label_gemini.py -v
```
Expected: all PASS (Gemini mocked)

- [ ] **Step 3: Commit**

```bash
git add tests/pipeline/test_label_gemini.py
git commit -m "test: add Gemini-mocked integration tests for label_gemini"
```

---

## Task 6: parse_reference

**Files:**
- Create: `src/pipeline/parse_reference.py`
- Create: `tests/pipeline/test_parse_reference.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # Create a minimal valid PDF using reportlab (or use a stub)
    # Since we can't depend on reportlab in tests, mock pdfplumber instead
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
        mock_doc.__getitem__ = lambda s, i: mock_page
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
        assert mock_doc.__getitem__.call_count <= MAX_PDF_IMAGE_PAGES


def test_parse_reference_returns_none_on_timeout():
    from src.pipeline.parse_reference import parse_reference
    with patch("src.pipeline.parse_reference._parse_sync", side_effect=Exception("fail")):
        result = parse_reference("/fake.pdf")
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/pipeline/test_parse_reference.py -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/pipeline/parse_reference.py`**

```python
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
```

- [ ] **Step 4: Run tests**

```
pytest tests/pipeline/test_parse_reference.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/parse_reference.py tests/pipeline/test_parse_reference.py
git commit -m "feat: add parse_reference — PDF text + scanned image fallback with guardrails"
```

---

## Task 7: propose_labels

**Files:**
- Create: `src/pipeline/propose_labels.py`
- Create: `tests/pipeline/test_propose_labels.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/pipeline/test_propose_labels.py -v
```
Expected: `FAILED — ModuleNotFoundError`

- [ ] **Step 3: Create `src/pipeline/propose_labels.py`**

```python
# src/pipeline/propose_labels.py
import base64
import json
import logging
import os
from typing import Optional

import cv2
from google import genai  # module-level: tests patch src.pipeline.propose_labels.genai

from src.pipeline.ingest import _blur_faces as _do_blur_faces

logger = logging.getLogger(__name__)

_GEMINI_MODEL = "gemini-2.5-pro"
_MAX_FRAMES = 20


def propose_labels(
    video_path: str,
    *,
    reference_context: Optional[str] = None,
    blur_faces: bool = False,
    model: str = _GEMINI_MODEL,
    max_labels: int = 12,
) -> list[str]:
    """Sample frames from video and ask Gemini to suggest work operation labels.
    Returns empty list on any failure (non-blocking)."""
    try:
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        frames = _sample_frames(video_path, blur_faces, _do_blur_faces)
        if not frames:
            return []

        ref_section = f"\n\nWork Standard Reference:\n{reference_context[:1500]}" if reference_context else ""
        prompt = (
            f"You are an industrial engineer. Analyze this factory work video and propose "
            f"up to {max_labels} distinct work operation labels in Japanese. "
            f"Labels should be specific action names (e.g. 部品取り出し, ネジ締め). "
            f"Output ONLY a JSON array of strings.{ref_section}"
        )

        parts: list = [prompt]
        for ts, b64 in frames:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
            parts.append(f"[t={ts:.0f}s]")

        resp = client.models.generate_content(
            model=model, contents=parts,
            config={"temperature": 0},
        )
        text = resp.text.strip()
        s, e = text.find("["), text.rfind("]") + 1
        if s == -1 or e == 0:
            return []
        labels = json.loads(text[s:e])
        return [str(l).strip() for l in labels if str(l).strip()][:max_labels]
    except Exception as exc:
        logger.warning("propose_labels failed: %s", exc)
        return []


def _sample_frames(
    video_path: str,
    blur_faces: bool,
    blur_fn,
) -> list[tuple[float, str]]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    interval = max(total / _MAX_FRAMES, 5.0)
    frames = []
    ts = 0.0
    while ts < total and len(frames) < _MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        if blur_faces:
            frame = blur_fn(frame)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        frames.append((ts, base64.b64encode(buf.tobytes()).decode()))
        ts += interval
    cap.release()
    return frames
```

- [ ] **Step 4: Run tests**

```
pytest tests/pipeline/test_propose_labels.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/propose_labels.py tests/pipeline/test_propose_labels.py
git commit -m "feat: add propose_labels — async vocabulary suggestion from video+PDF"
```

---

## Task 8: ids.py + jobs.py — TRACK_RUNNERS Registry

**Files:**
- Modify: `src/web/ids.py` (add `_ref_contexts`)
- Modify: `src/web/jobs.py` (TRACK_RUNNERS + track_std + reference_context)
- Modify: `tests/web/test_jobs.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/web/test_jobs.py`:

```python
def test_track_runners_registry_has_required_tracks():
    from src.web.jobs import TRACK_RUNNERS, _init_runners
    _init_runners()
    assert "b" in TRACK_RUNNERS
    assert "a" in TRACK_RUNNERS
    assert "std" in TRACK_RUNNERS


def test_register_stores_track():
    from src.web import jobs
    jobs.register("j1", "std")
    status = jobs.get_status("j1")
    assert status["track"] == "std"
    jobs._jobs.clear()
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/web/test_jobs.py::test_track_runners_registry_has_required_tracks -v
```
Expected: `FAILED — ImportError or AttributeError`

- [ ] **Step 3: Update `src/web/ids.py`**

Add `_ref_contexts` and helpers (append after existing `_registry`):

```python
# src/web/ids.py  — add these after line 13 (_registry: dict...)

# job_id → reference_context (parsed from PDF at upload time)
_ref_contexts: dict[str, str | None] = {}


def store_ref_context(job_id: str, context: str | None) -> None:
    _ref_contexts[job_id] = context


def get_ref_context(job_id: str) -> str | None:
    return _ref_contexts.get(job_id)
```

- [ ] **Step 4: Rewrite `src/web/jobs.py`**

```python
# src/web/jobs.py
"""ジョブ登録・状態遷移・別スレッド実行管理。"""
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=1)

# Registry: track_key → callable(video_path, label_vocabulary, **opts) -> SegmentList
# Populated lazily to avoid circular imports at module load time.
TRACK_RUNNERS: dict[str, Any] = {}


def _init_runners() -> None:
    if TRACK_RUNNERS:
        return

    from src.pipeline.label_zeroshot import label_zeroshot
    from src.pipeline.label_vlm_single import label_vlm_single
    from src.pipeline.label_gemini import label_gemini

    def _run_b(video_path, label_vocabulary, *, fps=1.0, penalty=10.0, blur_faces=False, **_):
        from src.pipeline.ingest import extract_frames
        from src.pipeline.embed import embed_frames
        from src.pipeline.presegment import detect_boundaries
        frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        return label_zeroshot(video_path, label_vocabulary,
                              fps=fps, boundary_timestamps=boundaries, blur_faces=blur_faces)

    def _run_a(video_path, label_vocabulary, *, blur_faces=False, **_):
        return label_vlm_single(video_path, label_vocabulary, blur_faces=blur_faces)

    def _run_std(video_path, label_vocabulary, *,
                 fps=1.0, penalty=10.0, blur_faces=False,
                 reference_context=None, hints=None, **_):
        from src.pipeline.ingest import extract_frames
        from src.pipeline.embed import embed_frames
        from src.pipeline.presegment import detect_boundaries
        frames = extract_frames(video_path, fps=fps, blur_faces=blur_faces)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        return label_gemini(
            video_path, label_vocabulary, boundaries,
            blur_faces=blur_faces,
            reference_context=reference_context,
            hints=hints,
        )

    TRACK_RUNNERS["b"] = _run_b
    TRACK_RUNNERS["a"] = _run_a
    TRACK_RUNNERS["std"] = _run_std


def register(job_id: str, track: str) -> None:
    _jobs[job_id] = {"status": "registered", "stage": "", "track": track, "error": None}


def get_status(job_id: str) -> dict[str, Any] | None:
    return _jobs.get(job_id)


def _run_pipeline(
    job_id: str,
    video_path: Path,
    label_list: list[str],
    track: str,
    output_dir: Path,
    blur_faces: bool,
    fps: float,
    penalty: float,
    reference_context: str | None = None,
    hints: list | None = None,
) -> None:
    try:
        _jobs[job_id]["status"] = "running"
        _init_runners()

        tracks = ["b", "a"] if track == "both" else [track]
        for t in tracks:
            runner = TRACK_RUNNERS.get(t)
            if runner is None:
                raise ValueError(f"Unknown track: {t!r}. Available: {list(TRACK_RUNNERS)}")
            _jobs[job_id]["stage"] = f"{t}: analyzing"
            seg_list = runner(
                str(video_path), label_list,
                fps=fps, penalty=penalty, blur_faces=blur_faces,
                reference_context=reference_context, hints=hints,
            )
            from src.pipeline.report import save_segments
            save_segments(seg_list, str(output_dir))

        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["stage"] = "complete"

    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


async def start_pipeline(
    job_id: str,
    video_path: Path,
    label_list: list[str],
    track: str,
    output_dir: Path,
    blur_faces: bool = False,
    fps: float = 1.0,
    penalty: float = 10.0,
    reference_context: str | None = None,
    hints: list | None = None,
) -> None:
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor, _run_pipeline,
        job_id, video_path, label_list, track, output_dir,
        blur_faces, fps, penalty, reference_context, hints,
    )
```

- [ ] **Step 5: Run tests**

```
pytest tests/web/test_jobs.py -v
```
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/web/ids.py src/web/jobs.py tests/web/test_jobs.py
git commit -m "feat: add TRACK_RUNNERS registry to jobs.py; add track_std; ids.py stores ref_context"
```

---

## Task 9: routes.py — /upload-pdf, /propose-labels, track_std, hints receiver

**Files:**
- Modify: `src/web/routes.py`
- Modify: `tests/web/test_routes.py`

> **Why a separate `/upload-pdf` endpoint:** the video form auto-submits on file selection
> (`requestSubmit()` in `index.html`), so the PDF — chosen later in `_label_form.html` —
> can never ride in the same multipart request. The PDF is attached *after* the video
> upload, keyed by `job_id`.

- [ ] **Step 1: Write failing tests**

Add to `tests/web/test_routes.py`:

```python
def test_upload_pdf_attaches_to_job(tmp_path):
    client.post("/upload", files={"file": ("v.mp4", b"d", "video/mp4")})
    job_id = list(ids_module.all_job_ids())[-1]
    with patch("src.web.routes._parse_pdf_safe", return_value="parsed context"):
        resp = client.post(
            "/upload-pdf",
            data={"job_id": job_id},
            files={"pdf": ("manual.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )
    assert resp.status_code == 200
    assert ids_module.get_ref_context(job_id) == "parsed context"


def test_upload_pdf_unregistered_job_returns_404():
    resp = client.post(
        "/upload-pdf",
        data={"job_id": "ghost"},
        files={"pdf": ("m.pdf", b"%PDF", "application/pdf")},
    )
    assert resp.status_code == 404


def test_upload_pdf_rejects_non_pdf():
    client.post("/upload", files={"file": ("v.mp4", b"d", "video/mp4")})
    job_id = list(ids_module.all_job_ids())[-1]
    resp = client.post(
        "/upload-pdf",
        data={"job_id": job_id},
        files={"pdf": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_analyze_accepts_hints_json(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    client.post("/upload", files={"file": ("v.mp4", b"d", "video/mp4")})
    job_id = list(ids_module.all_job_ids())[-1]
    with patch.object(jobs_module, "start_pipeline", new=AsyncMock()) as mock_start:
        resp = client.post(
            "/analyze",
            data={
                "job_id": job_id, "labels": "A,B", "track": "b",
                "hints": '[{"label": "ドライバー", "frame_sec": 12.3}]',
            },
        )
    assert resp.status_code == 200


def test_propose_labels_unregistered_job_returns_404():
    resp = client.post("/propose-labels", data={"job_id": "ghost"})
    assert resp.status_code == 404


def test_propose_labels_no_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client.post("/upload", files={"file": ("v.mp4", b"d", "video/mp4")})
    job_id = list(ids_module.all_job_ids())[-1]
    resp = client.post("/propose-labels", data={"job_id": job_id})
    assert resp.status_code == 200
    assert b"GEMINI_API_KEY" in resp.content


def test_analyze_track_std_without_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client.post("/upload", files={"file": ("v.mp4", b"d", "video/mp4")})
    job_id = list(ids_module.all_job_ids())[-1]
    resp = client.post("/analyze", data={"job_id": job_id, "labels": "A,B", "track": "std"})
    assert resp.status_code == 200
    assert b"GEMINI_API_KEY" in resp.content
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/web/test_routes.py::test_upload_pdf_attaches_to_job -v
```
Expected: `FAILED — 404 Not Found` (/upload-pdf route does not exist yet)

- [ ] **Step 3: Rewrite `src/web/routes.py`**

```python
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
    hints: Optional[str] = Form(None),   # extension receiver: JSON array of Hint dicts
) -> HTMLResponse:
    if not ids.is_registered(job_id):
        return HTMLResponse(content="<p class='error'>不明な job_id です。</p>", status_code=404)

    # Parse hints (extension point — passed through to label_gemini, no UI yet)
    hint_list = None
    if hints:
        try:
            import json as _json
            from src.schemas import Hint
            raw = _json.loads(hints)
            hint_list = [Hint(**{k: v for k, v in h.items()
                                 if k in Hint.__dataclass_fields__}) for h in raw]
        except Exception:
            hint_list = None  # malformed hints are ignored, not fatal

    needs_gemini = track in _GEMINI_TRACKS or track == "both"
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
        track = "b"  # "both" runs b+a; default the results view to b (tabs switch to a)

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
```

- [ ] **Step 4: Run all route tests**

```
pytest tests/web/test_routes.py -v
```
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/web/routes.py tests/web/test_routes.py
git commit -m "feat: routes — optional PDF in /upload, /propose-labels, track_std support"
```

---

## Task 10: UI — Gantt Bug Fix & Enrich Display

**Files:**
- Modify: `src/web/templates/_timeline.html`
- Modify: `src/web/templates/index.html`
- Modify: `src/web/static/app.css`

> No automated tests for template/CSS changes. Verify manually with `uvicorn src.web.app:app --reload` after this task.

- [ ] **Step 1: Fix `_timeline.html` — segments-loaded event + enrich columns + track_std tab**

Replace `src/web/templates/_timeline.html` entirely:

```html
<div class="timeline-header">
  <h3>タイムライン — {{ seg_list.video_id }}</h3>

  <div class="track-tabs">
    <button hx-get="/results/{{ job_id }}?track=std"
            hx-target="#timeline-content" hx-swap="innerHTML"
            class="tab {% if track == 'std' %}active{% endif %}">
      Track STD (CLIP+Gemini)
    </button>
    <button hx-get="/results/{{ job_id }}?track=b"
            hx-target="#timeline-content" hx-swap="innerHTML"
            class="tab {% if track == 'b' %}active{% endif %}">
      Track B (CLIP)
    </button>
    <button hx-get="/results/{{ job_id }}?track=a"
            hx-target="#timeline-content" hx-swap="innerHTML"
            class="tab {% if track == 'a' %}active{% endif %}">
      Track A (Gemini単発)
    </button>
  </div>
</div>

<table class="seg-table">
  <thead>
    <tr>
      <th>開始</th><th>終了</th><th>作業ラベル</th>
      <th>分類</th><th>信頼度</th>
    </tr>
  </thead>
  <tbody>
    {% for seg in seg_list.segments %}
    <tr class="seg-row"
        data-start="{{ seg.start_sec }}"
        data-end="{{ seg.end_sec }}"
        onclick="window.dispatchEvent(new CustomEvent('seek-to', {detail: {{ seg.start_sec }}}))"
        title="{{ seg.description or '' }}">
      <td>{{ '%02d:%05.2f' | format(seg.start_sec // 60, seg.start_sec % 60) }}</td>
      <td>{{ '%02d:%05.2f' | format(seg.end_sec // 60, seg.end_sec % 60) }}</td>
      <td class="label-cell">{{ seg.label }}</td>
      <td class="cat-cell cat-{{ seg.category or 'none' }}">
        {% if seg.category == 'seimi' %}正味
        {% elif seg.category == 'fuzui' %}付随
        {% elif seg.category == 'muda' %}ムダ
        {% else %}—
        {% endif %}
      </td>
      <td>{{ '%.2f' | format(seg.confidence) }}</td>
    </tr>
    {% if seg.improvement %}
    <tr class="seg-improvement">
      <td colspan="5">💡 {{ seg.improvement }}</td>
    </tr>
    {% endif %}
    {% endfor %}
  </tbody>
</table>

{% if metrics %}
<div class="metrics-section">
  <h4>評価指標（正解アノテーションとの比較）</h4>
  <table class="metrics-table">
    <thead>
      <tr><th>F1@10</th><th>F1@25</th><th>F1@50</th><th>Edit</th><th>Acc</th></tr>
    </thead>
    <tbody>
      <tr>
        <td>{{ '%.3f' | format(metrics['f1@10']) }}</td>
        <td>{{ '%.3f' | format(metrics['f1@25']) }}</td>
        <td>{{ '%.3f' | format(metrics['f1@50']) }}</td>
        <td>{{ '%.1f' | format(metrics['edit']) }}</td>
        <td>{{ '%.3f' | format(metrics['acc']) }}</td>
      </tr>
    </tbody>
  </table>
</div>
{% endif %}

<!-- Fire segments-loaded after DOM insertion so Gantt/stats build from this data -->
<script>
(function() {
  var segs = {{ segments_dicts | tojson }};
  window.dispatchEvent(new CustomEvent('segments-loaded', { detail: segs }));
})();
</script>
```

- [ ] **Step 2: Update `_label_form.html` — track std, PDF wiring, vocab prefill**

Three changes in `src/web/templates/_label_form.html`:

**(a) Track select — `std` becomes the first/default option:**

Replace the existing `<select name="track" ...>` block with:

```html
<select name="track" class="lf-input">
  <option value="std">標準（CLIP境界 + Gemini精緻化・推奨）</option>
  <option value="b">Track B（CLIPのみ・API不要）</option>
  <option value="a">Track A（Gemini単発）</option>
  <option value="both">B と A を比較</option>
</select>
```

**(b) PDF zone — wire to `/upload-pdf` (currently cosmetic):**

Replace the PDF section (`<div class="up-section">` ... `</div>` containing `pdfDropZone`) with an htmx form:

```html
<div class="up-section">
  <div class="up-section-label">
    <span class="up-badge up-badge-opt">任意</span>
    作業標準書（PDF）
    <span class="up-section-hint">ラベル生成・分析精度の向上に使用します</span>
  </div>
  <form id="pdfForm"
        hx-post="/upload-pdf"
        hx-target="#pdfParseStatus"
        hx-swap="innerHTML"
        hx-encoding="multipart/form-data">
    <input type="hidden" name="job_id" value="{{ job_id }}">
    <div class="pdf-drop-zone" id="pdfDropZone" onclick="document.getElementById('pdfInput').click()">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <rect x="4" y="2" width="16" height="20" rx="2" stroke="#9CA3AF" stroke-width="1.5" fill="#F9FAFB"/>
        <path d="M8 8h8M8 12h8M8 16h5" stroke="#9CA3AF" stroke-width="1.2" stroke-linecap="round"/>
      </svg>
      <span class="pdf-drop-text">PDFをドラッグ、または<em>ファイルを選択</em></span>
      <input type="file" id="pdfInput" name="pdf" accept="application/pdf" style="display:none"
             onchange="handlePdfSel(this); this.closest('form').requestSubmit()">
    </div>
    <div class="up-selected up-selected-pdf" id="pdfSelected" style="display:none">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1D4ED8" stroke-width="2">
        <rect x="4" y="2" width="16" height="20" rx="2"/>
        <path d="M8 8h8M8 12h8M8 16h5" stroke-linecap="round"/>
      </svg>
      <span class="up-sel-name" id="pdfSelectedName"></span>
      <span id="pdfParseStatus"></span>
      <button type="button" class="up-sel-clear" onclick="clearPdfSel()">✕</button>
    </div>
  </form>
</div>
```

(The existing `handlePdfSel` / `clearPdfSel` / drag-drop JS in the same file stays as-is —
only the `onchange` gains `requestSubmit()` and the zone is wrapped in the htmx form.)

**(c) Vocab prefill — fire `/propose-labels` when the form loads, fill the labels input:**

Add at the end of the `<script>` block in `_label_form.html`:

```javascript
/* Vocab auto-proposal: non-blocking prefill of the labels input */
(function() {
  const labelInput = document.querySelector('input[name="labels"]');
  if (!labelInput) return;
  fetch('/propose-labels', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body: 'job_id=' + encodeURIComponent('{{ job_id }}'),
  })
  .then(r => r.text())
  .then(html => {
    const m = html.match(/data-labels="([^"]*)"/);
    // Only prefill if the user hasn't typed anything yet
    if (m && m[1] && !labelInput.value.trim()) labelInput.value = m[1];
  })
  .catch(() => {});  /* proposal failure is silent — manual input always works */
})();
```

- [ ] **Step 3: Update `index.html` — category colors for Gantt + tooltip + sidebar**

In `index.html`, replace the `getColorForLabel` and `buildGantt` sections:

Find and replace in the `<script>` block (the `CAT_COLORS` block through `buildGantt`):

```javascript
/* ── Category colors (seimi/fuzui/muda) + label fallback ── */
const CATEGORY_COLORS = {
  seimi: {color:'#22C55E', lt:'#DCFCE7', dk:'#15803D'},
  fuzui: {color:'#F97316', lt:'#FFEDD5', dk:'#C2410C'},
  muda:  {color:'#3B82F6', lt:'#DBEAFE', dk:'#1D4ED8'},
};
const CAT_COLORS = [
  {color:'#8B5CF6', lt:'#EDE9FE', dk:'#6D28D9'},
  {color:'#EC4899', lt:'#FCE7F3', dk:'#BE185D'},
  {color:'#EAB308', lt:'#FEF9C3', dk:'#A16207'},
  {color:'#14B8A6', lt:'#CCFBF1', dk:'#0F766E'},
  {color:'#6B7280', lt:'#F3F4F6', dk:'#374151'},
  {color:'#F43F5E', lt:'#FFE4E6', dk:'#BE123C'},
];
let _colorMap = {};
function getColorForSeg(seg) {
  if (seg.category && CATEGORY_COLORS[seg.category]) return CATEGORY_COLORS[seg.category];
  if (_colorMap[seg.label] === undefined) {
    const idx = Object.keys(_colorMap).length % CAT_COLORS.length;
    _colorMap[seg.label] = idx;
  }
  return CAT_COLORS[_colorMap[seg.label]];
}
function getColorForLabel(label) {
  if (_colorMap[label] === undefined) {
    const idx = Object.keys(_colorMap).length % CAT_COLORS.length;
    _colorMap[label] = idx;
  }
  return CAT_COLORS[_colorMap[label]];
}
```

Replace `buildGantt` bar color and tooltip to use `getColorForSeg`:

```javascript
function buildGantt(segments) {
  _segments = segments;
  _dur = segments.length ? segments[segments.length - 1].end_sec : 1;

  const TICK = 9;
  document.getElementById('gRuler').innerHTML =
    Array.from({length: TICK + 1}, (_, i) => {
      const s = _dur / TICK * i;
      return `<div class="g-tick" style="left:${(i/TICK*100).toFixed(3)}%">${mmss(s)}</div>`;
    }).join('');

  const grids = Array.from({length: TICK - 1}, (_, i) =>
    `<div class="g-vline" style="left:calc(200px + (100% - 200px)*${((i+1)/TICK).toFixed(4)})"></div>`
  ).join('');

  const rows = segments.map((seg, idx) => {
    const c = getColorForSeg(seg);
    const l = (seg.start_sec / _dur * 100).toFixed(4);
    const w = ((seg.end_sec - seg.start_sec) / _dur * 100).toFixed(4);
    const desc = seg.description ? escHtml(seg.description) : '';
    const catName = seg.category === 'seimi' ? '正味' : seg.category === 'fuzui' ? '付随' : seg.category === 'muda' ? 'ムダ' : '';
    return `
    <div class="g-row">
      <div class="g-lbl">
        <span class="g-lbl-num">${idx+1}.</span>
        ${escHtml(seg.label)}
      </div>
      <div class="g-bars">
        <div class="g-bar"
          style="left:${l}%;width:${w}%;background:${c.color}"
          data-label="${escHtml(seg.label)}"
          data-cat="${escHtml(catName)}"
          data-desc="${desc}"
          data-improvement="${seg.improvement ? escHtml(seg.improvement) : ''}"
          data-start="${seg.start_sec}" data-end="${seg.end_sec}"
          onclick="window.dispatchEvent(new CustomEvent('seek-to',{detail:${seg.start_sec}}))">
        </div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('gBody').innerHTML =
    grids + rows +
    `<div class="g-ph" id="gPh"><div class="g-ph-badge" id="gBadge">${mmss(0)}</div></div>`;
  updatePh();
}
```

Update tooltip handler to show category + description + improvement:

```javascript
document.addEventListener('mouseover', e => {
  const bar = e.target.closest('.g-bar');
  if (!bar) return;
  const lbl = bar.dataset.label;
  const cat = bar.dataset.cat;
  const desc = bar.dataset.desc;
  const improvement = bar.dataset.improvement;
  const s = +bar.dataset.start, en = +bar.dataset.end;
  // Determine color
  const catKey = cat === '正味' ? 'seimi' : cat === '付随' ? 'fuzui' : cat === 'ムダ' ? 'muda' : null;
  const c = catKey ? CATEGORY_COLORS[catKey] : getColorForLabel(lbl);
  document.getElementById('tip-cat').textContent = cat || lbl;
  document.getElementById('tip-cat').style.background = c.lt;
  document.getElementById('tip-cat').style.color = c.dk;
  document.getElementById('tip-name').textContent = desc || '';
  document.getElementById('tip-time').textContent = `⏱ ${mmss1(s)} → ${mmss1(en)}`;
  document.getElementById('tip-dur').textContent = improvement ? `💡 ${improvement}` : `⌛ ${durStr(en-s)}`;
  tip.classList.add('on');
});
```

Also update the sidebar `getColor` call in the Alpine.js `app()` to use category:

In the sidebar `x-for` template, replace:
```html
<span class="tag"
      :style="'background:'+getColor(seg.label).lt+';color:'+getColor(seg.label).dk"
      x-text="seg.label"></span>
```
with:
```html
<span class="tag"
      :style="'background:'+getCatColor(seg).lt+';color:'+getCatColor(seg).dk"
      x-text="seg.label"></span>
```

Add `getCatColor` to the Alpine `app()`:
```javascript
getCatColor(seg) {
  if (seg.category && window.CATEGORY_COLORS && window.CATEGORY_COLORS[seg.category])
    return window.CATEGORY_COLORS[seg.category];
  return this.getColor(seg.label);
},
```

And add `window.CATEGORY_COLORS = CATEGORY_COLORS;` right after `const CATEGORY_COLORS = {...}`.

Finally, replace `buildStats` so the donut chart is **category-based when categories exist**
(spec §7.1: カテゴリ別円グラフ), falling back to label-based for track_b:

```javascript
function buildStats(segments) {
  if (!segments.length) return;
  const hasCategories = segments.some(s => s.category);
  const CAT_NAMES = {seimi: '正味作業', fuzui: '付随作業', muda: 'ムダ作業'};

  // Group either by category (enriched tracks) or by label (track_b fallback)
  const keyOf  = s => hasCategories ? (s.category || 'unknown') : s.label;
  const nameOf = k => hasCategories ? (CAT_NAMES[k] || '未分類') : k;
  const colorOf = k => hasCategories
    ? (CATEGORY_COLORS[k] || {color:'#6B7280', lt:'#F3F4F6', dk:'#374151'})
    : getColorForLabel(k);

  const totals = {}, counts = {};
  segments.forEach(seg => {
    const k = keyOf(seg);
    const d = seg.end_sec - seg.start_sec;
    totals[k] = (totals[k] || 0) + d;
    counts[k] = (counts[k] || 0) + 1;
  });
  const total = Object.values(totals).reduce((a, b) => a + b, 0);
  const keys = Object.keys(totals);

  // Donut
  const cx=140, cy=140, R=118, ri=66;
  let a = -Math.PI/2, paths = '';
  keys.forEach(k => {
    const c = colorOf(k);
    const sl = totals[k] / total;
    const ae = a + sl * Math.PI * 2;
    const [x1,y1]=[cx+R*Math.cos(a), cy+R*Math.sin(a)];
    const [x2,y2]=[cx+R*Math.cos(ae),cy+R*Math.sin(ae)];
    const [xi1,yi1]=[cx+ri*Math.cos(a), cy+ri*Math.sin(a)];
    const [xi2,yi2]=[cx+ri*Math.cos(ae),cy+ri*Math.sin(ae)];
    const lg = sl > .5 ? 1 : 0;
    const mid = a + sl*Math.PI, mr = (R+ri)/2;
    paths += `<path d="M${xi1},${yi1} L${x1},${y1} A${R},${R} 0 ${lg},1 ${x2},${y2} L${xi2},${yi2} A${ri},${ri} 0 ${lg},0 ${xi1},${yi1}Z" fill="${c.color}" stroke="#fff" stroke-width="3"/>`;
    if (sl > .05) paths += `<text x="${(cx+mr*Math.cos(mid)).toFixed(1)}" y="${(cy+mr*Math.sin(mid)).toFixed(1)}" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="13" font-weight="800">${(sl*100).toFixed(0)}%</text>`;
    a = ae;
  });
  const tm = Math.floor(total/60), ts = Math.floor(total%60);
  paths += `<text x="140" y="134" text-anchor="middle" fill="#374151" font-size="16" font-weight="800">${tm}分${ts}秒</text>
            <text x="140" y="154" text-anchor="middle" fill="#9CA3AF" font-size="12">合計</text>`;
  document.getElementById('pieSvg').innerHTML = paths;

  // Table (category rows when enriched, label rows otherwise)
  document.getElementById('sTable').innerHTML = keys.map(k => {
    const c = colorOf(k);
    const s = totals[k];
    const m = Math.floor(s/60), sec = Math.floor(s%60);
    return `
    <div class="s-row">
      <div class="s-dot" style="background:${c.color}"></div>
      <div class="s-name">${escHtml(nameOf(k))}</div>
      <div class="s-cnt">${counts[k]}件</div>
      <div class="s-val">
        <div class="s-t">${m}分${sec}秒</div>
        <div class="s-p">${(s/total*100).toFixed(1)}%</div>
      </div>
    </div>`;
  }).join('');
  document.getElementById('sTot').textContent = `${tm}分${ts}秒`;
}
```

- [ ] **Step 4: Add category CSS to `app.css`**

Add after existing `.tag` rules:

```css
/* Category badge colors */
.cat-seimi { color: #15803D; background: #DCFCE7; border-radius: 4px; padding: 1px 5px; }
.cat-fuzui { color: #C2410C; background: #FFEDD5; border-radius: 4px; padding: 1px 5px; }
.cat-muda  { color: #1D4ED8; background: #DBEAFE; border-radius: 4px; padding: 1px 5px; }
.cat-none  { color: #6B7280; }

/* Improvement row in table */
.seg-improvement td {
  font-size: 11px;
  color: #7C3AED;
  background: #F5F3FF;
  padding: 2px 8px 4px 20px;
  border-bottom: 1px solid #E9D5FF;
}

/* PDF parse status in selected chip */
.pdf-parse-status { font-size: 11px; color: #15803D; margin-left: 6px; }
```

- [ ] **Step 5: Manual verification**

```
uvicorn src.web.app:app --reload
```

Then open `http://localhost:8000`, upload a video, run Track B, and confirm:
- Gantt chart populates after analysis completes
- Bars use label-hash colors (track_b has no category)
- Stats donut groups by label (track_b fallback)
- Track selector shows STD / B / A tabs
- Track select in label form defaults to 標準（std）

- [ ] **Step 6: Commit**

```bash
git add src/web/templates/_label_form.html src/web/templates/_timeline.html src/web/templates/index.html src/web/static/app.css
git commit -m "fix: Gantt segments-loaded script tag; category colors, stats, PDF wiring, vocab prefill"
```

---

## Task 11: Evaluate — Boundary Deviation Log

**Files:**
- Modify: `src/evaluate/metrics.py`

- [ ] **Step 1: Add `boundary_deviation_log` function**

In `src/evaluate/metrics.py`, add after `compute_all`:

```python
def boundary_deviation_log(
    pred: SegmentList,
    gt: SegmentList,
) -> dict[str, list[float]]:
    """Return per-boundary deviation between pred and gt boundary timestamps.

    NOTE: track_std's Gemini boundary refinement may move boundaries 1-2s from the
    annotator's physical timestamps to IE-logical positions (e.g., 'tool touch = start').
    A lower F1@10 vs track_b does NOT necessarily mean worse quality — sample-review
    the video to verify logical correctness before drawing conclusions.
    """
    pred_bounds = sorted({s.start_sec for s in pred.segments} | {s.end_sec for s in pred.segments})
    gt_bounds   = sorted({s.start_sec for s in gt.segments}   | {s.end_sec for s in gt.segments})

    deviations: dict[str, list[float]] = {
        "matched_deviations": [],   # |pred_b - nearest_gt_b| for matched pairs
        "unmatched_pred": [],       # pred boundaries with no gt within 50s
        "unmatched_gt": [],         # gt boundaries with no pred within 50s
    }

    gt_used = set()
    for pb in pred_bounds:
        candidates = [(abs(pb - gb), i) for i, gb in enumerate(gt_bounds) if i not in gt_used]
        if not candidates:
            deviations["unmatched_pred"].append(pb)
            continue
        dist, idx = min(candidates)
        if dist <= 50.0:
            deviations["matched_deviations"].append(dist)
            gt_used.add(idx)
        else:
            deviations["unmatched_pred"].append(pb)

    for i, gb in enumerate(gt_bounds):
        if i not in gt_used:
            deviations["unmatched_gt"].append(gb)

    return deviations
```

- [ ] **Step 2: Add tests for boundary_deviation_log**

Add to `tests/evaluate/test_metrics.py`:

```python
def test_boundary_deviation_log_matched(ground_truth_segments, perfect_prediction):
    from src.evaluate.metrics import boundary_deviation_log
    dev = boundary_deviation_log(perfect_prediction, ground_truth_segments)
    # perfect prediction → every deviation is 0
    assert all(d == 0.0 for d in dev["matched_deviations"])
    assert dev["unmatched_pred"] == []
    assert dev["unmatched_gt"] == []


def test_boundary_deviation_log_empty_gt():
    from src.evaluate.metrics import boundary_deviation_log
    from src.schemas import Segment, SegmentList
    pred = SegmentList("v", 1.0, ["A"], [Segment(0.0, 10.0, "A", 1.0)], "track_std")
    gt = SegmentList("v", 1.0, ["A"], [], "ground_truth")
    dev = boundary_deviation_log(pred, gt)
    # no gt boundaries → all pred boundaries unmatched, no crash
    assert len(dev["unmatched_pred"]) == 2  # 0.0 and 10.0


def test_boundary_deviation_log_shifted_boundary(ground_truth_segments):
    from src.evaluate.metrics import boundary_deviation_log
    from src.schemas import Segment, SegmentList
    import copy
    pred = copy.deepcopy(ground_truth_segments)
    pred.source = "track_std"
    # shift one boundary by 2s (Gemini IE-logical correction scenario)
    pred.segments[0] = Segment(0.0, 12.0, "作業A", 1.0)
    pred.segments[1] = Segment(12.0, 20.0, "作業B", 1.0)
    dev = boundary_deviation_log(pred, ground_truth_segments)
    assert any(abs(d - 2.0) < 0.01 for d in dev["matched_deviations"])
```

- [ ] **Step 3: Run tests**

```
pytest tests/evaluate/ -v
```
Expected: all PASS (existing + 3 new)

- [ ] **Step 4: Commit**

```bash
git add src/evaluate/metrics.py tests/evaluate/test_metrics.py
git commit -m "feat: add boundary_deviation_log to evaluate/metrics for boundary drift diagnosis"
```

---

## Task 12: Integration Test Update

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Add track_std integration test (Gemini mocked)**

Add to `tests/test_integration.py`:

```python
import json
from unittest.mock import MagicMock, patch

import pytest


def _gemini_resp(segs):
    m = MagicMock()
    m.text = json.dumps(segs)
    return m


@patch("src.pipeline.label_gemini.genai")
def test_track_std_pipeline_produces_valid_segment_list(mock_genai, synthetic_video_path, tmp_path):
    from src.pipeline.ingest import extract_frames
    from src.pipeline.embed import embed_frames
    from src.pipeline.presegment import detect_boundaries
    from src.pipeline.label_gemini import label_gemini
    from src.pipeline.aggregate import aggregate
    from src.schemas import SegmentList

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.generate_content.return_value = _gemini_resp([
        {"start_sec": 0.0, "end_sec": 10.0, "label": "部品取り出し",
         "category": "fuzui", "description": "棚から取る", "improvement": None, "confidence": 0.9},
        {"start_sec": 10.0, "end_sec": 20.0, "label": "ネジ締め",
         "category": "seimi", "description": "4本締結", "improvement": None, "confidence": 0.95},
        {"start_sec": 20.0, "end_sec": 30.0, "label": "手待ち",
         "category": "muda", "description": "次工程待ち", "improvement": "同期化で削減可", "confidence": 0.8},
    ])

    with patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}):
        frames = extract_frames(synthetic_video_path, fps=1.0)
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=10.0)
        seg_list = label_gemini(synthetic_video_path, ["部品取り出し", "ネジ締め", "手待ち"], boundaries)

    # Invariant: continuous, non-overlapping, covers [0, total_duration]
    assert seg_list.segments[0].start_sec == pytest.approx(0.0, abs=0.1)
    for i in range(len(seg_list.segments) - 1):
        a, b = seg_list.segments[i], seg_list.segments[i + 1]
        assert a.end_sec <= b.start_sec + 0.01, f"Gap/overlap between seg {i} and {i+1}"

    # Aggregate works without error
    stats = aggregate(seg_list)
    assert stats["total_sec"] > 0
    assert "by_category" in stats

    # Enrich fields present
    cats = {s.category for s in seg_list.segments if s.category}
    assert cats <= {"seimi", "fuzui", "muda"}
```

- [ ] **Step 2: Run all tests**

```
pytest tests/ -v --tb=short
```
Expected: all PASS

- [ ] **Step 3: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add track_std integration test covering stitch invariant + aggregate"
```

---

## Task 13: CLI — run_pipeline.py track std

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Add std branch to the CLI**

In `scripts/run_pipeline.py`:

1. Add import at top: `from src.pipeline.label_gemini import label_gemini`
2. Update the `track` option help text: `help="Which track to run: std | a | b | both | all"`
3. Add this branch **before** the existing Track B block (std reuses B's Stage 0/1 outputs):

```python
    if track in ("std", "all"):
        typer.echo("\n── Track STD (CLIP boundaries + Gemini refinement) ──")
        frames = extract_frames(video, fps=fps, blur_faces=blur_faces)
        typer.echo(f"  Extracted {len(frames)} frames")
        timestamps, embeddings = embed_frames(frames)
        boundaries = detect_boundaries(timestamps, embeddings, penalty=penalty)
        typer.echo(f"  Boundaries: {[f'{b:.1f}s' for b in boundaries]}")
        seg_list = label_gemini(
            video, label_list, boundaries,
            blur_faces=blur_faces, raw_output_dir=output_dir,
        )
        path = save_segments(seg_list, output_dir)
        typer.echo(f"  Saved: {path}")
        typer.echo(to_timeline_markdown(seg_list))
```

4. Change the existing two conditions to also accept `"all"`:
   - `if track in ("b", "both"):` → `if track in ("b", "both", "all"):`
   - `if track in ("a", "both"):` → `if track in ("a", "both", "all"):`

(`run_evaluate.py` needs no change — it keys predictions by `seg.source`, so
`*_track_std.json` files flow through `compare_systems` automatically for 3-track comparison.)

- [ ] **Step 2: Smoke-check the CLI parses**

```
python -c "import scripts.run_pipeline"
```
Expected: no ImportError

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v --tb=short
```
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat: add --track std / all to run_pipeline CLI for 3-track benchmarking"
```

---

## Self-Review Checklist

### Spec Coverage
- [x] §3.1 Segment enrich fields → Task 1
- [x] §3.2 Category 3 values (seimi/fuzui/muda) → Task 4 `_CATEGORY_ALIASES`
- [x] §3.3 Hint dataclass → Task 1; /analyze hints receiver → Task 9
- [x] §4.2 label_gemini (window alignment, stitch, invariant, normalization) → Tasks 4-5
- [x] §4.2-5 J: label synonym normalization → Task 4 `_LABEL_SYNONYMS` + `_normalize_label`
- [x] §4.2-6 K: enrich core-priority merge → Task 4 `_merge_adjacent_enrich`
- [x] §4.2-7 reproducibility (raw responses, temperature=0) → Task 4 (`raw_output_dir`)
- [x] §4.4 N: TRACK_RUNNERS registry → Task 8
- [x] §5 propose_labels → Task 7; vocab prefill wiring → Task 10 Step 2(c)
- [x] §6 parse_reference + L guardrails → Task 6; deps → Task 0
- [x] §7.1 aggregate → Task 2; category stats UI → Task 10 buildStats
- [x] §7.2 report update → Task 3
- [x] §8.1 /upload-pdf + /propose-labels + /analyze track_std + hints → Task 9
- [x] §8.2 Gantt segments-loaded bug fix → Task 10
- [x] §8.3 Category colors → Task 10
- [x] §8.4 description/improvement display → Task 10
- [x] §8.5 graceful degradation (track_b no enrich → label hash color) → Task 10
- [x] §8.6 label form (std default, PDF wiring, vocab prefill) → Task 10 Step 2
- [x] §9.2 M: boundary deviation log → Task 11 (with tests)
- [x] §11 test plan (all Gemini mocked) → Tasks 1-5, 7-9, 11, 12
- [x] 実装順序7: 評価/CLI 3トラック対応 → Task 13

### No Placeholders
Reviewed — no TBD/TODO/placeholder steps found.

### Type Consistency
- `Segment` constructor with 7 positional params used consistently in `label_gemini.py`
- `_stitch` returns `list[Segment]` → `_merge_adjacent_enrich` accepts and returns same
- `TRACK_RUNNERS["std"]` expects `(video_path, label_vocabulary, **opts)` → `_run_std` matches
- `ids.get_ref_context` / `ids.store_ref_context` used in routes.py → added to ids.py in Task 8
- `hints` flows: routes.analyze → jobs.start_pipeline → _run_pipeline → runner → label_gemini
- `raw_output_dir` param: label_gemini default `"results"` matches web `_RESULTS_DIR`; CLI passes `output_dir`

### Mocking Consistency (re-reviewed)
- All patch targets (`src.pipeline.label_gemini.genai`, `src.pipeline.parse_reference.{pdfplumber,fitz,Image,genai}`, `src.pipeline.propose_labels.genai`) correspond to **module-level imports** in source — function-local imports would bypass the mocks (fixed in review)

---

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-06-10-ollo-tools-gemini.md`.

**1. Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
