# egocentric-work-analysis

Zero-shot action segmentation for first-person factory line videos.
Benchmarks against Ollo Factory Tools.

## 📖 Phase 1 Manual

👉 **[Phase 1 マニュアル (HTML)](docs/phase1-manual.html)** — セットアップから実行、評価まで

使い方：
1. `docs/phase1-manual.html` をブラウザで開く
2. コマンドをコピーして実行
3. 検証ポイントを確認

## Setup

```bash
pip install -e ".[dev]"
export GEMINI_API_KEY=your_key_here
```

## Quick Start

```bash
# Track B (staged pipeline) + Track A (Gemini):
python -m scripts.run_pipeline video.mp4 "部品取り出し,ネジ締め,検査" --track both

# Evaluate against ground truth annotation:
python -m scripts.run_evaluate annotations/gt.json results/video_track_b.json results/video_track_a.json
```

## Tracks

- **Track A**: Gemini 2.5 Pro single-pass (windowed, 5 min/window)
- **Track B**: CLIP ViT-B-32 embedding → ruptures PELT change-point detection → zero-shot labeling

## Test

```bash
python -m pytest tests/ -v --ignore=tests/pipeline/test_embed.py
# 37 tests passing (CLIP model download optional)
```

## Architecture

| Phase | Status | Content |
|-------|--------|---------|
| **Phase 1** | ✅ Completed | CLI pipeline (Track A/B), TAS metrics, evaluation framework |
| **Phase 2** | 🚧 Planned | Web UI (FastAPI + htmx + Alpine.js) |

## Resources

- Design specs: `docs/superpowers/specs/`
- Implementation plan: `docs/superpowers/plans/`
- Manual: `docs/phase1-manual.html` ← **Start here**
