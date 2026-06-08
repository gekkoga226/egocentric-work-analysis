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
