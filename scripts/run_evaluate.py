#!/usr/bin/env python3
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
