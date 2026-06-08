#!/usr/bin/env python3
"""Run zero-shot action segmentation on a video file."""
import typer
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
