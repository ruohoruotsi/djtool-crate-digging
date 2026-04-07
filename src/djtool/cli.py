"""Command-line interface for djtool."""

from __future__ import annotations

import logging
from pathlib import Path

import click

from djtool import __version__
from djtool.config import ClassificationConfig, DJToolConfig, SegmentationConfig


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool) -> None:
    """djtool-crate-digga - Zero-shot DJ tool retrieval from personal music libraries."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), default="output", help="Output directory.")
@click.option(
    "--format", "output_format",
    type=click.Choice(["folders", "csv", "json"]),
    default="folders",
    help="Output format.",
)
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu", help="Torch device.")
@click.option("--msaf-algorithm", default="sf", help="MSAF boundary algorithm.")
@click.option("--smad-model-path", type=click.Path(exists=True), default=None, help="Path to TVSM model checkpoint.")
@click.option("--smad-csv", type=click.Path(exists=True), default=None, help="Pre-computed SMAD CSV (skip TVSM inference).")
@click.option("--min-duration", type=float, default=3.0, help="Min segment duration for classification (seconds).")
@click.option("--threshold", type=float, default=0.1, help="Min confidence to keep a classification.")
def process(
    input_path: str,
    output_dir: str,
    output_format: str,
    device: str,
    msaf_algorithm: str,
    smad_model_path: str | None,
    smad_csv: str | None,
    min_duration: float,
    threshold: float,
) -> None:
    """Process an audio file or directory for DJ tool segments."""
    from djtool.pipeline import process_library, process_track

    config = DJToolConfig(
        output_dir=Path(output_dir),
        output_format=output_format,
        device=device,
        segmentation=SegmentationConfig(
            msaf_boundaries_id=msaf_algorithm,
            smad_model_path=Path(smad_model_path) if smad_model_path else None,
        ),
        classification=ClassificationConfig(
            min_duration_s=min_duration,
            confidence_threshold=threshold,
            device=device,
        ),
    )

    input_path = Path(input_path)
    if input_path.is_file():
        results = process_track(input_path, config, smad_csv_path=smad_csv)
    else:
        results = process_library(input_path, config)

    click.echo(f"Found {len(results)} DJ tool segments.")


@main.command()
@click.argument("segments_dir", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), default="output", help="Output directory.")
@click.option("--format", "output_format", type=click.Choice(["folders", "csv", "json"]), default="folders")
@click.option("--device", type=click.Choice(["cpu", "cuda", "mps"]), default="cpu")
@click.option("--threshold", type=float, default=0.1)
def classify(
    segments_dir: str,
    output_dir: str,
    output_format: str,
    device: str,
    threshold: float,
) -> None:
    """Classify pre-cut audio segments (CLAP only, no segmentation)."""
    from djtool.audio.export import ClassifiedSegment, export_to_csv, export_to_folders, export_to_json
    from djtool.classification.clap import classify_file

    segments_path = Path(segments_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in segments_path.rglob("*") if f.suffix.lower() in (".wav", ".mp3", ".flac")
    )

    classified: list[ClassifiedSegment] = []
    for f in audio_files:
        scores = classify_file(f, device=device)
        if scores is None:
            continue

        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]

        if confidence >= threshold:
            classified.append(ClassifiedSegment(
                source_file=str(f),
                segment_file=str(f),
                start_s=0.0,
                end_s=0.0,
                predicted_class=best_class,
                confidence=confidence,
                all_scores=scores,
            ))

    if output_format == "folders":
        export_to_folders(classified, output / "classified")
    elif output_format == "csv":
        export_to_csv(classified, output / "results.csv")
    elif output_format == "json":
        export_to_json(classified, output / "results.json")

    click.echo(f"Classified {len(classified)}/{len(audio_files)} segments.")


@main.command("download-models")
@click.option("--output-dir", type=click.Path(), default="models", help="Directory to save models.")
def download_models(output_dir: str) -> None:
    """Download required model checkpoints (TVSM)."""
    click.echo("Model download not yet implemented.")
    click.echo("Please manually download the TVSM checkpoint from:")
    click.echo("  https://github.com/biboamy/TVSM-dataset (see README for Google Drive link)")
    click.echo(f"  Place the .pt file in: {output_dir}/")
