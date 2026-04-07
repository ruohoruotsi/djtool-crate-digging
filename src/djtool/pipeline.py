"""End-to-end DJ tool retrieval pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from djtool.audio.export import ClassifiedSegment, export_to_csv, export_to_folders, export_to_json
from djtool.audio.io import extract_all_segments
from djtool.classification.clap import classify_file
from djtool.classification.prompts import DEFAULT_DJTOOL_CLASSES, get_class_names
from djtool.config import DJToolConfig
from djtool.segmentation.fusion import fuse_boundaries
from djtool.segmentation.msaf_adapter import get_structural_segments
from djtool.segmentation.smad_adapter import detect_speech_music, load_smad_csv

logger = logging.getLogger(__name__)


def process_track(
    audio_path: str | Path,
    config: DJToolConfig | None = None,
    smad_csv_path: str | Path | None = None,
) -> list[ClassifiedSegment]:
    """Run the full DJ tool retrieval pipeline on a single audio file.

    Args:
        audio_path: Path to the audio file to process.
        config: Pipeline configuration. Uses defaults if None.
        smad_csv_path: Optional pre-computed SMAD CSV. If None, runs TVSM inference.

    Returns:
        List of classified segments found in the track.
    """
    config = config or DJToolConfig()
    audio_path = Path(audio_path)
    seg_cfg = config.segmentation
    cls_cfg = config.classification

    logger.info("Processing: %s", audio_path.name)

    # Step 1: Structural segmentation (MSAF)
    msaf_segments = get_structural_segments(
        audio_path,
        boundaries_id=seg_cfg.msaf_boundaries_id,
        labels_id=seg_cfg.msaf_labels_id,
    )

    # Step 2: Speech/music detection (SMAD)
    if smad_csv_path is not None:
        speech_intervals, music_intervals = load_smad_csv(
            smad_csv_path,
            min_segment_duration_s=seg_cfg.min_segment_duration_s,
            min_silence_duration_s=seg_cfg.min_silence_duration_s,
        )
    else:
        raw_results = detect_speech_music(
            audio_path,
            model_path=seg_cfg.smad_model_path,
            music_threshold=seg_cfg.smad_music_threshold,
            speech_threshold=seg_cfg.smad_speech_threshold,
            device=config.device,
        )
        speech_intervals = [
            (r["start_time_s"], r["end_time_s"])
            for r in raw_results
            if r["speech_prob"] >= seg_cfg.smad_speech_threshold
        ]
        music_intervals = [
            (r["start_time_s"], r["end_time_s"])
            for r in raw_results
            if r["music_prob"] >= seg_cfg.smad_music_threshold
        ]

    # Step 3: Fuse MSAF boundaries with SMAD windows
    fused_segments = fuse_boundaries(
        msaf_segments,
        speech_intervals,
        music_intervals,
        tolerance_s=seg_cfg.fusion_tolerance_s,
        min_segment_duration_s=seg_cfg.min_segment_duration_s,
    )

    # Step 4: Extract audio segments
    track_output_dir = config.output_dir / "segments" / audio_path.stem
    segment_files = extract_all_segments(audio_path, fused_segments, track_output_dir)

    # Step 5: Classify each segment with CLAP
    class_names = get_class_names()
    classified: list[ClassifiedSegment] = []

    for seg_path, (start, end) in zip(segment_files, fused_segments):
        scores = classify_file(
            seg_path,
            sr=cls_cfg.sample_rate,
            model_name=cls_cfg.model_name,
            device=cls_cfg.device,
            min_duration_s=cls_cfg.min_duration_s,
        )
        if scores is None:
            continue

        best_class = max(scores, key=scores.get)
        confidence = scores[best_class]

        if confidence >= cls_cfg.confidence_threshold:
            classified.append(ClassifiedSegment(
                source_file=str(audio_path),
                segment_file=str(seg_path),
                start_s=start,
                end_s=end,
                predicted_class=best_class,
                confidence=confidence,
                all_scores=scores,
            ))

    logger.info(
        "Classified %d/%d segments as DJ tools from %s",
        len(classified), len(segment_files), audio_path.name,
    )

    # Step 6: Export
    _export_results(classified, config)

    return classified


def process_library(
    directory: str | Path,
    config: DJToolConfig | None = None,
    extensions: tuple[str, ...] = (".wav", ".mp3", ".flac", ".aiff", ".ogg"),
) -> list[ClassifiedSegment]:
    """Process all audio files in a directory.

    Args:
        directory: Directory containing audio files.
        config: Pipeline configuration.
        extensions: Audio file extensions to look for.

    Returns:
        Combined list of classified segments from all tracks.
    """
    config = config or DJToolConfig()
    directory = Path(directory)

    audio_files = sorted(
        f for f in directory.rglob("*") if f.suffix.lower() in extensions
    )
    logger.info("Found %d audio files in %s", len(audio_files), directory)

    all_results: list[ClassifiedSegment] = []
    for audio_file in audio_files:
        try:
            results = process_track(audio_file, config)
            all_results.extend(results)
        except Exception:
            logger.exception("Failed to process %s", audio_file)

    logger.info("Total: %d DJ tool segments from %d tracks", len(all_results), len(audio_files))
    return all_results


def _export_results(segments: list[ClassifiedSegment], config: DJToolConfig) -> None:
    """Export results in the configured format."""
    if not segments:
        return

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output_format == "folders":
        export_to_folders(segments, output_dir / "classified")
    elif config.output_format == "csv":
        export_to_csv(segments, output_dir / "results.csv")
    elif config.output_format == "json":
        export_to_json(segments, output_dir / "results.json")
    else:
        logger.warning("Unknown output format: %s", config.output_format)
