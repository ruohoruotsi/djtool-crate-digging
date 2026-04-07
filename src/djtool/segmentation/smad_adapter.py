"""Adapter for TVSM speech/music activity detection.

Wraps the vendored TVSM inference code and returns structured speech/music intervals.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _merge_intervals(
    intervals: list[list[float]],
    min_segment_duration_s: float = 2.0,
    min_silence_duration_s: float = 3.0,
) -> list[tuple[float, float]]:
    """Merge adjacent intervals and filter short ones.

    Ported from legacy process_segmentation.py merge_smad_output_per_activation().
    """
    intervals = [list(iv) for iv in intervals]
    results: list[list[float]] = []

    while intervals:
        if len(intervals) == 1:
            results.append(intervals.pop(0))
            continue

        gap = intervals[1][0] - intervals[0][1]
        if gap < min_silence_duration_s:
            intervals[0] = [intervals[0][0], max(intervals[0][1], intervals[1][1])]
            intervals.pop(1)
        else:
            results.append(intervals.pop(0))

    return [
        (r[0], r[1])
        for r in results
        if (r[1] - r[0]) >= min_segment_duration_s
    ]


def load_smad_csv(
    csv_path: str | Path,
    min_segment_duration_s: float = 2.0,
    min_silence_duration_s: float = 3.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Load and merge SMAD output from a TSV file.

    Args:
        csv_path: Path to the SMAD TSV output (start, end, label columns).
        min_segment_duration_s: Discard segments shorter than this.
        min_silence_duration_s: Merge segments with gaps shorter than this.

    Returns:
        Tuple of (speech_intervals, music_intervals), each a list of (start, end).
    """
    speech_raw: list[list[float]] = []
    music_raw: list[list[float]] = []

    with open(csv_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            start = round(float(row[0]), 3)
            end = round(float(row[1]), 3)
            label = row[2].strip()
            if label == "s":
                speech_raw.append([start, end])
            elif label == "m":
                music_raw.append([start, end])

    speech = _merge_intervals(speech_raw, min_segment_duration_s, min_silence_duration_s)
    music = _merge_intervals(music_raw, min_segment_duration_s, min_silence_duration_s)

    logger.info("SMAD: %d speech intervals, %d music intervals from %s", len(speech), len(music), csv_path)
    return speech, music


def detect_speech_music(
    audio_path: str | Path,
    model_path: str | Path | None = None,
    music_threshold: float = 0.5,
    speech_threshold: float = 0.5,
    device: str = "cpu",
) -> list[dict]:
    """Run TVSM inference on an audio file.

    Args:
        audio_path: Path to the audio file.
        model_path: Path to the TVSM model checkpoint.
        music_threshold: Threshold for music activation.
        speech_threshold: Threshold for speech activation.
        device: Torch device string.

    Returns:
        List of dicts with keys: start_time_s, end_time_s, music_prob, speech_prob.
    """
    from djtool._vendor.tvsm.detector import SMDetector

    if model_path is None:
        raise ValueError(
            "TVSM model path is required. Run 'djtool-crate-digga download-models' or provide --smad-model-path."
        )

    logger.info("Running TVSM inference on %s", audio_path)
    detector = SMDetector(str(model_path), device=device)
    results = detector.predict_audio(str(audio_path))
    logger.info("TVSM: %d frames detected", len(results))
    return results
