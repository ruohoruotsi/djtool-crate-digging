"""Adapter for MSAF structural segmentation.

Wraps msaf.process() and converts its output into a list of (start, end) segments.
"""

from __future__ import annotations

import logging
from pathlib import Path

import msaf

logger = logging.getLogger(__name__)


def get_structural_segments(
    audio_path: str | Path,
    boundaries_id: str = "sf",
    labels_id: str = "fmc2d",
) -> list[tuple[float, float]]:
    """Run MSAF on an audio file and return structural segment boundaries.

    Args:
        audio_path: Path to the audio file.
        boundaries_id: MSAF boundary algorithm id (e.g. "sf", "cnmf", "foote").
        labels_id: MSAF labeling algorithm id (e.g. "fmc2d", "cnmf", "scluster").

    Returns:
        List of (start_time_s, end_time_s) tuples for each structural segment.
    """
    audio_path = str(audio_path)
    logger.info("Running MSAF on %s (boundaries=%s, labels=%s)", audio_path, boundaries_id, labels_id)

    boundaries, labels = msaf.process(
        audio_path,
        boundaries_id=boundaries_id,
        labels_id=labels_id,
    )

    segments = []
    for i in range(len(boundaries) - 1):
        start = round(boundaries[i], 3)
        end = round(boundaries[i + 1], 3)
        if start < end:
            segments.append((start, end))

    logger.info("Found %d structural segments", len(segments))
    return segments
