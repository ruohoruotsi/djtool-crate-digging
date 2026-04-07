"""Fusion of MSAF structural boundaries with SMAD speech/music windows.

Implements Algorithm 1 from the ISMIR 2024 LBD paper: for each MSAF boundary,
snap to the nearest SMAD speech onset/offset within a tolerance window.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def fuse_boundaries(
    msaf_segments: list[tuple[float, float]],
    speech_intervals: list[tuple[float, float]],
    music_intervals: list[tuple[float, float]],
    tolerance_s: float = 1.0,
    min_segment_duration_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Adjust MSAF boundaries to align with SMAD speech onsets/offsets.

    For each MSAF segment boundary, look for the nearest speech interval
    edge within the tolerance window. If found, snap the boundary to it.
    This produces segments that more cleanly isolate speech-only or
    music-only regions for DJ tool extraction.

    Args:
        msaf_segments: Structural segments from MSAF as (start, end) tuples.
        speech_intervals: Merged speech intervals from SMAD.
        music_intervals: Merged music intervals from SMAD.
        tolerance_s: Max distance (seconds) to snap a boundary to a speech edge.
        min_segment_duration_s: Discard fused segments shorter than this.

    Returns:
        Refined list of (start, end) segment tuples.
    """
    if not msaf_segments:
        return []

    speech_edges = _collect_edges(speech_intervals)

    adjusted_boundaries: list[float] = []
    all_boundaries = _segments_to_boundaries(msaf_segments)

    for boundary in all_boundaries:
        nearest = _find_nearest_edge(boundary, speech_edges, tolerance_s)
        adjusted_boundaries.append(nearest if nearest is not None else boundary)

    adjusted_boundaries = sorted(set(adjusted_boundaries))

    segments: list[tuple[float, float]] = []
    for i in range(len(adjusted_boundaries) - 1):
        start = adjusted_boundaries[i]
        end = adjusted_boundaries[i + 1]
        if (end - start) >= min_segment_duration_s:
            segments.append((round(start, 3), round(end, 3)))

    logger.info(
        "Fusion: %d MSAF segments -> %d fused segments (tolerance=%.1fs)",
        len(msaf_segments),
        len(segments),
        tolerance_s,
    )
    return segments


def _segments_to_boundaries(segments: list[tuple[float, float]]) -> list[float]:
    """Extract unique sorted boundary times from a list of segments."""
    boundaries = set()
    for start, end in segments:
        boundaries.add(start)
        boundaries.add(end)
    return sorted(boundaries)


def _collect_edges(intervals: list[tuple[float, float]]) -> list[float]:
    """Collect all onset/offset times from a list of intervals."""
    edges = []
    for start, end in intervals:
        edges.append(start)
        edges.append(end)
    return sorted(edges)


def _find_nearest_edge(
    boundary: float, edges: list[float], tolerance: float
) -> float | None:
    """Find the edge nearest to boundary within tolerance, or None."""
    best = None
    best_dist = tolerance
    for edge in edges:
        dist = abs(edge - boundary)
        if dist <= best_dist:
            best = edge
            best_dist = dist
    return best
