"""Tests for the MSAF + SMAD boundary fusion logic."""

from djtool.segmentation.fusion import fuse_boundaries, _find_nearest_edge, _collect_edges


def test_fuse_boundaries_no_speech():
    """Without speech edges, MSAF segments pass through unchanged."""
    msaf = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    result = fuse_boundaries(msaf, speech_intervals=[], music_intervals=[], tolerance_s=1.0)
    assert result == msaf


def test_fuse_boundaries_snap_to_speech():
    """Boundaries snap to nearby speech edges within tolerance."""
    msaf = [(0.0, 10.0), (10.0, 20.0)]
    speech = [(9.5, 15.0)]
    result = fuse_boundaries(msaf, speech, [], tolerance_s=1.0, min_segment_duration_s=1.0)
    assert any(s == 9.5 or e == 9.5 for s, e in result)


def test_fuse_boundaries_empty_input():
    result = fuse_boundaries([], [], [])
    assert result == []


def test_find_nearest_edge_within_tolerance():
    edges = [5.0, 10.0, 15.0]
    assert _find_nearest_edge(10.3, edges, 0.5) == 10.0
    assert _find_nearest_edge(10.8, edges, 0.5) is None


def test_collect_edges():
    intervals = [(1.0, 3.0), (5.0, 7.0)]
    assert _collect_edges(intervals) == [1.0, 3.0, 5.0, 7.0]
