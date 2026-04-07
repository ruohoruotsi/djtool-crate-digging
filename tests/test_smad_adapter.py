"""Tests for SMAD adapter interval merging."""

from djtool.segmentation.smad_adapter import _merge_intervals


def test_merge_no_intervals():
    assert _merge_intervals([]) == []


def test_merge_single_interval():
    result = _merge_intervals([[1.0, 5.0]], min_segment_duration_s=2.0)
    assert result == [(1.0, 5.0)]


def test_merge_single_short_interval():
    result = _merge_intervals([[1.0, 2.0]], min_segment_duration_s=2.0)
    assert result == []


def test_merge_close_intervals():
    """Intervals with gaps smaller than min_silence get merged."""
    intervals = [[0.0, 3.0], [4.0, 8.0]]
    result = _merge_intervals(intervals, min_segment_duration_s=2.0, min_silence_duration_s=3.0)
    assert len(result) == 1
    assert result[0] == (0.0, 8.0)


def test_merge_distant_intervals():
    """Intervals with large gaps stay separate."""
    intervals = [[0.0, 3.0], [10.0, 15.0]]
    result = _merge_intervals(intervals, min_segment_duration_s=2.0, min_silence_duration_s=3.0)
    assert len(result) == 2
