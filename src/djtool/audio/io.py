"""Audio loading and segment extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)


def extract_segment(
    audio_path: str | Path,
    start_s: float,
    end_s: float,
    output_path: str | Path,
) -> Path:
    """Extract a time segment from an audio file and save it.

    Args:
        audio_path: Source audio file.
        start_s: Start time in seconds.
        end_s: End time in seconds.
        output_path: Where to save the extracted segment.

    Returns:
        Path to the saved segment file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    info = torchaudio.info(str(audio_path))
    sr = info.sample_rate
    start_frame = int(start_s * sr)
    num_frames = int((end_s - start_s) * sr)

    waveform, _ = torchaudio.load(
        str(audio_path), frame_offset=start_frame, num_frames=num_frames
    )

    torchaudio.save(str(output_path), waveform, sr)
    logger.debug("Extracted segment %.3f-%.3f -> %s", start_s, end_s, output_path)
    return output_path


def extract_all_segments(
    audio_path: str | Path,
    segments: list[tuple[float, float]],
    output_dir: str | Path,
) -> list[Path]:
    """Extract multiple segments from an audio file.

    Args:
        audio_path: Source audio file.
        segments: List of (start_s, end_s) tuples.
        output_dir: Directory to write segment files into.

    Returns:
        List of paths to the saved segment files.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    saved: list[Path] = []
    for i, (start, end) in enumerate(segments):
        filename = f"{i:02d}_{stem}_{start:.3f}_{end:.3f}.wav"
        out = extract_segment(audio_path, start, end, output_dir / filename)
        saved.append(out)

    logger.info("Extracted %d segments from %s -> %s", len(saved), audio_path.name, output_dir)
    return saved
