"""Export classified DJ tool segments to various formats."""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ClassifiedSegment:
    """A segment that has been classified as a DJ tool type."""

    source_file: str
    segment_file: str
    start_s: float
    end_s: float
    predicted_class: str
    confidence: float
    all_scores: dict[str, float]


def export_to_folders(
    segments: list[ClassifiedSegment],
    output_dir: Path,
) -> None:
    """Copy segment files into class-named subdirectories."""
    for seg in segments:
        class_dir = output_dir / seg.predicted_class
        class_dir.mkdir(parents=True, exist_ok=True)
        src = Path(seg.segment_file)
        if src.exists():
            shutil.copy2(src, class_dir / src.name)

    logger.info("Exported %d segments to folders in %s", len(segments), output_dir)


def export_to_csv(
    segments: list[ClassifiedSegment],
    output_path: Path,
) -> None:
    """Write classification results to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "start", "end", "class", "confidence", "segment_file"])
        for seg in segments:
            writer.writerow([
                seg.source_file,
                f"{seg.start_s:.3f}",
                f"{seg.end_s:.3f}",
                seg.predicted_class,
                f"{seg.confidence:.4f}",
                seg.segment_file,
            ])

    logger.info("Wrote %d results to %s", len(segments), output_path)


def export_to_json(
    segments: list[ClassifiedSegment],
    output_path: Path,
) -> None:
    """Write classification results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(seg) for seg in segments]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Wrote %d results to %s", len(segments), output_path)
