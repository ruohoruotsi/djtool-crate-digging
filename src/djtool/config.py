"""Configuration for the djtool pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SegmentationConfig:
    """Parameters for the segmentation stage (MSAF + SMAD)."""

    msaf_boundaries_id: str = "sf"
    msaf_labels_id: str = "fmc2d"
    smad_model_path: Path | None = None
    smad_music_threshold: float = 0.5
    smad_speech_threshold: float = 0.5
    min_segment_duration_s: float = 2.0
    min_silence_duration_s: float = 3.0
    fusion_tolerance_s: float = 1.0


@dataclass
class ClassificationConfig:
    """Parameters for the CLAP classification stage."""

    model_name: str = "laion/clap-htsat-unfused"
    min_duration_s: float = 3.0
    confidence_threshold: float = 0.1
    sample_rate: int = 48000
    device: str = "cpu"


@dataclass
class DJToolConfig:
    """Top-level configuration for the djtool pipeline."""

    output_dir: Path = Path("output")
    output_format: str = "folders"  # folders, csv, json
    device: str = "cpu"  # cpu, cuda, mps
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    def __post_init__(self):
        self.classification.device = self.device
