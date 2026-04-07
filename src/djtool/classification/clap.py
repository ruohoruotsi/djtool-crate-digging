"""CLAP zero-shot audio classification for DJ tool segments."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from djtool.classification.prompts import (
    DEFAULT_DJTOOL_CLASSES,
    DJToolClass,
    get_class_names,
    get_prompt_list,
)

logger = logging.getLogger(__name__)

_model = None
_processor = None


def _load_model(model_name: str = "laion/clap-htsat-unfused", device: str = "cpu"):
    """Lazily load the CLAP model and processor (once)."""
    global _model, _processor
    if _model is None:
        from transformers import AutoProcessor, ClapModel

        logger.info("Loading CLAP model: %s", model_name)
        _model = ClapModel.from_pretrained(model_name).to(device)
        _processor = AutoProcessor.from_pretrained(model_name)
    return _model, _processor


def classify_audio(
    audio: np.ndarray,
    sr: int,
    model_name: str = "laion/clap-htsat-unfused",
    device: str = "cpu",
    classes: list[DJToolClass] | None = None,
) -> dict[str, float]:
    """Classify an audio array against DJ tool classes.

    Args:
        audio: 1D numpy array of audio samples.
        sr: Sample rate.
        model_name: HuggingFace model identifier.
        device: Torch device string.
        classes: DJ tool classes to classify against.

    Returns:
        Dict mapping class name to probability.
    """
    model, processor = _load_model(model_name, device)
    prompts = get_prompt_list(classes)
    names = get_class_names(classes)

    inputs = processor(
        text=prompts, audios=audio, return_tensors="pt", padding=True, sampling_rate=sr
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    probs = outputs.logits_per_audio.softmax(dim=-1).cpu().detach().numpy().squeeze()

    return dict(zip(names, probs.tolist()))


def classify_file(
    audio_path: str | Path,
    sr: int = 48000,
    model_name: str = "laion/clap-htsat-unfused",
    device: str = "cpu",
    classes: list[DJToolClass] | None = None,
    min_duration_s: float = 3.0,
) -> dict[str, float] | None:
    """Load an audio file and classify it.

    Returns None if the file is shorter than min_duration_s.
    """
    audio, file_sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = len(audio) / sr

    if duration < min_duration_s:
        logger.debug("Skipping %s (%.1fs < %.1fs min)", audio_path, duration, min_duration_s)
        return None

    return classify_audio(audio, sr, model_name, device, classes)
