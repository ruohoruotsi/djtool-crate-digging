"""DJ tool class definitions and CLAP text prompts.

Each DJ tool class has a name and a text prompt used for zero-shot CLAP classification.
Prompts can be customized by passing a different list to the classifier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DJToolClass:
    name: str
    prompt: str


DEFAULT_DJTOOL_CLASSES: list[DJToolClass] = [
    DJToolClass(
        name="acapella",
        prompt="acapella, expressively sung human vocal with background instrumental music tracks",
    ),
    DJToolClass(
        name="instrumentals",
        prompt="piano, synths or strings, instrumental, guitar",
    ),
    DJToolClass(
        name="drums",
        prompt="drums, a drum loop, drum solo, breakbeat, percussive elements",
    ),
    DJToolClass(
        name="beatbox",
        prompt="beatboxing",
    ),
    DJToolClass(
        name="fx",
        prompt="siren, riser sound effects, whoosh, crash, synthetic, transitional effect",
    ),
    DJToolClass(
        name="vinyl_fx",
        prompt="vinyl scratch loop, turnatablist DJ battle sounds",
    ),
    DJToolClass(
        name="drops",
        prompt="a high energy, high tension, climactic, massive EDM drop",
    ),
]


def get_prompt_list(classes: list[DJToolClass] | None = None) -> list[str]:
    """Return the list of text prompts for CLAP inference."""
    classes = classes or DEFAULT_DJTOOL_CLASSES
    return [c.prompt for c in classes]


def get_class_names(classes: list[DJToolClass] | None = None) -> list[str]:
    """Return the list of class names."""
    classes = classes or DEFAULT_DJTOOL_CLASSES
    return [c.name for c in classes]
