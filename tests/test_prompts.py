"""Tests for DJ tool class prompts."""

from djtool.classification.prompts import (
    DEFAULT_DJTOOL_CLASSES,
    get_class_names,
    get_prompt_list,
)


def test_default_classes_count():
    assert len(DEFAULT_DJTOOL_CLASSES) == 7


def test_class_names():
    names = get_class_names()
    assert "acapella" in names
    assert "drums" in names
    assert "drops" in names


def test_prompt_list_matches_classes():
    prompts = get_prompt_list()
    names = get_class_names()
    assert len(prompts) == len(names)
