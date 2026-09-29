"""Attribute normalisation used only for *matching*. Stored attributes keep their original text."""

import re

from drone_agent.pipeline.config import NormalizationSettings


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_]", " ", value.strip().lower()))


def normalize_color(value: str, cfg: NormalizationSettings) -> str:
    """'Dark Blue' -> 'blue', 'light blue' -> 'blue', 'grey' -> 'gray'.

    Only leading shade words are dropped; anything else ('navy', 'teal') is left as is.
    """
    words = _clean(value).split(" ")
    while len(words) > 1 and words[0] in cfg.color_modifiers:
        words.pop(0)
    color = " ".join(words)
    return cfg.color_aliases.get(color, color)


def normalize_attribute(key: str, value: str, cfg: NormalizationSettings) -> str:
    if key == "color":
        return normalize_color(value, cfg)
    text = _clean(value)
    if key == "body_type":
        return cfg.body_type_aliases.get(text, text)
    return text


def normalize_plate(value: str) -> str:
    """Case, spaces and punctuation are ignored when comparing explicit plates."""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper()
