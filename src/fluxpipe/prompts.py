"""Prompts are assembled from the bible, never hand-typed per shot.

Each shot becomes one STRUCTURED prompt: a dict using the field names from BFL's
FLUX.2 JSON prompting guide (scene, subjects, style, color_palette, lighting, mood,
composition, camera). From that one dict we can:

  - send it as JSON (FLUX.2 reads JSON prompts directly), or
  - flatten it into natural language (as_text), and
  - diff two shots field by field, to prove they differ only where the story does.

The character's identity is copied word for word into every shot. That is the whole trick.
"""

from __future__ import annotations

import json

from .bible import Bible, Shot


def build_structured(bible: Bible, shot: Shot) -> dict:
    char = bible.characters[shot.character]
    loc = bible.locations[shot.location]
    prompt = {
        "scene": loc.description,
        "subjects": [
            {
                "description": f"{char.identity.describe()}, wearing {char.costumes[shot.costume]}",
                "position": shot.position,
                "action": shot.action,
            }
        ],
        "style": bible.style,
        "color_palette": loc.palette,
        "lighting": loc.lighting,
        "mood": shot.mood,
        "composition": shot.framing,
        "camera": bible.camera_for(shot),
    }
    # Drop empty fields rather than sending blanks.
    return {k: v for k, v in prompt.items() if v not in ("", [], {})}


def as_json(prompt: dict) -> str:
    """What gets sent in the API's `prompt` field."""
    return json.dumps(prompt, ensure_ascii=False)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def as_text(prompt: dict) -> str:
    """The same prompt as one natural-language paragraph, for reading or for the Playground."""
    subj = prompt["subjects"][0]
    parts = [
        f"{_cap(prompt['composition'])} of {subj['description']}.",
        f"{_cap(subj['action'])}, {subj['position']}.",
        f"Setting: {prompt['scene']}.",
        f"Lighting: {prompt['lighting']}.",
    ]
    if "mood" in prompt:
        parts.append(f"Mood: {prompt['mood']}.")
    if "camera" in prompt:
        parts.append("Camera: " + ", ".join(f"{v}" for v in prompt["camera"].values()) + ".")
    if "color_palette" in prompt:
        parts.append("Colour palette: " + ", ".join(prompt["color_palette"]) + ".")
    parts.append(f"{_cap(prompt['style'])}.")
    return " ".join(parts)


def _flatten(value, prefix: str = "") -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for k, v in value.items():
            out |= _flatten(v, f"{prefix}.{k}" if prefix else k)
        return out
    if isinstance(value, list) and value and isinstance(value[0], dict):
        out = {}
        for i, v in enumerate(value):
            out |= _flatten(v, f"{prefix}[{i}]")
        return out
    return {prefix: str(value)}


def diff(a: dict, b: dict) -> list[tuple[str, str | None, str | None]]:
    """Fields that differ between two structured prompts: (field, a_value, b_value)."""
    fa, fb = _flatten(a), _flatten(b)
    return [(k, fa.get(k), fb.get(k)) for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)]


def reference_images(bible: Bible, shot: Shot) -> list[str]:
    """The locked references to attach to this shot (filled in at M3)."""
    return [
        *bible.characters[shot.character].reference_images,
        *bible.locations[shot.location].reference_images,
    ]
