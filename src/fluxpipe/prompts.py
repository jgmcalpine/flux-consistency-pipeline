"""Prompts are assembled from the bible, never hand-typed per shot.

Each shot becomes one STRUCTURED prompt: a dict using the field names from BFL's
FLUX.2 JSON prompting guide (scene, subjects, style, color_palette, lighting, mood,
composition, camera). From that one dict we can:

  - send it as JSON (FLUX.2 reads JSON prompts directly), or
  - flatten it into natural language (as_text), and
  - diff two shots field by field, to prove they differ only where the story does.

The character's identity is copied word for word into every shot. That is the whole trick.
From M3 the locked reference images are attached too, and the prompt says which is which
("image 1", "image 2"... in attachment order, as BFL's multi-reference guide recommends).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .bible import Bible, Shot


@dataclass(frozen=True)
class Ref:
    role: str  # face | costume | prop:<id> | location
    path: str  # relative to the bible folder


def shot_refs(bible: Bible, shot: Shot) -> list[Ref]:
    """The locked references for a shot, in the order they are attached (image 1, 2, ...)."""
    char = bible.characters[shot.character]
    loc = bible.locations[shot.location]
    return [
        *(Ref("face", r) for r in char.reference_images),
        *(Ref("costume", r) for r in char.costume_references.get(shot.costume, [])),
        *(Ref(f"prop:{p}", r) for p in shot.props for r in bible.props[p].reference_images),
        *(Ref("location", r) for r in loc.reference_images),
    ]


def _images(refs: list[Ref], role: str) -> str:
    nums = [f"image {i}" for i, r in enumerate(refs, 1) if r.role == role]
    return " and ".join(nums)


def build_structured(bible: Bible, shot: Shot) -> dict:
    char = bible.characters[shot.character]
    loc = bible.locations[shot.location]
    refs = shot_refs(bible, shot)
    who, outfit, place = (_images(refs, r) for r in ("face", "costume", "location"))
    description = f"{char.identity.describe()}, wearing {char.costumes[shot.costume]}"
    if who:
        description = f"the same person as in {who}: {description}"
    if outfit:
        description += f", exactly the outfit shown in {outfit}"
    subjects = [{"description": description, "position": shot.position, "action": shot.action}]
    for prop_id in shot.props:
        # Each prop is its own subject, described word for word from the bible, like the character.
        seen_in = _images(refs, f"prop:{prop_id}")
        text = bible.props[prop_id].description
        subjects.append({"description": f"exactly the object shown in {seen_in}: {text}" if seen_in else text})
    prompt = {
        "scene": f"the location from {place}: {loc.description}" if place else loc.description,
        "subjects": subjects,
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
        *(f"{_cap(p['description'])}." for p in prompt["subjects"][1:]),
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
    """The locked reference paths for this shot, in attachment order."""
    return [r.path for r in shot_refs(bible, shot)]
