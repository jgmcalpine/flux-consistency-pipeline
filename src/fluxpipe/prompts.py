"""Prompts are assembled from the bible, never hand-typed per shot.

The same character is described with the same words in every shot, so the only
thing that changes between two prompts is what actually changes between the shots.
"""

from __future__ import annotations

from .bible import Bible, Shot

TEMPLATE = (
    "{framing} of {appearance}, wearing {costume}. {action}. "
    "Setting: {setting}. Lighting: {lighting}. {style}."
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def build_prompt(bible: Bible, shot: Shot) -> str:
    char = bible.characters[shot.character]
    loc = bible.locations[shot.location]
    return TEMPLATE.format(
        framing=_cap(shot.framing),
        appearance=char.appearance,
        costume=char.costumes[shot.costume],
        action=_cap(shot.action),
        setting=loc.description,
        lighting=loc.lighting,
        style=_cap(bible.style),
    )


def reference_images(bible: Bible, shot: Shot) -> list[str]:
    """The locked references to attach to this shot (filled in at M3)."""
    return [
        *bible.characters[shot.character].reference_images,
        *bible.locations[shot.location].reference_images,
    ]
