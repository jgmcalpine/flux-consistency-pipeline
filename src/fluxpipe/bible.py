"""Stage 1: the story bible. Loads the YAML files and checks every reference resolves."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Character(BaseModel):
    id: str
    name: str
    appearance: str
    costumes: dict[str, str]
    reference_images: list[str] = Field(default_factory=list)


class Location(BaseModel):
    id: str
    name: str
    description: str
    lighting: str
    palette: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)


class Shot(BaseModel):
    id: str
    character: str
    costume: str
    location: str
    framing: str
    action: str
    duration_s: float = 5


class Bible(BaseModel):
    style: str
    characters: dict[str, Character]
    locations: dict[str, Location]
    shots: list[Shot]

    def problems(self) -> list[str]:
        """Every broken reference in the bible. An empty list means it's valid."""
        issues: list[str] = []
        seen: set[str] = set()
        for shot in self.shots:
            if shot.id in seen:
                issues.append(f"{shot.id}: duplicate shot id")
            seen.add(shot.id)
            char = self.characters.get(shot.character)
            if char is None:
                issues.append(f"{shot.id}: unknown character '{shot.character}'")
            elif shot.costume not in char.costumes:
                issues.append(
                    f"{shot.id}: '{char.id}' has no costume '{shot.costume}' "
                    f"(has: {', '.join(char.costumes)})"
                )
            if shot.location not in self.locations:
                issues.append(f"{shot.id}: unknown location '{shot.location}'")
        return issues


def _load_dir(folder: Path, model: type[BaseModel]) -> dict:
    items = {}
    for path in sorted(folder.glob("*.yaml")):
        item = model.model_validate(yaml.safe_load(path.read_text()))
        items[item.id] = item
    return items


def load_bible(bible_dir: Path) -> Bible:
    shots_file = yaml.safe_load((bible_dir / "shots.yaml").read_text())
    return Bible(
        style=shots_file["style"],
        characters=_load_dir(bible_dir / "characters", Character),
        locations=_load_dir(bible_dir / "locations", Location),
        shots=[Shot.model_validate(s) for s in shots_file["shots"]],
    )
