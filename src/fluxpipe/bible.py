"""Stage 1: the story bible. Loads the YAML files and checks every reference resolves."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Strict(BaseModel):
    """Unknown keys are errors, so a typo in the YAML can't silently vanish."""

    model_config = ConfigDict(extra="forbid")


class Identity(Strict):
    """What must never change between shots. Every field is required on purpose."""

    who: str  # the noun: "a woman lighthouse keeper"
    age: str
    build: str
    face: str
    hair: str
    marks: str

    def describe(self) -> str:
        return f"{self.who} {self.age}, {self.build}, {self.face}, {self.hair}, {self.marks}"


class Character(Strict):
    id: str
    name: str
    identity: Identity
    costumes: dict[str, str]
    # Locked at M3. Paths are relative to the bible folder.
    reference_images: list[str] = Field(default_factory=list)  # the face, any costume
    costume_references: dict[str, list[str]] = Field(default_factory=dict)  # costume id -> sheet


class Location(Strict):
    id: str
    name: str
    description: str
    lighting: str
    palette: list[str] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)


class Camera(Strict):
    angle: str | None = None
    lens: str | None = None
    depth_of_field: str | None = None


class Shot(Strict):
    id: str
    character: str
    costume: str
    location: str
    framing: str
    action: str
    position: str = "centre frame"
    mood: str = ""
    camera: Camera = Field(default_factory=Camera)
    duration_s: float = 5


class Bible(Strict):
    style: str
    camera_defaults: Camera
    characters: dict[str, Character]
    locations: dict[str, Location]
    shots: list[Shot]
    _dir: Path | None = PrivateAttr(default=None)

    def ref_path(self, ref: str) -> Path:
        """A reference image path from the YAML, resolved against the bible folder."""
        return (self._dir or Path(".")) / ref

    def camera_for(self, shot: Shot) -> dict[str, str]:
        """Defaults first, then the shot's overrides. Fewer hand-typed values = less drift."""
        merged = self.camera_defaults.model_dump() | shot.camera.model_dump(exclude_none=True)
        return {k: v for k, v in merged.items() if v}

    def shot(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(f"No shot '{shot_id}'. Shots: {', '.join(s.id for s in self.shots)}")

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
        for char in self.characters.values():
            for costume in char.costume_references:
                if costume not in char.costumes:
                    issues.append(f"{char.id}: costume_references names unknown costume '{costume}'")
        if self._dir is not None:
            for owner, ref in self._all_refs():
                if not self.ref_path(ref).is_file():
                    issues.append(f"{owner}: reference image not found: {ref}")
        return issues

    def _all_refs(self):
        for char in self.characters.values():
            yield from ((char.id, r) for r in char.reference_images)
            for refs in char.costume_references.values():
                yield from ((char.id, r) for r in refs)
        for loc in self.locations.values():
            yield from ((loc.id, r) for r in loc.reference_images)


def _load_dir(folder: Path, model: type[BaseModel]) -> dict:
    items = {}
    for path in sorted(folder.glob("*.yaml")):
        item = model.model_validate(yaml.safe_load(path.read_text()))
        items[item.id] = item
    return items


def load_bible(bible_dir: Path) -> Bible:
    shots_file = yaml.safe_load((bible_dir / "shots.yaml").read_text())
    bible = Bible(
        style=shots_file["style"],
        camera_defaults=Camera.model_validate(shots_file.get("camera_defaults") or {}),
        characters=_load_dir(bible_dir / "characters", Character),
        locations=_load_dir(bible_dir / "locations", Location),
        shots=[Shot.model_validate(s) for s in shots_file["shots"]],
    )
    bible._dir = bible_dir
    return bible
