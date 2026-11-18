"""M3: canonical reference images, generated as candidates and then locked by hand.

Three kinds, made in this order:
  face   a neutral head-and-shoulders portrait. Text only: this is where the face is decided.
  sheet  the character in one costume, four views side by side. Attaches the LOCKED face,
         so every costume sheet shows the same person.
  plate  a location with no people in it, in its own lighting.
  prop   one story object alone on a plain background, so its shape and colour are fixed.

Candidates go to assets/refs/<id>/ (git-ignored). `fluxpipe lock` copies the one you
choose into bible/refs/<id>/ (committed) and records it in the bible YAML. From then on
every shot attaches the locked images, and its prompt says which is which.
"""

from __future__ import annotations

import re
import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML

from .bible import Bible
from .generate import Request, build_request

STUDIO = {
    "scene": "a plain light grey studio backdrop, nothing else",
    "style": "realistic photograph, sharp focus, natural skin texture, true-to-life colour",
    "lighting": "soft, even studio light from the front, no harsh shadows",
}
SIZES = {"face": (896, 1088), "sheet": (1920, 1088), "plate": (1280, 720), "prop": (992, 992)}
CANDIDATE = re.compile(r"^(?P<kind>face|sheet|plate|prop)(?:-(?P<costume>[\w-]+))?_s\d+\.png$")


@dataclass
class Candidate:
    kind: str
    target: str  # character or location id
    costume: str | None
    request: Request
    out: Path


def _seed(kind: str, target: str, costume: str | None, i: int) -> int:
    return zlib.crc32(f"{kind}:{target}:{costume or ''}".encode()) + i


def _face_prompt(bible: Bible, char_id: str) -> dict:
    char = bible.characters[char_id]
    return {
        **STUDIO,
        "subjects": [{
            "description": f"{char.identity.describe()}, wearing a plain grey crew-neck t-shirt",
            "position": "centre frame, facing the camera squarely",
            "action": "looking straight into the lens with a relaxed, neutral expression",
        }],
        "composition": "head and shoulders portrait, the face filling most of the frame",
        "camera": {"angle": "eye level", "lens": "85mm", "depth_of_field": "face fully sharp"},
    }


def _sheet_prompt(bible: Bible, char_id: str, costume: str) -> dict:
    char = bible.characters[char_id]
    return {
        **STUDIO,
        "subjects": [{
            "description": (
                f"the same person as in image 1: {char.identity.describe()}, "
                f"wearing {char.costumes[costume]}"
            ),
            "position": "the same person shown four times, evenly spaced in a row",
            "action": (
                "standing in a relaxed neutral pose, seen from the front, from a three-quarter "
                "angle, in side profile and from the back, full body from head to toe"
            ),
        }],
        "composition": "character reference sheet: four full-body views side by side, nothing else in frame",
        "camera": {"angle": "eye level", "lens": "50mm", "depth_of_field": "everything sharp"},
    }


def _plate_prompt(bible: Bible, loc_id: str) -> dict:
    loc = bible.locations[loc_id]
    return {
        "scene": loc.description,
        "style": bible.style,
        "color_palette": loc.palette,
        "lighting": loc.lighting,
        "composition": "empty establishing shot of the location, no people anywhere in frame",
        "camera": {"angle": "eye level", "lens": "24mm", "depth_of_field": "deep, everything in focus"},
    }


def _prop_prompt(bible: Bible, prop_id: str) -> dict:
    prop = bible.props[prop_id]
    return {
        **STUDIO,
        "subjects": [{"description": prop.description, "position": "centre frame"}],
        "lighting": "soft, even studio light, a gentle shadow beneath the object",
        "composition": "product photograph of a single object, filling the middle of the frame, nothing else",
        "camera": {"angle": "slightly above", "lens": "100mm macro", "depth_of_field": "the whole object sharp"},
    }


def candidates(
    bible: Bible, kind: str, target: str, *, out_dir: Path, n: int, model: str,
    costume: str | None = None,
) -> list[Candidate]:
    """The requests for n candidates of one reference (nothing is sent here)."""
    refs: list[Path] = []
    if kind == "face":
        prompt = _face_prompt(bible, _character(bible, target))
    elif kind == "sheet":
        char = bible.characters[_character(bible, target)]
        if costume not in char.costumes:
            raise ValueError(f"Pass --costume, one of: {', '.join(char.costumes)}")
        if not char.reference_images:
            raise ValueError(f"Lock a face for '{target}' first: sheets attach it so every costume shows the same person.")
        prompt = _sheet_prompt(bible, target, costume)
        refs = [bible.ref_path(char.reference_images[0])]
    elif kind == "plate":
        if target not in bible.locations:
            raise ValueError(f"No location '{target}'. Locations: {', '.join(bible.locations)}")
        prompt = _plate_prompt(bible, target)
    elif kind == "prop":
        if target not in bible.props:
            raise ValueError(f"No prop '{target}'. Props: {', '.join(bible.props) or '(none)'}")
        prompt = _prop_prompt(bible, target)
    else:
        raise ValueError("kind must be face, sheet, plate or prop")

    width, height = SIZES[kind]
    label = f"{kind}-{costume}" if costume else kind
    out = []
    for i in range(n):
        seed = _seed(kind, target, costume, i)
        req = build_request(model, prompt, width=width, height=height, seed=seed, ref_files=refs)
        out.append(Candidate(kind, target, costume, req, out_dir / target / f"{label}_s{seed}.png"))
    return out


def _character(bible: Bible, target: str) -> str:
    if target not in bible.characters:
        raise ValueError(f"No character '{target}'. Characters: {', '.join(bible.characters)}")
    return target


def lock(bible: Bible, bible_dir: Path, candidate: Path) -> tuple[str, list[str]]:
    """Copy a chosen candidate into bible/refs/ and record it in the YAML.

    Returns (what was locked, warnings). Replaces any earlier lock of the same kind.
    """
    m = CANDIDATE.match(candidate.name)
    if not m or not candidate.is_file():
        raise ValueError(f"{candidate} isn't a reference candidate made by `fluxpipe refs`.")
    kind, costume, target = m["kind"], m["costume"], candidate.parent.name
    label = f"{kind}-{costume}" if costume else kind
    rel = f"refs/{target}/{label}.png"
    dest = bible_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate, dest)

    warnings: list[str] = []
    if kind in ("plate", "prop"):
        folder = bible_dir / ("locations" if kind == "plate" else "props")
        _edit_yaml(folder, target, lambda d: d.__setitem__("reference_images", [rel]))
    elif kind == "face":
        char = bible.characters[target]
        if char.costume_references:
            warnings.append("The costume sheets were made from the previous face: regenerate and re-lock them.")
        _edit_yaml(bible_dir / "characters", target, lambda d: d.__setitem__("reference_images", [rel]))
    else:
        def set_sheet(d):
            if "costume_references" not in d:
                d["costume_references"] = {}
            d["costume_references"][costume] = [rel]
        _edit_yaml(bible_dir / "characters", target, set_sheet)
    return f"{target} {label} -> bible/{rel}", warnings


def _edit_yaml(folder: Path, item_id: str, change) -> None:
    """Round-trip edit, so the comments and layout of the owner's YAML survive."""
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # don't re-wrap long lines
    for path in sorted(folder.glob("*.yaml")):
        data = yaml.load(path.read_text())
        if data.get("id") == item_id:
            change(data)
            with path.open("w") as f:
                yaml.dump(data, f)
            return
    raise ValueError(f"No YAML in {folder} with id '{item_id}'")
