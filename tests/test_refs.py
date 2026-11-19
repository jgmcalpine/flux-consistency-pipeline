import base64
import json
import shutil
import struct
import zlib

import pytest
from ruamel.yaml import YAML

from fluxpipe.bible import load_bible
from fluxpipe.config import ROOT, estimate_cost
from fluxpipe.generate import keyframe_request
from fluxpipe.images import png_size
from fluxpipe.prompts import build_structured
from fluxpipe.refs import candidates, lock


def tiny_png(path, width, height):
    """A real (blank) PNG of a given size, so the price estimate sees true dimensions."""
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    raw = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return path


@pytest.fixture
def bible_dir(tmp_path):
    """A throwaway copy of the bible, so lock() never touches the real one."""
    d = tmp_path / "bible"
    shutil.copytree(ROOT / "bible", d, ignore=shutil.ignore_patterns("refs"))
    # Start from nothing locked, whatever the real bible has locked so far.
    yaml = YAML()
    for f in d.glob("*/*.yaml"):
        data = yaml.load(f.read_text())
        if isinstance(data, dict) and "reference_images" in data:
            data["reference_images"] = []
            data.pop("costume_references", None)
            with f.open("w") as out:
                yaml.dump(data, out)
    shots = yaml.load((d / "shots.yaml").read_text())
    for shot in shots["shots"]:
        shot.pop("edit_refs", None)  # these point into refs/ too
        shot.pop("prop_refs", None)
    with (d / "shots.yaml").open("w") as out:
        yaml.dump(shots, out)
    return d


def char_and_costume(bible):
    shot = bible.shots[0]
    return shot.character, shot.costume, shot.location


def test_png_size(tmp_path):
    assert png_size(tiny_png(tmp_path / "a.png", 32, 16)) == (32, 16)


def test_sheet_needs_a_locked_face_first(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    char, costume, _ = char_and_costume(bible)
    with pytest.raises(ValueError, match="Lock a face"):
        candidates(bible, "sheet", char, out_dir=tmp_path, n=1, model="flux-2-pro", costume=costume)


def test_face_candidates_use_different_seeds_and_no_refs(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    char, _, _ = char_and_costume(bible)
    cands = candidates(bible, "face", char, out_dir=tmp_path, n=3, model="flux-2-pro")
    assert len({c.request.payload["seed"] for c in cands}) == 3
    assert not any(k.startswith("input_image") for k in cands[0].request.payload)
    assert cands[0].request.est_cost_usd == 0.03  # 896x1088 stays under 1 MP


def test_plate_has_no_people(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    _, _, loc = char_and_costume(bible)
    prompt = json.loads(candidates(bible, "plate", loc, out_dir=tmp_path, n=1, model="flux-2-pro")[0]
                        .request.payload["prompt"])
    assert "subjects" not in prompt and "no people" in prompt["composition"]


def test_lock_face_then_sheet_then_shot_attaches_everything(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    char, costume, loc = char_and_costume(bible)
    out = tmp_path / "assets"

    # 1. lock a face
    face = candidates(bible, "face", char, out_dir=out, n=1, model="flux-2-pro")[0]
    tiny_png(face.out, 896, 1088)
    lock(bible, bible_dir, face.out)
    bible = load_bible(bible_dir)
    assert bible.characters[char].reference_images == [f"refs/{char}/face.png"]
    assert "# identity" in (bible_dir / "characters" / f"{char}.yaml").read_text()  # comments kept

    # 2. the sheet attaches the locked face as image 1
    sheet = candidates(bible, "sheet", char, out_dir=out, n=1, model="flux-2-pro", costume=costume)[0]
    assert base64.b64decode(sheet.request.payload["input_image"]) == face.out.read_bytes()
    assert "image 1" in sheet.request.payload["prompt"]
    tiny_png(sheet.out, 1920, 1088)
    lock(bible, bible_dir, sheet.out)

    # 3. a plate
    plate = candidates(load_bible(bible_dir), "plate", loc, out_dir=out, n=1, model="flux-2-pro")[0]
    tiny_png(plate.out, 1280, 720)
    lock(load_bible(bible_dir), bible_dir, plate.out)

    # 4. a shot now attaches face, sheet and plate, and says which image is which
    bible = load_bible(bible_dir)
    assert bible.problems() == []
    shot = bible.shots[0]
    req = keyframe_request(bible, shot, model="flux-2-pro")
    assert {"input_image", "input_image_2", "input_image_3"} <= req.payload.keys()
    desc = build_structured(bible, shot)["subjects"][0]["description"]
    assert desc.startswith("the same person as in image 1")
    assert "outfit shown in image 2" in desc
    assert build_structured(bible, shot)["scene"].startswith("the location from image 3")
    # three refs: each billed as 1 MP on top of the 1 MP output
    assert req.est_cost_usd == estimate_cost("flux-2-pro", width=1280, height=720, refs=[(1, 1)] * 3)


def test_relocking_the_face_warns_about_stale_sheets(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    char, costume, _ = char_and_costume(bible)
    face = candidates(bible, "face", char, out_dir=tmp_path, n=1, model="flux-2-pro")[0]
    lock(bible, bible_dir, tiny_png(face.out, 896, 1088))
    bible = load_bible(bible_dir)
    sheet = candidates(bible, "sheet", char, out_dir=tmp_path, n=1, model="flux-2-pro", costume=costume)[0]
    lock(bible, bible_dir, tiny_png(sheet.out, 1920, 1088))
    _, warnings = lock(load_bible(bible_dir), bible_dir, face.out)
    assert warnings and "regenerate" in warnings[0]


def test_missing_reference_file_is_a_problem(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    char, _, _ = char_and_costume(bible)
    bible.characters[char].reference_images = ["refs/nope.png"]
    assert any("not found" in p for p in bible.problems())


def test_lock_rejects_random_files(bible_dir, tmp_path):
    with pytest.raises(ValueError):
        lock(load_bible(bible_dir), bible_dir, tiny_png(tmp_path / "holiday.png", 8, 8))


def test_prop_reference_is_locked_and_attached_with_its_own_image_number(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    shot = next(s for s in bible.shots if s.props)
    prop_id = shot.props[0]
    cand = candidates(bible, "prop", prop_id, out_dir=tmp_path, n=1, model="flux-2-pro")[0]
    assert "product photograph" in json.loads(cand.request.payload["prompt"])["composition"]
    lock(bible, bible_dir, tiny_png(cand.out, 992, 992))
    bible = load_bible(bible_dir)
    assert bible.props[prop_id].reference_images == [f"refs/{prop_id}/prop.png"]
    subjects = build_structured(bible, bible.shot(shot.id))["subjects"]
    assert any(s["description"].startswith("exactly the object shown in image 1") for s in subjects)


def test_prop_with_another_locked_prop_attaches_it_beside(bible_dir, tmp_path):
    bible = load_bible(bible_dir)
    first, second = list(bible.props)[:2]
    with pytest.raises(ValueError, match="Lock"):
        candidates(bible, "prop", second, out_dir=tmp_path, n=1, model="flux-2-pro", with_props=[first])
    first_cand = candidates(bible, "prop", first, out_dir=tmp_path, n=1, model="flux-2-pro")[0]
    lock(bible, bible_dir, tiny_png(first_cand.out, 992, 992))
    bible = load_bible(bible_dir)
    cand = candidates(bible, "prop", second, out_dir=tmp_path, n=1, model="flux-2-pro", with_props=[first])[0]
    assert base64.b64decode(cand.request.payload["input_image"]) == first_cand.out.read_bytes()
    subjects = json.loads(cand.request.payload["prompt"])["subjects"]
    assert subjects[1]["description"].startswith("exactly the object shown in image 1")
    assert "beside" in subjects[1]["position"]
    plain = candidates(bible, "prop", second, out_dir=tmp_path, n=1, model="flux-2-pro")[0]
    assert plain.out != cand.out  # new seeds, so earlier candidates are never overwritten
