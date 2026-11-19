import json

import pytest
from pydantic import ValidationError

from fluxpipe.bible import Camera, Character, load_bible
from fluxpipe.config import ROOT
from fluxpipe.prompts import as_json, as_text, build_structured, diff, reference_images


@pytest.fixture
def bible():
    return load_bible(ROOT / "bible")


def test_sample_bible_is_valid(bible):
    assert bible.problems() == []


def test_broken_reference_is_caught(bible):
    bible.shots[0].costume = "tuxedo"
    assert any("no costume 'tuxedo'" in p for p in bible.problems())


def test_identity_fields_are_required():
    with pytest.raises(ValidationError):
        Character.model_validate(
            {"id": "x", "name": "X", "identity": {"age": "30s"}, "costumes": {"a": "coat"}}
        )


def test_identity_is_word_for_word_identical_in_every_shot(bible):
    for shot in bible.shots:
        identity = bible.characters[shot.character].identity.describe()
        assert identity in build_structured(bible, shot)["subjects"][0]["description"]


def test_shot_camera_overrides_only_what_it_sets(bible):
    shot = bible.shots[0].model_copy(update={"camera": Camera(lens="135mm")})
    cam = build_structured(bible, shot)["camera"]
    assert cam["lens"] == "135mm"  # overridden by the shot
    assert cam["angle"] == bible.camera_defaults.angle  # inherited from camera_defaults


def test_json_prompt_round_trips(bible):
    p = build_structured(bible, bible.shot("s01"))
    assert json.loads(as_json(p)) == p
    assert as_text(p).startswith("Wide shot")


def test_diff_never_touches_identity(bible):
    a = build_structured(bible, bible.shot("s01"))
    b = build_structured(bible, bible.shot("s02"))
    fields = [f for f, _, _ in diff(a, b)]
    assert "action" not in fields  # nested under subjects
    assert "subjects[0].action" in fields
    assert "style" not in fields


def test_yaml_typos_are_errors_not_silence():
    # Unquoted commas in a YAML {flow map} turn words into a stray key. Caught, not dropped.
    with pytest.raises(ValidationError):
        Camera.model_validate({"lens": "24mm", "depth_of_field": "deep", "everything in focus": None})


def test_props_are_described_word_for_word_in_every_shot_that_shows_them(bible):
    shots_with_props = [s for s in bible.shots if s.props]
    assert shots_with_props
    for shot in shots_with_props:
        descriptions = [x["description"] for x in build_structured(bible, shot)["subjects"][1:]]
        for prop_id in shot.props:
            assert any(bible.props[prop_id].description in d for d in descriptions)


def test_unknown_prop_is_caught(bible):
    bible.shots[0].props = ["golden_idol"]
    assert any("unknown prop 'golden_idol'" in p for p in bible.problems())


def test_plate_false_leaves_the_location_plate_off(bible):
    shot = bible.shot("s09")
    assert shot.plate is False
    assert not any("plate" in r for r in reference_images(bible, shot))
    assert "the location from image" not in build_structured(bible, shot)["scene"]


def test_hands_only_shots_attach_no_face_reference(bible):
    for shot in bible.shots:
        has_face_ref = any(r.endswith("face.png") for r in reference_images(bible, shot))
        assert has_face_ref == shot.face


def test_image_numbers_follow_what_is_attached(bible):
    p = build_structured(bible, bible.shot("s10"))  # no face ref: the costume sheet is image 1
    assert "exactly the outfit shown in image 1" in p["subjects"][0]["description"]
    assert "image 2" in p["subjects"][1]["description"] and "image 3" in p["subjects"][2]["description"]


def test_shot_overrides_for_scene_and_lighting(bible):
    loc = bible.locations["thatched_hut"]
    s11 = build_structured(bible, bible.shot("s11"))
    assert s11["lighting"] != loc.lighting and "burnt out" in s11["lighting"]
    assert build_structured(bible, bible.shot("s07"))["lighting"] == loc.lighting
    assert loc.description not in build_structured(bible, bible.shot("s09"))["scene"]


def test_hut_is_described_the_same_inside_and_out(bible):
    hut = "a small round hut with a steep conical thatched roof on a ring of wooden posts"
    assert hut in bible.locations["thatched_hut"].description
    assert hut in bible.locations["jungle_path"].description


def test_missing_edit_ref_is_caught(bible):
    bible.shot("s11").edit_refs.append("refs/nope/missing.png")
    assert "s11: reference image not found: refs/nope/missing.png" in bible.problems()


def test_a_shot_can_swap_a_props_reference_for_an_in_hand_crop(bible):
    from fluxpipe.prompts import shot_refs
    s04, s08 = bible.shot("s04"), bible.shot("s08")
    assert [r.path for r in shot_refs(bible, s04) if r.role == "prop:crystal"] == ["refs/crystal/in_hand.png"]
    assert [r.path for r in shot_refs(bible, s08) if r.role == "prop:crystal"] == ["refs/crystal/prop.png"]
    crystal = next(x["description"] for x in build_structured(bible, s04)["subjects"][1:]
                   if bible.props["crystal"].description in x["description"])
    assert "same size compared with the hand" in crystal


def test_prop_refs_must_name_a_prop_of_the_shot(bible):
    bible.shot("s08").prop_refs = {"satchel": ["refs/satchel/prop.png"]}
    assert any("prop_refs names 'satchel'" in p for p in bible.problems())
