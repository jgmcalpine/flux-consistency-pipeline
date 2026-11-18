import json

import pytest
from pydantic import ValidationError

from fluxpipe.bible import Camera, Character, load_bible
from fluxpipe.config import ROOT
from fluxpipe.prompts import as_json, as_text, build_structured, diff


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
