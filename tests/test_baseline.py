"""M4: the naive baseline must differ from the pipeline ONLY in what the pipeline adds."""

import json

import pytest

from fluxpipe.bible import load_bible
from fluxpipe.config import ROOT
from fluxpipe.generate import baseline_request, keyframe_request


@pytest.fixture
def bible():
    return load_bible(ROOT / "bible")


def test_every_shot_has_a_baseline_prompt(bible):
    assert bible.baseline is not None
    assert bible.baseline.shots.keys() == {s.id for s in bible.shots}


def test_missing_baseline_prompt_is_caught(bible):
    del bible.baseline.shots["s05"]
    assert "baseline.yaml: no prompt for shot 's05'" in bible.problems()


def test_baseline_attaches_no_references(bible):
    for shot in bible.shots:
        req = baseline_request(bible, shot)
        assert req.refs == []
        assert not any(k.startswith("input_image") for k in req.payload)


def test_baseline_is_plain_text_not_json(bible):
    prompt = baseline_request(bible, bible.shot("s01")).payload["prompt"]
    with pytest.raises(json.JSONDecodeError):
        json.loads(prompt)
    assert "#" not in prompt  # no hex colours: that's pipeline detail


def test_same_character_text_in_every_shot(bible):
    for shot in bible.shots:
        assert bible.baseline.character in baseline_request(bible, shot).payload["prompt"]


def test_only_the_pipeline_differs(bible):
    """Same model, size, seed and upsampling as the keyframe; only prompt and refs change."""
    for shot in bible.shots:
        naive = baseline_request(bible, shot).payload
        piped = keyframe_request(bible, shot, model="flux-2-pro").payload
        strip = lambda p: {k: v for k, v in p.items() if k != "prompt" and not k.startswith("input_image")}
        assert strip(naive) == strip(piped)
        assert baseline_request(bible, shot).endpoint == "flux-2-pro"
