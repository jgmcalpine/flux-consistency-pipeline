from fluxpipe.bible import load_bible
from fluxpipe.config import ROOT
from fluxpipe.prompts import build_prompt


def test_sample_bible_is_valid():
    bible = load_bible(ROOT / "bible")
    assert bible.problems() == []


def test_broken_reference_is_caught():
    bible = load_bible(ROOT / "bible")
    bible.shots[0].costume = "tuxedo"
    assert any("no costume 'tuxedo'" in p for p in bible.problems())


def test_same_character_same_words_in_every_prompt():
    bible = load_bible(ROOT / "bible")
    appearance = bible.characters["keeper"].appearance
    for shot in bible.shots:
        assert appearance in build_prompt(bible, shot)
