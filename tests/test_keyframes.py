"""M5 selection loop: fake generator and judge, so no API calls and no model."""

from pathlib import Path

from fluxpipe.generate import shot_seed
from fluxpipe.keyframes import BELOW, BY_EYE, DRY_RUN, PASS, candidate_seeds, run_shot, summary
from fluxpipe.qa.face import FaceScore


def face(sim, px=200):
    return FaceScore("x.png", 1 if sim is not None else 0, sim, px if sim is not None else None, 0.9)


def scripted(scores: list):
    """A generator that 'makes' images in order and a judge that returns the scripted scores."""
    made: list[int] = []
    table: dict[Path, FaceScore] = {}

    def generate(seed):
        path = Path(f"s{seed}.png")
        table[path] = scores[len(made)]
        made.append(seed)
        return path

    return generate, table.__getitem__, made


def test_first_candidate_uses_the_baseline_seed():
    seeds = candidate_seeds("s02", 0, 3)
    assert seeds[0] == shot_seed("s02")
    assert len(set(seeds + candidate_seeds("s02", 1, 3))) == 6  # round 2 never repeats round 1


def test_stops_paying_once_a_round_passes():
    gen, judge, made = scripted([face(0.3), face(0.62), face(0.4), face(0.9)])
    r = run_shot("s02", n=3, max_rounds=2, generate=gen, judge=judge)
    assert r.status == PASS
    assert len(made) == 3  # the fourth (0.9) is never generated
    assert r.pick.face.similarity == 0.62


def test_retries_then_keeps_the_best_below_threshold():
    gen, judge, made = scripted([face(0.2), face(0.3), face(0.45), face(0.1)])
    r = run_shot("s02", n=2, max_rounds=2, generate=gen, judge=judge)
    assert r.status == BELOW
    assert len(made) == 4
    assert r.pick.face.similarity == 0.45  # best across both rounds


def test_no_judgeable_face_does_not_retry():
    gen, judge, made = scripted([face(None), face(0.9, px=40), face(None), face(0.9)])
    r = run_shot("s03", n=3, max_rounds=2, generate=gen, judge=judge)
    assert r.status == BY_EYE
    assert len(made) == 3  # a tiny face (40 px) doesn't count, and we don't pay for round 2
    assert r.pick is r.candidates[0]


def test_dry_run_makes_one_round_and_picks_nothing():
    calls = []
    r = run_shot("s01", n=3, max_rounds=2, generate=lambda seed: calls.append(seed), judge=None)
    assert r.status == DRY_RUN and r.pick is None and len(calls) == 3


def test_summary_names_the_pick_and_every_score():
    gen, judge, _ = scripted([face(0.7)])
    s = summary(run_shot("s08", n=1, max_rounds=1, generate=gen, judge=judge))
    assert s["status"] == PASS and s["pick"] == s["candidates"][0]["image"]
    assert s["candidates"][0]["similarity"] == 0.7
