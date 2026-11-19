"""Face QA logic, with a fake detector so tests don't need InsightFace's 280 MB model."""

from pathlib import Path

import numpy as np
import pytest

from fluxpipe.qa.face import SAME_PERSON, FaceHit, FaceScore, cosine, largest, score_images


def unit(*v):
    a = np.array(v, dtype=float)
    return a / np.linalg.norm(a)


ME = unit(1, 0, 0)
NEAR_ME = unit(1, 0.3, 0)  # cosine ~0.96 with ME
STRANGER = unit(0, 1, 0)  # cosine 0 with ME


def fake(table: dict[str, list[FaceHit]]):
    return lambda path: table[Path(path).name]


def test_cosine_basics():
    assert cosine(ME, ME) == pytest.approx(1.0)
    assert cosine(ME, STRANGER) == pytest.approx(0.0)


def test_largest_face_is_the_subject():
    small, big = FaceHit(STRANGER, 40, 0.9), FaceHit(ME, 200, 0.9)
    assert largest([small, big]) is big
    assert largest([]) is None


def test_turned_away_faces_are_not_judged():
    down = FaceScore("x.png", 1, 0.39, 200, 0.9, pitch=-55.9, yaw=-43.2)
    profile = FaceScore("x.png", 1, 0.7, 200, 0.9, pitch=-4.0, yaw=55.7)  # s04: still judged
    assert down.verdict == "turned away" and profile.verdict == "same"
    assert FaceScore("x.png", 1, 0.39, 200, 0.9, yaw=-70).verdict == "turned away"
    assert down.as_dict()["pitch"] == -55.9


def test_scores_and_verdicts():
    detect = fake({
        "ref.png": [FaceHit(ME, 300, 0.99)],
        "a.png": [FaceHit(NEAR_ME, 200, 0.9)],
        "b.png": [FaceHit(STRANGER, 200, 0.9)],
        "c.png": [],
        "d.png": [FaceHit(ME, 40, 0.6)],  # wide shot: a tiny face
    })
    scores, _ = score_images(detect, Path("ref.png"), [Path(n) for n in "a.png b.png c.png d.png".split()])
    assert [s.verdict for s in scores] == ["same", "different", "no face", "too small"]
    assert scores[0].similarity > SAME_PERSON > scores[1].similarity


def test_pairs_skip_small_and_missing_faces():
    detect = fake({
        "ref.png": [FaceHit(ME, 300, 0.99)],
        "a.png": [FaceHit(ME, 200, 0.9)],
        "b.png": [FaceHit(NEAR_ME, 200, 0.9)],
        "c.png": [],
        "d.png": [FaceHit(STRANGER, 40, 0.6)],
    })
    _, pairs = score_images(detect, Path("ref.png"), [Path(n) for n in "a.png b.png c.png d.png".split()])
    assert len(pairs) == 1  # only a-b are both judgeable
    assert pairs[0] > SAME_PERSON


def test_reference_without_a_face_is_an_error():
    with pytest.raises(ValueError, match="No face"):
        score_images(fake({"ref.png": []}), Path("ref.png"), [])


def test_score_round_trips_to_json_shape():
    d = FaceScore("x.png", 1, 0.7, 150, 0.9).as_dict()
    assert d["verdict"] == "same" and d["similarity"] == 0.7
