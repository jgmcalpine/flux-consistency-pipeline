"""The rough cut: which clip each shot uses, and where the fades go. No ffmpeg calls."""

import shutil

import pytest

from fluxpipe.cut import CutShot, build, choose_clips, scene_fades


def _setup(tmp_path):
    rv, av = tmp_path / "results", tmp_path / "assets"
    for p in [rv / "s01.mp4", av / "s02" / "s02_draft1.mp4", av / "s02" / "s02_draft2.mp4",
              av / "s03" / "s03_draft1.mp4"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
    picks = {"s01": {"pick": "a1"}, "s02": {"pick": "b2"}, "s03": {"pick": "c2"}}
    clips = {
        "s01": {"takes": [{"file": "s01_draft1.mp4", "from_pick": "a1"}],
                "final": {"from": "s01_draft1.mp4", "from_pick": "a1"}},
        "s02": {"takes": [{"file": "s02_draft1.mp4", "from_pick": "b2"},
                          {"file": "s02_draft2.mp4", "from_pick": "b2"}], "final": None},
        "s03": {"takes": [{"file": "s03_draft1.mp4", "from_pick": "c1"}], "final": None},  # old pick
    }
    return rv, av, picks, clips


def test_final_first_then_newest_current_draft(tmp_path):
    rv, av, picks, clips = _setup(tmp_path)
    shots, problems = choose_clips([("s01", "river"), ("s02", "river")], picks, clips, rv, av)
    assert not problems
    assert [(s.id, s.kind, s.clip.name) for s in shots] == [("s01", "final", "s01.mp4"),
                                                           ("s02", "draft", "s02_draft2.mp4")]


def test_a_clip_from_an_old_pick_is_never_used(tmp_path):
    rv, av, picks, clips = _setup(tmp_path)
    _, problems = choose_clips([("s03", "hut")], picks, clips, rv, av)
    assert problems == ["s03: no clip from the current pick c2 (fluxpipe video s03)"]


def test_use_picks_a_take_but_still_checks_it(tmp_path):
    rv, av, picks, clips = _setup(tmp_path)
    shots, problems = choose_clips([("s02", "river")], picks, clips, rv, av, {"s02": "s02_draft1.mp4"})
    assert not problems and shots[0].clip.name == "s02_draft1.mp4"
    _, problems = choose_clips([("s03", "hut")], picks, clips, rv, av, {"s03": "s03_draft1.mp4"})
    assert "not the pick c2" in problems[0]


def test_fades_only_at_the_ends_and_at_a_change_of_place(tmp_path):
    shots = [CutShot(i, loc, tmp_path / f"{i}.mp4", "draft")
             for i, loc in [("s01", "river"), ("s02", "river"), ("s05", "path"), ("s07", "hut")]]
    assert scene_fades(shots) == [(True, False), (False, True), (True, True), (True, True)]


def test_build_joins_title_shots_and_end(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("build checks ffmpeg is installed first")
    shots = [CutShot("s01", "river", tmp_path / "a.mp4", "final"), CutShot("s02", "river", tmp_path / "b.mp4", "draft")]
    calls = []
    info = build(shots, tmp_path / "film.mp4", tmp_path / "work", title="T", subtitle="a rough cut",
                 end_lines=["one", "two"], run=lambda cmd, check: calls.append(cmd), probe=lambda clip: 5.041667)
    assert info["parts"] == ["00_title.mp4", "01_s01.mp4", "02_s02.mp4", "99_end.mp4"]
    assert info["story_s"] == 10.08  # each clip trimmed to whole frames: 2 x 121 / 24 s
    assert info["finals"] == ["s01"] and info["drafts"] == ["s02"]
    first_clip = calls[1][calls[1].index("-vf") + 1]
    shots[1].cut_to = 2.6
    info = build(shots, tmp_path / "film.mp4", tmp_path / "work", title="T", subtitle="", end_lines=["x"],
                 run=lambda cmd, check: calls.append(cmd), probe=lambda clip: 5.041667)
    assert info["story_s"] == 7.62  # 121/24 + 2.6 (62 whole frames = 2.583)
    assert "fade=in" in first_clip and "fade=out" not in first_clip  # hard cut into s02
