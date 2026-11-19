"""The side-by-side demo: what it shows (scores, headline) and how it's assembled. No ffmpeg needed."""

import json
import shutil

import pytest
from PIL import Image

from fluxpipe import demo
from fluxpipe.demo import DemoShot, Side, build, headline, load_scores, shot_cmd


def test_labels_say_what_the_score_means():
    assert Side(0.66, "same").label()[:2] == ("0.66", "same person")
    assert Side(0.26, "different").label()[:2] == ("0.26", "a different person")
    assert Side(0.26, "turned away").label()[0] == "n/a"  # a turned head isn't shown as a low score
    assert not Side(0.47, "too small").judged


def test_headline_only_counts_shots_judged_on_both_sides():
    base = {"s02": Side(0.265, "turned away"), "s04": Side(0.257, "different"), "s08": Side(0.344, "different")}
    pipe = {"s02": Side(0.657, "same"), "s04": Side(0.666, "same"), "s08": Side(0.750, "same")}
    h = headline(["s02", "s04", "s08"], base, pipe)
    assert h == {"shots": ["s04", "s08"], "baseline": 0.3, "pipeline": 0.708}
    assert headline(["s02"], base, pipe) is None


def test_load_scores_keys_by_shot(tmp_path):
    f = tmp_path / "faces.json"
    f.write_text(json.dumps({"shots": [
        {"image": "s04_flux-2-pro_s1.png", "similarity": 0.257, "verdict": "different"},
        {"image": "s08.png", "similarity": 0.75, "verdict": "same"},
    ]}))
    scores = load_scores(f)
    assert scores["s04"].similarity == 0.257 and scores["s08"].verdict == "same"


def test_shot_cmd_overlays_still_and_clip_on_the_card(tmp_path):
    cmd = shot_cmd(tmp_path / "c.png", tmp_path / "b.png", tmp_path / "v.mp4", 5.0, tmp_path / "o.mp4")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=1280:704" in graph and "zoompan" in graph and "d=120" in graph  # 5 s at 24 fps
    assert f"overlay={demo.LEFT_X}:{demo.PANEL_Y}" in graph and f"overlay={demo.RIGHT_X}:{demo.PANEL_Y}" in graph
    assert "-an" in cmd  # silent for now


def test_build_renders_every_part_then_joins_them(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("build checks ffmpeg is installed first")
    face = tmp_path / "face.png"
    Image.new("RGB", (896, 1088), (120, 120, 120)).save(face)
    shots = [DemoShot(s, tmp_path / f"{s}.png", tmp_path / f"{s}.mp4", Side(0.26, "different"), Side(0.7, "same"))
             for s in ("s04", "s08")]
    calls = []
    info = build(shots, face, tmp_path / "out.mp4", tmp_path / "work", spend_usd=12.5,
                 run=lambda cmd, check: calls.append(cmd), probe=lambda clip: 5.04)
    assert info["parts"] == ["00_intro.mp4", "01_s04.mp4", "02_s08.mp4", "99_outro.mp4"]
    assert info["headline"]["pipeline"] == 0.7
    assert len(calls) == 5 and calls[-1][calls[-1].index("-f") + 1] == "concat"
    cards = sorted(p.name for p in (tmp_path / "work").glob("*.png"))
    assert cards == ["00_intro.png", "01_s04.png", "02_s08.png", "99_outro.png"]
    assert Image.open(tmp_path / "work" / "01_s04.png").size == (demo.W, demo.H)
    listed = (tmp_path / "work" / "parts.txt").read_text().splitlines()
    assert len(listed) == 4 and listed[0].endswith("00_intro.mp4'")
