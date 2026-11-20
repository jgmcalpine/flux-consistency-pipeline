"""Background jitter: a still is steady, a tremble is jitter, a smooth move is a camera move."""

import numpy as np
import pytest

from fluxpipe.qa.background import Background, path_stats
from fluxpipe.qa.report import face_cell, markdown, row, totals


def test_a_still_path_has_no_jitter():
    assert path_stats([np.zeros(2)] * 30, [1.0] * 30) == (0.0, 0.0, 1.0)


def test_a_smooth_move_is_not_jitter_but_a_tremble_is():
    smooth = [np.array([0.25, 0.0])] * 60  # 0.5 px a frame at full size: a slow, steady pan
    jitter, moved, _ = path_stats(smooth, [1.0] * 60)
    assert jitter < 0.1 and moved == 30.0
    shake = [np.array([(-1) ** i * 0.5, 0.0]) for i in range(60)]  # 1 px back and forth every frame
    jitter, moved, _ = path_stats(shake, [1.0] * 60)
    assert jitter > 0.3 and moved < 1


def test_verdicts_and_camera_words():
    assert Background(120, 0.02, 1.9, 1.003, 1.0).verdict == "steady"
    assert Background(120, 0.57, 76, 1.2, 1.0).verdict == "slight jitter"
    assert Background(120, 2.6, 3, 1.0, 1.0).verdict == "jittery"
    assert Background(120, 0.1, 0, 1.0, 0.5).verdict == "not tracked"
    assert Background(120, 0.02, 1.9, 1.003, 1.0).camera == "still"
    assert Background(120, 0.3, 50.5, 1.168, 1.0).camera == "drifts 50 px, pushes in 17%"
    assert Background(120, 0.1, 3, 0.958, 1.0).camera == "pulls out 4%"


def test_measure_on_a_synthetic_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from fluxpipe.qa.background import measure

    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur((rng.random((800, 1400)) * 255).astype(np.uint8), (0, 0), 2)
    def write(name, offsets):
        path = tmp_path / name
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (1280, 704))
        for dx, dy in offsets:
            frame = texture[40 + dy:744 + dy, 60 + dx:1340 + dx]
            out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        out.release()
        return path
    still = measure(write("still.mp4", [(0, 0)] * 30))
    shaken = measure(write("shaken.mp4", [(2 * (-1) ** i, 0) for i in range(30)]))
    assert still.verdict == "steady" and still.camera == "still"
    assert shaken.verdict == "jittery"
    assert measure(write("trim.mp4", [(0, 0)] * 48), seconds=1.0).frames == 24


def test_report_rows_say_both_measurements():
    bg = Background(120, 0.02, 1.9, 1.003, 1.0).as_dict()
    face = {"frames": 10, "judged": 5, "below": 1, "first": 0.62, "min": 0.49}
    rows = [row("s04", "final", 5.04, face, True, bg), row("s03", "draft", 5.04, None, False, bg),
            row("s05", "draft", 5.04, {"frames": 10, "judged": 0}, True, bg)]
    assert rows[0]["face_verdict"] == "same person in 4 of 5 frames"
    assert face_cell(None, False)[1] == "n/a" and rows[2]["face_verdict"] == "by eye"
    t = totals(rows)
    assert (t["face_frames_same"], t["face_frames"], t["steady"]) == (4, 5, 3)
    table = markdown(rows)
    assert "| s04 | final | **same person in 4 of 5 frames**" in table and "0.02 px, steady" in table
