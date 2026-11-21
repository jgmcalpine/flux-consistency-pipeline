"""The README's evidence: the cost table from ledger rows, and the generated blocks in the README."""

import pytest
from PIL import Image

from fluxpipe.demo import Side
from fluxpipe.report import cost_markdown, cost_rows, fill_markers, headline_markdown, stills_figure


def _call(stage, est, real, model="flux-2-pro"):
    return {"stage": stage, "model": model, "est_cost_usd": est, "actual_cost_usd": real}


def test_cost_rows_count_exact_bills_and_sum_once():
    calls = [_call("refs", 0.075, 0.06), _call("refs", 0.045, 0.045), _call("video", 0.3, 0.3, "flux-3-video"),
             _call("keyframe", 0.105, None)]  # no reported cost: the estimate counts, but not as "exact"
    rows = cost_rows(calls)
    assert [r["stage"] for r in rows] == ["refs", "keyframe", "video"]  # the method's order, not the ledger's
    refs = rows[0]
    assert (refs["calls"], refs["exact"]) == (2, 1) and refs["billed"] == pytest.approx(0.105)
    table = cost_markdown(rows)
    assert "| **Total** | | **4** | **$0.53** | **$0.51** | **2 of 4** |" in table


def test_fill_markers_replaces_only_the_generated_block():
    text = "intro\n<!-- generated:costs -->\nold\n<!-- /generated:costs -->\noutro\n"
    out = fill_markers(text, {"costs": "| new |\n"})
    assert out == "intro\n<!-- generated:costs -->\n| new |\n<!-- /generated:costs -->\noutro\n"
    assert fill_markers(out, {"costs": "| new |\n"}) == out  # running it twice changes nothing
    with pytest.raises(ValueError):
        fill_markers("no markers", {"costs": "x"})


def test_stills_figure_has_a_row_per_shot(tmp_path):
    paths = {}
    for s in ("s01", "s02"):
        paths[s] = tmp_path / f"{s}.png"
        Image.new("RGB", (1280, 720), (90, 90, 90)).save(paths[s])
    img = stills_figure(["s01", "s02"], paths, paths, {"s01": Side(0.26, "different")},
                        {"s01": Side(0.7, "same")}, tile=(160, 90))
    assert img.size == (2 * 160 + 12, 64 + 2 * (90 + 12))


def test_headline_says_what_the_numbers_mean():
    cost = cost_rows([_call("video", 0.3, 0.3, "flux-3-video")])
    qa = {"face_frames_same": 17, "face_frames": 18, "steady": 10, "shots": 11, "jitter_median": 0.11}
    top = headline_markdown([0.15, 0.37], [0.59, 0.67], 5, 3,
                            {"shots": ["s04", "s08"], "baseline": 0.301, "pipeline": 0.708}, qa, cost)
    assert "best pair **0.37**: no two shots show the same man" in top
    assert "worst pair **0.59**: every pair is the same man" in top
    assert "| **0.30** | **0.71** |" in top and "**17 of 18**" in top and "$0.30** for 1 calls" in top


def test_cost_summary_says_what_the_ledger_guarantees_and_where_the_money_went():
    from fluxpipe.report import cost_summary

    calls = [_call("refs", 0.075, 0.06), _call("keyframe", 0.09, 0.09), _call("keyframe", 0.09, 0.09),
             {**_call("video", 0.3, 0.3, "flux-3-video"), "mode": "i2v", "draft": 1},
             {**_call("video", 0.75, 0.75, "flux-3-video"), "mode": "draft_enhance", "draft": None}]
    text = cost_summary(cost_rows(calls), calls, picks=1)
    assert "Of 5 calls, 4 were billed exactly as estimated (1 billed less)" in text
    assert "2 keyframe candidates for 1 picks" in text and "1 drafts at $0.30, and 1 full renders at $0.75" in text
    assert "<details>" in text and "| **Total** |" in text
