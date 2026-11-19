import base64
import json
import shutil

import httpx
import pytest

from fluxpipe.bfl_client import BFLClient
from fluxpipe.bible import load_bible
from fluxpipe.config import ROOT
from fluxpipe.ledger import Ledger
from fluxpipe.qa.face import FaceScore
from fluxpipe.qa.video import summarize
from fluxpipe.video import (HD_SIZE, draft_request, enhance_request, find_draft_cache,
                            fit_start_frame, seconds)

from test_refs import tiny_png


@pytest.fixture
def bible():
    return load_bible(ROOT / "bible")


def test_every_shot_has_motion(bible):
    assert all(s.motion for s in bible.shots)


def test_draft_is_motion_only_no_audio_no_seed(bible, tmp_path):
    shot = bible.shot("s02")
    start = tiny_png(tmp_path / "start.png", 16, 16)
    req = draft_request(shot, start)
    p = req.payload
    assert p["mode"] == "i2v" and p["draft"] is True and p["generate_audio"] is False
    assert p["prompt"] == shot.motion.strip()
    assert "seed" not in p  # FLUX 3 video has no seed field (additionalProperties: false)
    uri = p["keyframes"][0]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == start.read_bytes()
    assert req.endpoint == "flux-3-video"
    assert req.est_cost_usd == 0.06 * seconds(shot)


def test_full_render_prices_and_drafts_are_hd_only(bible, tmp_path):
    shot = bible.shot("s02")
    start = tiny_png(tmp_path / "start.png", 16, 16)
    assert draft_request(shot, start, draft=False).est_cost_usd == 0.15 * seconds(shot)
    with pytest.raises(ValueError):
        draft_request(shot, start, resolution="fhd")
    with pytest.raises(ValueError, match="motion"):
        draft_request(shot.model_copy(update={"motion": None}), start)


def test_durations_are_whole_seconds_between_5_and_20(bible):
    shot = bible.shot("s02")
    assert seconds(shot.model_copy(update={"duration_s": 3.2})) == 5
    assert seconds(shot.model_copy(update={"duration_s": 6.1})) == 7
    assert seconds(shot.model_copy(update={"duration_s": 40})) == 20


def test_enhance_sends_only_the_draft_cache(bible, tmp_path):
    cache = tmp_path / "d.cache.bin"
    cache.write_bytes(b"bundle")
    req = enhance_request(bible.shot("s02"), cache)
    assert req.payload == {"mode": "draft_enhance", "resolution": "hd",
                           "draft_cache": base64.b64encode(b"bundle").decode()}


def test_draft_cache_is_found_wherever_it_is():
    assert find_draft_cache({"result": {"sample": "x", "draft_cache": "u1"}}) == "u1"
    assert find_draft_cache({"draft_cache": "u2"}) == "u2"
    assert find_draft_cache({"result": {"files": [{"draft_cache": "u3"}]}}) == "u3"
    assert find_draft_cache({"result": {"sample": "x"}}) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_start_frame_is_cropped_to_hd(tmp_path):
    from fluxpipe.images import png_size
    out = fit_start_frame(tiny_png(tmp_path / "k.png", 1280, 720), tmp_path / "start.png")
    assert png_size(out) == HD_SIZE


def test_video_job_keeps_poll_and_leaves_blobs_out_of_the_ledger(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"id": "v1", "cost": 30,
                                             "polling_url": "https://api.bfl.ai/v1/get_result?id=v1"})
        if "get_result" in request.url.path:
            return httpx.Response(200, json={"status": "Ready", "result": {
                "sample": "https://delivery.bfl.ai/v.mp4", "draft_cache": "https://delivery.bfl.ai/c.bin"}})
        return httpx.Response(200, content=b"bytes")

    ledger = Ledger(tmp_path / "l.sqlite3", spend_cap_usd=5)
    client = BFLClient("k", ledger, dry_run=False, transport=httpx.MockTransport(handler),
                       sleep=lambda s: None)
    res = client.run_job(endpoint="flux-3-video", stage="video", out_path=tmp_path / "v.mp4",
                         payload={"mode": "i2v", "prompt": "he smiles", "keyframes": ["data:..."],
                                  "draft": True}, est_cost_usd=0.3)
    assert res.actual_cost_usd == 0.3
    assert find_draft_cache(res.poll) == "https://delivery.bfl.ai/c.bin"
    assert client.fetch(find_draft_cache(res.poll), tmp_path / "c.bin").read_bytes() == b"bytes"
    params = json.loads(ledger.last()["params"])
    assert "keyframes" not in params and params["draft"] is True


def test_clip_summary_reports_drift_and_skips_small_faces():
    def fs(sim, h=200):
        return FaceScore("f", 1, sim, h, 0.9)
    s = summarize([fs(0.8), fs(0.6), fs(0.3, h=40), fs(None, None), fs(0.7)])
    assert s == {"frames": 5, "judged": 3, "first": 0.8, "mean": 0.7, "min": 0.6, "drift": 0.2}
    assert summarize([fs(None, None)]) == {"frames": 1, "judged": 0}
