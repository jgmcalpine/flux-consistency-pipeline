import json

import httpx
import pytest

from fluxpipe.bfl_client import BFLClient, BFLError
from fluxpipe.bible import load_bible
from fluxpipe.config import ROOT, billed_mp, estimate_cost
from fluxpipe.generate import keyframe_request, shot_seed
from fluxpipe.ledger import Ledger, SpendCapExceeded

# --- pricing -------------------------------------------------------------------------


def test_megapixels_round_up_and_cap_at_four():
    assert billed_mp(1280, 720) == 1  # 0.92 MP
    assert billed_mp(1024, 1024) == 2  # 1.05 MP rounds up (pessimistic on purpose)
    assert billed_mp(4000, 4000) == 4  # FLUX.2 caps at 4 MP


def test_first_megapixel_costs_more_than_the_rest():
    assert estimate_cost("flux-2-pro", width=1280, height=720) == 0.03
    # 1920x1080 = 2.07 MP -> billed as 3: $0.03 + 2 x $0.015
    assert estimate_cost("flux-2-pro", width=1920, height=1080) == 0.06


def test_reference_images_are_billed():
    one_big_ref = estimate_cost("flux-2-pro", width=1280, height=720, refs=[(2048, 2048)])
    assert one_big_ref == round(0.03 + 4 * 0.015, 4)  # one ref: billed at its size (5 MP -> 4)
    three_refs = estimate_cost("flux-2-pro", width=1280, height=720, refs=[(2048, 2048)] * 3)
    assert three_refs == round(0.03 + 3 * 0.015, 4)  # several refs: 1 MP each


def test_klein_rejects_too_many_refs():
    with pytest.raises(ValueError):
        estimate_cost("flux-2-klein-4b", width=1280, height=720, refs=[(1024, 1024)] * 5)


def test_unknown_model_is_an_error():
    with pytest.raises(KeyError):
        estimate_cost("flux-9-ultra", width=1280, height=720)


# --- request building ----------------------------------------------------------------


@pytest.fixture
def bible():
    return load_bible(ROOT / "bible")


def test_request_is_reproducible_and_unrewritten(bible):
    req = keyframe_request(bible, bible.shot("s01"), model="flux-2-pro")
    assert req.payload["seed"] == shot_seed("s01")
    assert req.payload["disable_pup"] is True  # pro would otherwise rewrite our prompt
    assert json.loads(req.payload["prompt"])["subjects"][0]["action"] == bible.shot("s01").action
    assert "disable_pup" not in keyframe_request(bible, bible.shot("s01")).payload  # klein


def test_sizes_must_be_multiples_of_16(bible):
    with pytest.raises(ValueError):
        keyframe_request(bible, bible.shot("s01"), width=1000, height=563)


# --- client, against a fake BFL ------------------------------------------------------


def fake_bfl(statuses=("Pending", "Generating", "Ready"), submit_status=200, cost=1.4):
    """A stand-in for api.bfl.ai that records every request it sees."""
    seen: list[httpx.Request] = []
    polls = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "POST":
            if submit_status != 200:
                return httpx.Response(submit_status, json={"detail": "nope"})
            return httpx.Response(200, json={
                "id": "job-1", "polling_url": "https://api.us1.bfl.ai/v1/get_result?id=job-1",
                "cost": cost, "input_mp": 0, "output_mp": 0.92,
            })
        if "get_result" in request.url.path:
            status = next(polls)
            body = {"id": "job-1", "status": status}
            if status == "Ready":
                body["result"] = {"sample": "https://delivery.us1.bfl.ai/abc.png"}
            return httpx.Response(200, json=body)
        if request.url.host.startswith("delivery"):
            return httpx.Response(200, content=b"\x89PNG fake")
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.sqlite3", spend_cap_usd=1.0)


def run(client, tmp_path, est=0.014):
    return client.run_job(
        endpoint="flux-2-klein-4b", payload={"prompt": "{}", "seed": 1},
        est_cost_usd=est, stage="test", out_path=tmp_path / "out.png", shot_id="s01",
    )


def test_full_job_saves_image_and_records_real_cost(ledger, tmp_path):
    transport, seen = fake_bfl()
    client = BFLClient("k", ledger, dry_run=False, transport=transport, sleep=lambda s: None)
    result = run(client, tmp_path, est=0.02)
    assert (tmp_path / "out.png").read_bytes().startswith(b"\x89PNG")
    assert result.actual_cost_usd == 0.014  # 1.4 credits
    row = ledger.last()
    assert (row["status"], row["job_id"], row["actual_cost_usd"]) == ("done", "job-1", 0.014)
    assert ledger.total_spent() == 0.014  # the real cost replaces the estimate


def test_api_key_never_goes_to_the_delivery_cdn(ledger, tmp_path):
    transport, seen = fake_bfl()
    run(BFLClient("k", ledger, dry_run=False, transport=transport, sleep=lambda s: None), tmp_path)
    assert all("x-key" not in r.headers for r in seen if r.url.host.startswith("delivery"))
    assert all(r.headers["x-key"] == "k" for r in seen if "bfl.ai" in r.url.host and
               not r.url.host.startswith("delivery"))


def test_moderated_job_is_logged_as_failed(ledger, tmp_path):
    transport, _ = fake_bfl(statuses=("Pending", "Content Moderated"))
    client = BFLClient("k", ledger, dry_run=False, transport=transport, sleep=lambda s: None)
    with pytest.raises(BFLError, match="Content Moderated"):
        run(client, tmp_path)
    row = ledger.last()
    assert row["status"] == "failed" and row["actual_cost_usd"] == 0.014  # may still be billed


def test_out_of_credits_has_a_clear_message(ledger, tmp_path):
    transport, _ = fake_bfl(submit_status=402)
    client = BFLClient("k", ledger, dry_run=False, transport=transport, sleep=lambda s: None)
    with pytest.raises(BFLError, match="out of credits"):
        run(client, tmp_path)


def test_dry_run_sends_nothing(ledger, tmp_path):
    transport, seen = fake_bfl()
    result = run(BFLClient("", ledger, dry_run=True, transport=transport), tmp_path)
    assert result.dry_run and seen == [] and ledger.total_spent() == 0


def test_spend_cap_blocks_before_any_request(ledger, tmp_path):
    transport, seen = fake_bfl()
    client = BFLClient("k", ledger, dry_run=False, transport=transport, sleep=lambda s: None)
    with pytest.raises(SpendCapExceeded):
        run(client, tmp_path, est=5.0)
    assert seen == []


def test_fingerprint_ignores_seed_but_not_prompt_or_refs(bible, tmp_path):
    from fluxpipe.generate import fingerprint

    a = keyframe_request(bible, bible.shot("s02"), model="flux-2-pro", seed=1)
    b = keyframe_request(bible, bible.shot("s02"), model="flux-2-pro", seed=2)
    assert fingerprint(a.endpoint, a.payload, a.refs) == fingerprint(b.endpoint, b.payload, b.refs)
    changed = {**a.payload, "prompt": a.payload["prompt"] + " "}
    assert fingerprint(a.endpoint, changed, a.refs) != fingerprint(a.endpoint, a.payload, a.refs)
    relocked = tmp_path / "face.png"
    relocked.write_bytes(b"a different image")
    assert fingerprint(a.endpoint, a.payload, [relocked, *a.refs[1:]]) != fingerprint(a.endpoint, a.payload, a.refs)
