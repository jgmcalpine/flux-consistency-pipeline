"""The only module that talks to the Black Forest Labs API.

BFL uses an async job model (docs.bfl.ai/api_integration/integration_guidelines):
  1. POST the request to /v1/<model>. The reply has a job `id`, a `polling_url`
     and the job's `cost` in credits (1 credit = $0.01).
  2. GET the polling_url (you must use the one returned, not a fixed URL) until the
     status is "Ready". "Pending", "Reasoning" and "Generating" mean keep waiting;
     anything else ("Request Moderated", "Content Moderated", "Error", ...) is final.
  3. Download result.sample straight away: the signed URL expires after 10 minutes.

Every call goes through the ledger first, so the spend cap is checked before any money moves.

Checked against docs.bfl.ai on 17 Nov 2026; FLUX 3 video (preview) checked 19 Nov 2026.
Video results are an mp4 in result.sample; their URLs expire after about 2 hours.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import USD_PER_CREDIT
from .ledger import Ledger

BASE_URL = "https://api.bfl.ai/v1"  # global; api.eu.bfl.ai and api.us.bfl.ai also exist
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 600
IN_PROGRESS = {"Pending", "Reasoning", "Generating"}
# Payload fields that carry image/video data: never copied into the ledger (paths go in `refs`).
BLOB_FIELDS = {"keyframes", "draft_cache", "start_video"}
SUBMIT_RETRIES = 3  # only for 429 (too many concurrent jobs), which isn't billed


@dataclass
class JobResult:
    ledger_id: int
    job_id: str | None
    output_path: Path | None
    dry_run: bool
    actual_cost_usd: float | None = None
    poll: dict | None = None  # the final poll response (a video draft's draft_cache is in here)


class BFLError(RuntimeError):
    pass


class BFLClient:
    def __init__(
        self,
        api_key: str,
        ledger: Ledger,
        *,
        dry_run: bool = True,
        transport: httpx.BaseTransport | None = None,  # tests pass a fake one
        sleep=time.sleep,
    ):
        if not dry_run and not api_key:
            raise BFLError("BFL_API_KEY is empty. Add it to .env, or keep DRY_RUN=true.")
        self.ledger = ledger
        self.dry_run = dry_run
        self.api_key = api_key
        self._sleep = sleep
        self.http = httpx.Client(
            base_url=BASE_URL,
            headers={"x-key": api_key, "accept": "application/json"},
            timeout=60,
            transport=transport,
        )
        # A separate client for downloads, so the API key is never sent to the delivery CDN.
        self.download = httpx.Client(timeout=120, transport=transport)

    def credits(self) -> float:
        """Current balance in credits. Free; handy for checking costs before and after a call."""
        if not self.api_key:
            raise BFLError("BFL_API_KEY is empty. Add it to .env to check your balance.")
        resp = self.http.get("/credits")
        _raise_for_status(resp)
        return float(resp.json()["credits"])

    def run_job(
        self,
        *,
        endpoint: str,
        payload: dict,
        est_cost_usd: float,
        stage: str,
        out_path: Path,
        shot_id: str | None = None,
        refs: list[str] | None = None,
        poll_timeout_s: float = POLL_TIMEOUT_S,
    ) -> JobResult:
        """Reserve in the ledger, submit, poll, download. Logs failures (they may be billed)."""
        row_id = self.ledger.reserve(
            stage=stage,
            model=endpoint,
            prompt=payload.get("prompt", ""),
            seed=payload.get("seed"),
            refs=refs,
            # Reference image data is left out: `refs` records their paths instead.
            params={k: v for k, v in payload.items()
                    if k != "prompt" and not k.startswith("input_image") and k not in BLOB_FIELDS},
            est_cost_usd=est_cost_usd,
            shot_id=shot_id,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            return JobResult(row_id, None, None, dry_run=True)

        job: dict = {}
        cost: float | None = None
        try:
            job = self._submit(endpoint, payload)
            if job.get("cost") is not None:
                cost = round(float(job["cost"]) * USD_PER_CREDIT, 4)
            poll = self._poll(job["polling_url"], poll_timeout_s)
            image = self.download.get(poll["result"]["sample"])
            image.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image.content)
        except Exception as exc:  # log every failure, then re-raise
            self.ledger.fail(row_id, repr(exc), job_id=job.get("id"), actual_cost_usd=cost)
            raise
        self.ledger.complete(
            row_id, output_path=str(out_path), job_id=job.get("id"), actual_cost_usd=cost
        )
        return JobResult(row_id, job.get("id"), out_path, dry_run=False, actual_cost_usd=cost, poll=poll)

    def fetch(self, url: str, out_path: Path) -> Path:
        """Download a file BFL returned (no API key sent), e.g. a video draft's draft_cache."""
        resp = self.download.get(url)
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.content)
        return out_path

    def _submit(self, endpoint: str, payload: dict) -> dict:
        for attempt in range(SUBMIT_RETRIES + 1):
            resp = self.http.post(f"/{endpoint}", json=payload)
            if resp.status_code == 429 and attempt < SUBMIT_RETRIES:
                self._sleep(2**attempt * 2)  # 2s, 4s, 8s
                continue
            _raise_for_status(resp)
            return resp.json()
        raise AssertionError("unreachable")

    def _poll(self, polling_url: str, timeout_s: float = POLL_TIMEOUT_S) -> dict:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            resp = self.http.get(polling_url)  # absolute URL: overrides base_url, keeps x-key
            _raise_for_status(resp)
            body = resp.json()
            status = body.get("status")
            if status == "Ready":
                return body
            if status not in IN_PROGRESS:
                raise BFLError(f"Job ended with status {status!r}: {body.get('details') or body}")
            self._sleep(POLL_INTERVAL_S)
        raise BFLError(f"Timed out after {timeout_s}s waiting for {polling_url}")


def _raise_for_status(resp: httpx.Response) -> None:
    """Turn HTTP errors into messages that say what to do. Never includes the API key."""
    if resp.is_success:
        return
    hints = {
        401: "the API key was rejected. Check BFL_API_KEY in .env.",
        402: "out of credits. Top up at dashboard.bfl.ai.",
        422: "the request was invalid (a field name or value is wrong).",
        429: "too many concurrent jobs (the limit is 24). Wait and retry.",
    }
    hint = hints.get(resp.status_code, "unexpected error.")
    raise BFLError(f"HTTP {resp.status_code} from {resp.request.url.path}: {hint} {resp.text[:300]}")
