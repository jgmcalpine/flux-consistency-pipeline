"""The only module that talks to the Black Forest Labs API.

BFL uses an async job model:
  1. POST the request to an endpoint, get back a job id and a polling_url
  2. GET the polling_url until status is "Ready" (or an error)
  3. Download the result URL (it expires, so save it straight away)

Every call goes through the ledger first, so the spend cap is checked before any money moves.

M2 TODO: confirm endpoint names and request fields against https://docs.bfl.ai before
the first real call. FLUX 3 Video is still labelled preview, so check its fields at M6 too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .ledger import Ledger

BASE_URL = "https://api.bfl.ai/v1"
POLL_INTERVAL_S = 1.5
POLL_TIMEOUT_S = 600


@dataclass
class JobResult:
    ledger_id: int
    job_id: str | None
    output_path: Path | None
    dry_run: bool


class BFLError(RuntimeError):
    pass


class BFLClient:
    def __init__(self, api_key: str, ledger: Ledger, *, dry_run: bool = True):
        if not dry_run and not api_key:
            raise BFLError("BFL_API_KEY is empty. Add it to .env, or keep DRY_RUN=true.")
        self.ledger = ledger
        self.dry_run = dry_run
        self.http = httpx.Client(
            base_url=BASE_URL,
            headers={"x-key": api_key, "accept": "application/json"},
            timeout=60,
        )

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
    ) -> JobResult:
        """Reserve in the ledger, submit, poll, download. Logs failures (they may be billed)."""
        row_id = self.ledger.reserve(
            stage=stage,
            model=endpoint,
            prompt=payload.get("prompt", ""),
            seed=payload.get("seed"),
            refs=refs,
            params={k: v for k, v in payload.items() if k != "prompt"},
            est_cost_usd=est_cost_usd,
            shot_id=shot_id,
            dry_run=self.dry_run,
        )
        if self.dry_run:
            return JobResult(row_id, None, None, dry_run=True)

        try:
            submitted = self.http.post(f"/{endpoint}", json=payload)
            submitted.raise_for_status()
            job = submitted.json()
            result_url = self._poll(job["polling_url"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(httpx.get(result_url, timeout=120).content)
        except Exception as exc:  # log every failure, then re-raise
            self.ledger.fail(row_id, repr(exc))
            raise
        self.ledger.complete(row_id, output_path=str(out_path), job_id=job.get("id"))
        return JobResult(row_id, job.get("id"), out_path, dry_run=False)

    def _poll(self, polling_url: str) -> str:
        deadline = time.monotonic() + POLL_TIMEOUT_S
        while time.monotonic() < deadline:
            resp = self.http.get(polling_url)
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status")
            if status == "Ready":
                return body["result"]["sample"]
            if status not in {"Pending", "Processing", "Queued"}:
                raise BFLError(f"Job ended with status {status!r}: {body}")
            time.sleep(POLL_INTERVAL_S)
        raise BFLError(f"Timed out after {POLL_TIMEOUT_S}s waiting for {polling_url}")
