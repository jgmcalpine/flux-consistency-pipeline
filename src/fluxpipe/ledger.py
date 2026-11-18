"""The ledger: a SQLite log of every paid call, and the thing that enforces the spend cap.

Every paid call follows the same three steps:
  1. reserve()  - checks the cap and writes a 'pending' row BEFORE any money is spent
  2. (the API call happens)
  3. complete() or fail() - fills in the result

A pending row counts toward spend, so a crash mid-call can never hide money from the cap.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    stage         TEXT NOT NULL,      -- refs | keyframe | baseline | video
    shot_id       TEXT,
    model         TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    seed          INTEGER,
    refs          TEXT NOT NULL,      -- JSON list of reference image paths
    params        TEXT NOT NULL,      -- JSON of any other request parameters
    est_cost_usd  REAL NOT NULL,
    status        TEXT NOT NULL,      -- pending | done | failed | dry_run
    job_id        TEXT,
    output_path   TEXT,
    qa_score      REAL,
    error         TEXT
);
"""

# Rows with these statuses count toward the spend cap.
BILLABLE = ("pending", "done", "failed")


class SpendCapExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Path, spend_cap_usd: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.cap = spend_cap_usd

    def total_spent(self) -> float:
        marks = ",".join("?" * len(BILLABLE))
        row = self.db.execute(
            f"SELECT COALESCE(SUM(est_cost_usd), 0) FROM generations WHERE status IN ({marks})",
            BILLABLE,
        ).fetchone()
        return round(row[0], 4)

    def remaining(self) -> float:
        return round(self.cap - self.total_spent(), 4)

    def reserve(
        self,
        *,
        stage: str,
        model: str,
        prompt: str,
        est_cost_usd: float,
        shot_id: str | None = None,
        seed: int | None = None,
        refs: list[str] | None = None,
        params: dict | None = None,
        dry_run: bool = False,
    ) -> int:
        """Log a call before it's made. Raises SpendCapExceeded if it would pass the cap."""
        if not dry_run and self.total_spent() + est_cost_usd > self.cap:
            raise SpendCapExceeded(
                f"This call (~${est_cost_usd:.2f}) would pass the ${self.cap:.2f} cap; "
                f"${self.remaining():.2f} left. Raise SPEND_CAP_USD in .env if you mean it."
            )
        cur = self.db.execute(
            "INSERT INTO generations (created_at, stage, shot_id, model, prompt, seed, refs,"
            " params, est_cost_usd, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                stage,
                shot_id,
                model,
                prompt,
                seed,
                json.dumps(refs or []),
                json.dumps(params or {}),
                est_cost_usd,
                "dry_run" if dry_run else "pending",
            ),
        )
        self.db.commit()
        return cur.lastrowid

    def complete(self, row_id: int, *, output_path: str, job_id: str | None = None) -> None:
        self.db.execute(
            "UPDATE generations SET status = 'done', output_path = ?, job_id = ? WHERE id = ?",
            (output_path, job_id, row_id),
        )
        self.db.commit()

    def fail(self, row_id: int, error: str) -> None:
        # A failed call may still have been billed, so it keeps counting toward spend.
        self.db.execute(
            "UPDATE generations SET status = 'failed', error = ? WHERE id = ?", (error, row_id)
        )
        self.db.commit()

    def set_score(self, row_id: int, score: float) -> None:
        self.db.execute("UPDATE generations SET qa_score = ? WHERE id = ?", (score, row_id))
        self.db.commit()

    def summary_by_stage(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT stage, status, COUNT(*) AS calls, ROUND(SUM(est_cost_usd), 2) AS usd"
            " FROM generations GROUP BY stage, status ORDER BY stage, status"
        ).fetchall()
