"""The ledger: a SQLite log of every paid call, and the thing that enforces the spend cap.

Every paid call follows the same three steps:
  1. reserve()  - checks the cap and writes a 'pending' row BEFORE any money is spent
  2. (the API call happens)
  3. complete() or fail() - fills in the result

A pending row counts toward spend, so a crash mid-call can never hide money from the cap.
Once BFL reports the real cost, that replaces the estimate in the total.
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
    stage         TEXT NOT NULL,      -- test | refs | keyframe | baseline | video
    shot_id       TEXT,
    model         TEXT NOT NULL,
    prompt        TEXT NOT NULL,
    seed          INTEGER,
    refs          TEXT NOT NULL,      -- JSON list of reference image paths
    params        TEXT NOT NULL,      -- JSON of any other request parameters
    est_cost_usd  REAL NOT NULL,
    actual_cost_usd REAL,             -- what BFL reported; NULL until known
    status        TEXT NOT NULL,      -- pending | done | failed | dry_run
    job_id        TEXT,
    output_path   TEXT,
    qa_score      REAL,
    error         TEXT
);
"""

# Columns added after M0. Old ledgers get them on open (SQLite has no ADD COLUMN IF NOT EXISTS).
MIGRATIONS = {"actual_cost_usd": "ALTER TABLE generations ADD COLUMN actual_cost_usd REAL"}

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
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(generations)")}
        for column, sql in MIGRATIONS.items():
            if column not in have:
                self.db.execute(sql)
        self.db.commit()
        self.cap = spend_cap_usd

    def total_spent(self) -> float:
        marks = ",".join("?" * len(BILLABLE))
        row = self.db.execute(
            "SELECT COALESCE(SUM(COALESCE(actual_cost_usd, est_cost_usd)), 0)"
            f" FROM generations WHERE status IN ({marks})",
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

    def complete(
        self,
        row_id: int,
        *,
        output_path: str,
        job_id: str | None = None,
        actual_cost_usd: float | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE generations SET status = 'done', output_path = ?, job_id = ?,"
            " actual_cost_usd = ? WHERE id = ?",
            (output_path, job_id, actual_cost_usd, row_id),
        )
        self.db.commit()

    def fail(
        self,
        row_id: int,
        error: str,
        *,
        job_id: str | None = None,
        actual_cost_usd: float | None = None,
    ) -> None:
        # A failed call may still have been billed, so it keeps counting toward spend.
        self.db.execute(
            "UPDATE generations SET status = 'failed', error = ?, job_id = ?,"
            " actual_cost_usd = ? WHERE id = ?",
            (error, job_id, actual_cost_usd, row_id),
        )
        self.db.commit()

    def last(self) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM generations ORDER BY id DESC LIMIT 1").fetchone()

    def set_score(self, row_id: int, score: float) -> None:
        self.db.execute("UPDATE generations SET qa_score = ? WHERE id = ?", (score, row_id))
        self.db.commit()

    def summary_by_stage(self) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT stage, status, COUNT(*) AS calls,"
            " ROUND(SUM(COALESCE(actual_cost_usd, est_cost_usd)), 3) AS usd"
            " FROM generations GROUP BY stage, status ORDER BY stage, status"
        ).fetchall()
