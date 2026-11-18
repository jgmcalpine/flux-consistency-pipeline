"""M5: keyframes. Generate N candidates per shot, judge each one, keep the best, retry if none pass.

The generative-media lesson: even with references, every seed is a different roll of the dice.
Most rolls land close to the reference; some drift (a different jaw, an older face). Making a few
candidates and letting a measurement choose is cheaper and more honest than hand-tuning prompts
until one seed looks right. It's also what you'd do on a real production: shoot several takes.

Rules for a shot:
  - Round 1 makes N candidates. The first seed is the shot's fixed seed (the same one the baseline
    used), the rest follow it, so every candidate is reproducible.
  - If a candidate's face reaches SAME_PERSON, the shot passes and we stop paying.
  - If none does, another round of N, up to max_rounds. Then keep the best and flag it.
  - If no candidate has a face big enough to judge (hands-only and wide shots), retrying can't
    help: keep them all and flag the shot for a human to pick by eye.

Nothing here talks to the API or loads a model; the CLI passes those in. That keeps the logic testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .generate import shot_seed
from .qa.face import SAME_PERSON, FaceScore

PASS = "pass"  # a candidate reached SAME_PERSON
BELOW = "below"  # judged, but no candidate reached it within max_rounds
BY_EYE = "by eye"  # no judgeable face in the shot; a human picks
DRY_RUN = "dry run"  # nothing generated, nothing to judge


@dataclass
class Candidate:
    seed: int
    path: Path
    face: FaceScore | None = None

    @property
    def judgeable(self) -> bool:
        return self.face is not None and self.face.verdict in ("same", "different")


@dataclass
class ShotResult:
    shot_id: str
    status: str
    candidates: list[Candidate] = field(default_factory=list)
    pick: Candidate | None = None


def candidate_seeds(shot_id: str, round_: int, n: int) -> list[int]:
    """Round 0 starts at the shot's own seed; later rounds continue the sequence."""
    base = shot_seed(shot_id)
    return [(base + round_ * n + i) % 2**32 for i in range(n)]


# generate(seed) -> the saved image, or None in dry run. judge(path) -> its face score.
Generate = Callable[[int], Path | None]
Judge = Callable[[Path], FaceScore]


def run_shot(
    shot_id: str, *, n: int, max_rounds: int, generate: Generate, judge: Judge | None
) -> ShotResult:
    result = ShotResult(shot_id, DRY_RUN)
    for round_ in range(max_rounds):
        for seed in candidate_seeds(shot_id, round_, n):
            path = generate(seed)
            if path is None:  # dry run: one round is enough to see the cost
                continue
            result.candidates.append(Candidate(seed, path, judge(path) if judge else None))
        if not result.candidates:
            return result  # dry run

        judged = [c for c in result.candidates if c.judgeable]
        if not judged:
            result.status, result.pick = BY_EYE, result.candidates[0]
            return result  # no face to measure: another round would cost money for nothing
        best = max(judged, key=lambda c: c.face.similarity)
        result.pick = best
        if best.face.similarity >= SAME_PERSON:
            result.status = PASS
            return result
        result.status = BELOW
    return result


def summary(result: ShotResult) -> dict:
    """What goes in picks.json: every candidate's score, and which one we kept and why."""
    return {
        "shot": result.shot_id,
        "status": result.status,
        "pick": result.pick.path.name if result.pick else None,
        "candidates": [
            {**(c.face.as_dict() if c.face else {}), "seed": c.seed, "image": c.path.name}
            for c in result.candidates
        ],
    }
