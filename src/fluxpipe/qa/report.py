"""The film's QA report: one row per shot, the two measurements side by side.

  FACE: is it the same man? Frames sampled at 2/s, each scored against the locked face (0.5 or
        more = same person). We report the WORST judged frame, because one wrong frame is what
        a viewer notices, plus how many frames could be judged at all.
  BACKGROUND: does the set stay put? Jitter in pixels (under 0.5 = steady) and what the camera did.

Both are measured over exactly what the cut plays (a shot with `cut_to` is measured up to there).
"""

from __future__ import annotations

from .background import JITTERY_PX, STEADY_PX
from .face import SAME_PERSON


def face_cell(summary: dict | None, face_expected: bool) -> tuple[str, str]:
    """(text, verdict) for the face column."""
    if not face_expected:
        return "hands only, no face meant to be in frame", "n/a"
    if summary is None or not summary.get("judged"):
        return "too small or turned away in every frame: checked by eye", "by eye"
    below = summary.get("below", sum([summary["min"] < SAME_PERSON]))
    text = (f"worst {summary['min']:.2f}, first {summary['first']:.2f}"
            f" ({summary['judged']} of {summary['frames']} frames judged)")
    if not below:
        return text, "same person"
    return text, f"same person in {summary['judged'] - below} of {summary['judged']} frames"


def row(shot: str, kind: str, seconds: float, face_summary: dict | None, face_expected: bool,
        background: dict) -> dict:
    face_text, face_verdict = face_cell(face_summary, face_expected)
    return {
        "shot": shot, "render": kind, "seconds": round(seconds, 2),
        "face": face_summary, "face_text": face_text, "face_verdict": face_verdict,
        "background": background,
    }


def totals(rows: list[dict]) -> dict:
    judged = [r for r in rows if r["face_verdict"] not in ("n/a", "by eye")]
    bg = [r["background"]["jitter_px"] for r in rows]
    return {
        "shots": len(rows),
        "face_judged": len(judged),
        "face_same": sum(r["face_verdict"] == "same person" for r in judged),
        "face_worst": min((r["face"]["min"] for r in judged), default=None),
        "face_frames": sum(r["face"]["judged"] for r in judged),
        "face_frames_same": sum(r["face"]["judged"] - r["face"].get("below", 0) for r in judged),
        "steady": sum(r["background"]["verdict"] == "steady" for r in rows),
        "jitter_max": max(bg) if bg else None,
        "jitter_median": sorted(bg)[len(bg) // 2] if bg else None,
    }


def markdown(rows: list[dict]) -> str:
    """The table for the README, generated so it can't drift from the measurements."""
    lines = [
        "| Shot | Render | Face (vs the locked face, 0.5+ = same person) | Background jitter | Camera |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        bg = r["background"]
        face = r["face_text"] if r["face_verdict"] in ("n/a", "by eye") else f"**{r['face_verdict']}**: {r['face_text']}"
        lines.append(f"| {r['shot']} | {r['render']} | {face} | {bg['jitter_px']:.2f} px, {bg['verdict']} |"
                     f" {bg['camera']} |")
    t = totals(rows)
    lines += [
        "",
        f"Face: {t['face_frames_same']} of {t['face_frames']} judged frames are the same person, across the"
        f" {t['face_judged']} shots with a judgeable face"
        + (f" (worst frame {t['face_worst']:.2f})." if t["face_worst"] is not None else "."),
        f"Background: {t['steady']} of {t['shots']} shots steady (jitter under {STEADY_PX} px;"
        f" over {JITTERY_PX} px is visible shake); median {t['jitter_median']:.2f} px, worst {t['jitter_max']:.2f} px.",
    ]
    return "\n".join(lines) + "\n"
