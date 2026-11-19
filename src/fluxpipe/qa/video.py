"""Video QA: does the face stay his face while it moves?

A still can be on-model and the clip still drift: as the head turns or the expression changes,
the model re-imagines the face frame by frame. So we sample frames (2 a second is plenty for a
5 s clip), score each with the same face scorer as the stills, and report:
  - first: the opening frame, which should match the keyframe it started from;
  - min:   the worst moment in the clip;
  - drift: first - min, how far the identity wanders during the motion.
Frames whose face is too small or missing (wide shots, hands only) are left out, as for stills.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .face import FaceScore


def extract_frames(clip: Path, out_dir: Path, fps: float = 2.0) -> list[Path]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to sample frames (brew install ffmpeg)")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={fps}",
         str(out_dir / "f%03d.png")],
        check=True,
    )
    return sorted(out_dir.glob("f*.png"))


def summarize(scores: list[FaceScore]) -> dict:
    judged = [s.similarity for s in scores if s.verdict in ("same", "different")]
    if not judged:
        return {"frames": len(scores), "judged": 0}
    return {
        "frames": len(scores),
        "judged": len(judged),
        "first": judged[0],
        "mean": round(sum(judged) / len(judged), 3),
        "min": min(judged),
        "drift": round(judged[0] - min(judged), 3),
    }
