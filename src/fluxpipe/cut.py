"""M7: the rough cut. Every shot in story order, each from the take that matches its current pick.

The generative-media lesson: a cut is where continuity is finally judged. Each shot was checked
on its own (face scores, review by eye); only played in order do you see whether the place, the
light and the man carry from one shot to the next. So the cut is built straight from the
manifests: the bible's shot order, `picks.json` for which keyframe is current, `clips.json` for
which clips were animated from it. A clip made from an older pick is never used by accident.

For each shot it uses the enhanced final if it's up to date, else the newest draft made from
the current pick (or the take named with --use). Hard cuts inside a scene (same place, same
moment); a short fade through black where the location changes, which reads as time passing.
Clips are played at their native 1280x704: upscaling adds pixels, not detail.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .demo import BG, DIM, FPS, INK, clip_seconds, font

W, H = 1280, 704
SCENE_FADE_S = 0.4
ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
          "-video_track_timescale", "12288", "-an"]


@dataclass
class CutShot:
    id: str
    location: str
    clip: Path
    kind: str  # "final" or "draft"
    cut_to: float | None = None  # the bible's trim, if any


def choose_clips(shot_ids: list[tuple[str, str]], picks: dict, clips: dict, results_video: Path,
                 assets_video: Path, use: dict[str, str] | None = None) -> tuple[list[CutShot], list[str]]:
    """(shot id, location) in story order -> the clip to use for each, plus every problem found."""
    use = use or {}
    chosen, problems = [], []
    for sid, location in shot_ids:
        pick = (picks.get(sid) or {}).get("pick")
        entry = clips.get(sid) or {}
        final = entry.get("final") or {}
        current = [t for t in entry.get("takes", []) if t.get("from_pick") == pick]
        if sid in use:
            take = next((t for t in entry.get("takes", []) if t["file"] == use[sid]), None)
            if take is None:
                problems.append(f"{sid}: no take named {use[sid]} in clips.json")
            elif take.get("from_pick") != pick:
                problems.append(f"{sid}: {use[sid]} was animated from {take.get('from_pick')}, not the pick {pick}")
            else:
                chosen.append(CutShot(sid, location, assets_video / sid / use[sid], "draft"))
        elif final and final.get("from_pick") == pick and (results_video / f"{sid}.mp4").is_file():
            chosen.append(CutShot(sid, location, results_video / f"{sid}.mp4", "final"))
        elif current:
            chosen.append(CutShot(sid, location, assets_video / sid / current[-1]["file"], "draft"))
        else:
            problems.append(f"{sid}: no clip from the current pick {pick} (fluxpipe video {sid})")
    for s in chosen:
        if not s.clip.is_file():
            problems.append(f"{s.id}: {s.clip} is listed but missing")
    return chosen, problems


def scene_fades(shots: list[CutShot]) -> list[tuple[bool, bool]]:
    """(fade in, fade out) per shot: at the start and end, and either side of a change of place."""
    out = []
    for i, s in enumerate(shots):
        first = i == 0 or shots[i - 1].location != s.location
        last = i == len(shots) - 1 or shots[i + 1].location != s.location
        out.append((first, last))
    return out


def _fade_filter(duration: float, fade_in: bool, fade_out: bool) -> str:
    parts = []
    if fade_in:
        parts.append(f"fade=in:st=0:d={SCENE_FADE_S}")
    if fade_out:
        parts.append(f"fade=out:st={duration - SCENE_FADE_S:.3f}:d={SCENE_FADE_S}")
    return ",".join(parts + ["format=yuv420p"])


def clip_cmd(clip: Path, duration: float, fade_in: bool, fade_out: bool, out: Path) -> list[str]:
    return ["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-t", f"{duration:.3f}",
            "-vf", f"scale={W}:{H},fps={FPS},{_fade_filter(duration, fade_in, fade_out)}", *ENCODE, str(out)]


def card_cmd(png: Path, duration: float, out: Path) -> list[str]:
    return ["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{duration}",
            "-i", str(png), "-vf", _fade_filter(duration, True, True), *ENCODE, str(out)]


def title_card(title: str, subtitle: str) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W // 2, H // 2 - 10), title, font=font(64, "bold"), fill=INK, anchor="ms")
    d.text((W // 2, H // 2 + 50), subtitle, font=font(26), fill=DIM, anchor="ms")
    return img


def end_card(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    top = H // 2 - (len(lines) - 1) * 24
    for i, line in enumerate(lines):
        d.text((W // 2, top + i * 48), line, font=font(28 if i else 34, "medium" if i == 0 else "regular"),
               fill=INK if i == 0 else DIM, anchor="ms")
    return img


def build(shots: list[CutShot], out: Path, work: Path, *, title: str, subtitle: str, end_lines: list[str],
          title_s: float = 3.0, end_s: float = 4.0, run=subprocess.run, probe=None) -> dict:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to build the cut (brew install ffmpeg)")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts = []

    def card(img: Image.Image, name: str, seconds: float) -> None:
        png, mp4 = work / f"{name}.png", work / f"{name}.mp4"
        img.save(png)
        run(card_cmd(png, seconds, mp4), check=True)
        parts.append(mp4)

    card(title_card(title, subtitle), "00_title", title_s)
    total = 0.0
    for i, (s, (fin, fout)) in enumerate(zip(shots, scene_fades(shots)), 1):
        seconds = (probe or clip_seconds)(s.clip)
        if s.cut_to:
            seconds = min(seconds, s.cut_to)
        seconds = int(seconds * FPS) / FPS  # whole frames, so the parts join without a gap
        mp4 = work / f"{i:02d}_{s.id}.mp4"
        run(clip_cmd(s.clip, seconds, fin, fout, mp4), check=True)
        parts.append(mp4)
        total += seconds
    card(end_card(end_lines), "99_end", end_s)

    list_file = work / "parts.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    return {"parts": [p.name for p in parts], "story_s": round(total, 2),
            "finals": [s.id for s in shots if s.kind == "final"],
            "drafts": [s.id for s in shots if s.kind == "draft"]}
