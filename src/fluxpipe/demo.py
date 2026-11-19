"""M7-lite: the side-by-side demo. Plain prompts on the left, the pipeline on the right.

The generative-media lesson: a demo of consistency has to show the MEASUREMENT, not just the
pictures. Viewers can't hold a face in their head across cuts, so every shot carries the same
number on both sides: the face match against the one locked reference face (ArcFace cosine,
0.5 or more = the same person). The reference sits in the corner the whole time, so you can see
what both sides are being compared with.

The two sides differ only in what the pipeline adds: same model, size and seed for the stills
(see bible/baseline.yaml). The left side is the baseline still with a slow ffmpeg push-in: no
generation, so no spend, and it isn't pretending to be a video model's output. The right side
is the enhanced FLUX 3 clip animated from the pipeline's keyframe. The scores shown are the
STILLS' scores on both sides, so the comparison is like for like (a video frame loses face
detail to compression, see CLAUDE.md "Lessons").

This ffmpeg has no drawtext filter, so all text is drawn with Pillow onto a full-frame card, and
the two pictures are overlaid onto it with ffmpeg.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 24
PANEL_W, PANEL_H = 936, 514  # 1280x704 (the video's frame) scaled to fit two across with gutters
GUTTER = 16
LEFT_X, RIGHT_X = GUTTER, W - GUTTER - PANEL_W
PANEL_Y = 250
FADE_S = 0.3
SAME_PERSON = 0.5

BG = (17, 17, 17)
INK = (236, 236, 236)
DIM = (150, 150, 150)
RED = (226, 92, 80)
GREEN = (98, 196, 120)
AMBER = (224, 176, 72)

FONT_FILES = [  # first one found wins; Pillow's built-in font is the last resort
    ("/System/Library/Fonts/Avenir Next.ttc", {"bold": 0, "medium": 5, "regular": 7}),
    ("/System/Library/Fonts/HelveticaNeue.ttc", {"bold": 1, "medium": 10, "regular": 0}),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", {"bold": 0, "medium": 0, "regular": 0}),
]


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    for path, faces in FONT_FILES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size, index=faces[weight])
    return ImageFont.load_default(size)


@dataclass
class Side:
    """One side of a shot: its still's face score and what that means."""

    similarity: float | None
    verdict: str

    @property
    def judged(self) -> bool:
        return self.verdict in ("same", "different") and self.similarity is not None

    def label(self) -> tuple[str, str, tuple[int, int, int]]:
        """(big number, verdict words, colour)."""
        if not self.judged:
            why = {"turned away": "head turned too far to score", "too small": "face too small to score",
                   "no face": "no face found"}.get(self.verdict, self.verdict)
            return ("n/a", why, AMBER)
        if self.similarity >= SAME_PERSON:
            return (f"{self.similarity:.2f}", "same person", GREEN)
        return (f"{self.similarity:.2f}", "a different person", RED)


def load_scores(faces_json: Path) -> dict[str, Side]:
    """Shot id -> Side, from a faces.json written by `fluxpipe score`."""
    data = json.loads(faces_json.read_text())
    return {s["image"][:3]: Side(s["similarity"], s["verdict"]) for s in data["shots"]}


def headline(shots: list[str], baseline: dict[str, Side], pipeline: dict[str, Side]) -> dict | None:
    """Mean face match over the shots judged on BOTH sides, so neither side gets an easier set."""
    both = [s for s in shots if baseline[s].judged and pipeline[s].judged]
    if not both:
        return None
    mean = lambda side: round(sum(side[s].similarity for s in both) / len(both), 3)  # noqa: E731
    return {"shots": both, "baseline": mean(baseline), "pipeline": mean(pipeline)}


# --- cards (Pillow) ---------------------------------------------------------------------------

def _thumb(face_png: Path, height: int) -> Image.Image:
    img = Image.open(face_png).convert("RGB")
    return img.resize((round(img.width * height / img.height), height), Image.LANCZOS)


def _score_bar(d: ImageDraw.ImageDraw, x: int, y: int, side: Side, width: int = 420) -> None:
    """A 0-1 bar with the same-person line at 0.5, so the number has a scale."""
    d.rounded_rectangle((x, y, x + width, y + 14), 7, fill=(48, 48, 48))
    _, _, colour = side.label()
    if side.judged:
        fill = max(14, round(width * max(0.0, min(1.0, side.similarity))))
        d.rounded_rectangle((x, y, x + fill, y + 14), 7, fill=colour)
    tick = x + round(width * SAME_PERSON)
    d.line((tick, y - 8, tick, y + 22), fill=INK, width=2)
    d.text((tick, y + 28), "same person", font=font(20), fill=DIM, anchor="mt")


def shot_card(shot: str, index: int, total: int, base: Side, pipe: Side, face_png: Path) -> Image.Image:
    """The full-frame background for one shot: labels, reference face, both scores."""
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((LEFT_X, 150), "PLAIN PROMPTS", font=font(54, "bold"), fill=INK, anchor="ls")
    d.text((LEFT_X, 200), "one written description of him, no references", font=font(26), fill=DIM, anchor="ls")
    d.text((RIGHT_X + PANEL_W, 150), "THE PIPELINE", font=font(54, "bold"), fill=INK, anchor="rs")
    d.text((RIGHT_X + PANEL_W, 200), "story bible + locked references + face QA",
           font=font(26), fill=DIM, anchor="rs")

    thumb = _thumb(face_png, 168)
    tx = (W - thumb.width) // 2
    img.paste(thumb, (tx, 26))
    d.rectangle((tx - 1, 25, tx + thumb.width, 26 + thumb.height), outline=(90, 90, 90))
    d.text((W // 2, 26 + thumb.height + 12), "the locked face", font=font(20), fill=DIM, anchor="mt")

    y = PANEL_Y + PANEL_H + 40
    for x, side in ((LEFT_X, base), (RIGHT_X, pipe)):
        number, words, colour = side.label()
        d.text((x, y), number, font=font(88, "bold"), fill=colour, anchor="lt")
        d.text((x + 230, y + 14), words, font=font(36, "medium"), fill=colour, anchor="lt")
        d.text((x + 230, y + 60), "face match with the locked face", font=font(24), fill=DIM, anchor="lt")
        _score_bar(d, x, y + 130, side)

    d.text((W // 2, H - 30), f"{shot}  ·  {index} of {total}", font=font(22), fill=DIM, anchor="ms")
    return img


def intro_card(face_png: Path, n_shots: int) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    thumb = _thumb(face_png, 420)
    img.paste(thumb, (160, (H - thumb.height) // 2))
    x = 160 + thumb.width + 90
    d.text((x, 330), "Is it the same man in every shot?", font=font(64, "bold"), fill=INK, anchor="ls")
    lines = [
        f"{n_shots} shots of one character, made two ways with FLUX:",
        "left, plain prompts; right, the consistency pipeline.",
        "",
        "Every shot is scored against this one locked face",
        "(ArcFace face identity: 0.50 or more = the same person).",
    ]
    for i, line in enumerate(lines):
        d.text((x, 420 + i * 52), line, font=font(34), fill=DIM if not line.startswith("Every") else INK,
               anchor="ls")
    return img


def outro_card(head: dict | None, spend_usd: float | None) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    cx = W // 2
    if head:
        d.text((cx, 230), "Face match with the locked face", font=font(48, "medium"), fill=INK, anchor="ms")
        d.text((cx - 360, 470), f"{head['baseline']:.2f}", font=font(190, "bold"), fill=RED, anchor="ms")
        d.text((cx, 440), "vs", font=font(64, "medium"), fill=DIM, anchor="ms")
        d.text((cx + 360, 470), f"{head['pipeline']:.2f}", font=font(190, "bold"), fill=GREEN, anchor="ms")
        d.text((cx - 360, 540), "plain prompts", font=font(36), fill=DIM, anchor="ms")
        d.text((cx + 360, 540), "the pipeline", font=font(36), fill=DIM, anchor="ms")
        d.text((cx, 640), f"mean over the shots scored on both sides ({', '.join(head['shots'])});"
               " 0.50 or more = the same person", font=font(28), fill=DIM, anchor="ms")
    lines = ["References, not words, hold identity. Every call checked against a spend cap."]
    if spend_usd is not None:
        lines.append(f"Total FLUX API spend so far, every experiment included: ${spend_usd:.2f}")
    for i, line in enumerate(lines):
        d.text((cx, 800 + i * 56), line, font=font(34), fill=INK, anchor="ms")
    d.text((cx, H - 60), "fluxpipe: a consistency pipeline for FLUX", font=font(26), fill=DIM, anchor="ms")
    return img


# --- video (ffmpeg) ---------------------------------------------------------------------------

ENCODE = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
          "-video_track_timescale", "12288", "-an"]


def _fades(duration: float) -> str:
    return f"fade=in:st=0:d={FADE_S},fade=out:st={duration - FADE_S:.3f}:d={FADE_S}"


def card_cmd(card_png: Path, duration: float, out: Path) -> list[str]:
    return ["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{duration}",
            "-i", str(card_png), "-vf", f"{_fades(duration)},format=yuv420p", *ENCODE, str(out)]


def shot_cmd(card_png: Path, baseline_png: Path, clip: Path, duration: float, out: Path,
             push_in: float = 0.08) -> list[str]:
    """Card + the baseline still (slow push-in, centre) on the left + the clip on the right.

    The still is cropped to the clip's 1280x704 and upscaled before zoompan, which moves in whole
    pixels: on a 4x larger image those steps are a quarter of a pixel in the output, so no judder.
    """
    frames = round(duration * FPS)
    left = (f"[1:v]crop=1280:704,scale=5120:-2,"
            f"zoompan=z='1+{push_in}*on/{frames}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2'"
            f":d={frames}:s={PANEL_W}x{PANEL_H}:fps={FPS}[l]")
    right = f"[2:v]scale={PANEL_W}:{PANEL_H}:flags=lanczos,fps={FPS},setpts=PTS-STARTPTS[r]"
    stack = (f"[0:v][l]overlay={LEFT_X}:{PANEL_Y}[a];[a][r]overlay={RIGHT_X}:{PANEL_Y},"
             f"{_fades(duration)},format=yuv420p[v]")
    return ["ffmpeg", "-loglevel", "error", "-y",
            "-loop", "1", "-framerate", str(FPS), "-t", f"{duration}", "-i", str(card_png),
            "-i", str(baseline_png), "-i", str(clip),
            "-filter_complex", f"{left};{right};{stack}", "-map", "[v]", "-t", f"{duration}",
            *ENCODE, str(out)]


def concat_cmd(list_file: Path, out: Path) -> list[str]:
    return ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-c", "copy", "-movflags", "+faststart", str(out)]


def clip_seconds(clip: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(clip)], check=True, capture_output=True, text=True).stdout
    return float(out.strip())


@dataclass
class DemoShot:
    id: str
    baseline_png: Path
    clip: Path
    base: Side
    pipe: Side


def build(shots: list[DemoShot], face_png: Path, out: Path, work: Path, *,
          spend_usd: float | None = None, intro_s: float = 4.0, outro_s: float = 5.0,
          shot_s: float = 5.0, run=subprocess.run, probe=None) -> dict:
    """Render every part into `work`, join them into `out`. Returns what the demo shows."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to build the demo (brew install ffmpeg)")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    parts: list[Path] = []

    def card(img: Image.Image, name: str, seconds: float) -> None:
        png, mp4 = work / f"{name}.png", work / f"{name}.mp4"
        img.save(png)
        run(card_cmd(png, seconds, mp4), check=True)
        parts.append(mp4)

    card(intro_card(face_png, len(shots)), "00_intro", intro_s)
    for i, s in enumerate(shots, 1):
        png, mp4 = work / f"{i:02d}_{s.id}.png", work / f"{i:02d}_{s.id}.mp4"
        shot_card(s.id, i, len(shots), s.base, s.pipe, face_png).save(png)
        seconds = min(shot_s, (probe or clip_seconds)(s.clip))
        run(shot_cmd(png, s.baseline_png, s.clip, seconds, mp4), check=True)
        parts.append(mp4)
    head = headline([s.id for s in shots], {s.id: s.base for s in shots}, {s.id: s.pipe for s in shots})
    card(outro_card(head, spend_usd), "99_outro", outro_s)

    list_file = work / "parts.txt"
    list_file.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    out.parent.mkdir(parents=True, exist_ok=True)
    run(concat_cmd(list_file, out), check=True)
    return {"shots": [s.id for s in shots], "headline": head, "parts": [p.name for p in parts]}
