"""The README's figures and cost table, generated from the results and the ledger.

Nothing in the README is typed by hand if it can be computed: the cost table comes from the
ledger (every paid call, its estimate and what BFL billed), the face and background numbers from
`results/*/faces.json` and `results/video/qa.json`, the pictures from the committed stills and
videos. Rerun `fluxpipe report` after any change and the README's evidence can't go stale.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from .demo import BG, DIM, INK, Side, font

STAGES = [  # ledger stage -> what it was, in story order of the method
    ("test", "API checks"),
    ("refs", "Reference candidates (8 locked)"),
    ("baseline", "Plain-prompt baseline stills"),
    ("keyframe", "Keyframe candidates (11 picked)"),
    ("edit", "Edits (tried; none used)"),
    ("video", "FLUX 3 video: drafts and finals"),
]


def cost_rows(calls: list[dict]) -> list[dict]:
    """Ledger rows (status done or failed) -> one row per stage: calls, estimated, billed, exact."""
    out = []
    for stage, label in STAGES:
        rows = [c for c in calls if c["stage"] == stage]
        if not rows:
            continue
        billed = [c["actual_cost_usd"] if c["actual_cost_usd"] is not None else c["est_cost_usd"] for c in rows]
        exact = sum(c["actual_cost_usd"] is not None and abs(c["actual_cost_usd"] - c["est_cost_usd"]) < 1e-4
                    for c in rows)
        out.append({"stage": stage, "label": label, "calls": len(rows),
                    "models": sorted({c["model"] for c in rows}),
                    "estimated": sum(c["est_cost_usd"] for c in rows), "billed": sum(billed), "exact": exact})
    return out  # sums unrounded: rounding happens once, when printed


def cost_markdown(rows: list[dict]) -> str:
    lines = ["| Stage | Model | Calls | Estimated | Billed | Billed exactly as estimated |",
             "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['label']} | {', '.join(r['models'])} | {r['calls']} | ${r['estimated']:.2f} |"
                     f" ${r['billed']:.2f} | {r['exact']} of {r['calls']} |")
    calls = sum(r["calls"] for r in rows)
    lines.append(f"| **Total** | | **{calls}** | **${sum(r['estimated'] for r in rows):.2f}** |"
                 f" **${sum(r['billed'] for r in rows):.2f}** | **{sum(r['exact'] for r in rows)} of {calls}** |")
    return "\n".join(lines) + "\n"


def headline_markdown(base_pairs: list[float], pipe_pairs: list[float], base_judged: int, pipe_judged: int,
                      head: dict | None, qa_totals: dict | None, cost: list[dict]) -> str:
    """The README's first table, from the same files every other number comes from."""
    lines = ["| | Plain prompts | The pipeline |", "|---|---:|---:|"]
    if base_pairs and pipe_pairs:
        best, worst = max(base_pairs), min(pipe_pairs)
        b_words = "no two shots show the same man" if best < 0.5 else "some shots show the same man"
        p_words = "every pair is the same man" if worst >= 0.5 else "not every pair is the same man"
        lines.append(f"| **Same man from shot to shot** (stills, every pair of judgeable shots:"
                     f" {base_judged} plain, {pipe_judged} pipeline) | best pair **{best:.2f}**: {b_words} |"
                     f" worst pair **{worst:.2f}**: {p_words} |")
    if head:
        lines.append(f"| **Face match with the locked face** ({', '.join(head['shots'])}: judgeable on both sides) |"
                     f" **{head['baseline']:.2f}** | **{head['pipeline']:.2f}** |")
    if qa_totals:
        t = qa_totals
        lines.append(f"| **Face in the finished film** (frames sampled at 2/s, every judgeable frame) | |"
                     f" **{t['face_frames_same']} of {t['face_frames']}** frames the same person |")
        lines.append(f"| **Background jitter in the film** (under 0.5 px is steady) | |"
                     f" **{t['steady']} of {t['shots']}** shots steady, median **{t['jitter_median']:.2f} px** |")
    calls = sum(r["calls"] for r in cost)
    lines.append(f"| **Total API spend**, every experiment included | |"
                 f" **${sum(r['billed'] for r in cost):.2f}** for {calls} calls,"
                 f" {sum(r['exact'] for r in cost)} billed exactly as estimated |")
    return "\n".join(lines) + "\n"


def fill_markers(text: str, blocks: dict[str, str]) -> str:
    """Replace what sits between <!-- generated:NAME --> and <!-- /generated:NAME --> with blocks[NAME]."""
    for name, body in blocks.items():
        start, end = f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"README needs exactly one {start} ... {end} pair")
        before, rest = text.split(start)
        _, after = rest.split(end)
        text = f"{before}{start}\n{body.rstrip()}\n{end}{after}"
    return text


# --- figures ----------------------------------------------------------------------------------

def _fit(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to cover w x h, then centre-crop: every tile the same shape."""
    img = img.convert("RGB")
    s = max(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def _contain(img: Image.Image, w: int, h: int) -> Image.Image:
    """Scale to fit inside w x h on the background: nothing cropped (for references)."""
    img = img.convert("RGB")
    s = min(w / img.width, h / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    tile = Image.new("RGB", (w, h), BG)
    tile.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
    return tile


def _caption(tile: Image.Image, text: str, colour=INK, size: int = 22) -> Image.Image:
    d = ImageDraw.Draw(tile, "RGBA")
    f = font(size, "medium")
    w = d.textlength(text, font=f)
    d.rectangle((0, tile.height - size - 18, w + 24, tile.height), fill=(0, 0, 0, 170))
    d.text((12, tile.height - 9), text, font=f, fill=colour, anchor="ls")
    return tile


def _side_words(side: Side) -> tuple[str, tuple]:
    number, words, colour = side.label()
    return (f"{number} {words}" if side.judged else words), colour


def stills_figure(shots: list[str], baseline: dict[str, Path], pipeline: dict[str, Path],
                  base_scores: dict[str, Side], pipe_scores: dict[str, Side], tile=(480, 270)) -> Image.Image:
    """Every shot as a row: plain-prompt still | pipeline still, each labelled with its face score."""
    w, h = tile
    head = 64
    img = Image.new("RGB", (2 * w + 12, head + len(shots) * (h + 12)), BG)
    d = ImageDraw.Draw(img)
    d.text((w // 2, 44), "plain prompts", font=font(30, "bold"), fill=INK, anchor="ms")
    d.text((w + 12 + w // 2, 44), "the pipeline", font=font(30, "bold"), fill=INK, anchor="ms")
    for i, s in enumerate(shots):
        y = head + i * (h + 12)
        for x, path, score in ((0, baseline[s], base_scores.get(s)), (w + 12, pipeline[s], pipe_scores.get(s))):
            t = _fit(Image.open(path), w, h)
            words, colour = _side_words(score) if score else ("", INK)
            img.paste(_caption(t, f"{s}  {words}", colour if score and score.judged else DIM), (x, y))
    return img


def refs_figure(refs: list[tuple[str, Path]], tile=(320, 240)) -> Image.Image:
    """The locked references, labelled: what every shot is anchored to."""
    w, h = tile
    cols = 4
    rows = -(-len(refs) // cols)
    img = Image.new("RGB", (cols * w + (cols - 1) * 10, rows * (h + 10) - 10), BG)
    for i, (label, path) in enumerate(refs):
        t = _contain(Image.open(path), w, h)
        img.paste(_caption(t, label, size=18), ((i % cols) * (w + 10), (i // cols) * (h + 10)))
    return img


def frame_at(video: Path, seconds: float, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{seconds:.3f}", "-i", str(video),
                    "-frames:v", "1", str(out)], check=True)
    return out


def film_strip(frames: list[tuple[str, Path]], tile=(320, 176), cols: int = 4) -> Image.Image:
    w, h = tile
    rows = -(-len(frames) // cols)
    img = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * (h + 6) - 6), BG)
    for i, (label, path) in enumerate(frames):
        t = _fit(Image.open(path), w, h)
        img.paste(_caption(t, label, size=16), ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    return img


def web_video_cmd(src: Path, out: Path, width: int = 960, crf: int = 27) -> list[str]:
    """A smaller copy for pasting into a README on github.com (it plays uploaded videos inline)."""
    return ["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vf", f"scale={width}:-2",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(out)]

