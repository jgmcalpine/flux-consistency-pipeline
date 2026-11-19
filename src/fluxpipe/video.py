"""M6: animate the picked keyframes with FLUX 3 video, image-to-video.

The generative-media lesson: in image-to-video the start frame already decides the look (the
face, costume, props and set we spent M3-M5 getting right). The prompt's only job is MOTION:
what moves, how, and what the camera does. Re-describing the man or the hut invites the model
to redraw them, which is exactly the drift we want to avoid. So each shot's `motion:` in the
bible says only what happens next, starting from what its keyframe shows.

Money: drafts first. A draft is a fast hd preview at $0.06/s (vs $0.15/s for a full hd render).
Make a draft or two per shot, judge them, and only then pay to "enhance" the keeper: that renders
the same shot (same seed, same motion) at full quality from the draft's `draft_cache` bundle.
There is no seed field in FLUX 3 video, so the draft_cache is the only way back to a take.

API facts (docs.bfl.ml, 19 Nov 2026): POST /v1/flux-3-video, `mode: "i2v"`, `keyframes` (one
image = the opening frame; base64 with a data: prefix, as the docs' examples), `duration` 5-20 s,
24 fps, `resolution` hd = 1280x704 at 16:9, `generate_audio` defaults to TRUE, `draft: true`.
Enhance: `mode: "draft_enhance"` with only `draft_cache` (+ `resolution`, default fhd).
"""

from __future__ import annotations

import base64
import math
import shutil
import subprocess
from pathlib import Path

from .bible import Shot
from .config import estimate_cost
from .generate import Request

ENDPOINT = "flux-3-video"
HD_SIZE = (1280, 704)  # what hd renders at 16:9; our keyframes are 1280x720
DEFAULT_RESOLUTION = "hd"  # 1280x704, generated natively; the higher ones are upsampled from it


def fit_start_frame(keyframe: Path, out: Path, size: tuple[int, int] = HD_SIZE) -> Path:
    """Crop the keyframe to the video's frame size (1280x720 -> 1280x704: 8 px off top and bottom).

    How FLUX 3 fits a start frame of a different shape isn't documented. Cropping it ourselves
    means the model never has to guess, and we know exactly which pixels it starts from.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is needed to prepare start frames (brew install ffmpeg)")
    w, h = size
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(keyframe),
         "-vf", f"scale={w}:-2,crop={w}:{h}", str(out)],
        check=True,
    )
    return out


def _data_uri(png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")


def seconds(shot: Shot) -> int:
    """Whole seconds, 5-20 (what i2v accepts). Partial seconds are billed as whole anyway."""
    return max(5, min(20, math.ceil(shot.duration_s)))


def draft_request(shot: Shot, start_frame: Path, *, draft: bool = True,
                  resolution: str = DEFAULT_RESOLUTION) -> Request:
    """Image-to-video from the shot's keyframe. draft=True is the cheap hd preview."""
    if not shot.motion:
        raise ValueError(f"{shot.id} has no `motion` in shots.yaml")
    if draft and resolution != "hd":
        raise ValueError("drafts are hd only")
    payload = {
        "mode": "i2v",
        "prompt": shot.motion.strip(),
        "keyframes": [_data_uri(start_frame)],
        "duration": seconds(shot),
        "resolution": resolution,
        "aspect_ratio": "16:9",
        "generate_audio": False,  # on by default; the cut gets its sound in M7
        "draft": draft,
    }
    price = "flux-3-video-draft" if draft else f"flux-3-video-{resolution}"
    est = estimate_cost(price, seconds=seconds(shot))
    return Request(ENDPOINT, payload, est, [str(start_frame)])


def enhance_request(shot: Shot, draft_cache: Path, *, resolution: str = DEFAULT_RESOLUTION) -> Request:
    """Render a chosen draft at full quality: same shot, same seed, from its draft_cache bundle.

    The API's default here is fhd ($0.24/s, upsampled); we default to hd ($0.15/s), which is the
    resolution FLUX 3 actually generates at. Ask for fhd only for the final cut, if at all.
    """
    payload = {
        "mode": "draft_enhance",
        "draft_cache": base64.b64encode(draft_cache.read_bytes()).decode("ascii"),
        "resolution": resolution,
    }
    est = estimate_cost(f"flux-3-video-{resolution}", seconds=seconds(shot))
    return Request(ENDPOINT, payload, est, [str(draft_cache)])


def find_draft_cache(poll: dict) -> str | None:
    """The draft_cache download URL from a finished draft's poll response.

    The docs say a draft's result includes it but not where, so look anywhere in the response.
    """
    stack = [poll]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "draft_cache" and isinstance(v, str):
                    return v
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return None
