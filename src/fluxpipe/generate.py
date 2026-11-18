"""Turn one shot into one FLUX.2 API request: endpoint, payload and estimated cost.

Decisions that matter for consistency:
  - The seed is fixed per shot (derived from the shot id), so re-running a shot with
    the same prompt reproduces it. Change the seed on purpose, not by accident.
  - Prompt upsampling is OFF. pro and max quietly rewrite your prompt with an LLM by
    default, so the words the model reads would no longer be the words in the bible,
    and two shots could get different rewrites of the same character.
  - The default size is 1280x720: 16:9 and under 1 MP, so it bills as a single MP.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from pathlib import Path

from .bible import Bible, Shot
from .config import IMAGE_PRICES_USD, estimate_cost
from .images import as_base64, png_size
from .prompts import as_json, build_structured, reference_images

DEFAULT_MODEL = "flux-2-klein-4b"  # cheapest: ~$0.014 a still
DEFAULT_SIZE = (1280, 720)
SIZE_STEP = 16  # FLUX works in 16-pixel blocks; not stated for FLUX.2, so we stay safe

# Which upsampling switch each model's schema uses (klein has none).
UPSAMPLING_OFF = {
    "flux-2-pro": {"disable_pup": True},
    "flux-2-max": {"disable_pup": True},
    "flux-2-flex": {"prompt_upsampling": False},
}


@dataclass
class Request:
    endpoint: str
    payload: dict
    est_cost_usd: float
    refs: list[str]  # paths, for the ledger (the payload holds the image data)


def check_size(width: int, height: int) -> None:
    if width % SIZE_STEP or height % SIZE_STEP:
        raise ValueError(f"width and height must be multiples of {SIZE_STEP}")


def build_request(
    model: str, prompt: dict, *, width: int, height: int, seed: int, ref_files: list[Path]
) -> Request:
    """One FLUX.2 request: the JSON prompt, size, seed, references as input_image, input_image_2..."""
    if model not in IMAGE_PRICES_USD:
        raise ValueError(f"Unknown model '{model}'. Known: {', '.join(IMAGE_PRICES_USD)}")
    check_size(width, height)
    max_refs = IMAGE_PRICES_USD[model].max_refs
    if len(ref_files) > max_refs:
        raise ValueError(f"{model} takes at most {max_refs} reference images, got {len(ref_files)}")
    payload = {
        "prompt": as_json(prompt),
        "width": width,
        "height": height,
        "seed": seed,
        "output_format": "png",  # lossless, so QA measures the model and not JPEG artefacts
        **UPSAMPLING_OFF.get(model, {}),
    }
    for i, path in enumerate(ref_files, 1):
        payload["input_image" if i == 1 else f"input_image_{i}"] = as_base64(path)
    est = estimate_cost(model, width=width, height=height, refs=[png_size(p) for p in ref_files])
    return Request(model, payload, est, [str(p) for p in ref_files])


def shot_seed(shot_id: str) -> int:
    """A stable seed per shot: the same id always gives the same number."""
    return zlib.crc32(shot_id.encode())


def keyframe_request(
    bible: Bible,
    shot: Shot,
    *,
    model: str = DEFAULT_MODEL,
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
    seed: int | None = None,
) -> Request:
    return build_request(
        model,
        build_structured(bible, shot),
        width=width,
        height=height,
        seed=shot_seed(shot.id) if seed is None else seed,
        ref_files=[bible.ref_path(r) for r in reference_images(bible, shot)],
    )
