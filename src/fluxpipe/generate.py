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

import hashlib
import json
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
    model: str, prompt: dict | str, *, width: int, height: int, seed: int, ref_files: list[Path]
) -> Request:
    """One FLUX.2 request: the prompt (JSON, or plain text for the baseline), size, seed,
    references as input_image, input_image_2..."""
    if model not in IMAGE_PRICES_USD:
        raise ValueError(f"Unknown model '{model}'. Known: {', '.join(IMAGE_PRICES_USD)}")
    check_size(width, height)
    max_refs = IMAGE_PRICES_USD[model].max_refs
    if len(ref_files) > max_refs:
        raise ValueError(f"{model} takes at most {max_refs} reference images, got {len(ref_files)}")
    payload = {
        "prompt": prompt if isinstance(prompt, str) else as_json(prompt),
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


def fingerprint(endpoint: str, payload: dict, ref_files: list[str | Path]) -> str:
    """What a request asks for, minus the seed: model, prompt, settings, and the reference images'
    actual bytes (so re-locking a reference changes it too). Same fingerprint = same shot request."""
    h = hashlib.sha256()
    settings = {k: v for k, v in payload.items() if k != "seed" and not k.startswith("input_image")}
    h.update(json.dumps([endpoint, settings], sort_keys=True).encode())
    for f in ref_files:
        h.update(hashlib.sha256(Path(f).read_bytes()).digest())
    return h.hexdigest()[:16]


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


EDIT_KEEP = "Keep everything else exactly as it is in image 1."


def edit_request(
    shot: Shot, source: Path, *, seed: int, extra: list[Path] = (), model: str = "flux-2-pro",
    width: int = DEFAULT_SIZE[0], height: int = DEFAULT_SIZE[1],
) -> Request:
    """Edit a finished keyframe: the frame is image 1, the prompt says only what changes.
    `extra` (the shot's edit_refs) are attached after it as image 2, 3..., for the prompt to name.

    Same endpoint as generation; passing input_image is what makes it an edit (docs.bfl.ai, FLUX.2
    image editing). Size is sent explicitly: the docs don't say what an edit returns without it.
    """
    if not shot.edit:
        raise ValueError(f"{shot.id} has no `edit` in shots.yaml")
    prompt = f"{shot.edit.strip()} {EDIT_KEEP}"
    return build_request(model, prompt, width=width, height=height, seed=seed,
                         ref_files=[source, *extra])


BASELINE_MODEL = "flux-2-pro"  # same model as the keyframes, or the comparison isn't fair


def baseline_request(
    bible: Bible,
    shot: Shot,
    *,
    model: str = BASELINE_MODEL,
    width: int = DEFAULT_SIZE[0],
    height: int = DEFAULT_SIZE[1],
) -> Request:
    """M4: the naive "before" shot. Same model, size and seed as the pipeline's keyframe, so the
    only differences are the ones the pipeline adds: references and the structured bible prompt."""
    if bible.baseline is None:
        raise ValueError("No bible/baseline.yaml")
    return build_request(
        model,
        bible.baseline.prompt(shot.id),
        width=width,
        height=height,
        seed=shot_seed(shot.id),
        ref_files=[],
    )
