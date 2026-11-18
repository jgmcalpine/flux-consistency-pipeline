"""Settings and the price table.

Prices are ESTIMATES used by the ledger to enforce the spend cap before a call is made.
The API also reports the real cost of each job, which the ledger stores next to the
estimate. If the two disagree, update this table. Source: bfl.ai/pricing (FLUX.2 section
and FAQ), checked 17 Nov 2026.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

USD_PER_CREDIT = 0.01  # docs.bfl.ai/quick_start/pricing: "1 credit = $0.01"


@dataclass(frozen=True)
class ImagePrice:
    """FLUX.2 bills per megapixel: a price for the first MP, less for each extra one,
    plus a per-MP charge for every reference image you attach."""

    first_mp: float
    extra_mp: float
    ref_mp: float
    max_refs: int


IMAGE_PRICES_USD: dict[str, ImagePrice] = {
    "flux-2-klein-4b": ImagePrice(0.014, 0.001, 0.001, max_refs=4),
    "flux-2-klein-9b": ImagePrice(0.015, 0.002, 0.002, max_refs=4),
    "flux-2-pro": ImagePrice(0.03, 0.015, 0.015, max_refs=8),
    "flux-2-flex": ImagePrice(0.05, 0.05, 0.05, max_refs=8),
    "flux-2-max": ImagePrice(0.07, 0.03, 0.03, max_refs=8),
}

# USD per second of output video. Unverified; check at M6.
VIDEO_PRICES_USD_PER_S: dict[str, float] = {
    "flux-3-video-draft": 0.06,
    "flux-3-video-hd": 0.17,
}

MAX_MP = 4  # FLUX.2's maximum output (and single-reference) resolution


@dataclass(frozen=True)
class Settings:
    api_key: str
    spend_cap_usd: float
    dry_run: bool
    bible_dir: Path
    assets_dir: Path
    ledger_path: Path


def load_settings(root: Path = ROOT) -> Settings:
    load_dotenv(root / ".env")
    return Settings(
        api_key=os.getenv("BFL_API_KEY", ""),
        spend_cap_usd=float(os.getenv("SPEND_CAP_USD", "25")),
        dry_run=os.getenv("DRY_RUN", "true").strip().lower() in {"1", "true", "yes"},
        bible_dir=root / "bible",
        assets_dir=root / "assets",
        ledger_path=root / "assets" / "ledger.sqlite3",
    )


def billed_mp(width: int, height: int) -> int:
    """Megapixels as BFL bills them: rounded UP to a whole MP, capped at 4.

    Deliberately pessimistic: 1 MP is taken as 1,000,000 pixels, so 1024x1024 (1.05 MP)
    is estimated as 2 MP. The real cost from the first API call settles which it is.
    """
    return min(MAX_MP, max(1, math.ceil(width * height / 1_000_000)))


def estimate_cost(
    model: str,
    *,
    width: int = 0,
    height: int = 0,
    refs: list[tuple[int, int]] | None = None,
    seconds: float | None = None,
) -> float:
    """Estimated USD for one call. Raises KeyError for an unknown model, on purpose.

    Images: first MP + extra MPs of the output, plus each reference image. One reference
    is billed at its own size (up to 4 MP); with several, each is billed as 1 MP.
    """
    if model in VIDEO_PRICES_USD_PER_S:
        if seconds is None:
            raise ValueError(f"{model} is a video model: pass seconds=")
        return round(VIDEO_PRICES_USD_PER_S[model] * seconds, 4)

    price = IMAGE_PRICES_USD[model]
    if not (width and height):
        raise ValueError(f"{model} is billed by megapixel: pass width= and height=")
    refs = refs or []
    if len(refs) > price.max_refs:
        raise ValueError(f"{model} takes at most {price.max_refs} reference images")
    out_mp = billed_mp(width, height)
    ref_mp = billed_mp(*refs[0]) if len(refs) == 1 else len(refs)
    cost = price.first_mp + (out_mp - 1) * price.extra_mp + ref_mp * price.ref_mp
    return round(cost, 4)
