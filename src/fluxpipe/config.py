"""Settings and the price table.

Prices are ESTIMATES used by the ledger to enforce the spend cap before a call is made.
After your first real calls (M2), compare the ledger against the BFL dashboard and
update these numbers. Sources as of Nov 2026: docs.bfl.ai/quick_start/pricing and
help.bfl.ai ("What is FLUX 3?").
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# USD per image (at ~1 megapixel). FLUX.2 bills by megapixel, so bigger images cost more.
IMAGE_PRICES_USD: dict[str, float] = {
    "flux-2-klein-4b": 0.014,
    "flux-2-pro": 0.03,
    "flux-2-pro-edit": 0.045,  # with reference images
    "flux-2-max": 0.07,
}

# USD per second of output video.
VIDEO_PRICES_USD_PER_S: dict[str, float] = {
    "flux-3-video-draft": 0.06,
    "flux-3-video-hd": 0.17,
}


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


def estimate_cost(model: str, seconds: float | None = None, images: int = 1) -> float:
    """Estimated USD for one call. Raises KeyError for an unknown model, on purpose."""
    if model in VIDEO_PRICES_USD_PER_S:
        if seconds is None:
            raise ValueError(f"{model} is a video model: pass seconds=")
        return round(VIDEO_PRICES_USD_PER_S[model] * seconds, 4)
    return round(IMAGE_PRICES_USD[model] * images, 4)
