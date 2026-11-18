"""Small helpers for image files, without pulling in an imaging library."""

from __future__ import annotations

import base64
import struct
from pathlib import Path

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def png_size(path: Path) -> tuple[int, int]:
    """(width, height) from a PNG header. Needed to estimate what a reference image costs."""
    head = path.read_bytes()[:24]
    if not head.startswith(PNG_MAGIC):
        raise ValueError(f"{path} is not a PNG")
    return struct.unpack(">II", head[16:24])


def as_base64(path: Path) -> str:
    """How a local reference image is sent in input_image fields.

    The docs only show public URLs here; raw base64 is what other BFL endpoints accept.
    If FLUX.2 rejects it, the fallback is the image's public GitHub URL (locked refs are committed).
    """
    return base64.b64encode(path.read_bytes()).decode("ascii")
