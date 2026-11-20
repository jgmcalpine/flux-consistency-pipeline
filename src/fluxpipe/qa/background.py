"""Background QA: does the set stay put, or does it shake and shimmer?

The generative-media lesson: a video model draws every frame, and nothing forces frame 37's
trees to sit exactly where frame 36's were. When they don't, the whole background trembles a
pixel or two: "jitter", the tell-tale of generated video and a thing a film crew can't use. A real
camera can still move on purpose (a slow push-in), so we have to tell a smooth move from a tremble.

How it works:
  1. Track the background. ORB finds ~2000 corner-like points in each frame; matching them
     between consecutive frames and fitting one camera motion (shift + zoom + rotation) with
     RANSAC gives the motion most points agree on. Points on the moving man disagree and are
     thrown out as outliers, so what's left is the set.
  2. Follow the frame centre through those motions: a path, in pixels of the 1280x704 frame.
  3. Smooth the path over 5 frames (~0.2 s). A deliberate camera move survives smoothing; a
     tremble from one frame to the next doesn't. JITTER = the RMS distance between the path and
     its smoothed self. CAMERA MOVE = how far the centre travelled, plus the change of zoom.

Calibration (our own clips, 20 Nov): a frozen keyframe encoded like a clip measures 0.00 px; the
same still shaken +-0.5 px measures 0.74 px, shaken +-2 px measures 2.6 px; a smooth 13 px pan
measures 0.31 px (whole-pixel steps). So: under 0.5 px is steady, 0.5-1 px slight, over 1 px jittery.

Caveat: in a close-up with a blurred background there are few background points, and a move that
changes speed (s08's push-in) leaves a little residue. A slight score there means "look at it",
like an unjudged face.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TRACK_SCALE = 0.5  # track at 640x352: 4x faster, and results are reported in full-size pixels
SMOOTH_FRAMES = 5  # ~0.2 s at 24 fps: shorter than any deliberate camera move
STEADY_PX = 0.5
JITTERY_PX = 1.0


@dataclass
class Background:
    frames: int
    jitter_px: float  # RMS frame-to-frame tremble of the background, in 1280x704 pixels
    moved_px: float  # how far the frame centre travelled over the clip
    zoom: float  # total change of scale, e.g. 1.17 = pushed in 17%
    tracked: float  # share of frame pairs where the background could be tracked

    @property
    def verdict(self) -> str:
        if self.tracked < 0.8:
            return "not tracked"
        if self.jitter_px < STEADY_PX:
            return "steady"
        return "slight jitter" if self.jitter_px <= JITTERY_PX else "jittery"

    @property
    def camera(self) -> str:
        """What the camera did, in words: 'still', or the move and zoom."""
        zoom = self.zoom - 1
        if self.moved_px < 8 and abs(zoom) < 0.02:
            return "still"
        parts = [f"drifts {self.moved_px:.0f} px"] if self.moved_px >= 8 else []
        if abs(zoom) >= 0.02:
            parts.append(f"{'pushes in' if zoom > 0 else 'pulls out'} {abs(zoom):.0%}")
        return ", ".join(parts)

    def as_dict(self) -> dict:
        return asdict(self) | {"verdict": self.verdict, "camera": self.camera}


def read_frames(clip: Path, seconds: float | None = None) -> list[np.ndarray]:
    """Greyscale frames at TRACK_SCALE, every frame (jitter lives between neighbouring frames)."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    limit = int(seconds * fps) if seconds else None
    out = []
    while limit is None or len(out) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(grey, None, fx=TRACK_SCALE, fy=TRACK_SCALE, interpolation=cv2.INTER_AREA))
    cap.release()
    return out


def camera_motion(a: np.ndarray, b: np.ndarray, orb=None, matcher=None) -> np.ndarray | None:
    """The 2x3 shift+zoom+rotation most background points agree on, from frame a to b."""
    import cv2

    orb = orb or cv2.ORB_create(2000)
    matcher = matcher or cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ka, da = orb.detectAndCompute(a, None)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None:
        return None
    matches = matcher.match(da, db)
    if len(matches) < 12:
        return None
    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=1.5)
    return M


def path_stats(steps: list[np.ndarray], zooms: list[float]) -> tuple[float, float, float]:
    """(jitter, moved, zoom) from the centre's per-frame steps, in full-size pixels."""
    path = np.vstack([[0.0, 0.0], np.cumsum(np.array(steps).reshape(-1, 2), axis=0)]) / TRACK_SCALE
    k = SMOOTH_FRAMES
    padded = np.pad(path, ((k // 2, k // 2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(padded[:, i], np.ones(k) / k, mode="valid") for i in range(2)], axis=1)
    jitter = float(np.sqrt((np.linalg.norm(path - smooth, axis=1) ** 2).mean()))
    return round(jitter, 2), round(float(np.linalg.norm(path[-1])), 1), round(float(np.prod(zooms)), 3)


def measure(clip: Path, seconds: float | None = None) -> Background:
    import cv2

    frames = read_frames(clip, seconds)
    if len(frames) < SMOOTH_FRAMES + 1:
        raise ValueError(f"{clip}: too few frames to measure")
    h, w = frames[0].shape
    centre = np.array([w / 2, h / 2, 1.0])
    orb, matcher = cv2.ORB_create(2000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    steps, zooms, tracked = [], [], 0
    for a, b in zip(frames, frames[1:]):
        M = camera_motion(a, b, orb, matcher)
        if M is None:
            steps.append(np.zeros(2))  # untracked pair: assume no motion, and count it
            zooms.append(1.0)
            continue
        tracked += 1
        steps.append(M @ centre - centre[:2])  # where the centre went: a zoom about it isn't a shift
        zooms.append(float(np.hypot(M[0, 0], M[1, 0])))
    jitter, moved, zoom = path_stats(steps, zooms)
    return Background(len(frames), jitter, moved, zoom, round(tracked / (len(frames) - 1), 2))
