"""Face identity QA: is the person in this image the person in the locked face reference?

How it works (the generative-media lesson):
  A face-recognition model (InsightFace's ArcFace, "buffalo_l") turns a face into an embedding:
  512 numbers that describe *who* it is, trained to ignore pose, light, expression and clothes.
  Two photos of the same person point in nearly the same direction; different people don't.
  So we compare directions with cosine similarity: 1.0 = identical, ~0 = unrelated.

  This is the honest judge for the demo. A human (or a vision LLM) is fooled by matching
  dreadlocks and bandana; ArcFace mostly looks at the bones of the face.

Caveats:
  - Small faces (wide shots) embed badly. We report the face's height in pixels so a low
    score on a 40 px face isn't read as "wrong person".
  - Turned-away faces embed badly too. "Trained to ignore pose" holds for a few tens of degrees:
    in a video draft of s02 the same man fell from 0.80 to 0.39 as his head tipped 65 degrees down
    and 44 to the side. So we also measure head pose (InsightFace's 3D landmarks) and don't judge
    faces turned further than our stills have shown the scorer copes with.
  - SAME_PERSON is a rule of thumb for buffalo_l, not a law. We calibrate it on our own images.
  - buffalo_l's weights are licensed for non-commercial research use. Fine for this repo.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np

SAME_PERSON = 0.5  # cosine; above this, buffalo_l usually means "same person"
SMALL_FACE_PX = 80  # below this face height, treat the score as unreliable
# Head pose limits, in degrees from facing the camera. Our picked stills scored >= 0.6 at up to
# 23 deg down (s07) and 56 deg sideways (s04); beyond a little more than that we have no evidence.
MAX_PITCH_DEG = 30
MAX_YAW_DEG = 60


@dataclass
class FaceHit:
    embedding: np.ndarray  # L2-normalised, 512 numbers
    height_px: int
    det_score: float
    pitch: float | None = None  # degrees; negative = looking down
    yaw: float | None = None  # degrees; turned left/right


@dataclass
class FaceScore:
    image: str
    faces: int  # how many faces were detected
    similarity: float | None  # largest face vs the reference; None if no face
    height_px: int | None
    det_score: float | None
    pitch: float | None = None
    yaw: float | None = None

    @property
    def turned_away(self) -> bool:
        return (self.pitch is not None and abs(self.pitch) > MAX_PITCH_DEG) or (
            self.yaw is not None and abs(self.yaw) > MAX_YAW_DEG)

    @property
    def verdict(self) -> str:
        if self.similarity is None:
            return "no face"
        if self.height_px is not None and self.height_px < SMALL_FACE_PX:
            return "too small"
        if self.turned_away:
            return "turned away"
        return "same" if self.similarity >= SAME_PERSON else "different"

    def as_dict(self) -> dict:
        return asdict(self) | {"verdict": self.verdict}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Something that finds faces in an image file. The real one is InsightFace; tests pass a fake.
Detector = Callable[[Path], list[FaceHit]]


def insightface_detector(det_size: int = 640) -> Detector:
    """Load buffalo_l once (detection + recognition only; it downloads ~280 MB the first time)."""
    import cv2
    from insightface.app import FaceAnalysis

    # landmark_3d_68 gives head pose (pitch, yaw, roll); it doesn't change the embedding.
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "landmark_3d_68", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def detect(path: Path) -> list[FaceHit]:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Can't read image: {path}")
        return [
            FaceHit(f.normed_embedding, int(f.bbox[3] - f.bbox[1]), float(f.det_score),
                    pitch=round(float(f.pose[0]), 1), yaw=round(float(f.pose[1]), 1))
            for f in app.get(img)
        ]

    return detect


def largest(faces: list[FaceHit]) -> FaceHit | None:
    """The main subject is the biggest face in frame (shots have one character)."""
    return max(faces, key=lambda f: f.height_px, default=None)


def reference_face(detect: Detector, reference: Path) -> FaceHit:
    ref = largest(detect(reference))
    if ref is None:
        raise ValueError(f"No face found in the reference {reference}")
    return ref


def score_one(detect: Detector, ref: FaceHit, path: Path) -> tuple[FaceScore, FaceHit | None]:
    """One image's main face vs the reference face. Also returns the face, for pairwise checks."""
    faces = detect(path)
    main = largest(faces)
    score = FaceScore(
        image=path.name,
        faces=len(faces),
        similarity=None if main is None else round(cosine(ref.embedding, main.embedding), 3),
        height_px=None if main is None else main.height_px,
        det_score=None if main is None else round(main.det_score, 3),
        pitch=None if main is None else main.pitch,
        yaw=None if main is None else main.yaw,
    )
    return score, main


def score_images(
    detect: Detector, reference: Path, images: list[Path]
) -> tuple[list[FaceScore], list[float]]:
    """Each image's main face vs the reference, plus every pair of judgeable faces vs each other.

    The pairs answer a different question: not "is it OUR man?" but "is it the same man in
    every shot?". A set can fail the first and still pass the second (consistent, wrong face).
    """
    ref = reference_face(detect, reference)
    scores, judgeable = [], []
    for path in images:
        score, main = score_one(detect, ref, path)
        if score.verdict in ("same", "different"):
            judgeable.append(main.embedding)
        scores.append(score)
    pairs = [round(cosine(a, b), 3) for a, b in combinations(judgeable, 2)]
    return scores, pairs
