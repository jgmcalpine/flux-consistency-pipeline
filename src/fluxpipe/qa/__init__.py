"""Automated QA (M5-M6).

Planned modules:
  face.py     - face embeddings (InsightFace) + cosine similarity against the locked references
  palette.py  - colour histogram distance against a location's reference plate
  costume.py  - checklist questions to a vision model ("is she wearing the mustard coat?")
  video.py    - per-frame identity tracking and background jitter (optical flow)

Install the extra dependencies when you get here:  pip install -e ".[qa]"
"""
