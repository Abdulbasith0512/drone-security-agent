"""VideoSource: a validated local video file (OpenCV)."""

import hashlib
import math
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
from pydantic import Field

from drone_agent.models import DomainModel


class VideoError(ValueError):
    """The video is missing, unreadable, corrupt or unusable."""


class VideoMetadata(DomainModel):
    source_name: str  # file name only
    sha256: str = Field(min_length=64, max_length=64)  # of the file bytes: stable identity
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)  # as reported by the container (can be inexact)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoSource:
    """A video that was checked at construction: it exists, opens, has a usable FPS and frame
    count, and its first frame really decodes. Raises VideoError otherwise."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise VideoError(f"video file not found: {self.path}")
        if self.path.stat().st_size == 0:
            raise VideoError(f"video file is empty: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        try:
            if not cap.isOpened():
                raise VideoError(f"cannot open video (unsupported or corrupt): {self.path}")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or not 0 < fps <= 1000:
                raise VideoError(f"video reports an unusable FPS ({fps}): {self.path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                raise VideoError(f"video reports no frames ({frame_count}): {self.path}")
            ok, frame = cap.read()
            if not ok or frame is None:
                raise VideoError(f"cannot decode the first frame (corrupt video?): {self.path}")
        finally:
            cap.release()
        height, width = frame.shape[:2]
        self.metadata = VideoMetadata(
            source_name=self.path.name,
            sha256=_sha256_file(self.path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )

    @contextmanager
    def capture(self) -> Iterator[cv2.VideoCapture]:
        """A fresh OpenCV capture, always released afterwards (explicit lifetime)."""
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise VideoError(f"cannot re-open video: {self.path}")
        try:
            yield cap
        finally:
            cap.release()
