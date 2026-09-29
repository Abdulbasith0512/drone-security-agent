"""Deterministic time-based frame sampling.

Sample k is taken at offset k * every_seconds; its frame index is floor(offset * fps + 0.5), so the
same video and config always select the same frames, including for non-integer FPS. The offset
recorded for a sample is the decoded frame's own time (index / fps). Frames are decoded
sequentially with grab()/retrieve(): only the selected frames are converted to images.
"""

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

import cv2
import numpy as np
from pydantic import Field

from drone_agent.models import DomainModel
from drone_agent.video.source import VideoError, VideoMetadata, VideoSource


class SamplingConfig(DomainModel):
    every_seconds: float = Field(default=2.0, ge=0.01, allow_inf_nan=False)
    max_samples: int | None = Field(default=None, ge=1)


@dataclass
class SampledFrame:
    """One decoded sample. `image` (BGR uint8) exists only until `release()`; it is never stored.
    Callers process a frame and release it before moving on."""

    frame_id: int  # equals frame_index: unique within one video, meaningful as evidence
    frame_index: int
    offset_seconds: float
    source: str
    width: int
    height: int
    frame_key: str  # "<video sha256[:16]>:<frame_index>"
    content_hash: str  # of a 32x32 greyscale copy (audit only, decoder-dependent)
    image: np.ndarray | None = field(default=None, repr=False)

    def release(self) -> None:
        self.image = None


@dataclass
class SamplingReport:
    original_frames: int
    duration_seconds: float
    fps: float
    planned: int = 0  # sample candidates selected by the plan
    decoded: int = 0
    failed: int = 0  # planned frames whose decode failed
    beyond_end: int = 0  # planned frames never reached because the stream ended early


def plan_samples(meta: VideoMetadata, config: SamplingConfig) -> list[tuple[int, float]]:
    """(frame_index, offset_seconds) for every sample, in order. Pure and deterministic."""
    plan: list[tuple[int, float]] = []
    last = -1
    k = 0
    limit = math.ceil(meta.duration_seconds / config.every_seconds) + 1
    while k <= limit:
        index = math.floor(k * config.every_seconds * meta.fps + 0.5)
        if index >= meta.frame_count:
            break
        if index != last:  # very small intervals must not repeat a frame
            plan.append((index, index / meta.fps))
            last = index
        if config.max_samples is not None and len(plan) >= config.max_samples:
            break
        k += 1
    return plan


def content_hash(image: np.ndarray) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(small.tobytes()).hexdigest()[:16]


def sample_frames(
    source: VideoSource, config: SamplingConfig, report: SamplingReport | None = None
) -> Iterator[SampledFrame]:
    """Yield the planned frames in order; fill `report` as decoding proceeds."""
    meta = source.metadata
    plan = plan_samples(meta, config)
    if report is None:
        report = SamplingReport(meta.frame_count, meta.duration_seconds, meta.fps)
    report.planned = len(plan)
    wanted = dict(plan)
    if not plan:
        return
    last_index = plan[-1][0]
    prefix = meta.sha256[:16]
    with source.capture() as cap:
        position = 0
        while position <= last_index:
            if not cap.grab():
                report.beyond_end += sum(1 for i, _ in plan if i >= position)
                return
            if position in wanted:
                ok, image = cap.retrieve()
                if not ok or image is None:
                    report.failed += 1
                else:
                    report.decoded += 1
                    yield SampledFrame(
                        frame_id=position,
                        frame_index=position,
                        offset_seconds=wanted[position],
                        source=meta.source_name,
                        width=int(image.shape[1]),
                        height=int(image.shape[0]),
                        frame_key=f"{prefix}:{position}",
                        content_hash=content_hash(image),
                        image=image,
                    )
            position += 1


def read_frame_at(source: VideoSource, seconds: float) -> SampledFrame:
    """Decode exactly one frame: the one nearest `seconds` (same rounding as the sampler)."""
    meta = source.metadata
    if not math.isfinite(seconds) or seconds < 0:
        raise VideoError("time must be a non-negative number of seconds")
    index = math.floor(seconds * meta.fps + 0.5)
    if index >= meta.frame_count:
        raise VideoError(
            f"time {seconds:g}s is beyond the end of the video ({meta.duration_seconds:.2f}s)"
        )
    with source.capture() as cap:
        for _ in range(index):
            if not cap.grab():
                raise VideoError(f"video ended before frame {index}")
        ok, image = cap.read()
    if not ok or image is None:
        raise VideoError(f"could not decode frame {index}")
    return SampledFrame(
        frame_id=index,
        frame_index=index,
        offset_seconds=index / meta.fps,
        source=meta.source_name,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        frame_key=f"{meta.sha256[:16]}:{index}",
        content_hash=content_hash(image),
        image=image,
    )
