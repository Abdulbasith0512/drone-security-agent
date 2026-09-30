"""Frame-change gate: skip sampled frames that look almost identical to the last accepted one.

Metric: mean absolute pixel difference (0-255) between small greyscale copies of the two frames.
It is a cheap pixel-similarity check to save vision-model calls. It is NOT motion detection, NOT
object tracking, and it knows nothing about what is in the frame.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from pydantic import Field

from drone_agent.models import DomainModel


class GateConfig(DomainModel):
    enabled: bool = True
    resize_width: int = Field(default=64, ge=8, le=1024)
    resize_height: int = Field(default=36, ge=8, le=1024)
    # A frame is accepted when its score is >= threshold (0 accepts everything).
    threshold: float = Field(default=6.0, ge=0, le=255, allow_inf_nan=False)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    score: float | None  # None for the first frame and when the gate is disabled
    reason: str  # "first" | "changed" | "similar" | "gate_disabled"


class FrameGate:
    def __init__(self, config: GateConfig | None = None) -> None:
        self.config = config or GateConfig()
        self._reference: np.ndarray | None = None

    def _small(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = (self.config.resize_width, self.config.resize_height)
        return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.int16)

    def reset(self) -> None:
        """Forget the reference, so the next frame is accepted as 'first' (e.g. a new scene)."""
        self._reference = None

    def score(self, image: np.ndarray) -> float | None:
        """Difference from the reference without changing it (None: no reference yet).

        Lets another decision-maker (the adaptive sampler) use the pixel signal and decide for
        itself; it then calls `set_reference` for the frames it actually sends to the model.
        """
        if self._reference is None:
            return None
        return round(float(np.abs(self._small(image) - self._reference).mean()), 4)

    def set_reference(self, image: np.ndarray) -> None:
        self._reference = self._small(image)

    def check(self, image: np.ndarray) -> GateDecision:
        """Decide for one frame; an accepted frame becomes the new reference."""
        if not self.config.enabled:
            return GateDecision(True, None, "gate_disabled")
        small = self._small(image)
        if self._reference is None:
            self._reference = small
            return GateDecision(True, None, "first")
        score = round(float(np.abs(small - self._reference).mean()), 4)
        if score >= self.config.threshold:
            self._reference = small
            return GateDecision(True, score, "changed")
        return GateDecision(False, score, "similar")
