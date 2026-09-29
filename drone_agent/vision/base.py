"""VisionProvider: the ONE boundary every observation source goes through.

Everything downstream (store, events, entities, rules) consumes `Observation` and never asks
whether it came from a YAML script or from real video.
"""

from typing import Any, Protocol, runtime_checkable

from pydantic import AwareDatetime, Field

from drone_agent.models import DomainModel, Observation, Zone


class FrameInput(DomainModel):
    """A frame handed to a provider.

    Scripted frames only need `frame_id` and `timestamp`. Video frames also carry the zone from the
    SIMULATED patrol timeline, the identity of the sampled frame, and the decoded pixels.
    `image` is only valid while the provider call runs (it is not stored anywhere) and is excluded
    from serialisation; do not compare FrameInputs with `==`.
    """

    frame_id: int = Field(ge=0)
    timestamp: AwareDatetime
    zone: Zone | None = None
    source_name: str | None = None  # e.g. the video file name
    offset_seconds: float | None = Field(default=None, ge=0)  # time since video start
    frame_key: str | None = None  # stable identity: <video sha256 prefix>:<frame index>
    content_hash: str | None = None  # hash of a small greyscale copy of the pixels (audit only)
    image: Any = Field(default=None, exclude=True, repr=False)  # BGR uint8 ndarray


@runtime_checkable
class VisionProvider(Protocol):
    name: str

    def analyze(self, frame: FrameInput) -> Observation:
        """Return a validated Observation (without telemetry; that is attached separately)."""
        ...
