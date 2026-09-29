"""VisionProvider abstraction: turns a frame into a validated Observation."""

from typing import Protocol, runtime_checkable

from pydantic import AwareDatetime, Field

from drone_agent.models import DomainModel, Observation


class FrameInput(DomainModel):
    """A frame handed to a provider. Real-video providers (M5) will add image data here."""

    frame_id: int = Field(ge=0)
    timestamp: AwareDatetime


@runtime_checkable
class VisionProvider(Protocol):
    name: str

    def analyze(self, frame: FrameInput) -> Observation:
        """Return a validated Observation (without telemetry; that is attached separately)."""
        ...
