"""Event: deterministic temporal association of detections (NOT computer-vision tracking)."""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, ObjectType, Zone


class CloseReason(StrEnum):
    GAP = "gap"  # no matching detection within max_gap_seconds
    MISSED_FRAMES = "missed_frames"  # too many consecutive frames in the zone without a match
    ZONE_CHANGE = "zone_change"  # the drone observed a different zone
    SCENARIO_END = "scenario_end"  # input ended while the event was still open


class Event(DomainModel):
    event_id: str = Field(min_length=1)
    object_type: ObjectType
    zone: Zone
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    # Most common raw value per attribute key across the event's detections (ties: first seen).
    attributes: dict[str, str] = Field(default_factory=dict)
    # Distinct activities in order of first appearance.
    activities: list[str] = Field(default_factory=list)
    # Description of the event's first observation.
    description: str = ""
    observation_ids: list[str] = Field(min_length=1)
    frame_ids: list[int] = Field(min_length=1)
    close_reason: CloseReason
    # Set by the entity linker (vehicles only); never links people.
    entity_id: str | None = None

    @property
    def duration_seconds(self) -> float:
        """last_seen - first_seen. A single sighting has duration 0."""
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def evidence_count(self) -> int:
        return len(self.observation_ids)

    @model_validator(mode="after")
    def _consistent(self) -> "Event":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        if len(self.frame_ids) != len(self.observation_ids):
            raise ValueError("frame_ids and observation_ids must have the same length")
        return self
