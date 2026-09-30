"""Event: deterministic temporal association of detections (NOT computer-vision tracking)."""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, ObjectType, Zone


class CloseReason(StrEnum):
    """Why an event stopped being extended: the end of OBSERVATION COVERAGE, not of the object.

    Whether the object actually left is a separate question, answered only by
    `Event.departure_observed_at`: the drone kept looking at the same zone and the object was not
    there. A gap, a zone change and the end of the input say nothing about it.
    """

    GAP = "gap"  # no observation of the zone for longer than max_gap_seconds
    MISSED_FRAMES = "missed_frames"  # too many consecutive frames in the zone without a match
    ZONE_CHANGE = "zone_change"  # the drone observed a different zone
    SCENARIO_END = "scenario_end"  # input ended while the event was still open
    OPEN = "open"  # streaming snapshot: the event is still being extended


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
    # Mean detection confidence over the detections merged into this event. Providers that report
    # none contribute the documented placeholder, so this is not a calibrated probability.
    confidence: float | None = Field(default=None, ge=0, le=1)
    # When the drone first saw the same zone WITHOUT the object, in the run of empty frames that
    # closed the event (only `missed_frames` closures set it). This is the only evidence of
    # departure; None means departure was not observed (coverage ended or the event is open).
    departure_observed_at: AwareDatetime | None = None

    @property
    def departure_confirmed(self) -> bool:
        return self.departure_observed_at is not None

    @property
    def duration_seconds(self) -> float:
        """last_seen - first_seen. A single sighting has duration 0."""
        return (self.last_seen - self.first_seen).total_seconds()

    @property
    def evidence_count(self) -> int:
        return len(self.observation_ids)

    @property
    def is_open(self) -> bool:
        return self.close_reason is CloseReason.OPEN

    @model_validator(mode="after")
    def _consistent(self) -> "Event":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        if self.departure_observed_at is not None and self.departure_observed_at < self.last_seen:
            raise ValueError("departure_observed_at must not be before last_seen")
        if len(self.frame_ids) != len(self.observation_ids):
            raise ValueError("frame_ids and observation_ids must have the same length")
        return self
