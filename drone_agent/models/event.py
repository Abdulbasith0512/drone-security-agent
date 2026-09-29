"""Minimal Event and Alert schemas. Schema only: aggregation (M3) and rules (M4) come later."""

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, ObjectType, Severity, Zone


class Event(DomainModel):
    event_id: str = Field(min_length=1)
    object_type: ObjectType
    zone: Zone
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    observation_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Event":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        return self


class Alert(DomainModel):
    alert_id: str = Field(min_length=1)
    rule_id: str = Field(min_length=1)
    severity: Severity
    timestamp: AwareDatetime
    message: str = Field(min_length=1)
    event_ids: list[str] = Field(default_factory=list)
    frame_ids: list[int] = Field(default_factory=list)
