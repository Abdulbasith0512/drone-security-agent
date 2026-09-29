"""Observation: what was seen in one frame."""

from pydantic import AliasChoices, AwareDatetime, Field

from drone_agent.models.base import DomainModel, ObjectType, Zone
from drone_agent.models.telemetry import TelemetryAssociation


class DetectedObject(DomainModel):
    """One object detected in a frame. Scenario files may write `type` for `object_type`."""

    object_id: str | None = None
    object_type: ObjectType = Field(validation_alias=AliasChoices("object_type", "type"))
    attributes: dict[str, str] = Field(default_factory=dict)
    activity: str | None = None
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)


class Observation(DomainModel):
    observation_id: str = Field(min_length=1)
    frame_id: int = Field(ge=0)
    timestamp: AwareDatetime
    source: str = Field(min_length=1)
    zone: Zone
    description: str = Field(min_length=1)
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    telemetry: TelemetryAssociation | None = None
