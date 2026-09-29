"""Telemetry sample and its association with a frame/observation."""

from pydantic import AwareDatetime, Field

from drone_agent.models.base import DomainModel, Zone


class Telemetry(DomainModel):
    """One drone telemetry sample. Coordinates in this project are synthetic."""

    timestamp: AwareDatetime
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)
    altitude_m: float = Field(ge=0, le=500, allow_inf_nan=False)
    zone: Zone
    battery_percent: float = Field(ge=0, le=100, allow_inf_nan=False)


class TelemetryAssociation(DomainModel):
    """The nearest-prior telemetry sample for a frame, with its age and staleness."""

    telemetry: Telemetry
    age_seconds: float = Field(ge=0)
    stale: bool
