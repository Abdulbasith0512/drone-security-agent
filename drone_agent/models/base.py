"""Shared base class and enums for domain models."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Immutable, strict base: unknown fields are rejected, instances are frozen."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Zone(StrEnum):
    """Property zones. A frame's zone is the drone's patrol zone when it was captured."""

    MAIN_GATE = "main_gate"
    DRIVEWAY = "driveway"
    PARKING_AREA = "parking_area"
    WAREHOUSE = "warehouse"
    RESTRICTED_ZONE = "restricted_zone"


class ObjectType(StrEnum):
    PERSON = "person"
    VEHICLE = "vehicle"
    OTHER = "other"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
