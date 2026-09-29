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
    """Alert severity. Set per rule in configuration; never chosen by an LLM."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return ("low", "medium", "high", "critical").index(self.value)
