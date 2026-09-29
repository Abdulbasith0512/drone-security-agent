"""Entity memory: coarse grouping of separate vehicle events. Not identity recognition."""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, ObjectType


class MatchLevel(StrEnum):
    COARSE = "coarse_match"  # similar vehicle: matched on normalised coarse attributes only
    CONFIRMED = "confirmed"  # explicit, equal plate evidence supplied in the input


class Entity(DomainModel):
    """A vehicle seen in one or more separate events.

    Unless `match_level` is CONFIRMED this only claims a *similar* vehicle (same coarse
    type/colour), never that it is the same physical vehicle.
    """

    entity_id: str = Field(min_length=1)
    entity_type: ObjectType
    attributes: dict[str, str]  # normalised coarse attributes, e.g. {"color": "blue", ...}
    match_level: MatchLevel
    first_seen: AwareDatetime
    last_seen: AwareDatetime
    event_ids: list[str] = Field(min_length=1)

    @property
    def appearance_count(self) -> int:
        return len(self.event_ids)

    @property
    def label(self) -> str:
        """Human wording that does not overclaim identity."""
        parts = [self.attributes.get("color"), self.attributes.get("body_type")]
        text = " ".join(p for p in parts if p) or self.entity_type.value
        if self.match_level is MatchLevel.CONFIRMED:
            return f"confirmed {text} (plate {self.attributes.get('plate', '?')})"
        return f"similar {text}"

    @model_validator(mode="after")
    def _vehicles_only(self) -> "Entity":
        if self.entity_type is not ObjectType.VEHICLE:
            raise ValueError("entity memory is vehicle-only; people are never identity-linked")
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        return self
