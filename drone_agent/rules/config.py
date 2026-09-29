"""Security rule configuration (config/security_rules.yaml); invalid files fail clearly."""

import re
from datetime import time, timedelta, timezone
from pathlib import Path

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from drone_agent.models import DomainModel, ObjectType, Severity, Zone
from drone_agent.pipeline.config import ConfigError

_OFFSET = re.compile(r"^([+-])(\d{2}):(\d{2})$")


class ZoneThreshold(DomainModel):
    threshold_seconds: float = Field(gt=0)


class LoiteringRule(DomainModel):
    """A monitored object stays in a monitored zone for at least `threshold_seconds`."""

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    object_types: list[ObjectType] = Field(default_factory=lambda: [ObjectType.PERSON])
    zones: dict[Zone, ZoneThreshold] = Field(
        default_factory=lambda: {Zone.MAIN_GATE: ZoneThreshold(threshold_seconds=300)}
    )

    @model_validator(mode="after")
    def _has_targets(self) -> "LoiteringRule":
        if self.enabled and (not self.zones or not self.object_types):
            raise ValueError("loitering is enabled but has no zones or object_types to monitor")
        return self


class AfterHoursRule(DomainModel):
    """A monitored object is present at any time during the daily closed window.

    The window is [closed_from, closed_until): closed_from is inside, closed_until is outside. If
    closed_until <= closed_from the window crosses midnight (default 22:00 -> 06:00). Times are
    property-local wall-clock times: `utc_offset` (e.g. "+05:30") names the local zone; if it is
    null each event's own UTC offset is used. The computer's clock is never consulted.
    """

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    object_types: list[ObjectType] = Field(default_factory=lambda: [ObjectType.PERSON])
    closed_from: time = time(22, 0)
    closed_until: time = time(6, 0)
    utc_offset: str | None = None
    zones: list[Zone] = Field(default_factory=list)  # empty = every zone

    @field_validator("closed_from", "closed_until", mode="before")
    @classmethod
    def _quoted_clock_time(cls, value: object) -> object:
        if isinstance(value, int):
            raise ValueError(
                'write the time as a quoted string such as "22:00" (YAML reads 22:00 as 1320)'
            )
        return value

    @field_validator("utc_offset")
    @classmethod
    def _offset_format(cls, value: str | None) -> str | None:
        if value is not None and not _OFFSET.match(value):
            raise ValueError('utc_offset must look like "+05:30" or "-08:00"')
        return value

    @model_validator(mode="after")
    def _valid_window(self) -> "AfterHoursRule":
        if self.closed_from == self.closed_until:
            raise ValueError("closed_from and closed_until must differ")
        if self.enabled and not self.object_types:
            raise ValueError("after_hours_presence is enabled but has no object_types")
        return self

    @property
    def offset(self) -> timezone | None:
        """The configured local zone as a fixed offset (None: use each event's own offset)."""
        if self.utc_offset is None:
            return None
        sign, hours, minutes = _OFFSET.match(self.utc_offset).groups()  # type: ignore[union-attr]
        delta = timedelta(hours=int(hours), minutes=int(minutes))
        return timezone(-delta if sign == "-" else delta)


class RepeatVehicleRule(DomainModel):
    """A similar vehicle (coarse Entity) has at least `minimum_appearances` separate events."""

    enabled: bool = True
    severity: Severity = Severity.LOW
    minimum_appearances: int = Field(default=3, ge=2)


class RestrictedZoneRule(DomainModel):
    """A monitored object is observed in a restricted zone (one alert per continuous event)."""

    enabled: bool = True
    severity: Severity = Severity.HIGH
    zones: list[Zone] = Field(default_factory=lambda: [Zone.RESTRICTED_ZONE])
    object_types: list[ObjectType] = Field(
        default_factory=lambda: [ObjectType.PERSON, ObjectType.VEHICLE]
    )

    @model_validator(mode="after")
    def _has_targets(self) -> "RestrictedZoneRule":
        if self.enabled and (not self.zones or not self.object_types):
            raise ValueError("restricted_zone_entry is enabled but has no zones or object_types")
        return self


class SecurityRulesConfig(DomainModel):
    loitering: LoiteringRule = Field(default_factory=LoiteringRule)
    after_hours_presence: AfterHoursRule = Field(default_factory=AfterHoursRule)
    repeat_vehicle: RepeatVehicleRule = Field(default_factory=RepeatVehicleRule)
    restricted_zone_entry: RestrictedZoneRule = Field(default_factory=RestrictedZoneRule)


def load_rules_config(path: str | Path | None = None) -> SecurityRulesConfig:
    """Load a rules YAML, or the built-in defaults (identical to the shipped YAML) for None."""
    if path is None:
        return SecurityRulesConfig()
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return SecurityRulesConfig.model_validate(raw or {})
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read rules config {path}: {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(f"invalid rules config {path}:\n{exc}") from exc
