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
    # Optional second rung: once the same event lasts this long the SAME alert is escalated to
    # CRITICAL (it is not replaced by a new alert). None = a single-severity rule.
    critical_after_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ladder_ascends(self) -> "ZoneThreshold":
        if (
            self.critical_after_seconds is not None
            and self.critical_after_seconds <= self.threshold_seconds
        ):
            raise ValueError("critical_after_seconds must be greater than threshold_seconds")
        return self


class LoiteringRule(DomainModel):
    """A monitored object stays in a monitored zone for at least `threshold_seconds`.

    The alert opens at `severity` (the WARNING rung) after `threshold_seconds`; if the zone also
    sets `critical_after_seconds` the same alert escalates to CRITICAL after that long.
    """

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    object_types: list[ObjectType] = Field(default_factory=lambda: [ObjectType.PERSON])
    zones: dict[Zone, ZoneThreshold] = Field(
        default_factory=lambda: {
            Zone.MAIN_GATE: ZoneThreshold(threshold_seconds=120, critical_after_seconds=300)
        }
    )
    recommended_action: str | None = None  # None = the built-in wording (rules/common.py)
    critical_action: str | None = None

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
    recommended_action: str | None = None

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
    recommended_action: str | None = None


class RestrictedZoneRule(DomainModel):
    """A monitored object is observed in a restricted zone (one alert per continuous event)."""

    enabled: bool = True
    severity: Severity = Severity.HIGH
    zones: list[Zone] = Field(default_factory=lambda: [Zone.RESTRICTED_ZONE])
    object_types: list[ObjectType] = Field(
        default_factory=lambda: [ObjectType.PERSON, ObjectType.VEHICLE]
    )
    recommended_action: str | None = None

    @model_validator(mode="after")
    def _has_targets(self) -> "RestrictedZoneRule":
        if self.enabled and (not self.zones or not self.object_types):
            raise ValueError("restricted_zone_entry is enabled but has no zones or object_types")
        return self


class LowBatteryRule(DomainModel):
    """Battery at or below `warning_percent` (severity), at or below `critical_percent` (CRITICAL).

    One alert per low-battery episode; it resolves when the battery recovers to `recover_percent`.
    """

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    warning_percent: float = Field(default=20.0, gt=0, le=100)
    critical_percent: float = Field(default=10.0, ge=0, le=100)
    recover_percent: float = Field(default=25.0, gt=0, le=100)
    recommended_action: str | None = None
    critical_action: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "LowBatteryRule":
        if not self.critical_percent < self.warning_percent < self.recover_percent:
            raise ValueError(
                "low_battery needs critical_percent < warning_percent < recover_percent"
            )
        return self


class SignalLossRule(DomainModel):
    """Link quality. Degraded below `degraded_below_percent` (severity); lost below
    `lost_below_percent`, or when no telemetry arrives for more than `gap_seconds` (HIGH)."""

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    lost_severity: Severity = Severity.HIGH
    degraded_below_percent: float = Field(default=40.0, gt=0, le=100)
    lost_below_percent: float = Field(default=15.0, ge=0, le=100)
    gap_seconds: float = Field(default=30.0, gt=0)
    recommended_action: str | None = None
    lost_action: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "SignalLossRule":
        if not self.lost_below_percent < self.degraded_below_percent:
            raise ValueError("signal_loss needs lost_below_percent < degraded_below_percent")
        if self.lost_severity.rank < self.severity.rank:
            raise ValueError("signal_loss lost_severity must not be lower than severity")
        return self


class AltitudeRule(DomainModel):
    """Altitude outside the safe patrol envelope [min_m, max_m] (severity); beyond the severe
    limits it is HIGH. One alert per excursion; it resolves when the drone is back in range."""

    enabled: bool = True
    severity: Severity = Severity.MEDIUM
    min_m: float = Field(default=5.0, ge=0)
    max_m: float = Field(default=120.0, gt=0)
    severe_min_m: float = Field(default=2.0, ge=0)
    severe_max_m: float = Field(default=150.0, gt=0)
    recommended_action: str | None = None
    severe_action: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "AltitudeRule":
        if not self.severe_min_m <= self.min_m < self.max_m <= self.severe_max_m:
            raise ValueError("altitude needs severe_min_m <= min_m < max_m <= severe_max_m")
        return self


class GeofenceRule(DomainModel):
    """The drone is further than `radius_m` from the property's centre (HIGH), or further than
    `radius_m * critical_factor` (CRITICAL). Resolves when it is back inside."""

    enabled: bool = True
    severity: Severity = Severity.HIGH
    center_latitude: float = Field(default=0.0004, ge=-90, le=90)
    center_longitude: float = Field(default=0.00045, ge=-180, le=180)
    radius_m: float = Field(default=100.0, gt=0)
    critical_factor: float = Field(default=1.5, gt=1.0)
    recommended_action: str | None = None
    critical_action: str | None = None


class SecurityRulesConfig(DomainModel):
    loitering: LoiteringRule = Field(default_factory=LoiteringRule)
    after_hours_presence: AfterHoursRule = Field(default_factory=AfterHoursRule)
    repeat_vehicle: RepeatVehicleRule = Field(default_factory=RepeatVehicleRule)
    restricted_zone_entry: RestrictedZoneRule = Field(default_factory=RestrictedZoneRule)
    # Drone-safety rules over telemetry (no camera involved).
    low_battery: LowBatteryRule = Field(default_factory=LowBatteryRule)
    signal_loss: SignalLossRule = Field(default_factory=SignalLossRule)
    altitude_anomaly: AltitudeRule = Field(default_factory=AltitudeRule)
    geofence_violation: GeofenceRule = Field(default_factory=GeofenceRule)


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
