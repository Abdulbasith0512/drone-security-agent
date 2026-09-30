"""Deterministic telemetry simulator for a fixed-property patrol.

The drone hovers in each patrol zone for a configured duration. Positions are SYNTHETIC
coordinates near (0, 0) and do not represent any real property. Scenarios can inject anomalies
(battery cliff, signal fade, altitude excursion, geofence drift) so the safety rules have
something to detect; they are scripted, not measured.
"""

import random
from typing import Literal

from pydantic import Field, model_validator

from drone_agent.clock import SimulationClock
from drone_agent.models import DomainModel, Telemetry, Zone
from drone_agent.telemetry.geo import east_offset_degrees

# Synthetic zone centres (degrees), a few metres to ~100 m apart near 0,0 (open ocean).
ZONE_CENTERS: dict[Zone, tuple[float, float]] = {
    Zone.MAIN_GATE: (0.0000, 0.0000),
    Zone.DRIVEWAY: (0.0002, 0.0001),
    Zone.PARKING_AREA: (0.0004, 0.0003),
    Zone.WAREHOUSE: (0.0006, 0.0006),
    Zone.RESTRICTED_ZONE: (0.0008, 0.0009),
    Zone.GARAGE: (0.0003, 0.0007),
}
_POSITION_JITTER_DEG = 0.00001  # about 1 m
_PATROL_ALTITUDE_M = 30.0
_ALTITUDE_JITTER_M = 1.5
_SIGNAL_JITTER = 2.0


class PatrolLeg(DomainModel):
    zone: Zone
    duration_s: int = Field(gt=0)


class Dropout(DomainModel):
    """Telemetry samples with start <= offset <= end are not delivered (simulated link loss)."""

    start_offset_s: float = Field(ge=0)
    end_offset_s: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "Dropout":
        if self.end_offset_s < self.start_offset_s:
            raise ValueError("dropout end_offset_s must not be before start_offset_s")
        return self


class Anomaly(DomainModel):
    """One scripted telemetry anomaly.

    * `battery_drop`: at `start_offset_s` the battery falls to `value` percent and keeps draining.
    * `signal_fade`: from `start_offset_s` to `end_offset_s` the signal ramps down to `value`
      percent over `ramp_s` seconds and holds; it recovers at `end_offset_s`.
    * `altitude_excursion`: between the offsets the altitude is `value` metres (plus jitter).
    * `geofence_drift`: between the offsets the drone drifts east, reaching `value` metres from
      its planned position at `end_offset_s`, then returns.
    """

    kind: Literal["battery_drop", "signal_fade", "altitude_excursion", "geofence_drift"]
    start_offset_s: float = Field(ge=0)
    end_offset_s: float | None = Field(default=None, ge=0)
    value: float
    ramp_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _valid(self) -> "Anomaly":
        if self.kind == "battery_drop":
            if not 0 <= self.value <= 100:
                raise ValueError("battery_drop value must be a percentage in 0-100")
            return self
        if self.end_offset_s is None:
            raise ValueError(f"{self.kind} needs end_offset_s")
        if self.end_offset_s <= self.start_offset_s:
            raise ValueError(f"{self.kind} end_offset_s must be after start_offset_s")
        if self.kind == "signal_fade" and not 0 <= self.value <= 100:
            raise ValueError("signal_fade value must be a percentage in 0-100")
        if self.kind == "altitude_excursion" and not 0 <= self.value <= 500:
            raise ValueError("altitude_excursion value must be metres in 0-500")
        if self.kind == "geofence_drift" and self.value <= 0:
            raise ValueError("geofence_drift value must be a positive distance in metres")
        return self

    def active(self, offset: float) -> bool:
        end = self.end_offset_s if self.end_offset_s is not None else float("inf")
        return self.start_offset_s <= offset <= end


class TelemetryConfig(DomainModel):
    seed: int = 0
    sample_interval_s: int = Field(default=10, gt=0)
    stale_after_s: float = Field(default=30.0, ge=0)
    battery_start_percent: float = Field(default=100.0, ge=0, le=100)
    battery_drain_percent_per_min: float = Field(default=0.4, ge=0)
    # Baseline link quality. None = this source reports no signal at all (the old behaviour).
    signal_baseline_percent: float | None = Field(default=None, ge=0, le=100)
    patrol: list[PatrolLeg] = Field(min_length=1)
    dropouts: list[Dropout] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)

    @property
    def duration_s(self) -> int:
        return sum(leg.duration_s for leg in self.patrol)

    def zone_at(self, offset_s: float) -> Zone:
        """Patrol zone at `offset_s`. Leg boundaries belong to the leg that starts there."""
        if offset_s < 0 or offset_s > self.duration_s:
            raise ValueError(f"offset {offset_s} outside patrol duration 0..{self.duration_s}")
        elapsed = 0
        for leg in self.patrol:
            elapsed += leg.duration_s
            if offset_s < elapsed:
                return leg.zone
        return self.patrol[-1].zone  # offset == duration


def _battery(config: TelemetryConfig, offset: int) -> float:
    start, since = config.battery_start_percent, 0.0
    for drop in sorted(
        (a for a in config.anomalies if a.kind == "battery_drop"), key=lambda a: a.start_offset_s
    ):
        if offset >= drop.start_offset_s:
            start, since = drop.value, drop.start_offset_s
    return start - config.battery_drain_percent_per_min * (offset - since) / 60


def _signal(config: TelemetryConfig, offset: int, noise: float) -> float | None:
    if config.signal_baseline_percent is None:
        return None
    level = config.signal_baseline_percent + noise
    for fade in (a for a in config.anomalies if a.kind == "signal_fade" and a.active(offset)):
        progress = min(1.0, (offset - fade.start_offset_s) / fade.ramp_s)
        level = level + (fade.value - level) * progress
    return round(min(100.0, max(0.0, level)), 1)


def simulate_telemetry(clock: SimulationClock, config: TelemetryConfig) -> list[Telemetry]:
    """Telemetry samples every `sample_interval_s` from offset 0 to the end of the patrol.

    Same clock start and config (including seed) always give identical output. Random draws are
    made for every sample before dropouts are applied, so dropouts never change other samples.
    """
    rng = random.Random(config.seed)
    samples: list[Telemetry] = []
    for offset in range(0, config.duration_s + 1, config.sample_interval_s):
        d_lat = rng.uniform(-_POSITION_JITTER_DEG, _POSITION_JITTER_DEG)
        d_lon = rng.uniform(-_POSITION_JITTER_DEG, _POSITION_JITTER_DEG)
        d_alt = rng.uniform(-_ALTITUDE_JITTER_M, _ALTITUDE_JITTER_M)
        # Drawn only when a signal is simulated, so older configs keep their exact sequence.
        d_sig = (
            rng.uniform(-_SIGNAL_JITTER, _SIGNAL_JITTER)
            if config.signal_baseline_percent is not None
            else 0.0
        )
        if any(d.start_offset_s <= offset <= d.end_offset_s for d in config.dropouts):
            continue
        zone = config.zone_at(offset)
        lat, lon = ZONE_CENTERS[zone]
        altitude = _PATROL_ALTITUDE_M
        for anomaly in (a for a in config.anomalies if a.active(offset)):
            if anomaly.kind == "altitude_excursion":
                altitude = anomaly.value
            elif anomaly.kind == "geofence_drift":
                assert anomaly.end_offset_s is not None
                span = anomaly.end_offset_s - anomaly.start_offset_s
                east_m = anomaly.value * (offset - anomaly.start_offset_s) / span
                lon += east_offset_degrees(east_m, lat)
        samples.append(
            Telemetry(
                timestamp=clock.at(offset),
                latitude=round(lat + d_lat, 6),
                longitude=round(lon + d_lon, 6),
                altitude_m=round(min(500.0, max(0.0, altitude + d_alt)), 1),
                zone=zone,
                battery_percent=round(min(100.0, max(0.0, _battery(config, offset))), 1),
                signal_percent=_signal(config, offset, d_sig),
            )
        )
    return samples
