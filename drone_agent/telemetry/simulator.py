"""Deterministic telemetry simulator for a fixed-property patrol.

The drone hovers in each patrol zone for a configured duration. Positions are SYNTHETIC
coordinates near (0, 0) and do not represent any real property.
"""

import random

from pydantic import Field, model_validator

from drone_agent.clock import SimulationClock
from drone_agent.models import DomainModel, Telemetry, Zone

# Synthetic zone centres (degrees), a few metres to ~100 m apart near 0,0 (open ocean).
ZONE_CENTERS: dict[Zone, tuple[float, float]] = {
    Zone.MAIN_GATE: (0.0000, 0.0000),
    Zone.DRIVEWAY: (0.0002, 0.0001),
    Zone.PARKING_AREA: (0.0004, 0.0003),
    Zone.WAREHOUSE: (0.0006, 0.0006),
    Zone.RESTRICTED_ZONE: (0.0008, 0.0009),
}
_POSITION_JITTER_DEG = 0.00001  # about 1 m
_PATROL_ALTITUDE_M = 30.0
_ALTITUDE_JITTER_M = 1.5


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


class TelemetryConfig(DomainModel):
    seed: int = 0
    sample_interval_s: int = Field(default=10, gt=0)
    stale_after_s: float = Field(default=30.0, ge=0)
    battery_start_percent: float = Field(default=100.0, ge=0, le=100)
    battery_drain_percent_per_min: float = Field(default=0.4, ge=0)
    patrol: list[PatrolLeg] = Field(min_length=1)
    dropouts: list[Dropout] = Field(default_factory=list)

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
        if any(d.start_offset_s <= offset <= d.end_offset_s for d in config.dropouts):
            continue
        zone = config.zone_at(offset)
        lat, lon = ZONE_CENTERS[zone]
        battery = config.battery_start_percent - config.battery_drain_percent_per_min * offset / 60
        samples.append(
            Telemetry(
                timestamp=clock.at(offset),
                latitude=round(lat + d_lat, 6),
                longitude=round(lon + d_lon, 6),
                altitude_m=round(_PATROL_ALTITUDE_M + d_alt, 1),
                zone=zone,
                battery_percent=round(max(0.0, battery), 1),
            )
        )
    return samples
