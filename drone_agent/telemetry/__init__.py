"""Telemetry simulator and telemetry-to-frame synchronisation."""

from drone_agent.telemetry.geo import (
    GEOFENCE_CENTER,
    GEOFENCE_RADIUS_M,
    distance_m,
    east_offset_degrees,
)
from drone_agent.telemetry.simulator import (
    ZONE_CENTERS,
    Anomaly,
    Dropout,
    PatrolLeg,
    TelemetryConfig,
    simulate_telemetry,
)
from drone_agent.telemetry.sync import TelemetrySynchronizer

__all__ = [
    "GEOFENCE_CENTER",
    "GEOFENCE_RADIUS_M",
    "ZONE_CENTERS",
    "Anomaly",
    "Dropout",
    "PatrolLeg",
    "TelemetryConfig",
    "TelemetrySynchronizer",
    "distance_m",
    "east_offset_degrees",
    "simulate_telemetry",
]
