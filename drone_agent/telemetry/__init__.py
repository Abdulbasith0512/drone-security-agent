"""Telemetry simulator and telemetry-to-frame synchronisation."""

from drone_agent.telemetry.simulator import (
    ZONE_CENTERS,
    Dropout,
    PatrolLeg,
    TelemetryConfig,
    simulate_telemetry,
)
from drone_agent.telemetry.sync import TelemetrySynchronizer

__all__ = [
    "ZONE_CENTERS",
    "Dropout",
    "PatrolLeg",
    "TelemetryConfig",
    "TelemetrySynchronizer",
    "simulate_telemetry",
]
