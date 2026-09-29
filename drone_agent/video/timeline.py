"""SIMULATED telemetry timeline for a video.

A demo video contains no drone telemetry. This maps video time onto a configured simulated clock
and a simulated patrol (zone per time span), then reuses the M1 telemetry simulator and
nearest-prior synchronisation. Nothing here is read from the video: the zone and telemetry are
assumptions the operator supplies.
"""

import math
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import AwareDatetime, ValidationError

from drone_agent.clock import SimulationClock
from drone_agent.models import DomainModel, Zone
from drone_agent.telemetry import PatrolLeg, TelemetryConfig


class TimelineError(ValueError):
    """The timeline file could not be read or validated, or does not cover the video."""


class VideoTimeline(DomainModel):
    start: AwareDatetime  # the simulated wall-clock time of video offset 0
    telemetry: TelemetryConfig

    @property
    def clock(self) -> SimulationClock:
        return SimulationClock(self.start)

    def check_covers(self, last_offset_seconds: float) -> None:
        if last_offset_seconds > self.telemetry.duration_s:
            raise TimelineError(
                f"the simulated patrol lasts {self.telemetry.duration_s}s but the video needs "
                f"{last_offset_seconds:.1f}s; extend the patrol in the timeline file"
            )


def single_zone_timeline(
    start: datetime, zone: Zone, video_duration_seconds: float, seed: int = 0
) -> VideoTimeline:
    """The whole video is one simulated patrol leg in `zone`."""
    duration = math.ceil(video_duration_seconds) + 1
    return VideoTimeline(
        start=start,
        telemetry=TelemetryConfig(seed=seed, patrol=[PatrolLeg(zone=zone, duration_s=duration)]),
    )


def load_timeline(path: str | Path) -> VideoTimeline:
    """YAML with `start:` and a `telemetry:` block (same shape as in a scenario file)."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TimelineError(f"cannot read timeline {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise TimelineError(f"timeline {path} must be a YAML mapping")
    try:
        return VideoTimeline.model_validate(raw)
    except ValidationError as exc:
        raise TimelineError(f"invalid timeline {path}:\n{exc}") from exc
