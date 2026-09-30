"""Scenario files: scripted frames plus a telemetry/patrol configuration (YAML)."""

from pathlib import Path

import yaml
from pydantic import AwareDatetime, Field, ValidationError, model_validator

from drone_agent.clock import SimulationClock
from drone_agent.expectations import Expectations
from drone_agent.models import DetectedObject, DomainModel, Zone
from drone_agent.telemetry import TelemetryConfig


class ScenarioError(ValueError):
    """A scenario file could not be read, parsed or validated."""


class ScenarioFrame(DomainModel):
    frame_id: int = Field(ge=0)
    timestamp_offset: float = Field(ge=0)  # seconds after scenario start
    zone: Zone
    description: str = Field(min_length=1)
    objects: list[DetectedObject] = Field(default_factory=list)


class Scenario(DomainModel):
    name: str = Field(min_length=1)
    description: str = ""
    start: AwareDatetime
    telemetry: TelemetryConfig
    frames: list[ScenarioFrame] = Field(min_length=1)
    # Optional ground truth used by the evaluation (see drone_agent/expectations.py).
    expect: Expectations | None = None

    @property
    def clock(self) -> SimulationClock:
        return SimulationClock(self.start)

    @model_validator(mode="after")
    def _frames_consistent(self) -> "Scenario":
        ids = [f.frame_id for f in self.frames]
        if len(set(ids)) != len(ids):
            raise ValueError("frame_id values must be unique")
        offsets = [f.timestamp_offset for f in self.frames]
        if any(b <= a for a, b in zip(offsets, offsets[1:], strict=False)):
            raise ValueError("frames must have strictly increasing timestamp_offset")
        duration = self.telemetry.duration_s
        for f in self.frames:
            if f.timestamp_offset > duration:
                raise ValueError(
                    f"frame {f.frame_id} offset {f.timestamp_offset} is after the patrol ends "
                    f"({duration}s)"
                )
            patrol_zone = self.telemetry.zone_at(f.timestamp_offset)
            if f.zone != patrol_zone:
                raise ValueError(
                    f"frame {f.frame_id}: zone {f.zone} does not match the drone's patrol zone "
                    f"{patrol_zone} at offset {f.timestamp_offset}"
                )
        return self


def load_scenario(path: str | Path) -> Scenario:
    """Load and validate a scenario YAML file. Raises ScenarioError on any problem."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ScenarioError(f"cannot read scenario {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"scenario {path} must be a YAML mapping at the top level")
    try:
        return Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(f"invalid scenario {path}:\n{exc}") from exc
