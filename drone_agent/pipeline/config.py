"""Event aggregation and vehicle entity matching thresholds (defaults mirror config/events.yaml)."""

from pathlib import Path

import yaml
from pydantic import Field, ValidationError

from drone_agent.models import DomainModel, ObjectType


class ConfigError(ValueError):
    """The configuration file could not be read or validated."""


class AggregationSettings(DomainModel):
    # An event ends when the next matching detection is more than this many seconds
    # after the last one.
    max_gap_seconds: float = Field(default=180.0, gt=0)
    # ... or when more than this many consecutive frames in the event's zone lack a matching
    # detection.
    # (Flicker tolerance: the drone looked, the detector just missed.)
    max_missed_frames: int = Field(default=2, ge=0)
    # Attributes that must agree (after normalisation) for a detection to join an event.
    # A key missing on either side is treated as unknown and does not block a match.
    stable_attributes: dict[ObjectType, list[str]] = Field(
        default_factory=lambda: {
            ObjectType.PERSON: ["clothing"],
            ObjectType.VEHICLE: ["color", "body_type"],
            ObjectType.OTHER: [],
        }
    )


class NormalizationSettings(DomainModel):
    color_modifiers: list[str] = Field(
        default_factory=lambda: ["dark", "light", "deep", "pale", "bright"]
    )
    color_aliases: dict[str, str] = Field(default_factory=lambda: {"grey": "gray"})
    body_type_aliases: dict[str, str] = Field(
        default_factory=lambda: {"pickup truck": "pickup", "pick up": "pickup"}
    )


class VehicleMatchSettings(DomainModel):
    # Both events need a value for every required attribute and the normalised values must be
    # equal.
    # Missing evidence prevents a link; nothing is guessed.
    required_attributes: list[str] = Field(default_factory=lambda: ["color", "body_type"])
    # Optional explicit plate evidence (only used if the scenario/VLM supplied it).
    plate_attribute: str = "plate"


class EventConfig(DomainModel):
    aggregation: AggregationSettings = Field(default_factory=AggregationSettings)
    normalization: NormalizationSettings = Field(default_factory=NormalizationSettings)
    vehicle_matching: VehicleMatchSettings = Field(default_factory=VehicleMatchSettings)


def load_config(path: str | Path | None = None) -> EventConfig:
    """Load a YAML config, or return the built-in defaults when `path` is None."""
    if path is None:
        return EventConfig()
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return EventConfig.model_validate(raw or {})
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read config {path}: {exc}") from exc
    except ValidationError as exc:
        raise ConfigError(f"invalid config {path}:\n{exc}") from exc
