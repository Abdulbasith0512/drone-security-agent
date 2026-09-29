"""What a real vision model must return, and how it becomes an Observation.

A real provider implements only `FrameDescriber.describe(frame) -> ProviderOutput`: it looks at
pixels and reports what is VISIBLE. It never chooses the zone, time or ids (those come from the
pipeline's simulated timeline) and it never draws security conclusions (those belong to the
deterministic rules). `DescriberProvider` turns any describer into a `VisionProvider`, so real
video, cassette replay and the scripted provider all yield the same validated `Observation`.
"""

import json
import math
import re
from typing import Protocol, runtime_checkable

from pydantic import Field, ValidationError, field_validator

from drone_agent.models import DetectedObject, DomainModel, ObjectType, Observation
from drone_agent.vision.base import FrameInput

CONTRACT_VERSION = "1"

# Guidance a real provider should put in its prompt (kept here so it is versioned with the schema).
PROVIDER_INSTRUCTIONS = """\
You describe ONE image from a fixed-property security drone. Reply with a single JSON object and
nothing else, matching the given schema.
Report only what is visibly present: people, vehicles and other notable objects.
For each object give: type (person | vehicle | other), activity if it is visible (for example
standing, walking, parked, driving), coarse visible attributes only (person: clothing;
vehicle: color, body_type), and your confidence between 0 and 1.
Do NOT infer or state identity, intent, criminality, names, plate numbers you cannot read, or
anything that is not visible. Do NOT decide whether something is suspicious or dangerous.
If nothing notable is visible, return an empty objects list.
Keep the description to one factual sentence.
"""

_KEY = re.compile(r"^[a-z][a-z0-9_]{0,29}$")


class ProviderOutputError(ValueError):
    """A provider returned something that does not satisfy the output schema."""


class ProviderObject(DomainModel):
    type: ObjectType
    activity: str | None = Field(default=None, min_length=1, max_length=100)
    attributes: dict[str, str] = Field(default_factory=dict, max_length=10)
    # None = the provider does not report confidence (see default_confidence in DescriberProvider).
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)

    @field_validator("attributes")
    @classmethod
    def _coarse_attributes(cls, value: dict[str, str]) -> dict[str, str]:
        for key, text in value.items():
            if not _KEY.match(key):
                raise ValueError(f"attribute key '{key}' must be lower_snake_case (max 30 chars)")
            if not text.strip() or len(text) > 100:
                raise ValueError(f"attribute '{key}' must be 1-100 non-blank characters")
        return value


class ProviderOutput(DomainModel):
    """Strict schema for one frame; unknown keys, wrong types and bad ranges are rejected."""

    description: str = Field(min_length=1, max_length=1000)
    objects: list[ProviderObject] = Field(default_factory=list, max_length=50)

    @field_validator("description")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("description must not be blank")
        return value


def provider_output_schema() -> dict:
    """JSON Schema to hand to a real provider (structured-output / prompt)."""
    return ProviderOutput.model_json_schema()


def parse_provider_output(raw: str | dict) -> ProviderOutput:
    """Validate provider output (a JSON string or an already-parsed dict).

    Policy: the only normalisation is removing ONE surrounding markdown code fence. Nothing else is
    repaired or guessed; missing fields are errors, never filled in. Raises ProviderOutputError.
    """
    data = raw
    if isinstance(raw, str):
        text = raw.strip()
        fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ProviderOutputError(f"provider output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ProviderOutputError("provider output must be a JSON object")
    try:
        return ProviderOutput.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'output'}: {e['msg']}" for e in exc.errors()
        )
        raise ProviderOutputError(f"provider output failed validation: {problems}") from exc


@runtime_checkable
class FrameDescriber(Protocol):
    """What a real VLM adapter (or a cassette) implements."""

    name: str

    def describe(self, frame: FrameInput) -> ProviderOutput:
        """Describe `frame.image`. Must return a validated ProviderOutput (or raise)."""
        ...


def observation_from_output(
    frame: FrameInput,
    output: ProviderOutput,
    source: str,
    default_confidence: float | None = None,
) -> Observation:
    """Build the pipeline's Observation from a provider's output.

    A missing confidence is an error unless `default_confidence` was configured explicitly; that
    value is a placeholder meaning "not reported", not a probability. No other field is invented.
    """
    if frame.zone is None:
        raise ValueError("a video frame needs a zone from the simulated timeline")
    if default_confidence is not None and not (
        math.isfinite(default_confidence) and 0 <= default_confidence <= 1
    ):
        raise ValueError("default_confidence must be between 0 and 1")
    objects: list[DetectedObject] = []
    for i, obj in enumerate(output.objects):
        confidence = obj.confidence if obj.confidence is not None else default_confidence
        if confidence is None:
            raise ProviderOutputError(
                f"object {i} ({obj.type.value}) has no confidence and no default_confidence "
                "is configured"
            )
        objects.append(
            DetectedObject(
                object_type=obj.type,
                activity=obj.activity,
                attributes=obj.attributes,
                confidence=confidence,
            )
        )
    return Observation(
        observation_id=f"obs-{frame.frame_id}",
        frame_id=frame.frame_id,
        timestamp=frame.timestamp,
        source=source,
        zone=frame.zone,
        description=output.description,
        detected_objects=objects,
    )


class DescriberProvider:
    """Adapts any FrameDescriber to the VisionProvider interface."""

    def __init__(self, describer: FrameDescriber, default_confidence: float | None = None) -> None:
        self.describer = describer
        self.default_confidence = default_confidence

    @property
    def name(self) -> str:
        return self.describer.name

    def analyze(self, frame: FrameInput) -> Observation:
        output = self.describer.describe(frame)
        return observation_from_output(frame, output, self.name, self.default_confidence)
