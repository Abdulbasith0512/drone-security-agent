from datetime import datetime

import pytest
from conftest import IST
from pydantic import ValidationError

from drone_agent.models import (
    Alert,
    DetectedObject,
    Event,
    ObjectType,
    Observation,
    Severity,
    Zone,
)

T0 = datetime(2026, 9, 29, 23, 55, tzinfo=IST)


def _obs(**overrides) -> Observation:
    data = dict(
        observation_id="obs-1",
        frame_id=1,
        timestamp=T0,
        source="scripted",
        zone=Zone.MAIN_GATE,
        description="Person standing near the main gate",
        detected_objects=[
            DetectedObject(
                object_type=ObjectType.PERSON,
                activity="standing",
                attributes={"clothing": "dark jacket"},
                confidence=0.94,
            )
        ],
    )
    data.update(overrides)
    return Observation(**data)


def test_observation_round_trips_through_json(make_telemetry):
    from drone_agent.models import TelemetryAssociation

    assoc = TelemetryAssociation(telemetry=make_telemetry(T0), age_seconds=4.0, stale=False)
    obs = _obs(telemetry=assoc)
    restored = Observation.model_validate_json(obs.model_dump_json())
    assert restored == obs
    assert restored.timestamp.utcoffset() == T0.utcoffset()
    assert obs.model_dump(mode="json")["zone"] == "main_gate"


def test_scenario_style_type_key_is_accepted():
    obj = DetectedObject.model_validate({"type": "vehicle", "confidence": 0.5})
    assert obj.object_type is ObjectType.VEHICLE
    assert obj.model_dump()["object_type"] == ObjectType.VEHICLE


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan")])
def test_confidence_must_be_between_0_and_1(confidence):
    with pytest.raises(ValidationError):
        DetectedObject(object_type=ObjectType.PERSON, confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_confidence_bounds_are_inclusive(confidence):
    assert DetectedObject(object_type=ObjectType.PERSON, confidence=confidence)


@pytest.mark.parametrize(
    "overrides",
    [
        {"latitude": 90.1},
        {"latitude": -90.1},
        {"longitude": 180.1},
        {"altitude_m": -0.5},
        {"altitude_m": 501},
        {"battery_percent": -1},
        {"battery_percent": 100.5},
        {"battery_percent": float("nan")},
        {"zone": "moon_base"},
    ],
)
def test_invalid_telemetry_rejected(make_telemetry, overrides):
    with pytest.raises(ValidationError):
        make_telemetry(T0, **overrides)


def test_naive_timestamps_rejected(make_telemetry):
    naive = datetime(2026, 9, 29, 23, 55)
    with pytest.raises(ValidationError):
        make_telemetry(naive)
    with pytest.raises(ValidationError):
        _obs(timestamp=naive)


def test_unknown_fields_and_empty_description_rejected():
    with pytest.raises(ValidationError):
        _obs(colour="blue")
    with pytest.raises(ValidationError):
        _obs(description="")


def test_models_are_immutable():
    with pytest.raises(ValidationError):
        _obs().description = "changed"


def test_event_requires_ordered_times_and_observations():
    kwargs = dict(
        event_id="e1",
        object_type=ObjectType.PERSON,
        zone=Zone.MAIN_GATE,
        first_seen=T0,
        last_seen=T0,
        observation_ids=["obs-1"],
    )
    assert Event(**kwargs)
    with pytest.raises(ValidationError):
        Event(**{**kwargs, "last_seen": datetime(2026, 9, 29, 23, 0, tzinfo=IST)})
    with pytest.raises(ValidationError):
        Event(**{**kwargs, "observation_ids": []})


def test_alert_schema_accepts_severity_and_rejects_unknown():
    kwargs = dict(alert_id="a1", rule_id="loitering", timestamp=T0, message="m")
    assert Alert(severity=Severity.WARNING, **kwargs).frame_ids == []
    with pytest.raises(ValidationError):
        Alert(severity="urgent", **kwargs)
