from datetime import datetime, timedelta

import pytest
from conftest import IST
from pydantic import ValidationError

from drone_agent.models import (
    Alert,
    CloseReason,
    DetectedObject,
    Entity,
    Event,
    MatchLevel,
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


def test_event_requires_ordered_times_and_matching_evidence():
    kwargs = dict(
        event_id="e1",
        object_type=ObjectType.PERSON,
        zone=Zone.MAIN_GATE,
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=90),
        observation_ids=["obs-1", "obs-2"],
        frame_ids=[1, 2],
        close_reason=CloseReason.GAP,
    )
    event = Event(**kwargs)
    assert event.duration_seconds == 90
    assert event.evidence_count == 2
    assert Event.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError):
        Event(**{**kwargs, "last_seen": datetime(2026, 9, 29, 23, 0, tzinfo=IST)})
    with pytest.raises(ValidationError):
        Event(**{**kwargs, "observation_ids": []})
    with pytest.raises(ValidationError, match="same length"):
        Event(**{**kwargs, "frame_ids": [1]})


def test_entity_is_vehicle_only_and_labels_do_not_overclaim():
    kwargs = dict(
        entity_id="ENT-V-001",
        entity_type=ObjectType.VEHICLE,
        attributes={"color": "blue", "body_type": "pickup"},
        match_level=MatchLevel.COARSE,
        first_seen=T0,
        last_seen=T0,
        event_ids=["EVT-V-001", "EVT-V-002"],
    )
    entity = Entity(**kwargs)
    assert entity.appearance_count == 2
    assert entity.label == "similar blue pickup"
    confirmed = Entity(
        **{
            **kwargs,
            "match_level": MatchLevel.CONFIRMED,
            "attributes": {**kwargs["attributes"], "plate": "AB12CD"},
        }
    )
    assert confirmed.label.startswith("confirmed") and "AB12CD" in confirmed.label
    with pytest.raises(ValidationError, match="vehicle-only"):
        Entity(**{**kwargs, "entity_type": ObjectType.PERSON})


def _alert(**overrides) -> Alert:
    data = dict(
        alert_id="ALR-LOITER-EVT-P-001",
        rule_id="loitering",
        severity=Severity.MEDIUM,
        title="Person loitering at main_gate",
        message="Person stayed for 480s",
        triggered_at=T0 + timedelta(seconds=300),
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=480),
        zone=Zone.MAIN_GATE,
        event_ids=["EVT-P-001"],
        frame_ids=[1, 2, 3],
        facts={"duration_seconds": 480.0, "threshold_seconds": 300.0, "zones": ["main_gate"]},
    )
    data.update(overrides)
    return Alert(**data)


def test_alert_round_trips_and_carries_facts_and_evidence_links():
    alert = _alert(entity_id="ENT-V-001")
    assert Alert.model_validate_json(alert.model_dump_json()) == alert
    assert alert.facts["threshold_seconds"] == 300.0
    assert alert.event_ids == ["EVT-P-001"] and alert.frame_ids == [1, 2, 3]


def test_alert_validation_rejects_bad_severity_times_and_missing_evidence():
    with pytest.raises(ValidationError):
        _alert(severity="urgent")
    with pytest.raises(ValidationError):
        _alert(severity="warning")  # the pre-M4 name no longer exists
    with pytest.raises(ValidationError, match="triggered_at"):
        _alert(triggered_at=T0 - timedelta(seconds=1))
    with pytest.raises(ValidationError):
        _alert(event_ids=[])
    with pytest.raises(ValidationError):
        _alert(frame_ids=[])
    with pytest.raises(ValidationError):
        _alert(facts={"nested": {"not": "allowed"}})
    with pytest.raises(ValidationError):
        _alert(triggered_at=datetime(2026, 9, 29, 23, 45))  # naive


def test_severity_is_ordered():
    ranks = [s.rank for s in (Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL)]
    assert ranks == [0, 1, 2, 3]
