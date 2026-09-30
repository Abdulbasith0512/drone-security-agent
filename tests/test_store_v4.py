"""Schema v4: streaming persistence, alert lifecycle, safety-alert evidence, provenance."""

import sqlite3
from datetime import datetime, timedelta

import pytest
from conftest import IST
from pydantic import ValidationError

from drone_agent.models import (
    Alert,
    AlertStatus,
    AlertTransition,
    CloseReason,
    DetectedObject,
    Entity,
    Event,
    LadderStep,
    MatchLevel,
    ObjectType,
    Observation,
    Provenance,
    Severity,
    TelemetryAssociation,
    TransitionKind,
    Verdict,
    Zone,
)
from drone_agent.store import Store
from drone_agent.store import schema as schema_module

T0 = datetime(2026, 9, 29, 23, 40, tzinfo=IST)


def obs(frame_id: int, seconds: int = 0, provenance=None) -> Observation:
    return Observation(
        observation_id=f"obs-{frame_id}",
        frame_id=frame_id,
        timestamp=T0 + timedelta(seconds=seconds),
        source="scripted",
        zone=Zone.MAIN_GATE,
        description="Person at the gate",
        detected_objects=[
            DetectedObject(
                object_type=ObjectType.PERSON, attributes={"clothing": "jacket"}, confidence=0.9
            )
        ],
        provenance=provenance,
    )


def event(seconds_last: int, *, closed: bool, frames=(1,)) -> Event:
    return Event(
        event_id="EVT-P-001",
        object_type=ObjectType.PERSON,
        zone=Zone.MAIN_GATE,
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=seconds_last),
        attributes={"clothing": "jacket"},
        observation_ids=[f"obs-{f}" for f in frames],
        frame_ids=list(frames),
        close_reason=CloseReason.GAP if closed else CloseReason.OPEN,
        confidence=0.9,
    )


def alert(severity=Severity.MEDIUM, **overrides) -> Alert:
    data = dict(
        alert_id="ALR-LOITER-EVT-P-001",
        rule_id="loitering",
        severity=severity,
        title="Person loitering at main_gate",
        message="stayed 130s",
        triggered_at=T0 + timedelta(seconds=120),
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=130),
        zone=Zone.MAIN_GATE,
        event_ids=["EVT-P-001"],
        frame_ids=[1, 2],
        confidence=0.9,
        verdict=Verdict.SUSPICIOUS,
        recommended_action="Verify visually",
    )
    data.update(overrides)
    return Alert(**data)


def transition(seq, kind, to_status, to_sev, at_s, **kw) -> AlertTransition:
    at = T0 + timedelta(seconds=at_s)
    return AlertTransition(
        seq=seq,
        kind=kind,
        at=at,
        effective_at=at,
        to_status=to_status,
        to_severity=to_sev,
        reason=kw.pop("reason", "test"),
        **kw,
    )


@pytest.fixture
def seeded(store, make_telemetry):
    """A store with two observations, matching telemetry and one open event."""
    sample = make_telemetry(T0, signal_percent=82.5)
    store.insert_telemetry(sample)
    first = obs(1, 0, Provenance.SCRIPTED).model_copy(
        update={"telemetry": TelemetryAssociation(telemetry=sample, age_seconds=0, stale=False)}
    )
    store.insert_observation(first)
    store.insert_observation(obs(2, 130))
    store.save_events([event(0, closed=False)])
    return store


# ---- schema ---------------------------------------------------------------------------------
def test_schema_v4_adds_the_new_tables_and_columns(store):
    names = {r["name"] for r in store._conn.execute("SELECT name FROM sqlite_master")}
    assert {"alert_history", "alert_telemetry", "scenarios", "agent_runs"} <= names
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(alerts)")}
    assert {"status", "confidence", "verdict", "recommended_action", "ladder_json"} <= cols
    assert schema_module.SCHEMA_VERSION >= 4


def test_v3_database_is_upgraded_in_place_and_keeps_its_data(tmp_path):
    path = tmp_path / "v3.db"
    conn = sqlite3.connect(path)
    for version in (1, 2, 3):
        for statement in schema_module.MIGRATIONS[version]:
            conn.execute(statement)
    conn.execute("PRAGMA user_version = 3")
    conn.execute(
        "INSERT INTO frames VALUES (1, '2026-09-29T23:41:00+05:30',"
        " '2026-09-29T18:11:00.000000Z', 'scripted', 'main_gate')"
    )
    conn.execute(
        "INSERT INTO observations (observation_id, frame_id, timestamp, ts_utc, source, zone,"
        " description) VALUES ('obs-1', 1, '2026-09-29T23:41:00+05:30',"
        " '2026-09-29T18:11:00.000000Z', 'scripted', 'main_gate', 'old data')"
    )
    conn.commit()
    conn.close()
    with Store.open(path) as s:
        assert s.migrated_from == 3
        old = s.get_observation("obs-1")
        assert old.description == "old data" and old.provenance is None
        assert s.count_alerts() == 0 and s.list_agent_runs() == []


def test_reapplying_a_column_migration_is_harmless(tmp_path):
    path = tmp_path / "again.db"
    with Store.open(path) as s:
        s._conn.execute("PRAGMA user_version = 3")  # pretend the last step never ran
    with Store.open(path) as s:  # every ALTER ... ADD COLUMN now finds its column already there
        assert s.migrated_from == 3


# ---- telemetry and provenance -----------------------------------------------------------------
def test_telemetry_signal_and_provenance_round_trip(seeded):
    first, second = seeded.query_observations()
    assert first.provenance is Provenance.SCRIPTED and second.provenance is None
    assert first.telemetry.telemetry.signal_percent == 82.5
    assert seeded.list_telemetry()[0].signal_percent == 82.5


def test_signal_percent_is_optional_and_validated(make_telemetry):
    assert make_telemetry(T0).signal_percent is None
    with pytest.raises(ValidationError):
        make_telemetry(T0, signal_percent=101)


def test_telemetry_reads_by_time_and_id(store, make_telemetry):
    ids = [store.insert_telemetry(make_telemetry(T0 + timedelta(seconds=10 * i))) for i in range(3)]
    assert [t.timestamp for t in store.list_telemetry(start=T0 + timedelta(seconds=10))] == [
        T0 + timedelta(seconds=10),
        T0 + timedelta(seconds=20),
    ]
    assert store.get_telemetry(ids[1]).timestamp == T0 + timedelta(seconds=10)
    assert store.get_telemetry(999) is None
    assert store.telemetry_id_for(T0 + timedelta(seconds=20)) == ids[2]
    assert store.telemetry_id_for(T0 + timedelta(seconds=5)) is None
    tid, sample = store.telemetry_before(T0 + timedelta(seconds=15))
    assert tid == ids[1] and sample.timestamp == T0 + timedelta(seconds=10)
    assert store.telemetry_before(T0 - timedelta(seconds=1)) is None


# ---- open events ------------------------------------------------------------------------------
def test_open_events_are_persisted_and_can_be_extended_in_place(seeded):
    (stored,) = seeded.query_events()
    assert stored.close_reason is CloseReason.OPEN and stored.is_open
    assert seeded.query_events(open_only=True)[0].event_id == "EVT-P-001"

    seeded.save_events([event(130, closed=False, frames=(1, 2))])
    (grown,) = seeded.query_events()
    assert grown.frame_ids == [1, 2] and grown.duration_seconds == 130 and grown.is_open

    seeded.save_events([event(130, closed=True, frames=(1, 2))])
    (closed,) = seeded.query_events()
    assert closed.close_reason is CloseReason.GAP and not closed.is_open
    assert seeded.query_events(open_only=True) == []
    assert seeded.count_events() == 1


def test_event_time_range_filter_selects_overlapping_events(seeded):
    assert len(seeded.query_events(start=T0 - timedelta(minutes=5), end=T0)) == 1
    assert seeded.query_events(start=T0 + timedelta(minutes=5)) == []
    assert seeded.query_events(end=T0 - timedelta(seconds=1)) == []


def test_upserting_an_event_keeps_alerts_that_reference_it(seeded):
    seeded.save_events([event(130, closed=False, frames=(1, 2))])
    seeded.save_alert(alert())
    seeded.save_events([event(200, closed=False, frames=(1, 2))])  # must not violate the FK
    assert seeded.get_alert("ALR-LOITER-EVT-P-001").event_ids == ["EVT-P-001"]


# ---- alert lifecycle ----------------------------------------------------------------------------
def test_alert_lifecycle_history_and_ladder_round_trip(seeded):
    seeded.save_events([event(130, closed=False, frames=(1, 2))])
    ladder = [
        LadderStep(at=T0 + timedelta(seconds=120), severity=Severity.MEDIUM, label="warning"),
        LadderStep(at=T0 + timedelta(seconds=125), severity=Severity.CRITICAL, label="critical"),
    ]
    history = [
        transition(1, TransitionKind.RAISED, AlertStatus.OPEN, Severity.MEDIUM, 120),
        transition(
            2,
            TransitionKind.ESCALATED,
            AlertStatus.ESCALATED,
            Severity.CRITICAL,
            125,
            from_status=AlertStatus.OPEN,
            from_severity=Severity.MEDIUM,
        ),
    ]
    a = alert(
        Severity.CRITICAL,
        status=AlertStatus.ESCALATED,
        ladder=ladder,
        history=history,
        detected_at=T0 + timedelta(seconds=120),
        updated_at=T0 + timedelta(seconds=130),
    )
    seeded.save_alert(a)
    back = seeded.get_alert(a.alert_id)
    assert back.status is AlertStatus.ESCALATED and back.severity is Severity.CRITICAL
    assert [s.label for s in back.ladder] == ["warning", "critical"]
    assert [t.kind for t in back.history] == [TransitionKind.RAISED, TransitionKind.ESCALATED]
    assert back.history[1].from_severity is Severity.MEDIUM
    assert back.verdict is Verdict.SUSPICIOUS and back.recommended_action == "Verify visually"
    assert back.detected_at == T0 + timedelta(seconds=120)
    assert back.detection_latency_seconds == 0

    resolved = a.model_copy(
        update={
            "status": AlertStatus.RESOLVED,
            "resolved_at": T0 + timedelta(seconds=300),
            "history": [
                *history,
                transition(
                    3, TransitionKind.RESOLVED, AlertStatus.RESOLVED, Severity.CRITICAL, 300
                ),
            ],
        }
    )
    seeded.save_alert(resolved)  # an update, not a duplicate
    assert seeded.count_alerts() == 1
    assert seeded.get_alert(a.alert_id).history[-1].kind is TransitionKind.RESOLVED
    assert seeded.query_alerts(active_only=True) == []
    assert len(seeded.query_alerts(status=AlertStatus.RESOLVED)) == 1


def test_safety_alert_cites_telemetry_and_needs_no_event(store, make_telemetry):
    t = T0 + timedelta(seconds=30)
    store.insert_telemetry(make_telemetry(T0, battery_percent=15.0))
    store.insert_telemetry(make_telemetry(t, battery_percent=8.0))
    a = Alert(
        alert_id="ALR-BATTERY-LOW",
        rule_id="low_battery",
        severity=Severity.CRITICAL,
        title="Battery critical",
        message="8%",
        triggered_at=T0,
        first_seen=T0,
        last_seen=t,
        telemetry_at=[T0, t],
    )
    store.save_alert(a)
    back = store.get_alert("ALR-BATTERY-LOW")
    assert back.event_ids == [] and back.frame_ids == [] and back.telemetry_at == [T0, t]


def test_alert_model_rejects_inconsistent_lifecycle_data():
    with pytest.raises(ValidationError, match="evidence"):
        alert(event_ids=[], frame_ids=[])
    with pytest.raises(ValidationError, match="frame_ids"):
        alert(frame_ids=[])
    with pytest.raises(ValidationError, match="resolved_at"):
        alert(status=AlertStatus.RESOLVED)
    step = LadderStep(at=T0, severity=Severity.LOW, label="warning")
    with pytest.raises(ValidationError, match="last ladder step"):
        alert(Severity.MEDIUM, ladder=[step])
    down = [
        LadderStep(at=T0, severity=Severity.HIGH, label="a"),
        LadderStep(at=T0 + timedelta(seconds=1), severity=Severity.MEDIUM, label="b"),
    ]
    with pytest.raises(ValidationError, match="rising"):
        alert(Severity.MEDIUM, ladder=down)


def test_batch_replace_alerts_writes_the_new_fields(seeded):
    seeded.save_events([event(130, closed=False, frames=(1, 2))])
    seeded.replace_alerts([alert(status=AlertStatus.ACKNOWLEDGED)])
    assert seeded.get_alert("ALR-LOITER-EVT-P-001").status is AlertStatus.ACKNOWLEDGED
    assert seeded.replace_alerts([]) == 1 and seeded.count_alerts() == 0


# ---- entities with scores ---------------------------------------------------------------------
def test_entity_scores_and_components_round_trip(store):
    for frame in (1, 2):
        store.insert_observation(obs(frame, frame * 10))
    events = [
        event(10, closed=True).model_copy(
            update={
                "event_id": "EVT-V-001",
                "entity_id": "ENT-V-001",
                "object_type": ObjectType.VEHICLE,
            }
        ),
        event(20, closed=True, frames=(2,)).model_copy(
            update={
                "event_id": "EVT-V-002",
                "entity_id": "ENT-V-001",
                "object_type": ObjectType.VEHICLE,
            }
        ),
    ]
    entity = Entity(
        entity_id="ENT-V-001",
        entity_type=ObjectType.VEHICLE,
        attributes={"color": "blue"},
        match_level=MatchLevel.COARSE,
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=20),
        event_ids=["EVT-V-001", "EVT-V-002"],
        match_score=0.83,
        link_scores={"EVT-V-002": 0.83},
        link_components={"EVT-V-002": {"color": 1.0, "body_type": 0.8, "temporal": 0.97}},
    )
    store.replace_derived(events, [entity])
    back = store.get_entity("ENT-V-001")
    assert back.match_score == 0.83 and back.link_scores == {"EVT-V-002": 0.83}
    assert back.link_components["EVT-V-002"]["temporal"] == 0.97


# ---- scenarios and agent runs -----------------------------------------------------------------
def test_scenario_registry_and_agent_runs(store):
    store.upsert_scenario("demo", "d", T0.isoformat(), "abc", {"alerts": []})
    store.upsert_scenario("demo", "d2", T0.isoformat(), "abd", {"alerts": [1]})
    assert store.get_scenario("demo")["description"] == "d2"
    assert store.get_scenario()["expect"] == {"alerts": [1]}
    assert store.get_scenario("missing") is None

    run_id = store.add_agent_run("patrol", "demo", "scripted", "running", metrics={"frames": 0})
    store.update_agent_run(run_id, status="finished", metrics={"frames": 27}, trace=[{"step": 1}])
    other = store.add_agent_run(
        "analyst", "q", "template", "ok", input_text="why?", output_text="a"
    )
    assert other == run_id + 1
    (patrol,) = store.list_agent_runs("patrol")
    assert patrol["status"] == "finished" and patrol["metrics"] == {"frames": 27}
    assert patrol["trace"] == [{"step": 1}]
    assert len(store.list_agent_runs()) == 2
