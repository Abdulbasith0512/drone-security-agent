"""T02: a loitering alert resolves on OBSERVED departure, never on lost observation coverage.

A camera moving to another zone, a gap in the observations, dropped frames and the end of the input
all end an event's coverage. None of them shows the person leaving. Only the drone looking at the
same zone, more than `max_missed_frames` times in a row, and not seeing the person does.
"""

import shutil
import sqlite3
from datetime import timedelta

import pytest
from conftest import SCENARIO_DIR

from drone_agent.agents import PerceptionAgent, SamplerConfig
from drone_agent.ingest import ingest_result
from drone_agent.models import AlertStatus, CloseReason, TransitionKind
from drone_agent.pipeline import build_events
from drone_agent.rules import AlertManager, analyze
from drone_agent.runtime import EngineOptions, PatrolEngine
from drone_agent.scenario import Scenario, load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store
from drone_agent.store import schema as schema_module
from drone_agent.vision import ScriptedProvider

PERSON = {"type": "person", "activity": "standing", "attributes": {"clothing": "dark"}}
ALERT_ID = "ALR-LOITER-EVT-P-001"


def patrol(frames, legs=(("main_gate", 900),)):
    """`frames`: (offset_s, zone, person_present). The person stands at the gate for 60-240 s."""
    return Scenario.model_validate(
        {
            "name": "departure",
            "start": "2026-09-29T23:00:00+05:30",
            "telemetry": {
                "seed": 1,
                "sample_interval_s": 30,
                "patrol": [{"zone": zone, "duration_s": seconds} for zone, seconds in legs],
            },
            "frames": [
                {
                    "frame_id": i,
                    "timestamp_offset": offset,
                    "zone": zone,
                    "description": "Person at the gate" if present else f"{zone} clear at {offset}",
                    "objects": [{**PERSON, "confidence": 0.9}] if present else [],
                }
                for i, (offset, zone, present) in enumerate(frames, start=1)
            ],
        }
    )


SEEN_AT_GATE = [(60, "main_gate", True), (120, "main_gate", True), (180, "main_gate", True)]
SEEN_AT_GATE.append((240, "main_gate", True))  # 180 s of loitering: past the 120 s warning


def run(scenario, **kwargs):
    with Store.open(":memory:") as store:
        PatrolEngine(
            scenario,
            store,
            options=EngineOptions(sampler=SamplerConfig(enabled=False)),
            **kwargs,
        ).run()
        return (
            store.get_alert(ALERT_ID),
            store.get_event("EVT-P-001"),
            store.query_events(),
        )


# ---- 1. the camera moves away ------------------------------------------------------------------
def test_camera_moving_to_another_zone_does_not_resolve_the_alert():
    scenario = patrol(
        [*SEEN_AT_GATE, (300, "driveway", False), (360, "driveway", False)],
        legs=(("main_gate", 270), ("driveway", 600)),
    )
    alert, event, _ = run(scenario)
    assert event.close_reason is CloseReason.ZONE_CHANGE  # coverage ended...
    assert event.departure_observed_at is None  # ...but nobody saw the gate empty
    assert alert.status is AlertStatus.OPEN and alert.resolved_at is None
    assert TransitionKind.RESOLVED not in [h.kind for h in alert.history]


# ---- 2. a gap in the observations --------------------------------------------------------------
def test_a_long_gap_in_observations_does_not_prove_departure():
    # nothing is seen for 360 s (longer than the 180 s gap limit), then the gate is seen empty once
    scenario = patrol([*SEEN_AT_GATE, (600, "main_gate", False)])
    alert, event, _ = run(scenario)
    assert event.close_reason is CloseReason.GAP and event.departure_observed_at is None
    assert alert.status is AlertStatus.OPEN and alert.resolved_at is None


def test_dropped_frames_and_the_end_of_the_input_do_not_prove_departure():
    scenario = patrol([*SEEN_AT_GATE, (300, "main_gate", False), (360, "main_gate", False)])
    provider = _Flaky(scenario, fail_frames={5, 6})  # the perception service is down afterwards
    alert, event, _ = run(scenario, perception=PerceptionAgent([provider]))
    assert event.close_reason is CloseReason.SCENARIO_END and event.departure_observed_at is None
    assert alert.status is AlertStatus.OPEN and alert.resolved_at is None


class _Flaky:
    name = "flaky"

    def __init__(self, scenario, fail_frames):
        self.inner = ScriptedProvider(scenario)
        self.fail_frames = fail_frames

    def analyze(self, frame):
        if frame.frame_id in self.fail_frames:
            raise RuntimeError("simulated vision outage")
        return self.inner.analyze(frame)


# ---- 3. explicit same-zone departure evidence ---------------------------------------------------
EMPTY_GATE = [(300, "main_gate", False), (360, "main_gate", False), (420, "main_gate", False)]


def test_same_zone_departure_evidence_resolves_the_alert_at_the_observed_timestamp():
    scenario = patrol([*SEEN_AT_GATE, *EMPTY_GATE])
    alert, event, _ = run(scenario)
    start = scenario.start
    # the gate was first seen empty at +300 s; three empty looks (more than the tolerance of 2)
    # confirmed it at +420 s
    assert event.close_reason is CloseReason.MISSED_FRAMES
    assert event.departure_observed_at == start + timedelta(seconds=300)
    assert event.last_seen == start + timedelta(seconds=240)
    assert alert.status is AlertStatus.RESOLVED
    assert alert.resolved_at == start + timedelta(seconds=300)
    resolved = alert.history[-1]
    assert resolved.kind is TransitionKind.RESOLVED
    assert resolved.effective_at == start + timedelta(seconds=300)  # when the person was gone
    assert resolved.at == start + timedelta(seconds=420)  # when the system could be sure


def test_one_or_two_empty_looks_are_within_the_tolerance_and_prove_nothing():
    scenario = patrol(
        [*SEEN_AT_GATE, *EMPTY_GATE[:2], (480, "driveway", False)],
        legs=(("main_gate", 450), ("driveway", 600)),
    )
    alert, event, _ = run(scenario)
    assert event.close_reason is CloseReason.ZONE_CHANGE and event.departure_observed_at is None
    assert alert.status is AlertStatus.OPEN


def test_a_reappearing_person_clears_a_pending_absence():
    """Two empty looks, then the person is back: the earlier misses must not become a departure."""
    frames = [*SEEN_AT_GATE, *EMPTY_GATE[:2], (420, "main_gate", True)]
    frames += [(480, "main_gate", False), (540, "main_gate", False), (600, "main_gate", False)]
    alert, event, events = run(patrol(frames))
    assert len(events) == 1  # one continuous event: the tolerance absorbed the two misses
    # the departure time is the start of the LAST run of empty looks, not the earlier one
    assert event.departure_observed_at == patrol(frames).start + timedelta(seconds=480)
    assert alert.resolved_at == event.departure_observed_at


# ---- 4. the operator can still resolve ---------------------------------------------------------
def test_an_operator_can_resolve_an_alert_the_system_left_open():
    scenario = patrol([*SEEN_AT_GATE, (300, "driveway", False)],
                      legs=(("main_gate", 270), ("driveway", 600)))  # fmt: skip
    with Store.open(":memory:") as store:
        PatrolEngine(scenario, store).run()
        manager = AlertManager(store.query_alerts())
        assert manager.get(ALERT_ID).status is AlertStatus.OPEN
        at = scenario.start + timedelta(seconds=400)
        resolved = manager.resolve(ALERT_ID, at, "guard confirmed the gate is clear").alert
        assert resolved.status is AlertStatus.RESOLVED and resolved.resolved_at == at
        assert resolved.history[-1].actor == "operator"


# ---- 5. determinism and persistence ------------------------------------------------------------
@pytest.mark.parametrize(
    "frames", [[*SEEN_AT_GATE, *EMPTY_GATE], SEEN_AT_GATE], ids=["left", "stayed"]
)
def test_streaming_and_batch_agree_on_events_and_the_resolution(frames):
    scenario = patrol(frames)
    with Store.open(":memory:") as live, Store.open(":memory:") as ref:
        PatrolEngine(
            scenario, live, options=EngineOptions(sampler=SamplerConfig(enabled=False))
        ).run()
        ingest_result(ref, run_scenario(scenario))
        build_events(ref)
        analyze(ref)
        assert live.query_events() == ref.query_events()  # includes departure_observed_at
        assert [(a.alert_id, a.status, a.resolved_at) for a in live.query_alerts()] == [
            (a.alert_id, a.status, a.resolved_at) for a in ref.query_alerts()
        ]


def test_the_departure_timestamp_is_persisted_and_old_rows_are_not_reinterpreted(tmp_path):
    scenario = patrol([*SEEN_AT_GATE, *EMPTY_GATE])
    db = tmp_path / "v6.db"
    with Store.open(db) as store:
        ingest_result(store, run_scenario(scenario))
        build_events(store)
    with Store.open(db) as store:  # reopened: read back from SQLite
        (event,) = store.query_events()
        assert event.departure_observed_at == scenario.start + timedelta(seconds=300)

    # simulate a database written before schema v6: no column, user_version 5
    old = tmp_path / "v5.db"
    shutil.copy(db, old)
    conn = sqlite3.connect(old)
    conn.execute("ALTER TABLE events DROP COLUMN departure_observed_at")
    conn.execute("PRAGMA user_version = 5")
    conn.commit()
    conn.close()
    with Store.open(old) as migrated:
        assert migrated.migrated_from == 5
        (event,) = migrated.query_events()
        assert event.close_reason is CloseReason.MISSED_FRAMES  # kept as recorded
        assert event.departure_observed_at is None  # but never promoted to a confirmed departure
        assert schema_module.SCHEMA_VERSION == 6


def test_a_departure_before_the_last_sighting_is_rejected():
    scenario = patrol([*SEEN_AT_GATE, *EMPTY_GATE])
    with Store.open(":memory:") as store:
        PatrolEngine(scenario, store).run()
        event = store.get_event("EVT-P-001")
    with pytest.raises(ValueError, match="departure_observed_at"):
        type(event).model_validate(
            {**event.model_dump(), "departure_observed_at": event.last_seen - timedelta(seconds=1)}
        )


def test_the_shipped_security_scenario_no_longer_claims_the_loiterer_left():
    scenario = load_scenario(SCENARIO_DIR / "security_events.yaml")
    with Store.open(":memory:") as store:
        PatrolEngine(scenario, store).run()
        alert = store.get_alert(ALERT_ID)
        person = store.get_event("EVT-P-001")
    assert person.close_reason is CloseReason.ZONE_CHANGE  # the drone flew on to the driveway
    assert alert.status is AlertStatus.ESCALATED and alert.resolved_at is None
