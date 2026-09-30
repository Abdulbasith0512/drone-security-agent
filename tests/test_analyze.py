import sqlite3
from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.ingest import ingest_result
from drone_agent.models import AlertStatus, ObjectType, Severity, TransitionKind, Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import AlertManager, RuleEngine, analyze, load_rules_config
from drone_agent.rules.analyze import data_end
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store, StoreError

SECURITY = SCENARIO_DIR / "security_events.yaml"
NORMAL = SCENARIO_DIR / "normal_day.yaml"


def prepared(path, store):
    ingest_result(store, run_scenario(load_scenario(path)))
    build_events(store)
    return analyze(store)


def run_cli(*argv) -> tuple[int, str]:
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=240))
    return code, out.getvalue()


# ---- security_events ------------------------------------------------------------------------
def test_security_events_gate_person_loiters(store):
    prepared(SECURITY, store)
    (alert,) = store.query_alerts(rule_id="loitering")
    (event,) = [store.get_event(e) for e in alert.event_ids]
    assert (event.object_type, event.zone) == (ObjectType.PERSON, Zone.MAIN_GATE)
    assert alert.facts["duration_seconds"] == 480 and alert.facts["threshold_seconds"] == 120
    assert alert.facts["critical_after_seconds"] == 300 and alert.facts["zone"] == "main_gate"
    assert alert.frame_ids == [1, 2, 3, 4, 6, 7, 8, 9]  # the tolerated frame 5 did not matter


def test_security_events_loitering_alert_escalates_and_stays_open_when_the_camera_moves_on(store):
    prepared(SECURITY, store)
    (alert,) = store.query_alerts(rule_id="loitering")
    assert alert.severity is Severity.CRITICAL  # 480 s passed the 300 s critical rung
    assert [(s.label, s.severity) for s in alert.ladder] == [
        ("warning", Severity.MEDIUM),
        ("critical", Severity.CRITICAL),
    ]
    assert (alert.triggered_at - alert.first_seen).total_seconds() == 120
    assert [t.kind for t in alert.history] == [
        TransitionKind.RAISED,
        TransitionKind.ESCALATED,
    ]
    # The event ended because the drone flew on to the driveway, not because the drone saw the
    # gate empty, so there is no evidence the person left: the alert stays open (escalated).
    assert alert.status is AlertStatus.ESCALATED and alert.resolved_at is None
    assert alert.confidence is not None and 0.9 <= alert.confidence <= 0.95
    assert alert.recommended_action and "Dispatch security" in alert.recommended_action


def test_security_events_blue_pickup_entity_triggers_repeat_vehicle_with_careful_wording(store):
    prepared(SECURITY, store)
    (alert,) = store.query_alerts(rule_id="repeat_vehicle")
    entity = store.get_entity(alert.entity_id)
    assert entity.label == "similar blue pickup" and entity.appearance_count == 3
    assert alert.event_ids == entity.event_ids
    assert alert.severity is Severity.LOW
    assert "similar blue pickup was observed in 3 separate appearances" in alert.message
    assert alert.facts["match_level"] == "coarse_match"
    assert alert.frame_ids == [10, 11, 13, 14, 16, 17]


def test_security_events_white_sedan_is_not_a_repeat_vehicle(store):
    prepared(SECURITY, store)
    sedan = next(
        e for e in store.query_events(object_type=ObjectType.VEHICLE) if e.frame_ids == [26]
    )
    assert not any(sedan.event_id in a.event_ids for a in store.query_alerts())
    assert sedan.entity_id not in {a.entity_id for a in store.query_alerts()}


def test_security_events_after_hours_is_evaluated_for_each_person_event(store):
    prepared(SECURITY, store)
    alerts = store.query_alerts(rule_id="after_hours_presence")
    assert [a.frame_ids for a in alerts] == [[1, 2, 3, 4, 6, 7, 8, 9], [19, 20], [22, 23, 24]]
    warehouse = next(a for a in alerts if a.zone is Zone.WAREHOUSE)
    assert warehouse.facts["event_local_time"] == "00:11:00"  # after midnight, inside 22:00-06:00
    assert (warehouse.facts["closed_from"], warehouse.facts["closed_until"]) == ("22:00", "06:00")
    assert warehouse.triggered_at.day == 30  # the simulated clock crossed midnight


def test_security_events_restricted_zone_entry_is_one_alert_for_the_continuous_event(store):
    prepared(SECURITY, store)
    (alert,) = store.query_alerts(rule_id="restricted_zone_entry")
    assert alert.severity is Severity.HIGH and alert.zone is Zone.RESTRICTED_ZONE
    assert alert.frame_ids == [22, 23, 24]  # three frames, one alert


def test_security_events_overlap_policy_and_actual_totals(store):
    summary = prepared(SECURITY, store)
    assert summary.by_rule == {
        "after_hours_presence": 3,
        "loitering": 1,
        "repeat_vehicle": 1,
        "restricted_zone_entry": 1,
        "signal_loss": 1,  # the scenario's 140 s telemetry dropout is a real link loss
    }
    assert summary.alerts == 7 and (summary.events_analyzed, summary.entities_analyzed) == (7, 2)
    assert summary.by_severity == {"critical": 1, "high": 2, "low": 1, "medium": 3}
    # a night-time loiterer and a restricted-zone intruder each have TWO alerts (different risks)
    per_event: dict[str, list[str]] = {}
    for alert in store.query_alerts():
        for event_id in alert.event_ids:
            per_event.setdefault(event_id, []).append(alert.rule_id)
    assert sorted(per_event["EVT-P-001"]) == ["after_hours_presence", "loitering"]
    assert sorted(per_event["EVT-P-003"]) == ["after_hours_presence", "restricted_zone_entry"]
    assert per_event["EVT-P-002"] == ["after_hours_presence"]


def test_normal_day_produces_no_alerts(store):
    summary = prepared(NORMAL, store)
    assert (summary.events_analyzed, summary.entities_analyzed) == (5, 2)
    assert summary.alerts == 0 and store.query_alerts() == []
    assert summary.by_rule == {} and summary.by_severity == {}


# ---- persistence, idempotency, integrity ----------------------------------------------------
def test_alerts_round_trip_through_sqlite_exactly(store):
    prepared(SECURITY, store)
    now = data_end(store)
    snapshots = RuleEngine().run(
        store.query_events(), store.list_entities(), store.list_telemetry(), now=now
    )
    manager = AlertManager()
    manager.reconcile(snapshots, now, live=False)
    expected = manager.alerts
    assert store.query_alerts() == expected
    assert store.get_alert(expected[0].alert_id) == expected[0]
    assert store.get_alert("ALR-NOPE") is None
    first = expected[0]
    assert first.triggered_at.utcoffset().total_seconds() == 5.5 * 3600  # timezone preserved


def test_alert_ids_are_deterministic_across_databases(tmp_path):
    ids = []
    for name in ("a.db", "b.db"):
        with Store.open(tmp_path / name) as s:
            prepared(SECURITY, s)
            ids.append([a.alert_id for a in s.query_alerts()])
    assert ids[0] == ids[1]
    assert ids[0] == [
        "ALR-AFTERHOURS-EVT-P-001",
        "ALR-LOITER-EVT-P-001",
        "ALR-REPEAT-ENT-V-001",
        "ALR-SIGNAL-20260929T183630Z",  # no telemetry for 30 s after the last good sample
        "ALR-AFTERHOURS-EVT-P-002",
        "ALR-AFTERHOURS-EVT-P-003",
        "ALR-RESTRICTED-EVT-P-003",
    ]


def test_analyze_twice_never_duplicates_alerts(store):
    first = prepared(SECURITY, store)
    before = store.query_alerts()
    second = analyze(store)
    assert store.query_alerts() == before
    assert store.count_alerts() == first.alerts == 7
    assert (first.replaced_alerts, second.replaced_alerts) == (0, 7)
    links = store._conn.execute("SELECT COUNT(*) FROM alert_events").fetchone()[0]
    assert links == sum(len(a.event_ids) for a in before)


def test_changing_the_rules_config_replaces_the_alerts(store, tmp_path):
    prepared(SECURITY, store)
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "loitering: {zones: {main_gate: {threshold_seconds: 600}}}\n"
        "after_hours_presence: {enabled: false}\n"
        "repeat_vehicle: {minimum_appearances: 4}\n",
        encoding="utf-8",
    )
    summary = analyze(store, load_rules_config(rules))
    # 480 s < 600 s; 3 appearances < 4; the telemetry outage and the restricted zone remain
    assert summary.by_rule == {"restricted_zone_entry": 1, "signal_loss": 1}
    assert summary.replaced_alerts == 7 and store.count_alerts() == 2


def test_rebuilding_events_discards_alerts_that_cited_them(store):
    prepared(SECURITY, store)
    summary = build_events(store)
    assert summary.discarded_alerts == 7 and store.count_alerts() == 0
    assert analyze(store).alerts == 7  # regenerated from the rebuilt events


def test_alert_evidence_chain_is_intact_down_to_frames(store):
    prepared(SECURITY, store)
    assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    for alert in store.query_alerts():
        frames: set[int] = set()
        for event_id in alert.event_ids:
            event = store.get_event(event_id)
            observations = store.get_event_observations(event_id)
            assert [o.frame_id for o in observations] == event.frame_ids
            frames |= set(event.frame_ids)
        assert alert.frame_ids == sorted(frames)
        assert alert.first_seen <= alert.triggered_at <= alert.last_seen
        if alert.entity_id:
            assert store.get_entity(alert.entity_id).event_ids == alert.event_ids


def test_foreign_keys_protect_alert_links(store):
    prepared(SECURITY, store)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._conn.execute(
            "INSERT INTO alert_events (alert_id, event_id) VALUES ('ALR-LOITER-EVT-P-001', 'EVT-X')"
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._conn.execute(
            "INSERT INTO alert_events (alert_id, event_id) VALUES ('ALR-NOPE', 'EVT-P-001')"
        )
    with pytest.raises(sqlite3.IntegrityError):  # the severity vocabulary is enforced in the schema
        store._conn.execute("UPDATE alerts SET severity = 'urgent'")


def test_failed_alert_replace_keeps_previous_alerts(store):
    prepared(SECURITY, store)
    before = store.query_alerts()
    dangling_event = before[0].model_copy(update={"event_ids": ["EVT-NOPE"]})
    dangling_entity = before[0].model_copy(update={"entity_id": "ENT-NOPE"})
    for bad in (dangling_event, dangling_entity):
        with pytest.raises(StoreError):
            store.replace_alerts([bad])
        assert store.query_alerts() == before


def test_alert_filters(store):
    prepared(SECURITY, store)
    assert len(store.query_alerts(rule_id="after_hours_presence")) == 3
    assert [a.rule_id for a in store.query_alerts(severity=Severity.HIGH)] == [
        "signal_loss",
        "restricted_zone_entry",
    ]
    # 2 for the gate person + repeat vehicle + the link loss, which began while at the gate
    assert len(store.query_alerts(zone=Zone.MAIN_GATE)) == 4
    assert store.query_alerts(rule_id="loitering", zone=Zone.WAREHOUSE) == []


def test_analyze_on_an_empty_database_is_harmless(store):
    summary = analyze(store)
    assert (summary.events_analyzed, summary.alerts) == (0, 0)


# ---- CLI ------------------------------------------------------------------------------------
@pytest.fixture
def cli_db(tmp_path):
    db = tmp_path / "security.db"
    for step in (
        ["ingest", SECURITY, "--db", db],
        ["build-events", "--db", db],
    ):
        assert run_cli(*step)[0] == 0
    return db


def test_cli_analyze_summary_and_rerun(cli_db):
    code, text = run_cli("analyze", "--db", cli_db)
    assert code == 0
    for expected in (
        "Events analyzed",
        "Alerts generated",
        "7",
        "after_hours_presence: 3",
        "loitering: 1",
        "signal_loss: 1",
        "critical: 1",
        "high: 2",
        "medium: 3",
        "low: 1",
        "no LLM",
    ):
        assert expected in text
    code, text = run_cli("analyze", "--db", cli_db)
    assert code == 0 and "Replaced previous alerts" in text and "7" in text


def test_cli_alert_listing_and_filters(cli_db):
    run_cli("analyze", "--db", cli_db)
    code, text = run_cli("alerts", "--db", cli_db)
    assert code == 0 and "7 alert(s)" in text
    assert "ALR-LOITER-EVT-P-001" in text and "Person loitering at main_gate" in text
    code, text = run_cli("alerts", "--db", cli_db, "--severity", "high")
    assert "2 alert(s)" in text and "ALR-RESTRICTED-EVT-P-003" in text
    assert "ALR-SIGNAL-20260929T183630Z" in text
    code, text = run_cli("alerts", "--db", cli_db, "--rule", "repeat_vehicle")
    assert "1 alert(s)" in text and "Repeat vehicle: similar blue pickup" in text


def test_cli_show_loitering_alert_exposes_facts_and_evidence(cli_db):
    run_cli("analyze", "--db", cli_db)
    code, text = run_cli("alerts", "--db", cli_db, "--show", "ALR-LOITER-EVT-P-001")
    assert code == 0
    for expected in (
        "severity:",
        "critical",
        "reason:",
        "480s",
        "120s",
        "300s",
        "duration_seconds",
        "threshold_seconds",
        "EVT-P-001",
        "obs-9",
        "obs-1 ",
        "trace: ALR-LOITER-EVT-P-001 -> events EVT-P-001 -> frames 1-4,6-9",
    ):
        assert expected in text
    assert "obs-5" not in text  # the tolerated missed frame is not evidence


def test_cli_show_repeat_vehicle_alert_keeps_the_uncertainty(cli_db):
    run_cli("analyze", "--db", cli_db)
    code, text = run_cli("alerts", "--db", cli_db, "--show", "ALR-REPEAT-ENT-V-001")
    assert code == 0
    assert "similar blue pickup was observed in 3 separate appearances" in text
    assert "coarse_match" in text and "does not confirm" in text
    assert "EVT-V-001" in text and "EVT-V-002" in text and "EVT-V-003" in text
    assert "ENT-V-001" in text


def test_cli_alerts_before_analyze_unknown_alert_and_bad_rules(cli_db, tmp_path):
    code, text = run_cli("alerts", "--db", cli_db)
    assert code == 0 and "no alerts found" in text
    assert run_cli("alerts", "--db", cli_db, "--show", "ALR-NOPE")[0] == 1
    bad = tmp_path / "bad.yaml"
    bad.write_text("repeat_vehicle: {minimum_appearances: 1}\n", encoding="utf-8")
    code, text = run_cli("analyze", "--db", cli_db, "--rules", bad)
    assert code == 1 and "invalid rules config" in text
    good = tmp_path / "good.yaml"
    good.write_text("restricted_zone_entry: {enabled: false}\n", encoding="utf-8")
    code, text = run_cli("analyze", "--db", cli_db, "--rules", good)
    assert code == 0 and "restricted_zone_entry" not in text


def test_cli_analyze_before_build_events_prints_a_hint(tmp_path):
    db = tmp_path / "empty.db"
    run_cli("ingest", SECURITY, "--db", db)
    code, text = run_cli("analyze", "--db", db)
    assert code == 0 and "no events to analyze" in text


def test_cli_build_events_reports_discarded_alerts(cli_db):
    run_cli("analyze", "--db", cli_db)
    code, text = run_cli("build-events", "--db", cli_db)
    assert code == 0 and "Alerts discarded" in text and "run `analyze` again" in text
