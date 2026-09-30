"""CLI: run, scenarios, alert lifecycle actions."""

import json
from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.models import AlertStatus
from drone_agent.store import Store


def run_cli(*argv) -> tuple[int, str]:
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200))
    return code, out.getvalue()


def test_scenarios_lists_what_ships():
    code, text = run_cli("scenarios")
    assert code == 0 and "security_events" in text and "normal_day" in text
    assert "python -m drone_agent run --scenario" in text


def test_run_processes_a_named_scenario_and_prints_the_lifecycle_summary(tmp_path):
    db = tmp_path / "run.db"
    code, text = run_cli("run", "--scenario", "security_events", "--db", db, "--no-live", "--quiet")
    assert code == 0
    for expected in (
        "Patrol summary: security_events",
        "SIMULATED",
        "7 alert(s) with lifecycle",
        "ALR-LOITER-EVT-P-001",
        "medium > critical",  # the ladder
        "Recommended actions",
        "SCRIPTED",
    ):
        assert expected in text
    with Store.open(db) as store:
        assert store.count_alerts() == 7 and store.count_events() == 7
        assert store.list_agent_runs("patrol")[0]["status"] == "finished"


def test_run_accepts_a_path_and_prints_a_live_log_unless_quiet(tmp_path):
    code, text = run_cli(
        "run",
        "--scenario",
        SCENARIO_DIR / "normal_day.yaml",
        "--db",
        tmp_path / "n.db",
        "--no-live",
    )
    assert code == 0 and "PATROL STARTED" in text and "FRAME" in text and "0 alert" not in text
    assert "no alerts were raised" in text


def test_run_refuses_to_overwrite_unless_asked(tmp_path):
    db = tmp_path / "again.db"
    assert run_cli("run", "--scenario", "normal_day", "--db", db, "--quiet")[0] == 0
    code, text = run_cli("run", "--scenario", "normal_day", "--db", db, "--quiet")
    assert code == 1 and "already exists" in text and "--overwrite" in text
    assert run_cli("run", "--scenario", "normal_day", "--db", db, "--quiet", "--overwrite")[0] == 0


def test_an_unknown_scenario_names_the_available_ones(tmp_path):
    code, text = run_cli("run", "--scenario", "moon_base", "--db", tmp_path / "x.db")
    assert code == 2 and "unknown scenario 'moon_base'" in text and "security_events" in text


def test_a_bad_rules_file_fails_clearly(tmp_path):
    bad = tmp_path / "rules.yaml"
    bad.write_text("loitering: {typo: 1}\n", encoding="utf-8")
    code, text = run_cli(
        "run", "--scenario", "normal_day", "--db", tmp_path / "x.db", "--rules", bad
    )
    assert code == 2 and "invalid rules config" in text


def test_no_sampler_analyses_every_frame_and_json_summary_is_written(tmp_path):
    summary = tmp_path / "out" / "summary.json"
    code, text = run_cli(
        "run", "--scenario", "security_events", "--db", tmp_path / "s.db", "--quiet",
        "--no-sampler", "--json", summary,
    )  # fmt: skip
    assert code == 0
    data = json.loads(summary.read_text())
    assert data["model_calls"] == data["frames"] == 27 and data["avoided_pct"] == 0.0
    assert data["sampler"]["by_reason"] == {"sampler_disabled": 27}


def test_a_stricter_rules_file_changes_what_a_run_alerts_on(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "loitering: {zones: {main_gate: {threshold_seconds: 600}}}\nsignal_loss: {enabled: false}\n",
        encoding="utf-8",
    )
    db = tmp_path / "strict.db"
    code, _ = run_cli(
        "run", "--scenario", "security_events", "--db", db, "--quiet", "--rules", rules
    )
    assert code == 0
    with Store.open(db) as store:
        assert {a.rule_id for a in store.query_alerts()} == {
            "after_hours_presence",
            "repeat_vehicle",
            "restricted_zone_entry",
        }


# ---- operator actions on alerts ----------------------------------------------------------------
def make_db(tmp_path):
    db = tmp_path / "ops.db"
    assert run_cli("run", "--scenario", "security_events", "--db", db, "--quiet")[0] == 0
    return db


def test_an_operator_can_acknowledge_then_resolve_and_history_records_who(tmp_path):
    db = make_db(tmp_path)
    alert_id = "ALR-RESTRICTED-EVT-P-003"
    at = "2026-09-30T00:20:00+05:30"
    code, text = run_cli("alert-action", "--db", db, alert_id, "acknowledge", "--at", at)
    assert code == 0 and "acknowledged by operator" in text and "status acknowledged" in text
    code, text = run_cli(
        "alert-action", "--db", db, alert_id, "resolve", "--at", at, "--reason", "guard attended"
    )
    assert code == 0 and "status resolved" in text
    with Store.open(db) as store:
        alert = store.get_alert(alert_id)
        assert alert.status is AlertStatus.RESOLVED and alert.resolved_at is not None
        acted = [h for h in alert.history if h.actor == "operator"]
        assert [h.kind.value for h in acted] == ["acknowledged", "resolved"]
        assert acted[-1].reason == "guard attended"


def test_illegal_operator_actions_exit_with_a_clear_message(tmp_path):
    db = make_db(tmp_path)
    assert run_cli("alert-action", "--db", db, "ALR-NOPE", "acknowledge")[0] == 1
    run_cli("alert-action", "--db", db, "ALR-RESTRICTED-EVT-P-003", "acknowledge")
    code, text = run_cli("alert-action", "--db", db, "ALR-RESTRICTED-EVT-P-003", "acknowledge")
    assert code == 1 and "already acknowledged" in text
    code, text = run_cli("alert-action", "--db", db, "ALR-LOITER-EVT-P-001", "acknowledge")
    assert code == 1 and "already resolved" in text  # the person left: it resolved itself
    with pytest.raises(SystemExit) as exc:  # argparse rejects a time without a UTC offset
        run_cli("alert-action", "--db", db, "ALR-X", "resolve", "--at", "2026-09-30T00:00")
    assert exc.value.code == 2
