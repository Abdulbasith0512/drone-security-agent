from dataclasses import replace
from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.ingest import ingest_result
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import DuplicateRecordError, Store

SECURITY = SCENARIO_DIR / "security_events.yaml"
NORMAL = SCENARIO_DIR / "normal_day.yaml"


@pytest.fixture(scope="module")
def security_result():
    return run_scenario(load_scenario(SECURITY))


def run_cli(*argv) -> tuple[int, str]:
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=220))
    return code, out.getvalue()


def test_security_events_ingest_round_trips_exactly(store, security_result):
    summary = ingest_result(store, security_result)
    expected_objects = sum(len(o.detected_objects) for o in security_result.observations)
    assert (summary.frames, summary.observations, summary.telemetry_samples) == (27, 27, 288)
    assert (summary.objects, summary.stale_associations) == (expected_objects, 1)
    assert (store.count_frames(), store.count_observations()) == (27, 27)
    assert (store.count_detected_objects(), store.count_telemetry()) == (expected_objects, 288)

    stored = store.query_observations()
    assert (
        stored == security_result.observations
    )  # domain models rebuilt from rows, incl. telemetry
    assert store.get_observation_by_frame(18).telemetry.stale is True


def test_normal_day_ingest(store):
    result = run_scenario(load_scenario(NORMAL))
    summary = ingest_result(store, result)
    assert (summary.frames, summary.telemetry_samples, summary.stale_associations) == (11, 121, 0)
    assert store.query_observations() == result.observations


def test_scenario_queries_find_the_expected_evidence(store, security_result):
    ingest_result(store, security_result)
    ids = lambda **kw: sorted(o.frame_id for o in store.query_observations(**kw))  # noqa: E731
    assert ids(text="blue pickup") == [10, 11, 13, 14, 16, 17]
    assert ids(text="person main gate") == [1, 2, 3, 4, 6, 7, 8, 9]
    assert ids(text="vehicle warehouse") == []
    assert ids(text="restricted") == [22, 23, 24, 25]


def test_reingesting_the_same_scenario_is_rejected_atomically(store, security_result):
    ingest_result(store, security_result)
    before = (store.count_frames(), store.count_detected_objects(), store.count_telemetry())
    with pytest.raises(DuplicateRecordError):
        ingest_result(store, security_result)
    assert (store.count_frames(), store.count_detected_objects(), store.count_telemetry()) == before


def test_failed_ingest_leaves_nothing_behind(store, security_result):
    poisoned = replace(
        security_result,
        observations=[*security_result.observations, security_result.observations[0]],
    )
    with pytest.raises(DuplicateRecordError):
        ingest_result(store, poisoned)  # last observation duplicates frame 1
    assert (store.count_frames(), store.count_observations()) == (0, 0)
    assert (store.count_detected_objects(), store.count_telemetry()) == (0, 0)
    assert store.query_observations(text="person") == []


def test_ingest_into_new_store_after_failure_still_works(tmp_path, security_result):
    with Store.open(tmp_path / "a.db") as s:
        ingest_result(s, security_result)
        assert s.count_observations() == 27


# ---- CLI ------------------------------------------------------------------------------------
def test_cli_ingest_then_queries(tmp_path):
    db = tmp_path / "out" / "security.db"  # parent directory is created
    code, text = run_cli("ingest", SECURITY, "--db", db)
    assert code == 0
    for expected in (
        "security_events",
        "Frames ingested",
        "27",
        "Telemetry samples",
        "288",
        "Stale telemetry associations",
        "1",
    ):
        assert expected in text

    code, text = run_cli("query", "--db", db, "--zone", "main_gate")
    assert code == 0 and "12 observation(s)" in text
    assert "obs-1" in text and "Blue pickup truck stopped" in text

    code, text = run_cli("query", "--db", db, "--object", "person")
    assert code == 0 and "13 observation(s)" in text and "restricted" in text

    code, text = run_cli("query", "--db", db, "--search", "blue pickup")
    assert code == 0 and "6 observation(s)" in text

    code, text = run_cli("query", "--db", db, "--zone", "warehouse", "--search", "vehicle")
    assert code == 0 and "0 observation(s)" in text


def test_cli_ingest_twice_reports_duplicate_and_changes_nothing(tmp_path):
    db = tmp_path / "s.db"
    assert run_cli("ingest", SECURITY, "--db", db)[0] == 0
    code, text = run_cli("ingest", SECURITY, "--db", db)
    assert code == 1 and "already" in text and "nothing" in text
    with Store.open(db) as s:
        assert s.count_observations() == 27


def test_cli_explain_and_bad_input(tmp_path):
    db = tmp_path / "s.db"
    run_cli("ingest", SECURITY, "--db", db)
    code, text = run_cli("query", "--db", db, "--zone", "main_gate", "--explain")
    assert code == 0 and "idx_observations_zone_ts" in text
    code, text = run_cli("query", "--db", db, "--search", "  ")
    assert code == 1 and "error" in text
    code, text = run_cli("ingest", tmp_path / "missing.yaml", "--db", db)
    assert code == 2


def test_cli_errors_containing_rich_markup_are_printed_literally(tmp_path):
    # The validation error echoes the bad value. Unescaped, "[/oops]" is a Rich closing tag.
    bad = tmp_path / "bad.yaml"
    lines = [
        "name: t",
        'start: "2026-09-29T10:00:00+05:30"',
        "telemetry: {patrol: [{zone: main_gate, duration_s: 60}]}",
        'frames: [{frame_id: 1, timestamp_offset: 0, zone: "[/oops]", description: d}]',
    ]
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    code, text = run_cli("ingest", bad, "--db", tmp_path / "x.db")
    assert code == 2 and "[/oops]" in text
