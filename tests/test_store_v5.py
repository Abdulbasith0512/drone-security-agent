"""Schema v5 and the bounded read paths: limits, counts and targeted lookups that keep the analyst's
queries independent of the size of the run (found and measured by the M13 benchmark)."""

import pytest
from conftest import SCENARIO_DIR

from drone_agent.agent.tools import AnalystTools
from drone_agent.eval.scale_bench import Stream
from drone_agent.ingest import ingest_observations, ingest_result
from drone_agent.models import AlertStatus, ObjectType, Severity, Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store
from drone_agent.store import schema as schema_module


@pytest.fixture(scope="module")
def security_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("v5") / "security.db"
    with Store.open(path) as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        build_events(s)
        analyze(s)
    return path


@pytest.fixture(scope="module")
def big_db(tmp_path_factory):
    """A few thousand generated observations: many events and alerts, so caps really bite."""
    path = tmp_path_factory.mktemp("v5big") / "big.db"
    with Store.open(path) as s:
        ingest_observations(s, "generated", [], list(Stream(3000)))
        build_events(s)
        analyze(s)
    return path


@pytest.fixture
def store(security_db):
    with Store.open(security_db) as s:
        yield s


@pytest.fixture
def big(big_db):
    with Store.open(big_db) as s:
        yield s


# ---- the migration ----------------------------------------------------------------------------
def test_schema_v5_adds_only_indexes():
    assert schema_module.SCHEMA_VERSION == 5
    assert all(stmt.startswith("CREATE INDEX") for stmt in schema_module.MIGRATIONS[5])


def test_a_v4_database_is_migrated_and_keeps_its_data(security_db, tmp_path):
    import shutil
    import sqlite3

    copy = tmp_path / "old.db"
    shutil.copy(security_db, copy)
    conn = sqlite3.connect(copy)
    for name in (
        "idx_events_first",
        "idx_alert_history_effective",
        "idx_alerts_rule",
        "idx_alerts_entity",
    ):
        conn.execute(f"DROP INDEX {name}")
    conn.execute("PRAGMA user_version = 4")
    conn.commit()
    conn.close()
    with Store.open(copy) as migrated:
        assert migrated.migrated_from == 4 and migrated.count_alerts() == 7
        names = {r["name"] for r in migrated._conn.execute("PRAGMA index_list(events)")}
        assert "idx_events_first" in names
        assert migrated._conn.execute("PRAGMA user_version").fetchone()[0] == 5


# ---- events -----------------------------------------------------------------------------------
def test_event_limit_keeps_the_earliest_and_the_count_is_exact(big):
    everything = big.query_events()
    assert len(everything) > 50
    first = big.query_events(limit=10)
    assert [e.event_id for e in first] == [e.event_id for e in everything[:10]]
    assert big.count_events_matching() == len(everything)
    filters = {"zone": Zone.MAIN_GATE, "object_type": ObjectType.PERSON}
    assert big.count_events_matching(**filters) == len(big.query_events(**filters))


def test_closed_and_open_filters_agree_with_the_events(store):
    events = store.query_events()
    assert store.count_events_matching(closed_only=True) == sum(not e.is_open for e in events)
    assert store.count_events_matching(open_only=True) == sum(e.is_open for e in events)


def test_a_time_window_selects_overlapping_events_only(big):
    everything = big.query_events()
    mid = everything[len(everything) // 2]
    window = big.query_events(start=mid.first_seen, end=mid.last_seen)
    assert mid.event_id in {e.event_id for e in window}
    assert all(e.last_seen >= mid.first_seen and e.first_seen <= mid.last_seen for e in window)
    assert big.count_events_matching(start=mid.first_seen, end=mid.last_seen) == len(window)


# ---- alerts -----------------------------------------------------------------------------------
def test_alert_limit_counts_and_group_by(big):
    everything = big.query_alerts()
    assert len(everything) > 30
    assert [a.alert_id for a in big.query_alerts(limit=5)] == [a.alert_id for a in everything[:5]]
    assert big.count_alerts_matching() == len(everything)
    assert big.count_alerts_matching(severity=Severity.HIGH) == sum(
        a.severity is Severity.HIGH for a in everything
    )
    by_rule = big.alert_counts_by("rule")
    assert sum(by_rule.values()) == len(everything)
    assert by_rule == {
        r: sum(a.rule_id == r for a in everything) for r in {a.rule_id for a in everything}
    }
    assert sum(big.alert_counts_by("status").values()) == len(everything)
    with pytest.raises(ValueError):
        big.alert_counts_by("title; DROP TABLE alerts")


def test_alerts_are_found_by_entity_and_by_event(store):
    entity_alerts = store.query_alerts(entity_id="ENT-V-001")
    assert [a.alert_id for a in entity_alerts] == ["ALR-REPEAT-ENT-V-001"]
    assert store.count_alerts_matching(entity_id="ENT-V-001") == 1
    cited = store.alerts_for_events(["EVT-P-003"])
    assert {a.rule_id for a in cited} == {"after_hours_presence", "restricted_zone_entry"}
    assert store.alerts_for_events([]) == []
    mapping = store.alert_ids_by_event(["EVT-P-003", "EVT-V-004"])
    assert set(mapping) == {"EVT-P-003"} and len(mapping["EVT-P-003"]) == 2


def test_status_filters_still_work_with_limits(store):
    unresolved = store.query_alerts(active_only=True, limit=3)
    assert len(unresolved) == 3 and all(a.status is not AlertStatus.RESOLVED for a in unresolved)


# ---- entities and the run span ----------------------------------------------------------------
def test_entity_limit_reads_links_only_for_the_returned_entities(big):
    every = big.list_entities()
    assert len(every) > 5
    limited = big.list_entities(limit=3)
    assert [e.entity_id for e in limited] == [e.entity_id for e in every[:3]]
    assert limited[0].event_ids == every[0].event_ids
    assert limited[0].link_scores == every[0].link_scores


def test_observation_span_matches_the_stored_extremes(big):
    span = big.observation_span()
    observations = big.query_observations()
    assert span == (observations[0].timestamp, observations[-1].timestamp)


# ---- the analyst's bounded tools --------------------------------------------------------------
def test_a_capped_timeline_is_exactly_the_prefix_of_the_full_timeline(big):
    tools = AnalystTools(big)
    full = tools.get_timeline(limit=40)
    assert full["truncated"] and full["returned"] == 40
    for k in (1, 2, 5, 13, 27):
        capped = tools.get_timeline(limit=k)["timeline"]
        assert capped == full["timeline"][:k], k


def test_the_timeline_total_counts_every_item_in_the_window(security_db):
    with Store.open(security_db) as store:
        tools = AnalystTools(store)
        everything = tools.get_timeline(limit=40)
        assert not everything["truncated"] and everything["total"] == everything["returned"]
        tiny = tools.get_timeline(limit=3)
        assert tiny["total"] == everything["total"] and tiny["returned"] == 3 and tiny["truncated"]


def test_tool_totals_are_exact_even_when_the_list_is_capped(big):
    tools = AnalystTools(big)
    events = tools.query_events(limit=7)
    assert events["returned"] == 7 and events["truncated"]
    assert events["total"] == big.count_events_matching()
    alerts = tools.list_alerts(limit=4)
    assert alerts["returned"] == 4 and alerts["total"] == big.count_alerts_matching()
    closed = tools.query_events(status="closed")
    assert closed["total"] == big.count_events_matching(closed_only=True)


def test_get_frame_lists_only_alerts_that_cite_its_events(store):
    tools = AnalystTools(store)
    frame = store.query_events(zone=Zone.RESTRICTED_ZONE)[0].frame_ids[0]
    alerts = tools.get_frame(frame)["alerts"]
    assert {a["rule_id"] for a in alerts} == {"after_hours_presence", "restricted_zone_entry"}
    assert tools.get_frame(1)["alerts"]  # the loitering person's first frame
