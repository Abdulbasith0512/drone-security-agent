import sqlite3
from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.ingest import ingest_result
from drone_agent.models import CloseReason, MatchLevel, ObjectType, Zone
from drone_agent.pipeline import EntityLinker, EventConfig, aggregate_observations, build_events
from drone_agent.pipeline.config import AggregationSettings
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store, StoreError
from drone_agent.store import schema as schema_module

SECURITY = SCENARIO_DIR / "security_events.yaml"
NORMAL = SCENARIO_DIR / "normal_day.yaml"


def built_store(scenario_path, store):
    ingest_result(store, run_scenario(load_scenario(scenario_path)))
    return build_events(store)


def run_cli(*argv) -> tuple[int, str]:
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=240))
    return code, out.getvalue()


# ---- security_events ground truth -----------------------------------------------------------
def test_security_events_gate_loiterer_is_one_event_despite_the_missed_detection(store):
    summary = built_store(SECURITY, store)
    gate = store.query_events(zone=Zone.MAIN_GATE, object_type=ObjectType.PERSON)
    assert len(gate) == 1
    ev = gate[0]
    assert ev.frame_ids == [1, 2, 3, 4, 6, 7, 8, 9]  # frame 5 was the flicker
    assert ev.duration_seconds == 480
    assert (ev.first_seen.hour, ev.first_seen.minute) == (23, 41)
    assert ev.attributes == {"clothing": "dark jacket"}
    assert ev.entity_id is None
    assert summary.observations == 27 and summary.detections == 20


def test_security_events_blue_pickup_is_three_events_but_one_similar_vehicle_entity(store):
    built_store(SECURITY, store)
    vehicles = store.query_events(object_type=ObjectType.VEHICLE)
    pickups = [e for e in vehicles if e.attributes["body_type"] == "pickup"]
    assert [(e.zone, e.frame_ids) for e in pickups] == [
        (Zone.PARKING_AREA, [10, 11]),
        (Zone.DRIVEWAY, [13, 14]),
        (Zone.MAIN_GATE, [16, 17]),
    ]
    assert {e.attributes["color"] for e in pickups} == {"blue", "dark blue"}  # raw text kept
    (entity_id,) = {e.entity_id for e in pickups}
    entity = store.get_entity(entity_id)
    assert entity.appearance_count == 3
    assert entity.match_level is MatchLevel.COARSE
    assert entity.label == "similar blue pickup"
    assert entity.event_ids == [e.event_id for e in pickups]


def test_security_events_white_sedan_is_not_the_blue_pickup(store):
    built_store(SECURITY, store)
    (sedan,) = [
        e for e in store.query_events(object_type=ObjectType.VEHICLE) if e.frame_ids == [26]
    ]
    pickup_entity = store.get_event(sedan.event_id).entity_id
    other = store.query_events(entity_id="ENT-V-001")
    assert pickup_entity == "ENT-V-002"
    assert sedan.event_id not in {e.event_id for e in other}
    assert store.get_entity("ENT-V-002").appearance_count == 1


def test_security_events_people_events_are_separate_and_never_identity_linked(store):
    built_store(SECURITY, store)
    people = store.query_events(object_type=ObjectType.PERSON)
    assert [(e.zone, e.frame_ids) for e in people] == [
        (Zone.MAIN_GATE, [1, 2, 3, 4, 6, 7, 8, 9]),
        (Zone.WAREHOUSE, [19, 20]),
        (Zone.RESTRICTED_ZONE, [22, 23, 24]),
    ]
    assert all(e.entity_id is None for e in people)
    assert {e.entity_type for e in store.list_entities()} == {ObjectType.VEHICLE}


def test_security_events_metrics(store):
    s = built_store(SECURITY, store)
    assert (s.events, s.person_events, s.vehicle_events, s.other_events) == (7, 3, 4, 0)
    assert (s.entities, s.multi_appearance_entities) == (2, 1)
    assert s.compression_ratio == pytest.approx(20 / 7)
    # every detection belongs to exactly one event (nothing lost, nothing double-counted)
    assert sum(e.evidence_count for e in store.query_events()) == s.detections
    assert (s.replaced_events, s.replaced_entities) == (0, 0)


# ---- normal_day -----------------------------------------------------------------------------
def test_normal_day_events_are_sensible_and_unrelated_things_are_not_merged(store):
    s = built_store(NORMAL, store)
    events = store.query_events()
    assert [(e.object_type.value, e.zone.value, e.frame_ids) for e in events] == [
        ("vehicle", "main_gate", [1, 2, 3]),  # delivery van
        ("person", "main_gate", [2, 3]),  # courier: a separate event from the van
        ("vehicle", "parking_area", [5, 6, 7]),  # staff hatchback
        ("person", "parking_area", [6]),  # staff member walking past
        ("person", "warehouse", [8, 9]),  # same clothing, different zone: not merged
    ]
    assert max(e.duration_seconds for e in events) == 120
    assert (s.entities, s.multi_appearance_entities) == (2, 0)
    assert s.detections == 11 and s.events == 5


# ---- persistence ----------------------------------------------------------------------------
def test_events_and_entities_round_trip_through_sqlite(store):
    result = run_scenario(load_scenario(SECURITY))
    ingest_result(store, result)
    build_events(store)
    events, entities = EntityLinker().link(aggregate_observations(result.observations))
    assert store.query_events() == sorted(events, key=lambda e: (e.first_seen, e.event_id))
    assert store.list_entities() == entities
    first = store.query_events()[0]
    assert first.first_seen.utcoffset().total_seconds() == 5.5 * 3600  # timezone preserved
    assert store.get_event("nope") is None and store.get_entity("nope") is None


def test_evidence_chain_event_to_observations_to_frames_is_intact(store):
    built_store(SECURITY, store)
    ev = store.query_events(zone=Zone.MAIN_GATE, object_type=ObjectType.PERSON)[0]
    observations = store.get_event_observations(ev.event_id)
    assert [o.observation_id for o in observations] == ev.observation_ids
    assert [o.frame_id for o in observations] == ev.frame_ids
    assert all(o.zone is Zone.MAIN_GATE for o in observations)
    assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []
    # reverse lookup: which events used frame 10 (observation obs-10)?
    rows = store._conn.execute(
        "SELECT event_id FROM event_observations WHERE observation_id = 'obs-10'"
    ).fetchall()
    assert [r[0] for r in rows] == ["EVT-V-001"]


def test_foreign_keys_protect_evidence_links(store):
    built_store(SECURITY, store)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._conn.execute(
            "INSERT INTO event_observations (event_id, observation_id, position)"
            " VALUES ('EVT-P-001', 'obs-does-not-exist', 99)"
        )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._conn.execute(
            "INSERT INTO entity_events (entity_id, event_id) VALUES ('ENT-V-999', 'EVT-P-001')"
        )


def test_failed_replace_leaves_previous_derived_data_untouched(store):
    built_store(SECURITY, store)
    before = (store.query_events(), store.list_entities())
    good = store.query_events()[0]
    bad = good.model_copy(update={"observation_ids": ["obs-missing"]})
    with pytest.raises(StoreError):
        store.replace_derived([bad], [])
    assert (store.query_events(), store.list_entities()) == before


def test_replace_rejects_inconsistent_entity_membership(store):
    built_store(SECURITY, store)
    events = store.query_events()
    entities = store.list_entities()
    with pytest.raises(StoreError, match="disagrees"):
        store.replace_derived([e.model_copy(update={"entity_id": None}) for e in events], entities)


# ---- idempotency ----------------------------------------------------------------------------
def test_build_events_twice_replaces_and_never_duplicates(store):
    first = built_store(SECURITY, store)
    events_1, entities_1 = store.query_events(), store.list_entities()
    second = build_events(store)
    assert (store.query_events(), store.list_entities()) == (events_1, entities_1)
    assert store.count_events() == first.events == 7
    assert (second.replaced_events, second.replaced_entities) == (7, 2)
    evidence_rows = store._conn.execute("SELECT COUNT(*) FROM event_observations").fetchone()[0]
    assert evidence_rows == sum(e.evidence_count for e in events_1)


def test_rebuilding_with_a_different_config_replaces_the_old_events(store):
    built_store(SECURITY, store)
    config = EventConfig(aggregation=AggregationSettings(max_missed_frames=0))
    build_events(store, config)  # the flicker frame now splits the gate person
    gate = store.query_events(zone=Zone.MAIN_GATE, object_type=ObjectType.PERSON)
    assert [e.frame_ids for e in gate] == [[1, 2, 3, 4], [6, 7, 8, 9]]
    assert gate[0].close_reason is CloseReason.MISSED_FRAMES


def test_build_on_empty_database_produces_nothing(store):
    s = build_events(store)
    assert (s.observations, s.events, s.entities, s.compression_ratio) == (0, 0, 0, None)


# ---- schema versioning ----------------------------------------------------------------------
def make_v1_database(path):
    conn = sqlite3.connect(path)
    for statement in schema_module.MIGRATIONS[1]:
        conn.execute(statement)
    conn.execute("PRAGMA user_version = 1")
    conn.execute(
        "INSERT INTO frames VALUES (1, '2026-09-29T23:41:00+05:30', '2026-09-29T18:11:00.000000Z',"
        " 'scripted', 'main_gate')"
    )
    conn.execute(
        "INSERT INTO observations (observation_id, frame_id, timestamp, ts_utc, source, zone,"
        " description) VALUES ('obs-1', 1, '2026-09-29T23:41:00+05:30',"
        " '2026-09-29T18:11:00.000000Z', 'scripted', 'main_gate', 'old data')"
    )
    conn.commit()
    conn.close()


def test_m2_database_is_upgraded_in_place_without_losing_data(tmp_path):
    path = tmp_path / "m2.db"
    make_v1_database(path)
    with Store.open(path) as store:
        assert store.migrated_from == 1
        assert store.get_observation("obs-1").description == "old data"
        assert store.count_events() == 0 and store.count_entities() == 0
        assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 2
    with Store.open(path) as store:
        assert store.migrated_from is None  # already current


def test_database_from_a_newer_version_is_refused(tmp_path):
    path = tmp_path / "future.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(StoreError, match="newer"):
        Store.open(path)


# ---- CLI ------------------------------------------------------------------------------------
@pytest.fixture
def cli_db(tmp_path):
    db = tmp_path / "security.db"
    assert run_cli("ingest", SECURITY, "--db", db)[0] == 0
    return db


def test_cli_build_events_summary_and_rebuild(cli_db):
    code, text = run_cli("build-events", "--db", cli_db)
    assert code == 0
    for expected in (
        "Observations analyzed",
        "27",
        "20 detections -> 7 events = 2.86",
        "person 3",
        "vehicle 4",
        "Multi-appearance entities",
        "0 events, 0 entities",
    ):
        assert expected in text
    code, text = run_cli("build-events", "--db", cli_db)
    assert code == 0 and "7 events, 2 entities" in text  # replaced, not duplicated


def test_cli_events_listing_and_filters(cli_db):
    run_cli("build-events", "--db", cli_db)
    code, text = run_cli("events", "--db", cli_db)
    assert code == 0 and "7 event(s)" in text
    assert "EVT-P-001" in text and "8 obs: 1-4,6-9" in text and "ENT-V-001" in text
    code, text = run_cli("events", "--db", cli_db, "--zone", "main_gate", "--object", "person")
    assert "1 event(s)" in text and "EVT-P-001" in text and "EVT-V-003" not in text
    code, text = run_cli("events", "--db", cli_db, "--entity", "ENT-V-001")
    assert "3 event(s)" in text


def test_cli_event_evidence_shows_frames_and_the_tolerated_gap(cli_db):
    run_cli("build-events", "--db", cli_db)
    code, text = run_cli("events", "--db", cli_db, "--evidence", "EVT-P-001")
    assert code == 0
    assert text.count("EVIDENCE") == 8
    assert "no detection (gap tolerated)" in text and "obs-5" in text
    assert "closed by: zone_change" in text
    code, text = run_cli("events", "--db", cli_db, "--evidence", "EVT-P-999")
    assert code == 1 and "no event" in text


def test_cli_entity_evidence_uses_careful_wording(cli_db):
    run_cli("build-events", "--db", cli_db)
    code, text = run_cli("entities", "--db", cli_db)
    assert code == 0 and "similar blue pickup" in text and "similar white sedan" in text
    code, text = run_cli("entities", "--db", cli_db, "--show", "ENT-V-001")
    assert code == 0
    assert "3 appearance(s)" in text and "COARSE match" in text
    assert "EVT-V-001" in text and "EVT-V-002" in text and "EVT-V-003" in text
    assert "does not prove it is the same physical vehicle" in text
    assert run_cli("entities", "--db", cli_db, "--show", "ENT-V-404")[0] == 1


def test_cli_events_before_build_and_bad_config(cli_db, tmp_path):
    code, text = run_cli("events", "--db", cli_db)
    assert code == 0 and "no events found" in text
    bad = tmp_path / "bad.yaml"
    bad.write_text("aggregation: {max_gap_seconds: 0}\n", encoding="utf-8")
    code, text = run_cli("build-events", "--db", cli_db, "--config", bad)
    assert code == 1 and "invalid config" in text


def test_cli_custom_config_changes_the_result(cli_db, tmp_path):
    strict = tmp_path / "strict.yaml"
    strict.write_text("aggregation: {max_missed_frames: 0}\n", encoding="utf-8")
    code, text = run_cli("build-events", "--db", cli_db, "--config", strict)
    assert code == 0 and "20 detections -> 8 events" in text


def test_cli_prints_markup_like_descriptions_literally(tmp_path):
    from datetime import datetime

    from conftest import IST

    from drone_agent.models import Observation

    db = tmp_path / "m.db"
    with Store.open(db) as s:
        s.insert_observation(
            Observation(
                observation_id="obs-1",
                frame_id=1,
                timestamp=datetime(2026, 9, 29, 12, 0, tzinfo=IST),
                source="scripted",
                zone=Zone.MAIN_GATE,
                description="odd [/oops] [bold]text[/bold]",
            )
        )
    code, text = run_cli("query", "--db", db)
    assert code == 0 and "[/oops] [bold]text[/bold]" in text
