import sqlite3
from datetime import UTC, datetime, timedelta, timezone

import pytest
from conftest import IST

from drone_agent.models import (
    DetectedObject,
    ObjectType,
    Observation,
    Telemetry,
    TelemetryAssociation,
    Zone,
)
from drone_agent.store import DuplicateRecordError, Fts5UnavailableError, Store, StoreError
from drone_agent.store import schema as schema_module

T0 = datetime(2026, 9, 29, 23, 40, tzinfo=IST)


def person(activity="standing", **attrs) -> DetectedObject:
    return DetectedObject(
        object_type=ObjectType.PERSON, activity=activity, attributes=attrs, confidence=0.9
    )


def vehicle(color="blue", body="pickup", activity="parked") -> DetectedObject:
    return DetectedObject(
        object_type=ObjectType.VEHICLE,
        activity=activity,
        attributes={"color": color, "body_type": body},
        confidence=0.8,
    )


def make_obs(frame_id, minutes=0, zone=Zone.MAIN_GATE, description="d", objects=(), **extra):
    return Observation(
        observation_id=f"obs-{frame_id}",
        frame_id=frame_id,
        timestamp=T0 + timedelta(minutes=minutes),
        source="scripted",
        zone=zone,
        description=description,
        detected_objects=list(objects),
        **extra,
    )


def counts(store: Store) -> tuple[int, int, int, int]:
    return (
        store.count_frames(),
        store.count_observations(),
        store.count_detected_objects(),
        store.count_telemetry(),
    )


# ---- schema ---------------------------------------------------------------------------------
def test_schema_created_with_tables_indexes_and_version(store):
    names = {r["name"] for r in store._conn.execute("SELECT name FROM sqlite_master").fetchall()}
    assert {"telemetry", "frames", "observations", "detected_objects", "observations_fts"} <= names
    assert {"idx_observations_ts", "idx_observations_zone_ts", "idx_objects_type_obs"} <= names
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == schema_module.SCHEMA_VERSION


def test_foreign_keys_are_enforced(store):
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        store._conn.execute(
            "INSERT INTO observations (observation_id, frame_id, timestamp, ts_utc, source,"
            " zone, description) VALUES ('x', 99, 't', 't', 's', 'main_gate', 'd')"
        )


def test_reopening_existing_database_keeps_data(tmp_path):
    path = tmp_path / "again.db"
    with Store.open(path) as s:
        s.insert_observation(make_obs(1, description="Person at gate", objects=[person()]))
    with Store.open(path) as s:
        assert s.count_observations() == 1
        assert s.query_observations(text="gate")[0].frame_id == 1


def test_missing_fts5_is_reported_not_silently_replaced(tmp_path, monkeypatch):
    broken = (*schema_module.STATEMENTS[:-1], "CREATE VIRTUAL TABLE x USING fts5_missing(a)")
    monkeypatch.setattr("drone_agent.store.store.STATEMENTS", broken)
    with pytest.raises(Fts5UnavailableError):
        Store.open(tmp_path / "nofts.db")


# ---- round trips ----------------------------------------------------------------------------
def test_observation_round_trip_with_objects_attributes_and_telemetry(store, make_telemetry):
    sample = make_telemetry(T0, zone=Zone.MAIN_GATE)
    store.insert_telemetry(sample)
    obs = make_obs(
        1,
        description="Person and pickup at the gate",
        objects=[
            DetectedObject(
                object_id="p-7",
                object_type=ObjectType.PERSON,
                activity=None,
                attributes={"clothing": "dark jacket", "bag": "backpack"},
                confidence=0.75,
            ),
            vehicle(color="dark blue"),
            DetectedObject(object_type=ObjectType.OTHER, confidence=0.5),
        ],
        telemetry=TelemetryAssociation(telemetry=sample, age_seconds=4.5, stale=True),
    )
    store.insert_observation(obs)
    assert store.get_observation("obs-1") == obs
    assert store.get_observation_by_frame(1) == obs
    assert store.get_observation("missing") is None
    assert store.get_observation_by_frame(404) is None
    assert counts(store) == (1, 1, 3, 1)


def test_observation_without_telemetry_round_trips(store):
    obs = make_obs(1, objects=[person()])
    store.insert_observation(obs)
    assert store.get_observation("obs-1").telemetry is None


def test_attributes_are_stored_as_canonical_json_and_keep_unicode(store):
    obj = DetectedObject(
        object_type=ObjectType.PERSON, attributes={"b": "2", "a": "café"}, confidence=1.0
    )
    store.insert_observation(make_obs(1, objects=[obj]))
    raw = store._conn.execute("SELECT attributes_json FROM detected_objects").fetchone()[0]
    assert raw == '{"a":"café","b":"2"}'
    assert store.get_observation("obs-1").detected_objects[0].attributes == {"a": "café", "b": "2"}


def test_timezones_are_preserved_and_ordering_uses_the_instant(store):
    tz_west = timezone(timedelta(hours=-8))
    utc = UTC
    a = make_obs(1, objects=[])  # 23:40 +05:30 == 18:10 UTC
    b = make_obs(2, objects=[]).model_copy(
        update={"timestamp": datetime(2026, 9, 29, 10, 20, tzinfo=tz_west)}  # 18:20 UTC
    )
    c = make_obs(3, objects=[]).model_copy(
        update={"timestamp": datetime(2026, 9, 29, 18, 15, tzinfo=utc)}  # 18:15 UTC
    )
    for o in (a, b, c):
        store.insert_observation(o)
    got = {o.frame_id: o.timestamp for o in store.query_observations()}
    assert got[1].utcoffset() == timedelta(hours=5, minutes=30)
    assert got[2].utcoffset() == timedelta(hours=-8)
    assert got[3].utcoffset() == timedelta(0)
    assert [o.frame_id for o in store.query_observations()] == [1, 3, 2]  # by instant, not text


# ---- structured queries ---------------------------------------------------------------------
@pytest.fixture
def populated(store):
    rows = [
        make_obs(
            1, 0, Zone.MAIN_GATE, "Person standing near the gate", [person(clothing="dark jacket")]
        ),
        make_obs(2, 5, Zone.MAIN_GATE, "Blue pickup truck at the gate", [vehicle()]),
        make_obs(3, 10, Zone.PARKING_AREA, "Blue pickup truck parked", [vehicle(), person()]),
        make_obs(4, 15, Zone.WAREHOUSE, "Person walking near the warehouse", [person("walking")]),
        make_obs(5, 20, Zone.WAREHOUSE, "Warehouse exterior clear"),
    ]
    for r in rows:
        store.insert_observation(r)
    return store


def frames(rows) -> list[int]:
    return [o.frame_id for o in rows]


def test_time_range_is_inclusive_and_mixed_timezone_safe(populated):
    utc = UTC
    start = (T0 + timedelta(minutes=5)).astimezone(utc)
    end = (T0 + timedelta(minutes=15)).astimezone(timezone(timedelta(hours=-8)))
    assert frames(populated.query_observations(start=start, end=end)) == [2, 3, 4]
    assert frames(populated.query_observations(start=start)) == [2, 3, 4, 5]
    assert frames(populated.query_observations(end=T0)) == [1]
    assert frames(populated.query_observations(start=T0 + timedelta(hours=1))) == []


def test_query_by_zone_and_object_type(populated):
    assert frames(populated.query_observations(zone=Zone.MAIN_GATE)) == [1, 2]
    assert frames(populated.query_observations(zone=Zone.RESTRICTED_ZONE)) == []
    assert frames(populated.query_observations(object_type=ObjectType.PERSON)) == [1, 3, 4]
    assert frames(populated.query_observations(object_type=ObjectType.VEHICLE)) == [2, 3]
    assert frames(populated.query_observations(object_type=ObjectType.OTHER)) == []


def test_combined_filters_are_anded(populated):
    got = populated.query_observations(
        zone=Zone.PARKING_AREA, object_type=ObjectType.PERSON, start=T0, end=T0 + timedelta(hours=1)
    )
    assert frames(got) == [3]
    assert (
        frames(populated.query_observations(zone=Zone.WAREHOUSE, object_type=ObjectType.VEHICLE))
        == []
    )
    assert frames(populated.query_observations(zone=Zone.MAIN_GATE, text="blue pickup")) == [2]
    assert frames(populated.query_observations(object_type=ObjectType.PERSON, limit=2)) == [1, 3]


def test_naive_query_time_is_rejected(populated):
    with pytest.raises(ValueError, match="timezone-aware"):
        populated.query_observations(start=datetime(2026, 9, 29, 23, 40))


# ---- FTS5 -----------------------------------------------------------------------------------
def test_fts_search_matches_words_from_description_objects_and_zone(populated):
    assert sorted(frames(populated.query_observations(text="blue pickup"))) == [2, 3]
    assert frames(populated.query_observations(text="person warehouse")) == [4]
    # 'parking' stems to 'park', so it also matches frame 2's activity 'parked'
    assert frames(populated.query_observations(text="parking")) == [3, 2]
    # object-derived text: frame 1 description has no 'jacket'; its object attribute does
    assert frames(populated.query_observations(text="jacket")) == [1]


def test_fts_is_case_insensitive_and_stems_but_has_no_synonyms(populated):
    assert frames(populated.query_observations(text="WALK")) == [4]  # walking -> walk (porter)
    assert frames(populated.query_observations(text="car")) == []  # no synonym for pickup/truck
    assert frames(populated.query_observations(text="pick")) == []  # no prefix matching


def test_fts_no_result_returns_empty_list(populated):
    assert populated.query_observations(text="helicopter") == []
    assert populated.query_observations(text="vehicle warehouse") == []


def test_fts_results_are_ranked_by_relevance(store):
    store.insert_observation(make_obs(1, 0, description="Gate clear, one pickup far away"))
    store.insert_observation(make_obs(2, 1, description="Pickup pickup pickup"))
    assert frames(store.query_observations(text="pickup")) == [2, 1]


@pytest.mark.parametrize(
    "text",
    [
        "'; DROP TABLE observations; --",
        '"unbalanced',
        "blue AND OR NEAR(",
        "main_gate:",
        "a-b",
    ],
)
def test_search_text_is_treated_as_data_not_sql_or_fts_syntax(populated, text):
    populated.query_observations(text=text)  # must not raise
    assert populated.count_observations() == 5  # nothing dropped


@pytest.mark.parametrize("text", ["  ---  ", "*", ""])
def test_search_with_no_words_is_rejected(populated, text):
    with pytest.raises(ValueError, match="no searchable words"):
        populated.query_observations(text=text)


def test_zone_name_is_searchable_even_when_description_lacks_it(store):
    store.insert_observation(
        make_obs(1, zone=Zone.RESTRICTED_ZONE, description="Nothing to report")
    )
    assert frames(store.query_observations(text="restricted zone")) == [1]


def test_lookup_by_id_is_parameterized(populated):
    assert populated.get_observation("obs-1' OR '1'='1") is None


def test_words_with_sql_metacharacters_in_content_round_trip(store):
    nasty = 'O\'Brien said "hi"; DROP TABLE frames;--'
    store.insert_observation(make_obs(1, description=nasty))
    assert store.get_observation("obs-1").description == nasty
    assert frames(store.query_observations(text="O'Brien")) == [1]
    assert store.count_frames() == 1


# ---- duplicates and atomicity ---------------------------------------------------------------
def test_duplicate_frame_or_observation_is_rejected_and_changes_nothing(populated):
    before = counts(populated)
    with pytest.raises(DuplicateRecordError):
        populated.insert_observation(make_obs(1, description="again", objects=[person()]))
    dup_id = make_obs(99, description="x").model_copy(update={"observation_id": "obs-1"})
    with pytest.raises(DuplicateRecordError):
        populated.insert_observation(dup_id)
    assert counts(populated) == before
    assert populated.get_observation("obs-1").description == "Person standing near the gate"


def test_duplicate_telemetry_timestamp_is_rejected(store, make_telemetry):
    store.insert_telemetry(make_telemetry(T0))
    with pytest.raises(DuplicateRecordError):
        store.insert_telemetry(make_telemetry(T0.astimezone(UTC), battery_percent=10))
    assert store.count_telemetry() == 1


def test_malformed_object_rolls_back_the_whole_observation_graph(store):
    good = person()
    # model_construct skips validation, so this reaches the DB CHECK constraint (confidence <= 1)
    bad = DetectedObject.model_construct(
        object_id=None, object_type=ObjectType.VEHICLE, attributes={}, activity=None, confidence=5.0
    )
    with pytest.raises(StoreError, match="constraint"):
        store.insert_observation(
            make_obs(1, description="Person and bad object", objects=[good, bad])
        )
    assert counts(store) == (0, 0, 0, 0)
    assert store.query_observations(text="person") == []  # FTS row rolled back too


def test_observation_referencing_unknown_or_different_telemetry_is_rejected(store, make_telemetry):
    sample = make_telemetry(T0)
    assoc = TelemetryAssociation(telemetry=sample, age_seconds=0, stale=False)
    with pytest.raises(StoreError, match="not stored"):
        store.insert_observation(make_obs(1, telemetry=assoc))
    store.insert_telemetry(make_telemetry(T0, battery_percent=50))
    with pytest.raises(StoreError, match="differs"):
        store.insert_observation(make_obs(1, telemetry=assoc))
    assert store.count_frames() == 0


def test_nested_transactions_use_savepoints(store):
    with store.transaction():
        store.insert_observation(make_obs(1, description="kept"))
        with pytest.raises(DuplicateRecordError):
            store.insert_observation(make_obs(1, description="dup"))
        store.insert_observation(make_obs(2, description="also kept"))
    assert store.count_observations() == 2

    with pytest.raises(RuntimeError), store.transaction():
        store.insert_observation(make_obs(3))
        raise RuntimeError("boom")
    assert store.count_observations() == 2


# ---- query plans ----------------------------------------------------------------------------
def plan(store, **filters) -> str:
    return " | ".join(store.explain_query(**filters))


def test_indexes_are_used_for_representative_queries(populated):
    zone_time = plan(populated, zone=Zone.MAIN_GATE, start=T0, end=T0 + timedelta(hours=1))
    assert "idx_observations_zone_ts" in zone_time
    assert "idx_observations_ts" in plan(populated, start=T0, end=T0 + timedelta(hours=1))
    assert "idx_objects_type_obs" in plan(populated, object_type=ObjectType.PERSON)
    assert "SCAN" not in plan(populated, zone=Zone.WAREHOUSE).replace("SCAN t", "")


def test_unknown_filter_values_are_rejected(populated):
    with pytest.raises(ValueError):
        populated.query_observations(zone="moon_base")


def test_telemetry_row_conversion_preserves_values(store):
    sample = Telemetry(
        timestamp=T0,
        latitude=0.000123,
        longitude=-0.5,
        altitude_m=29.9,
        zone=Zone.DRIVEWAY,
        battery_percent=77.7,
    )
    store.insert_telemetry(sample)
    row = store._conn.execute("SELECT * FROM telemetry").fetchone()
    from drone_agent.store.store import _telemetry_from_row

    assert _telemetry_from_row(row) == sample
