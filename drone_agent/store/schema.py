"""SQLite schema. Explicit DDL applied as ordered, additive migrations (tracked by user_version).

Timestamps are stored twice: `timestamp` is the exact ISO-8601 text with its UTC offset (round-trips
the original timezone), `ts_utc` is a fixed-width UTC string used for ordering, range filters and
indexes (text comparison of ISO strings with different offsets would be wrong).

Version 1 (M2): telemetry, frames, observations, detected_objects, observations_fts.
Version 2 (M3): events, event_observations, entities, entity_events (derived data).
Migrations only ever ADD objects, so an older database keeps all of its data when upgraded.
"""

_V1: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS telemetry (
        telemetry_id    INTEGER PRIMARY KEY,
        timestamp       TEXT NOT NULL,
        ts_utc          TEXT NOT NULL UNIQUE,
        latitude        REAL NOT NULL CHECK (latitude BETWEEN -90 AND 90),
        longitude       REAL NOT NULL CHECK (longitude BETWEEN -180 AND 180),
        altitude_m      REAL NOT NULL CHECK (altitude_m BETWEEN 0 AND 500),
        zone            TEXT NOT NULL,
        battery_percent REAL NOT NULL CHECK (battery_percent BETWEEN 0 AND 100)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frames (
        frame_id  INTEGER PRIMARY KEY CHECK (frame_id >= 0),
        timestamp TEXT NOT NULL,
        ts_utc    TEXT NOT NULL,
        source    TEXT NOT NULL,
        zone      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        observation_id        TEXT PRIMARY KEY,
        frame_id              INTEGER NOT NULL UNIQUE REFERENCES frames(frame_id),
        timestamp             TEXT NOT NULL,
        ts_utc                TEXT NOT NULL,
        source                TEXT NOT NULL,
        zone                  TEXT NOT NULL,
        description           TEXT NOT NULL,
        telemetry_id          INTEGER REFERENCES telemetry(telemetry_id),
        telemetry_age_seconds REAL CHECK (telemetry_age_seconds >= 0),
        telemetry_stale       INTEGER CHECK (telemetry_stale IN (0, 1)),
        CHECK ((telemetry_id IS NULL) = (telemetry_age_seconds IS NULL)
               AND (telemetry_id IS NULL) = (telemetry_stale IS NULL))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detected_objects (
        object_pk       INTEGER PRIMARY KEY,
        observation_id  TEXT NOT NULL REFERENCES observations(observation_id),
        position        INTEGER NOT NULL CHECK (position >= 0),
        object_id       TEXT,
        object_type     TEXT NOT NULL,
        activity        TEXT,
        confidence      REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
        attributes_json TEXT NOT NULL,
        UNIQUE (observation_id, position)
    )
    """,
    # Time-only range queries.
    "CREATE INDEX IF NOT EXISTS idx_observations_ts ON observations(ts_utc)",
    # Zone filter, optionally with a time range (zone is the leading column).
    "CREATE INDEX IF NOT EXISTS idx_observations_zone_ts ON observations(zone, ts_utc)",
    # Object-type filter: find observation ids for a type without scanning all objects.
    "CREATE INDEX IF NOT EXISTS idx_objects_type_obs "
    "ON detected_objects(object_type, observation_id)",
    # Keyword search. observation_id is UNINDEXED (stored, not searchable) and is the join key.
    # 'porter unicode61': case-insensitive, splits on punctuation ('main_gate' -> main, gate)
    # and stems English words (walking -> walk).
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
        observation_id UNINDEXED, description, objects_text, zone,
        tokenize = 'porter unicode61'
    )
    """,
)

_V2: tuple[str, ...] = (
    # Derived data: everything below can be regenerated from observations (see build-events).
    # Duration and evidence counts are not stored; they are derived from first/last_seen and
    # event_observations.
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id        TEXT PRIMARY KEY,
        object_type     TEXT NOT NULL,
        zone            TEXT NOT NULL,
        first_seen      TEXT NOT NULL,
        first_seen_utc  TEXT NOT NULL,
        last_seen       TEXT NOT NULL,
        last_seen_utc   TEXT NOT NULL,
        attributes_json TEXT NOT NULL,
        activities_json TEXT NOT NULL,
        description     TEXT NOT NULL,
        close_reason    TEXT NOT NULL
            CHECK (close_reason IN ('gap', 'missed_frames', 'zone_change', 'scenario_end')),
        CHECK (last_seen_utc >= first_seen_utc)
    )
    """,
    # Evidence: event -> observations (and through them, frames). `position` is chronological.
    """
    CREATE TABLE IF NOT EXISTS event_observations (
        event_id       TEXT NOT NULL REFERENCES events(event_id),
        observation_id TEXT NOT NULL REFERENCES observations(observation_id),
        position       INTEGER NOT NULL CHECK (position >= 0),
        PRIMARY KEY (event_id, observation_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS entities (
        entity_id       TEXT PRIMARY KEY,
        entity_type     TEXT NOT NULL CHECK (entity_type = 'vehicle'),
        attributes_json TEXT NOT NULL,
        match_level     TEXT NOT NULL CHECK (match_level IN ('coarse_match', 'confirmed')),
        first_seen      TEXT NOT NULL,
        first_seen_utc  TEXT NOT NULL,
        last_seen       TEXT NOT NULL,
        last_seen_utc   TEXT NOT NULL
    )
    """,
    # An event belongs to at most one entity (UNIQUE event_id).
    """
    CREATE TABLE IF NOT EXISTS entity_events (
        entity_id TEXT NOT NULL REFERENCES entities(entity_id),
        event_id  TEXT NOT NULL UNIQUE REFERENCES events(event_id),
        PRIMARY KEY (entity_id, event_id)
    )
    """,
    # Event listings by zone / type over time (M4 rules and M6 questions filter on these).
    "CREATE INDEX IF NOT EXISTS idx_events_zone_first ON events(zone, first_seen_utc)",
    "CREATE INDEX IF NOT EXISTS idx_events_type_first ON events(object_type, first_seen_utc)",
    # Reverse evidence lookup: which events used this observation/frame.
    "CREATE INDEX IF NOT EXISTS idx_event_obs_observation ON event_observations(observation_id)",
)

# target schema version -> statements that bring a (version - 1) database up to it
MIGRATIONS: dict[int, tuple[str, ...]] = {1: _V1, 2: _V2}
SCHEMA_VERSION = max(MIGRATIONS)
