"""SQLite schema (version 1). Explicit DDL, no ORM.

Timestamps are stored twice: `timestamp` is the exact ISO-8601 text with its UTC offset (round-trips
the original timezone), `ts_utc` is a fixed-width UTC string used for ordering, range filters and
indexes (text comparison of ISO strings with different offsets would be wrong).
"""

SCHEMA_VERSION = 1

# Executed one statement at a time so failures point at a single statement.
STATEMENTS: tuple[str, ...] = (
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
