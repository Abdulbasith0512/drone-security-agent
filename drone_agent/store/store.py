"""SQLite store for telemetry, frames, observations and detected objects (stdlib sqlite3)."""

import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from drone_agent.models import (
    DetectedObject,
    ObjectType,
    Observation,
    Telemetry,
    TelemetryAssociation,
    Zone,
)
from drone_agent.store.codec import canonical_json, utc_key
from drone_agent.store.derived import DerivedMixin
from drone_agent.store.errors import (
    Fts5UnavailableError,
    StoreError,
    translate_integrity_error,
)
from drone_agent.store.schema import MIGRATIONS, SCHEMA_VERSION

_IN_CHUNK = 500  # stay well below SQLite's bound-parameter limit


def _fts_query(text: str) -> str:
    """Turn free text into a safe FTS5 query: each word quoted, implicitly ANDed.

    Quoting neutralises FTS5 operators and punctuation (AND, OR, NEAR, -, :, *, quotes).
    """
    words = re.findall(r"\w+", text)
    if not words:
        raise ValueError("search text contains no searchable words")
    return " ".join(f'"{w}"' for w in words)


def _objects_text(obs: Observation) -> str:
    """Searchable text for detected objects: type, activity and attribute values."""
    parts: list[str] = []
    for obj in obs.detected_objects:
        parts.append(obj.object_type.value)
        if obj.activity:
            parts.append(obj.activity)
        parts.extend(v for _, v in sorted(obj.attributes.items()))
    return " ".join(parts)


class Store(DerivedMixin):
    """Explicit persistence API. One frame maps to one observation (frame_id is unique).

    `migrated_from` is the schema version an existing database was upgraded from (None if it was
    new or already current).
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._depth = 0
        self.migrated_from: int | None = None

    # ---- lifecycle -------------------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> "Store":
        """Open (creating if needed) a database and ensure the schema exists."""
        target = str(path)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target, isolation_level=None)  # explicit BEGIN/COMMIT below
        conn.row_factory = sqlite3.Row
        store = cls(conn)
        try:
            store._initialize()
        except Exception:
            conn.close()
            raise
        return store

    def _initialize(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self._conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise StoreError("could not enable SQLite foreign key enforcement")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StoreError(f"database schema version {version} is newer than {SCHEMA_VERSION}")
        if 0 < version < SCHEMA_VERSION:
            self.migrated_from = version
        try:
            with self.transaction():
                for target in range(version + 1, SCHEMA_VERSION + 1):
                    for statement in MIGRATIONS[target]:
                        self._conn.execute(statement)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except sqlite3.OperationalError as exc:
            if "fts5" in str(exc).lower():
                raise Fts5UnavailableError(
                    "this SQLite build does not provide FTS5; search cannot be enabled"
                ) from exc
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """All-or-nothing block. The outermost use is a transaction, nested uses are savepoints."""
        name = f"sp_{self._depth}"
        self._conn.execute("BEGIN" if self._depth == 0 else f"SAVEPOINT {name}")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._conn.execute("ROLLBACK")
            else:
                self._conn.execute(f"ROLLBACK TO {name}")
                self._conn.execute(f"RELEASE {name}")
            raise
        else:
            self._depth -= 1
            self._conn.execute("COMMIT" if self._depth == 0 else f"RELEASE {name}")

    # ---- writes ----------------------------------------------------------------------------
    def insert_telemetry(self, sample: Telemetry) -> int:
        """Store one telemetry sample; returns its telemetry_id. Timestamps are unique."""
        with self.transaction():
            try:
                cur = self._conn.execute(
                    "INSERT INTO telemetry (timestamp, ts_utc, latitude, longitude, altitude_m,"
                    " zone, battery_percent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        sample.timestamp.isoformat(),
                        utc_key(sample.timestamp),
                        sample.latitude,
                        sample.longitude,
                        sample.altitude_m,
                        sample.zone.value,
                        sample.battery_percent,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise self._translate(exc, f"telemetry at {sample.timestamp.isoformat()}") from exc
            assert cur.lastrowid is not None
            return cur.lastrowid

    def insert_observation(self, obs: Observation) -> None:
        """Atomically store frame + observation + detected objects + search index entry.

        If the observation has a telemetry association, that telemetry sample must already be
        stored and identical to the stored row. Any failure leaves nothing behind.
        """
        with self.transaction():
            ts_iso, ts_utc = obs.timestamp.isoformat(), utc_key(obs.timestamp)
            telemetry_id = age = stale = None
            if obs.telemetry is not None:
                telemetry_id = self._stored_telemetry_id(obs.telemetry.telemetry)
                age, stale = obs.telemetry.age_seconds, int(obs.telemetry.stale)
            try:
                self._conn.execute(
                    "INSERT INTO frames (frame_id, timestamp, ts_utc, source, zone)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (obs.frame_id, ts_iso, ts_utc, obs.source, obs.zone.value),
                )
                self._conn.execute(
                    "INSERT INTO observations (observation_id, frame_id, timestamp, ts_utc,"
                    " source, zone, description, telemetry_id, telemetry_age_seconds,"
                    " telemetry_stale) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        obs.observation_id,
                        obs.frame_id,
                        ts_iso,
                        ts_utc,
                        obs.source,
                        obs.zone.value,
                        obs.description,
                        telemetry_id,
                        age,
                        stale,
                    ),
                )
                self._conn.executemany(
                    "INSERT INTO detected_objects (observation_id, position, object_id,"
                    " object_type, activity, confidence, attributes_json)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            obs.observation_id,
                            pos,
                            obj.object_id,
                            obj.object_type.value,
                            obj.activity,
                            obj.confidence,
                            canonical_json(obj.attributes),
                        )
                        for pos, obj in enumerate(obs.detected_objects)
                    ],
                )
                self._conn.execute(
                    "INSERT INTO observations_fts (observation_id, description, objects_text, zone)"
                    " VALUES (?, ?, ?, ?)",
                    (obs.observation_id, obs.description, _objects_text(obs), obs.zone.value),
                )
            except sqlite3.IntegrityError as exc:
                raise self._translate(
                    exc, f"observation {obs.observation_id} (frame {obs.frame_id})"
                ) from exc

    def _stored_telemetry_id(self, sample: Telemetry) -> int:
        row = self._conn.execute(
            "SELECT * FROM telemetry WHERE ts_utc = ?", (utc_key(sample.timestamp),)
        ).fetchone()
        if row is None:
            raise StoreError(
                f"telemetry sample at {sample.timestamp.isoformat()} is not stored; "
                "insert telemetry before the observations that reference it"
            )
        if _telemetry_from_row(row) != sample:
            raise StoreError(
                f"telemetry association at {sample.timestamp.isoformat()} differs from "
                "the stored sample"
            )
        return row["telemetry_id"]

    @staticmethod
    def _translate(exc: sqlite3.IntegrityError, what: str) -> StoreError:
        return translate_integrity_error(exc, what)

    # ---- reads -----------------------------------------------------------------------------
    def get_observation(self, observation_id: str) -> Observation | None:
        rows = self._select_observations("o.observation_id = ?", [observation_id])
        return rows[0] if rows else None

    def get_observation_by_frame(self, frame_id: int) -> Observation | None:
        rows = self._select_observations("o.frame_id = ?", [frame_id])
        return rows[0] if rows else None

    def query_observations(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        zone: Zone | None = None,
        object_type: ObjectType | None = None,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[Observation]:
        """Observations matching ALL given filters. `start`/`end` are inclusive.

        Ordered by time (then frame_id), or by BM25 relevance when `text` is given.
        """
        sql, params = self._build_query(start, end, zone, object_type, text, limit)
        return self._observations_from_rows(self._conn.execute(sql, params).fetchall())

    def explain_query(
        self,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        zone: Zone | None = None,
        object_type: ObjectType | None = None,
        text: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """EXPLAIN QUERY PLAN lines for the same query `query_observations` would run."""
        sql, params = self._build_query(start, end, zone, object_type, text, limit)
        rows = self._conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
        return [row["detail"] for row in rows]

    def count_observations(self) -> int:
        return self._count("observations")

    def count_frames(self) -> int:
        return self._count("frames")

    def count_telemetry(self) -> int:
        return self._count("telemetry")

    def count_detected_objects(self) -> int:
        return self._count("detected_objects")

    def _count(self, table: str) -> int:
        return self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ---- query building / row conversion ---------------------------------------------------
    _COLUMNS = (
        "SELECT o.observation_id, o.frame_id, o.timestamp, o.source, o.zone, o.description,"
        " o.telemetry_age_seconds, o.telemetry_stale,"
        " t.timestamp AS t_timestamp, t.latitude AS t_latitude, t.longitude AS t_longitude,"
        " t.altitude_m AS t_altitude_m, t.zone AS t_zone, t.battery_percent AS t_battery_percent"
    )
    _FROM = " FROM observations o"
    _TELEMETRY_JOIN = " LEFT JOIN telemetry t ON t.telemetry_id = o.telemetry_id"

    def _build_query(
        self,
        start: datetime | None,
        end: datetime | None,
        zone: Zone | None,
        object_type: ObjectType | None,
        text: str | None,
        limit: int | None,
    ) -> tuple[str, list[object]]:
        joins, where, params = "", [], []
        order = "o.ts_utc, o.frame_id"
        if text is not None:
            joins = " JOIN observations_fts ON observations_fts.observation_id = o.observation_id"
            where.append("observations_fts MATCH ?")
            params.append(_fts_query(text))
            order = "bm25(observations_fts), o.ts_utc, o.frame_id"
        if start is not None:
            where.append("o.ts_utc >= ?")
            params.append(utc_key(start))
        if end is not None:
            where.append("o.ts_utc <= ?")
            params.append(utc_key(end))
        if zone is not None:
            where.append("o.zone = ?")
            params.append(Zone(zone).value)
        if object_type is not None:
            where.append(
                "EXISTS (SELECT 1 FROM detected_objects d"
                " WHERE d.observation_id = o.observation_id AND d.object_type = ?)"
            )
            params.append(ObjectType(object_type).value)
        sql = self._COLUMNS + self._FROM + joins + self._TELEMETRY_JOIN
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        return sql, params

    def _select_observations(self, condition: str, params: Sequence[object]) -> list[Observation]:
        rows = self._conn.execute(
            f"{self._COLUMNS}{self._FROM}{self._TELEMETRY_JOIN} WHERE {condition}", list(params)
        ).fetchall()
        return self._observations_from_rows(rows)

    def _observations_from_rows(self, rows: list[sqlite3.Row]) -> list[Observation]:
        objects: dict[str, list[DetectedObject]] = {r["observation_id"]: [] for r in rows}
        ids = list(objects)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            obj_rows = self._conn.execute(
                "SELECT observation_id, object_id, object_type, activity, confidence,"
                f" attributes_json FROM detected_objects WHERE observation_id IN ({marks})"
                " ORDER BY observation_id, position",
                chunk,
            )
            for r in obj_rows:
                objects[r["observation_id"]].append(
                    DetectedObject.model_validate(
                        {
                            "object_id": r["object_id"],
                            "object_type": r["object_type"],
                            "activity": r["activity"],
                            "confidence": r["confidence"],
                            "attributes": json.loads(r["attributes_json"]),
                        }
                    )
                )
        return [_observation_from_row(r, objects[r["observation_id"]]) for r in rows]


def _telemetry_from_row(row: sqlite3.Row) -> Telemetry:
    return Telemetry.model_validate(
        {
            "timestamp": datetime.fromisoformat(row["timestamp"]),
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "altitude_m": row["altitude_m"],
            "zone": row["zone"],
            "battery_percent": row["battery_percent"],
        }
    )


def _observation_from_row(row: sqlite3.Row, objects: list[DetectedObject]) -> Observation:
    association = None
    if row["t_timestamp"] is not None:
        sample = Telemetry.model_validate(
            {
                "timestamp": datetime.fromisoformat(row["t_timestamp"]),
                "latitude": row["t_latitude"],
                "longitude": row["t_longitude"],
                "altitude_m": row["t_altitude_m"],
                "zone": row["t_zone"],
                "battery_percent": row["t_battery_percent"],
            }
        )
        association = TelemetryAssociation(
            telemetry=sample,
            age_seconds=row["telemetry_age_seconds"],
            stale=bool(row["telemetry_stale"]),
        )
    return Observation.model_validate(
        {
            "observation_id": row["observation_id"],
            "frame_id": row["frame_id"],
            "timestamp": datetime.fromisoformat(row["timestamp"]),
            "source": row["source"],
            "zone": row["zone"],
            "description": row["description"],
            "detected_objects": objects,
            "telemetry": association,
        }
    )
