"""Persistence of derived data: events, their evidence links, and entities (schema v2).

Everything here is regenerable from observations, so it is written with replace semantics
(`replace_derived`), unlike ingested observations which are rejected if duplicated.
"""

import json
import sqlite3
from datetime import datetime

from drone_agent.models import (
    CloseReason,
    Entity,
    Event,
    MatchLevel,
    ObjectType,
    Observation,
    Zone,
)
from drone_agent.store.codec import canonical_json, utc_key
from drone_agent.store.errors import StoreError, translate_integrity_error

_IN_CHUNK = 500


class DerivedMixin:
    """Event/entity methods of `Store` (uses Store's `_conn` and `transaction`)."""

    _conn: sqlite3.Connection

    # ---- writes ----------------------------------------------------------------------------
    def replace_derived(self, events: list[Event], entities: list[Entity]) -> tuple[int, int]:
        """Atomically replace ALL stored events and entities. Returns the previous counts.

        Evidence links must point at stored observations (foreign keys). An event's `entity_id`
        must agree with the entity that lists it.
        """
        owner = {eid: ent.entity_id for ent in entities for eid in ent.event_ids}
        for ev in events:
            if ev.entity_id != owner.get(ev.event_id):
                raise StoreError(f"event {ev.event_id} entity_id disagrees with entity membership")
        unknown = set(owner) - {ev.event_id for ev in events}
        if unknown:
            raise StoreError(f"entities reference unknown events: {sorted(unknown)}")

        with self.transaction():  # type: ignore[attr-defined]
            previous = self.count_events(), self.count_entities()
            self.clear_derived()
            try:
                for ev in events:
                    self._insert_event(ev)
                for ent in entities:
                    self._insert_entity(ent)
            except sqlite3.IntegrityError as exc:
                raise translate_integrity_error(exc, "event/entity data") from exc
            return previous

    def clear_derived(self) -> None:
        """Delete all events, evidence links and entities (children first)."""
        with self.transaction():  # type: ignore[attr-defined]
            for table in ("entity_events", "entities", "event_observations", "events"):
                self._conn.execute(f"DELETE FROM {table}")

    def _insert_event(self, ev: Event) -> None:
        self._conn.execute(
            "INSERT INTO events (event_id, object_type, zone, first_seen, first_seen_utc,"
            " last_seen, last_seen_utc, attributes_json, activities_json, description,"
            " close_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ev.event_id,
                ev.object_type.value,
                ev.zone.value,
                ev.first_seen.isoformat(),
                utc_key(ev.first_seen),
                ev.last_seen.isoformat(),
                utc_key(ev.last_seen),
                canonical_json(ev.attributes),
                json.dumps(ev.activities, ensure_ascii=False),
                ev.description,
                ev.close_reason.value,
            ),
        )
        self._conn.executemany(
            "INSERT INTO event_observations (event_id, observation_id, position) VALUES (?, ?, ?)",
            [(ev.event_id, oid, pos) for pos, oid in enumerate(ev.observation_ids)],
        )

    def _insert_entity(self, ent: Entity) -> None:
        self._conn.execute(
            "INSERT INTO entities (entity_id, entity_type, attributes_json, match_level,"
            " first_seen, first_seen_utc, last_seen, last_seen_utc)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ent.entity_id,
                ent.entity_type.value,
                canonical_json(ent.attributes),
                ent.match_level.value,
                ent.first_seen.isoformat(),
                utc_key(ent.first_seen),
                ent.last_seen.isoformat(),
                utc_key(ent.last_seen),
            ),
        )
        self._conn.executemany(
            "INSERT INTO entity_events (entity_id, event_id) VALUES (?, ?)",
            [(ent.entity_id, eid) for eid in ent.event_ids],
        )

    # ---- reads -----------------------------------------------------------------------------
    def get_event(self, event_id: str) -> Event | None:
        rows = self._event_rows("e.event_id = ?", [event_id])
        return rows[0] if rows else None

    def query_events(
        self,
        *,
        zone: Zone | None = None,
        object_type: ObjectType | None = None,
        entity_id: str | None = None,
    ) -> list[Event]:
        """Events matching ALL given filters, ordered by first_seen then event_id."""
        where, params = [], []
        if zone is not None:
            where.append("e.zone = ?")
            params.append(Zone(zone).value)
        if object_type is not None:
            where.append("e.object_type = ?")
            params.append(ObjectType(object_type).value)
        if entity_id is not None:
            where.append("ee.entity_id = ?")
            params.append(entity_id)
        return self._event_rows(" AND ".join(where) or "1 = 1", params)

    def get_event_observations(self, event_id: str) -> list[Observation]:
        """The evidence for an event: its observations (with frame ids), chronologically."""
        ids = [
            r["observation_id"]
            for r in self._conn.execute(
                "SELECT observation_id FROM event_observations WHERE event_id = ?"
                " ORDER BY position",
                (event_id,),
            )
        ]
        found = self._select_observations(  # type: ignore[attr-defined]
            "o.observation_id IN"
            " (SELECT observation_id FROM event_observations WHERE event_id = ?)",
            [event_id],
        )
        by_id = {o.observation_id: o for o in found}
        return [by_id[i] for i in ids]

    def get_entity(self, entity_id: str) -> Entity | None:
        rows = self._entity_rows("entity_id = ?", [entity_id])
        return rows[0] if rows else None

    def list_entities(self) -> list[Entity]:
        return self._entity_rows("1 = 1", [])

    def count_events(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def count_entities(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def _event_rows(self, condition: str, params: list[object]) -> list[Event]:
        rows = self._conn.execute(
            "SELECT e.*, ee.entity_id FROM events e"
            " LEFT JOIN entity_events ee ON ee.event_id = e.event_id"
            f" WHERE {condition} ORDER BY e.first_seen_utc, e.event_id",
            params,
        ).fetchall()
        evidence: dict[str, list[tuple[str, int]]] = {r["event_id"]: [] for r in rows}
        ids = list(evidence)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                "SELECT eo.event_id, eo.observation_id, o.frame_id FROM event_observations eo"
                " JOIN observations o ON o.observation_id = eo.observation_id"
                f" WHERE eo.event_id IN ({marks}) ORDER BY eo.event_id, eo.position",
                chunk,
            ):
                evidence[r["event_id"]].append((r["observation_id"], r["frame_id"]))
        return [
            Event.model_validate(
                {
                    "event_id": r["event_id"],
                    "object_type": r["object_type"],
                    "zone": r["zone"],
                    "first_seen": datetime.fromisoformat(r["first_seen"]),
                    "last_seen": datetime.fromisoformat(r["last_seen"]),
                    "attributes": json.loads(r["attributes_json"]),
                    "activities": json.loads(r["activities_json"]),
                    "description": r["description"],
                    "observation_ids": [o for o, _ in evidence[r["event_id"]]],
                    "frame_ids": [f for _, f in evidence[r["event_id"]]],
                    "close_reason": CloseReason(r["close_reason"]),
                    "entity_id": r["entity_id"],
                }
            )
            for r in rows
        ]

    def _entity_rows(self, condition: str, params: list[object]) -> list[Entity]:
        rows = self._conn.execute(
            f"SELECT * FROM entities WHERE {condition} ORDER BY first_seen_utc, entity_id",
            params,
        ).fetchall()
        members: dict[str, list[str]] = {r["entity_id"]: [] for r in rows}
        for r in self._conn.execute(
            "SELECT ee.entity_id, ee.event_id FROM entity_events ee"
            " JOIN events e ON e.event_id = ee.event_id ORDER BY e.first_seen_utc, e.event_id"
        ):
            if r["entity_id"] in members:
                members[r["entity_id"]].append(r["event_id"])
        return [
            Entity.model_validate(
                {
                    "entity_id": r["entity_id"],
                    "entity_type": r["entity_type"],
                    "attributes": json.loads(r["attributes_json"]),
                    "match_level": MatchLevel(r["match_level"]),
                    "first_seen": datetime.fromisoformat(r["first_seen"]),
                    "last_seen": datetime.fromisoformat(r["last_seen"]),
                    "event_ids": members[r["entity_id"]],
                }
            )
            for r in rows
        ]
