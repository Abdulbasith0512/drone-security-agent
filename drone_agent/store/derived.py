"""Persistence of derived data: events, their evidence links, entities and alerts.

Two write styles share the same row encoders:

* Batch (`replace_derived`, `replace_alerts`): everything here is regenerable from observations,
  so it is written with replace semantics.
* Streaming (`save_events`, `save_entities`, `save_alert`): a run persists events while they are
  still open and alerts as their lifecycle advances, so rows are upserted, never deleted, and the
  foreign keys from alerts to events stay valid throughout.
"""

import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime

from drone_agent.models import (
    Alert,
    AlertStatus,
    AlertTransition,
    CloseReason,
    Entity,
    Event,
    LadderStep,
    MatchLevel,
    ObjectType,
    Observation,
    Severity,
    Verdict,
    Zone,
)
from drone_agent.store.codec import canonical_json, utc_key
from drone_agent.store.errors import StoreError, translate_integrity_error

_IN_CHUNK = 500

# An OPEN event keeps this placeholder in `close_reason` (which has its own CHECK); the real
# discriminator is the `status` column.
_OPEN_PLACEHOLDER = CloseReason.SCENARIO_END.value


class DerivedMixin:
    """Event/entity/alert methods of `Store` (uses Store's `_conn` and `transaction`)."""

    _conn: sqlite3.Connection

    # ---- batch writes ----------------------------------------------------------------------
    def replace_derived(self, events: list[Event], entities: list[Entity]) -> tuple[int, int, int]:
        """Atomically replace ALL stored events and entities.

        Returns the previous (events, entities, alerts) counts. Alerts reference events, so they
        are discarded too; run the rule engine again afterwards.

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
            previous = self.count_events(), self.count_entities(), self.count_alerts()
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
        """Delete all alerts, events, evidence links and entities (children first)."""
        with self.transaction():  # type: ignore[attr-defined]
            tables = (
                "alert_history",
                "alert_telemetry",
                "alert_events",
                "alerts",
                "entity_events",
                "entities",
                "event_observations",
                "events",
            )
            for table in tables:
                self._conn.execute(f"DELETE FROM {table}")

    _EVENT_COLUMNS = (
        "event_id, object_type, zone, first_seen, first_seen_utc, last_seen, last_seen_utc,"
        " attributes_json, activities_json, description, close_reason, status, confidence,"
        " departure_observed_at"
    )

    @staticmethod
    def _event_row(ev: Event) -> tuple:
        opened = ev.is_open
        return (
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
            _OPEN_PLACEHOLDER if opened else ev.close_reason.value,
            "open" if opened else "closed",
            ev.confidence,
            ev.departure_observed_at.isoformat() if ev.departure_observed_at else None,
        )

    def _insert_event(self, ev: Event) -> None:
        self._conn.execute(
            f"INSERT INTO events ({self._EVENT_COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self._event_row(ev),
        )
        self._conn.executemany(
            "INSERT INTO event_observations (event_id, observation_id, position) VALUES (?, ?, ?)",
            [(ev.event_id, oid, pos) for pos, oid in enumerate(ev.observation_ids)],
        )

    def _insert_entity(self, ent: Entity) -> None:
        self._conn.execute(
            "INSERT INTO entities (entity_id, entity_type, attributes_json, match_level,"
            " first_seen, first_seen_utc, last_seen, last_seen_utc, match_score)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                ent.entity_id,
                ent.entity_type.value,
                canonical_json(ent.attributes),
                ent.match_level.value,
                ent.first_seen.isoformat(),
                utc_key(ent.first_seen),
                ent.last_seen.isoformat(),
                utc_key(ent.last_seen),
                ent.match_score,
            ),
        )
        self._insert_entity_links(ent)

    def _insert_entity_links(self, ent: Entity) -> None:
        self._conn.executemany(
            "INSERT INTO entity_events (entity_id, event_id, link_score, link_components_json)"
            " VALUES (?, ?, ?, ?)",
            [
                (
                    ent.entity_id,
                    eid,
                    ent.link_scores.get(eid),
                    canonical_json(ent.link_components[eid])
                    if eid in ent.link_components
                    else None,
                )
                for eid in ent.event_ids
            ],
        )

    # ---- streaming writes ------------------------------------------------------------------
    def save_events(self, events: Iterable[Event]) -> None:
        """Insert or update events (open or closed) and their evidence links, in place.

        Uses an upsert, not delete + insert, so alerts and entity links that already reference an
        event stay valid while the event grows.
        """
        with self.transaction():  # type: ignore[attr-defined]
            for ev in events:
                cols = self._EVENT_COLUMNS.split(", ")
                updates = ", ".join(f"{c} = excluded.{c}" for c in cols[1:])
                try:
                    self._conn.execute(
                        f"INSERT INTO events ({self._EVENT_COLUMNS})"
                        f" VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                        f" ON CONFLICT(event_id) DO UPDATE SET {updates}",
                        self._event_row(ev),
                    )
                    self._conn.execute(
                        "DELETE FROM event_observations WHERE event_id = ?", (ev.event_id,)
                    )
                    self._conn.executemany(
                        "INSERT INTO event_observations (event_id, observation_id, position)"
                        " VALUES (?, ?, ?)",
                        [(ev.event_id, oid, pos) for pos, oid in enumerate(ev.observation_ids)],
                    )
                except sqlite3.IntegrityError as exc:
                    raise translate_integrity_error(exc, f"event {ev.event_id}") from exc

    def save_entities(self, entities: Iterable[Entity]) -> None:
        """Make the stored entities equal `entities` without touching alerts' entity references.

        Entities are recomputed from all events after each streaming step, so membership can
        change; rows are upserted and membership rewritten. An entity that no longer exists is
        removed only if no alert cites it.
        """
        entities = list(entities)
        with self.transaction():  # type: ignore[attr-defined]
            self._conn.execute("DELETE FROM entity_events")
            for ent in entities:
                self._conn.execute(
                    "INSERT INTO entities (entity_id, entity_type, attributes_json, match_level,"
                    " first_seen, first_seen_utc, last_seen, last_seen_utc, match_score)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(entity_id) DO UPDATE SET"
                    " attributes_json = excluded.attributes_json,"
                    " match_level = excluded.match_level, first_seen = excluded.first_seen,"
                    " first_seen_utc = excluded.first_seen_utc, last_seen = excluded.last_seen,"
                    " last_seen_utc = excluded.last_seen_utc, match_score = excluded.match_score",
                    (
                        ent.entity_id,
                        ent.entity_type.value,
                        canonical_json(ent.attributes),
                        ent.match_level.value,
                        ent.first_seen.isoformat(),
                        utc_key(ent.first_seen),
                        ent.last_seen.isoformat(),
                        utc_key(ent.last_seen),
                        ent.match_score,
                    ),
                )
                self._insert_entity_links(ent)
            keep = [e.entity_id for e in entities]
            marks = ",".join("?" * len(keep)) or "NULL"
            self._conn.execute(
                f"DELETE FROM entities WHERE entity_id NOT IN ({marks})"
                " AND entity_id NOT IN (SELECT entity_id FROM alerts WHERE entity_id IS NOT NULL)",
                keep,
            )

    # ---- event / entity reads ----------------------------------------------------------------
    def get_event(self, event_id: str) -> Event | None:
        rows = self._event_rows("e.event_id = ?", [event_id])
        return rows[0] if rows else None

    @staticmethod
    def _event_filter(
        zone: Zone | None,
        object_type: ObjectType | None,
        entity_id: str | None,
        start: datetime | None,
        end: datetime | None,
        open_only: bool,
        closed_only: bool = False,
    ) -> tuple[str, list[object]]:
        where: list[str] = []
        params: list[object] = []
        if zone is not None:
            where.append("e.zone = ?")
            params.append(Zone(zone).value)
        if object_type is not None:
            where.append("e.object_type = ?")
            params.append(ObjectType(object_type).value)
        if entity_id is not None:
            where.append("ee.entity_id = ?")
            params.append(entity_id)
        if start is not None:
            where.append("e.last_seen_utc >= ?")
            params.append(utc_key(start))
        if end is not None:
            where.append("e.first_seen_utc <= ?")
            params.append(utc_key(end))
        if open_only:
            where.append("e.status = 'open'")
        if closed_only:
            where.append("e.status = 'closed'")
        return " AND ".join(where) or "1 = 1", params

    def query_events(
        self,
        *,
        zone: Zone | None = None,
        object_type: ObjectType | None = None,
        entity_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        open_only: bool = False,
        closed_only: bool = False,
        limit: int | None = None,
    ) -> list[Event]:
        """Events matching ALL given filters, ordered by first_seen then event_id (earliest
        first; `limit` keeps the first N, so a bounded caller never loads the rest).

        `start`/`end` select events whose [first_seen, last_seen] overlaps that time range.
        """
        condition, params = self._event_filter(
            zone, object_type, entity_id, start, end, open_only, closed_only
        )
        return self._event_rows(condition, params, limit)

    def count_events_matching(
        self,
        *,
        zone: Zone | None = None,
        object_type: ObjectType | None = None,
        entity_id: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        open_only: bool = False,
        closed_only: bool = False,
    ) -> int:
        """How many events `query_events` would return without a limit."""
        condition, params = self._event_filter(
            zone, object_type, entity_id, start, end, open_only, closed_only
        )
        return self._conn.execute(
            "SELECT COUNT(*) FROM events e LEFT JOIN entity_events ee ON ee.event_id = e.event_id"
            f" WHERE {condition}",
            params,
        ).fetchone()[0]

    def count_timeline_event_items(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> int:
        """Timeline items the events in a window contribute: one "opened" each, plus one
        "closed" for every closed event that lasted more than a single instant."""
        condition, params = self._event_filter(None, None, None, start, end, False)
        row = self._conn.execute(
            "SELECT COUNT(*) AS opened, COALESCE(SUM(e.status = 'closed'"
            " AND e.last_seen_utc != e.first_seen_utc), 0) AS closed FROM events e"
            f" WHERE {condition}",
            params,
        ).fetchone()
        return row["opened"] + row["closed"]

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

    def event_ids_for_observation(self, observation_id: str) -> list[str]:
        """The events whose evidence includes this observation (reverse of the evidence link)."""
        return [
            r["event_id"]
            for r in self._conn.execute(
                "SELECT eo.event_id FROM event_observations eo"
                " JOIN events e ON e.event_id = eo.event_id WHERE eo.observation_id = ?"
                " ORDER BY e.first_seen_utc, e.event_id",
                (observation_id,),
            )
        ]

    def get_entity(self, entity_id: str) -> Entity | None:
        rows = self._entity_rows("entity_id = ?", [entity_id])
        return rows[0] if rows else None

    def list_entities(self, limit: int | None = None) -> list[Entity]:
        """Entities earliest first; `limit` keeps the first N."""
        return self._entity_rows("1 = 1", [], limit)

    def count_events(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def count_entities(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    def _event_rows(
        self, condition: str, params: list[object], limit: int | None = None
    ) -> list[Event]:
        sql = (
            "SELECT e.*, ee.entity_id FROM events e"
            " LEFT JOIN entity_events ee ON ee.event_id = e.event_id"
            f" WHERE {condition} ORDER BY e.first_seen_utc, e.event_id"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        rows = self._conn.execute(sql, params).fetchall()
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
                    "close_reason": (
                        CloseReason.OPEN
                        if r["status"] == "open"
                        else CloseReason(r["close_reason"])
                    ),
                    "entity_id": r["entity_id"],
                    "confidence": r["confidence"],
                    "departure_observed_at": (
                        datetime.fromisoformat(r["departure_observed_at"])
                        if r["departure_observed_at"]
                        else None
                    ),
                }
            )
            for r in rows
        ]

    def _entity_rows(
        self, condition: str, params: list[object], limit: int | None = None
    ) -> list[Entity]:
        sql = f"SELECT * FROM entities WHERE {condition} ORDER BY first_seen_utc, entity_id"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        rows = self._conn.execute(sql, params).fetchall()
        members: dict[str, list[str]] = {r["entity_id"]: [] for r in rows}
        scores: dict[str, dict[str, float]] = {r["entity_id"]: {} for r in rows}
        components: dict[str, dict[str, dict[str, float]]] = {r["entity_id"]: {} for r in rows}
        ids = list(members)
        for i in range(0, len(ids), _IN_CHUNK):  # only the returned entities' links are read
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                "SELECT ee.entity_id, ee.event_id, ee.link_score, ee.link_components_json"
                " FROM entity_events ee JOIN events e ON e.event_id = ee.event_id"
                f" WHERE ee.entity_id IN ({marks}) ORDER BY e.first_seen_utc, e.event_id",
                chunk,
            ):
                members[r["entity_id"]].append(r["event_id"])
                if r["link_score"] is not None:
                    scores[r["entity_id"]][r["event_id"]] = r["link_score"]
                if r["link_components_json"]:
                    components[r["entity_id"]][r["event_id"]] = json.loads(
                        r["link_components_json"]
                    )
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
                    "match_score": r["match_score"],
                    "link_scores": scores[r["entity_id"]],
                    "link_components": components[r["entity_id"]],
                }
            )
            for r in rows
            if members[r["entity_id"]]
        ]

    # ---- alerts ----------------------------------------------------------------------------
    def replace_alerts(self, alerts: list[Alert]) -> int:
        """Atomically replace ALL stored alerts with `alerts`. Returns the previous count.

        Alerts are derived (events + entities + rules), so a re-run replaces them; deterministic
        alert ids mean unchanged inputs give identical rows. Links must reference stored events,
        entities and telemetry samples (foreign keys).
        """
        with self.transaction():  # type: ignore[attr-defined]
            previous = self.count_alerts()
            for table in ("alert_history", "alert_telemetry", "alert_events", "alerts"):
                self._conn.execute(f"DELETE FROM {table}")
            try:
                for alert in alerts:
                    self._insert_alert(alert)
            except sqlite3.IntegrityError as exc:
                raise translate_integrity_error(exc, "alert data") from exc
            return previous

    def save_alert(self, alert: Alert) -> None:
        """Insert or update one alert with its links and full lifecycle history (streaming)."""
        with self.transaction():  # type: ignore[attr-defined]
            try:
                exists = self._conn.execute(
                    "SELECT 1 FROM alerts WHERE alert_id = ?", (alert.alert_id,)
                ).fetchone()
                if exists:
                    for table in ("alert_history", "alert_telemetry", "alert_events"):
                        self._conn.execute(
                            f"DELETE FROM {table} WHERE alert_id = ?", (alert.alert_id,)
                        )
                    self._conn.execute("DELETE FROM alerts WHERE alert_id = ?", (alert.alert_id,))
                self._insert_alert(alert)
            except sqlite3.IntegrityError as exc:
                raise translate_integrity_error(exc, f"alert {alert.alert_id}") from exc

    _ALERT_COLUMNS = (
        "alert_id, rule_id, severity, title, message, triggered_at, triggered_at_utc,"
        " first_seen, first_seen_utc, last_seen, last_seen_utc, zone, entity_id, facts_json,"
        " status, confidence, verdict, recommended_action, ladder_json, detected_at, updated_at,"
        " resolved_at"
    )

    def _insert_alert(self, a: Alert) -> None:
        ladder = [
            {"at": s.at.isoformat(), "severity": s.severity.value, "label": s.label}
            for s in a.ladder
        ]
        self._conn.execute(
            f"INSERT INTO alerts ({self._ALERT_COLUMNS}) VALUES ({','.join('?' * 22)})",
            (
                a.alert_id,
                a.rule_id,
                a.severity.value,
                a.title,
                a.message,
                a.triggered_at.isoformat(),
                utc_key(a.triggered_at),
                a.first_seen.isoformat(),
                utc_key(a.first_seen),
                a.last_seen.isoformat(),
                utc_key(a.last_seen),
                a.zone.value if a.zone else None,
                a.entity_id,
                canonical_json(a.facts),
                a.status.value,
                a.confidence,
                a.verdict.value if a.verdict else None,
                a.recommended_action,
                json.dumps(ladder, sort_keys=True),
                a.detected_at.isoformat() if a.detected_at else None,
                a.updated_at.isoformat() if a.updated_at else None,
                a.resolved_at.isoformat() if a.resolved_at else None,
            ),
        )
        self._conn.executemany(
            "INSERT INTO alert_events (alert_id, event_id) VALUES (?, ?)",
            [(a.alert_id, eid) for eid in a.event_ids],
        )
        self._conn.executemany(
            "INSERT INTO alert_telemetry (alert_id, telemetry_ts_utc) VALUES (?, ?)",
            [(a.alert_id, utc_key(ts)) for ts in a.telemetry_at],
        )
        self._conn.executemany(
            "INSERT INTO alert_history (alert_id, seq, kind, at, at_utc, effective_at,"
            " effective_at_utc, from_status, to_status, from_severity, to_severity, reason, actor)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    a.alert_id,
                    t.seq,
                    t.kind.value,
                    t.at.isoformat(),
                    utc_key(t.at),
                    t.effective_at.isoformat(),
                    utc_key(t.effective_at),
                    t.from_status.value if t.from_status else None,
                    t.to_status.value,
                    t.from_severity.value if t.from_severity else None,
                    t.to_severity.value,
                    t.reason,
                    t.actor,
                )
                for t in a.history
            ],
        )

    def count_alerts(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def get_alert(self, alert_id: str) -> Alert | None:
        rows = self._alert_rows("a.alert_id = ?", [alert_id])
        return rows[0] if rows else None

    @staticmethod
    def _alert_filter(
        rule_id: str | None,
        severity: Severity | None,
        zone: Zone | None,
        status: AlertStatus | None,
        active_only: bool,
        entity_id: str | None = None,
    ) -> tuple[str, list[object]]:
        where: list[str] = []
        params: list[object] = []
        if rule_id is not None:
            where.append("a.rule_id = ?")
            params.append(rule_id)
        if severity is not None:
            where.append("a.severity = ?")
            params.append(Severity(severity).value)
        if zone is not None:
            where.append("a.zone = ?")
            params.append(Zone(zone).value)
        if status is not None:
            where.append("a.status = ?")
            params.append(AlertStatus(status).value)
        if active_only:
            where.append("a.status != 'resolved'")
        if entity_id is not None:
            where.append("a.entity_id = ?")
            params.append(entity_id)
        return " AND ".join(where) or "1 = 1", params

    def query_alerts(
        self,
        *,
        rule_id: str | None = None,
        severity: Severity | None = None,
        zone: Zone | None = None,
        status: AlertStatus | None = None,
        active_only: bool = False,
        entity_id: str | None = None,
        limit: int | None = None,
    ) -> list[Alert]:
        """Alerts matching ALL given filters, ordered by triggered_at then alert_id (earliest
        first; `limit` keeps the first N, so a bounded caller never loads the rest)."""
        condition, params = self._alert_filter(
            rule_id, severity, zone, status, active_only, entity_id
        )
        return self._alert_rows(condition, params, limit)

    def count_alerts_matching(
        self,
        *,
        rule_id: str | None = None,
        severity: Severity | None = None,
        zone: Zone | None = None,
        status: AlertStatus | None = None,
        active_only: bool = False,
        entity_id: str | None = None,
    ) -> int:
        """How many alerts `query_alerts` would return without a limit."""
        condition, params = self._alert_filter(
            rule_id, severity, zone, status, active_only, entity_id
        )
        return self._conn.execute(
            f"SELECT COUNT(*) FROM alerts a WHERE {condition}", params
        ).fetchone()[0]

    def alert_counts_by(self, column: str) -> dict[str, int]:
        """Alert counts per severity, rule or status (a fixed whitelist, never user text)."""
        names = {"severity": "severity", "rule": "rule_id", "status": "status"}
        if column not in names:
            raise ValueError("column must be 'severity', 'rule' or 'status'")
        field_name = names[column]
        rows = self._conn.execute(
            f"SELECT {field_name} AS k, COUNT(*) AS n FROM alerts GROUP BY {field_name}"
            " ORDER BY MIN(triggered_at_utc), k"
        )
        return {r["k"]: r["n"] for r in rows}

    def alerts_for_events(self, event_ids: list[str], limit: int | None = None) -> list[Alert]:
        """Alerts that cite any of these events (earliest first)."""
        if not event_ids:
            return []
        marks = ",".join("?" * len(event_ids))
        return self._alert_rows(
            f"a.alert_id IN (SELECT alert_id FROM alert_events WHERE event_id IN ({marks}))",
            list(event_ids),
            limit,
        )

    def alert_ids_by_event(self, event_ids: list[str]) -> dict[str, list[str]]:
        """For each event, the ids of the alerts that cite it (in chronological alert order)."""
        out: dict[str, list[str]] = {}
        for i in range(0, len(event_ids), _IN_CHUNK):
            chunk = event_ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                "SELECT ae.event_id, ae.alert_id FROM alert_events ae"
                " JOIN alerts a ON a.alert_id = ae.alert_id"
                f" WHERE ae.event_id IN ({marks}) ORDER BY a.triggered_at_utc, a.alert_id",
                chunk,
            ):
                out.setdefault(r["event_id"], []).append(r["alert_id"])
        return out

    def timeline_alert_items(
        self, start: datetime | None, end: datetime | None, limit: int | None = None
    ) -> list[dict]:
        """Alert lifecycle changes whose effective time falls in [start, end], earliest first."""
        where, params = self._history_window(start, end)
        sql = (
            "SELECT h.alert_id, a.title, h.kind, h.effective_at, h.to_severity, h.to_status,"
            " h.actor FROM alert_history h JOIN alerts a ON a.alert_id = h.alert_id"
            f" WHERE {where} ORDER BY h.effective_at_utc,"
            " CASE h.kind WHEN 'raised' THEN 1 ELSE 2 END, h.alert_id, h.seq"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        return [dict(r) for r in self._conn.execute(sql, params)]

    def count_timeline_alert_items(
        self, start: datetime | None = None, end: datetime | None = None
    ) -> int:
        where, params = self._history_window(start, end)
        return self._conn.execute(
            f"SELECT COUNT(*) FROM alert_history h WHERE {where}", params
        ).fetchone()[0]

    @staticmethod
    def _history_window(start: datetime | None, end: datetime | None) -> tuple[str, list[object]]:
        where, params = [], []
        if start is not None:
            where.append("h.effective_at_utc >= ?")
            params.append(utc_key(start))
        if end is not None:
            where.append("h.effective_at_utc <= ?")
            params.append(utc_key(end))
        return " AND ".join(where) or "1 = 1", params

    def _alert_rows(
        self, condition: str, params: list[object], limit: int | None = None
    ) -> list[Alert]:
        sql = f"SELECT a.* FROM alerts a WHERE {condition} ORDER BY a.triggered_at_utc, a.alert_id"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        rows = self._conn.execute(sql, params).fetchall()
        events: dict[str, list[str]] = {r["alert_id"]: [] for r in rows}
        frames: dict[str, set[int]] = {r["alert_id"]: set() for r in rows}
        telemetry: dict[str, list[datetime]] = {r["alert_id"]: [] for r in rows}
        history: dict[str, list[AlertTransition]] = {r["alert_id"]: [] for r in rows}
        ids = list(events)
        for i in range(0, len(ids), _IN_CHUNK):
            chunk = ids[i : i + _IN_CHUNK]
            marks = ",".join("?" * len(chunk))
            for r in self._conn.execute(
                "SELECT ae.alert_id, ae.event_id FROM alert_events ae"
                " JOIN events e ON e.event_id = ae.event_id"
                f" WHERE ae.alert_id IN ({marks}) ORDER BY e.first_seen_utc, e.event_id",
                chunk,
            ):
                events[r["alert_id"]].append(r["event_id"])
            for r in self._conn.execute(
                "SELECT ae.alert_id, o.frame_id FROM alert_events ae"
                " JOIN event_observations eo ON eo.event_id = ae.event_id"
                " JOIN observations o ON o.observation_id = eo.observation_id"
                f" WHERE ae.alert_id IN ({marks})",
                chunk,
            ):
                frames[r["alert_id"]].add(r["frame_id"])
            for r in self._conn.execute(
                "SELECT atl.alert_id, t.timestamp FROM alert_telemetry atl"
                " JOIN telemetry t ON t.ts_utc = atl.telemetry_ts_utc"
                f" WHERE atl.alert_id IN ({marks}) ORDER BY t.ts_utc",
                chunk,
            ):
                telemetry[r["alert_id"]].append(datetime.fromisoformat(r["timestamp"]))
            for r in self._conn.execute(
                f"SELECT * FROM alert_history WHERE alert_id IN ({marks}) ORDER BY alert_id, seq",
                chunk,
            ):
                history[r["alert_id"]].append(
                    AlertTransition.model_validate(
                        {
                            "seq": r["seq"],
                            "kind": r["kind"],
                            "at": datetime.fromisoformat(r["at"]),
                            "effective_at": datetime.fromisoformat(r["effective_at"]),
                            "from_status": r["from_status"],
                            "to_status": r["to_status"],
                            "from_severity": r["from_severity"],
                            "to_severity": r["to_severity"],
                            "reason": r["reason"],
                            "actor": r["actor"],
                        }
                    )
                )
        return [
            Alert.model_validate(
                {
                    "alert_id": r["alert_id"],
                    "rule_id": r["rule_id"],
                    "severity": r["severity"],
                    "title": r["title"],
                    "message": r["message"],
                    "triggered_at": datetime.fromisoformat(r["triggered_at"]),
                    "first_seen": datetime.fromisoformat(r["first_seen"]),
                    "last_seen": datetime.fromisoformat(r["last_seen"]),
                    "zone": r["zone"],
                    "event_ids": events[r["alert_id"]],
                    "entity_id": r["entity_id"],
                    "frame_ids": sorted(frames[r["alert_id"]]),
                    "facts": json.loads(r["facts_json"]),
                    "status": r["status"],
                    "confidence": r["confidence"],
                    "verdict": Verdict(r["verdict"]) if r["verdict"] else None,
                    "recommended_action": r["recommended_action"],
                    "ladder": [
                        LadderStep(
                            at=datetime.fromisoformat(s["at"]),
                            severity=Severity(s["severity"]),
                            label=s["label"],
                        )
                        for s in json.loads(r["ladder_json"])
                    ],
                    "detected_at": (
                        datetime.fromisoformat(r["detected_at"]) if r["detected_at"] else None
                    ),
                    "updated_at": (
                        datetime.fromisoformat(r["updated_at"]) if r["updated_at"] else None
                    ),
                    "resolved_at": (
                        datetime.fromisoformat(r["resolved_at"]) if r["resolved_at"] else None
                    ),
                    "telemetry_at": telemetry[r["alert_id"]],
                    "history": history[r["alert_id"]],
                }
            )
            for r in rows
        ]
