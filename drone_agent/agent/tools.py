"""Read-only analyst tools over the stored evidence.

These are the ONLY way the analyst reaches the database. They are plain methods with typed,
validated arguments (no SQL, no free-form queries), each returning a bounded JSON-safe dict that
carries citation refs. Invalid arguments come back as {"error": ...} instead of raising, so a
model can correct itself. Nothing here writes to the store. No LangChain here (see llm.py).
"""

from datetime import datetime

from drone_agent.agent.evidence import (
    ToolLimits,
    alert_view,
    clip,
    collect_refs,
    entity_view,
    event_view,
    observation_view,
)
from drone_agent.agent.summary import run_facts
from drone_agent.models import ObjectType, Severity, Zone
from drone_agent.store import Store

TOOL_NAMES = (
    "search_observations",
    "list_events",
    "get_event",
    "get_event_evidence",
    "list_entities",
    "get_entity",
    "get_entity_history",
    "list_alerts",
    "get_alert",
    "get_run_summary",
)


def _enum(enum_cls, value, name: str):
    if value in (None, ""):
        return None
    try:
        return enum_cls(str(value).lower())
    except ValueError:
        allowed = ", ".join(m.value for m in enum_cls)
        raise ValueError(f"{name} must be one of: {allowed}") from None


def _time(value, name: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{name} must be an ISO-8601 timestamp with an offset") from None
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a UTC offset, e.g. 2026-09-29T23:40:00+05:30")
    return parsed


class AnalystTools:
    """Bounded read-only views. `retrieved` accumulates every ref returned, for citation checks."""

    def __init__(self, store: Store, limits: ToolLimits | None = None) -> None:
        self.store = store
        self.limits = limits or ToolLimits()
        self.retrieved: set[str] = set()

    # -- internal ---------------------------------------------------------------------------
    def _ok(self, result: dict) -> dict:
        self.retrieved |= collect_refs(result)
        return result

    @staticmethod
    def _bad(exc: Exception) -> dict:
        return {"error": str(exc)}

    def _cap(self, requested: int | None, default: int, maximum: int) -> int:
        n = default if requested is None else int(requested)
        return max(1, min(n, maximum))

    # -- observations -----------------------------------------------------------------------
    def search_observations(
        self,
        query: str | None = None,
        zone: str | None = None,
        object_type: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Keyword (FTS5) and/or filtered search over observations; newest evidence last."""
        try:
            filters = {
                "zone": _enum(Zone, zone, "zone"),
                "object_type": _enum(ObjectType, object_type, "object_type"),
                "start": _time(start, "start"),
                "end": _time(end, "end"),
                "text": query if query and query.strip() else None,
            }
            cap = self._cap(limit, self.limits.default_observations, self.limits.max_observations)
            total = self.store.count_matching_observations(**filters)
            found = self.store.query_observations(**filters, limit=cap)
        except ValueError as exc:
            return self._bad(exc)
        return self._ok(
            {
                "total": total,
                "returned": len(found),
                "truncated": total > len(found),
                "observations": [observation_view(o, self.limits) for o in found],
            }
        )

    # -- events -----------------------------------------------------------------------------
    def list_events(
        self, zone: str | None = None, object_type: str | None = None, limit: int | None = None
    ) -> dict:
        try:
            events = self.store.query_events(
                zone=_enum(Zone, zone, "zone"),
                object_type=_enum(ObjectType, object_type, "object_type"),
            )
        except ValueError as exc:
            return self._bad(exc)
        cap = self._cap(limit, self.limits.max_events, self.limits.max_events)
        return self._ok(
            {
                "total": len(events),
                "returned": min(len(events), cap),
                "truncated": len(events) > cap,
                "events": [event_view(e, self.limits) for e in events[:cap]],
            }
        )

    def get_event(self, event_id: str) -> dict:
        ev = self.store.get_event(str(event_id))
        if ev is None:
            return {"error": f"no event with id '{clip(str(event_id), 60)}'"}
        return self._ok({"event": event_view(ev, self.limits)})

    def get_event_evidence(self, event_id: str) -> dict:
        """The event and the observations (frames) it was built from, bounded."""
        ev = self.store.get_event(str(event_id))
        if ev is None:
            return {"error": f"no event with id '{clip(str(event_id), 60)}'"}
        cap = self.limits.max_evidence
        obs = self.store.get_event_observations(ev.event_id)
        return self._ok(
            {
                "event": event_view(ev, self.limits),
                "observations_total": len(obs),
                "observations_returned": min(len(obs), cap),
                "truncated": len(obs) > cap,
                "observations": [observation_view(o, self.limits) for o in obs[:cap]],
            }
        )

    # -- entities ---------------------------------------------------------------------------
    def list_entities(self) -> dict:
        entities = self.store.list_entities()
        cap = self.limits.max_entities
        return self._ok(
            {
                "total": len(entities),
                "returned": min(len(entities), cap),
                "truncated": len(entities) > cap,
                "entities": [entity_view(e) for e in entities[:cap]],
            }
        )

    def get_entity(self, entity_id: str) -> dict:
        ent = self.store.get_entity(str(entity_id))
        if ent is None:
            return {"error": f"no entity with id '{clip(str(entity_id), 60)}'"}
        return self._ok({"entity": entity_view(ent)})

    def get_entity_history(self, entity_id: str) -> dict:
        """Every separate appearance (event) of a vehicle entity, plus alerts about it."""
        ent = self.store.get_entity(str(entity_id))
        if ent is None:
            return {"error": f"no entity with id '{clip(str(entity_id), 60)}'"}
        cap = self.limits.max_events
        appearances = [self.store.get_event(e) for e in ent.event_ids[:cap]]
        alerts = [a for a in self.store.query_alerts() if a.entity_id == ent.entity_id]
        return self._ok(
            {
                "entity": entity_view(ent),
                "note": "coarse attribute match; identity is not confirmed unless match_level "
                "is 'confirmed'",
                "appearances_total": len(ent.event_ids),
                "appearances": [event_view(e, self.limits) for e in appearances if e],
                "alerts": [alert_view(a, self.limits) for a in alerts[: self.limits.max_alerts]],
            }
        )

    # -- alerts -----------------------------------------------------------------------------
    def list_alerts(
        self,
        rule_id: str | None = None,
        severity: str | None = None,
        zone: str | None = None,
        limit: int | None = None,
    ) -> dict:
        try:
            alerts = self.store.query_alerts(
                rule_id=rule_id or None,
                severity=_enum(Severity, severity, "severity"),
                zone=_enum(Zone, zone, "zone"),
            )
        except ValueError as exc:
            return self._bad(exc)
        cap = self._cap(limit, self.limits.max_alerts, self.limits.max_alerts)
        return self._ok(
            {
                "total": len(alerts),
                "returned": min(len(alerts), cap),
                "truncated": len(alerts) > cap,
                "alerts": [alert_view(a, self.limits) for a in alerts[:cap]],
            }
        )

    def get_alert(self, alert_id: str) -> dict:
        """One alert with the events (and entity) that triggered it; alerts are rule output."""
        alert = self.store.get_alert(str(alert_id))
        if alert is None:
            return {"error": f"no alert with id '{clip(str(alert_id), 60)}'"}
        events = [self.store.get_event(e) for e in alert.event_ids[: self.limits.max_events]]
        result = {
            "alert": alert_view(alert, self.limits),
            "events": [event_view(e, self.limits) for e in events if e],
        }
        if alert.entity_id and (ent := self.store.get_entity(alert.entity_id)):
            result["entity"] = entity_view(ent)
        return self._ok(result)

    # -- run --------------------------------------------------------------------------------
    def get_run_summary(self) -> dict:
        return self._ok(run_facts(self.store, self.limits))
