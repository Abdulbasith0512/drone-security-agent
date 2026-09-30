"""Read-only analyst tools over the stored evidence.

These are the ONLY way the analyst reaches the database. They are plain methods with typed,
validated arguments (no SQL, no free-form queries), each returning a bounded JSON-safe dict that
carries citation refs. Invalid arguments come back as {"error": ...} instead of raising, so a
model can correct itself. Nothing here writes to the store. No LangChain here (see llm.py).

The tools a language model sees (TOOL_NAMES) are the investigation set:

    query_events   search_events   search_frames  get_event    get_frame     get_alert
    get_evidence   entity_history  list_entities  list_alerts  telemetry_at  get_timeline
    summarize_run

A few older methods (list_events, get_entity, ...) remain because the deterministic template
analyst calls them directly; they are not offered to a model.
"""

import json
from datetime import datetime

from drone_agent.agent.evidence import (
    ToolLimits,
    alert_view,
    clip,
    collect_refs,
    entity_view,
    event_view,
    observation_view,
    ref,
    telemetry_view,
)
from drone_agent.agent.summary import run_facts
from drone_agent.models import AlertStatus, ObjectType, Severity, Telemetry, Zone
from drone_agent.pipeline.recurrence import MAX_SCORE, similar_earlier_person_events
from drone_agent.store import Store

TOOL_NAMES = (
    "query_events",
    "search_events",
    "search_frames",
    "get_event",
    "get_frame",
    "get_alert",
    "get_evidence",
    "entity_history",
    "list_entities",
    "list_alerts",
    "telemetry_at",
    "get_timeline",
    "summarize_run",
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
    """Bounded read-only views. `retrieved` accumulates every ref returned, for citation checks.

    `fit_results` keeps each result within the model's tool-result size cap (see `_fit`); the
    template analyst renders text, not JSON, so it leaves this off.
    """

    def __init__(
        self, store: Store, limits: ToolLimits | None = None, fit_results: bool = False
    ) -> None:
        self.store = store
        self.limits = limits or ToolLimits()
        self.fit_results = fit_results
        self.retrieved: set[str] = set()
        self.calls: list[dict] = []  # a trace of every tool call this session (name, args, outcome)

    # -- internal ---------------------------------------------------------------------------
    def _ok(self, result: dict) -> dict:
        result = self._fit(result)
        self.retrieved |= collect_refs(result)  # only what is actually returned counts as seen
        return result

    def _fit(self, result: dict) -> dict:
        """Keep one tool result within `max_result_chars` by dropping items from the end of its
        longest list, so a busy run degrades to "showing N of M" instead of an unusable oversized
        result. The totals stay and the result says it was trimmed; nothing is silently dropped."""

        if not self.fit_results:
            return result

        def size(node: object) -> int:
            return len(json.dumps(node, default=str))

        trimmed: list[str] = []
        while size(result) > self.limits.max_result_chars:
            lists = [k for k, v in result.items() if isinstance(v, list) and len(v) > 1]
            if not lists:
                break
            key = max(lists, key=lambda k: size(result[k]))
            result[key] = result[key][:-1]
            trimmed.append(key)
        for key in trimmed:
            if f"{key}_truncated" in result:
                result[f"{key}_truncated"] = True
            if "returned" in result:
                result["returned"] = len(result[key])
        if trimmed:
            result["truncated"] = True
            result["trimmed_to_fit"] = sorted(set(trimmed))
        return result

    @staticmethod
    def _bad(exc: Exception) -> dict:
        return {"error": str(exc)}

    def _cap(self, requested: int | None, default: int, maximum: int) -> int:
        n = default if requested is None else int(requested)
        return max(1, min(n, maximum))

    def trace(self, name: str, args: dict, result: dict) -> None:
        """Record one call for the run trace: what was asked and how much came back."""
        self.calls.append(
            {
                "tool": name,
                "args": {k: v for k, v in args.items() if v not in (None, "")},
                "ok": "error" not in result,
                "refs": len(collect_refs(result)),
                **({"error": clip(str(result["error"]), 120)} if "error" in result else {}),
            }
        )

    def _obs_view(self, obs) -> dict:
        telemetry_id = None
        if obs.telemetry is not None:
            telemetry_id = self.store.telemetry_id_for(obs.telemetry.telemetry.timestamp)
        return observation_view(obs, self.limits, telemetry_id)

    def _alert_view(self, alert, detail: bool = True) -> dict:
        ids = {}
        for stamp in alert.telemetry_at:
            found = self.store.telemetry_id_for(stamp)
            if found is not None:
                ids[stamp] = found
        return alert_view(alert, self.limits, ids, detail)

    # -- frames and observations --------------------------------------------------------------
    def search_frames(
        self,
        query: str | None = None,
        zone: str | None = None,
        object_type: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Keyword (FTS5) and/or filtered search over frame observations."""
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
                "observations": [self._obs_view(o) for o in found],
            }
        )

    search_observations = search_frames  # the name the template analyst uses

    def get_frame(self, frame_id: int) -> dict:
        """One frame's observation with its telemetry, the events it is evidence for and their
        alerts."""
        try:
            frame = int(frame_id)
        except (TypeError, ValueError):
            return {"error": f"frame_id must be an integer, got '{clip(str(frame_id), 40)}'"}
        obs = self.store.get_observation_by_frame(frame)
        if obs is None:
            return {"error": f"no frame {frame} is stored"}
        events = [
            self.store.get_event(e)
            for e in self.store.event_ids_for_observation(obs.observation_id)
        ]
        events = [e for e in events if e]
        alerts = self.store.alerts_for_events(
            [e.event_id for e in events], limit=self.limits.max_alerts
        )
        return self._ok(
            {
                "observation": self._obs_view(obs),
                "events": [event_view(e, self.limits) for e in events],
                "alerts": [self._alert_view(a, detail=False) for a in alerts],
                "note": "frames are text descriptions here; no image is stored",
            }
        )

    # -- events -----------------------------------------------------------------------------
    def query_events(
        self,
        zone: str | None = None,
        object_type: str | None = None,
        start: str | None = None,
        end: str | None = None,
        entity_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Events filtered by zone, type, time overlap ([start, end]), entity and open/closed."""
        try:
            if status not in (None, "") and str(status).lower() not in ("open", "closed"):
                raise ValueError("status must be one of: open, closed")
            filters = {
                "zone": _enum(Zone, zone, "zone"),
                "object_type": _enum(ObjectType, object_type, "object_type"),
                "entity_id": entity_id or None,
                "start": _time(start, "start"),
                "end": _time(end, "end"),
                "open_only": str(status).lower() == "open" if status else False,
                "closed_only": str(status).lower() == "closed" if status else False,
            }
            cap = self._cap(limit, self.limits.max_events, self.limits.max_events)
            total = self.store.count_events_matching(**filters)
            events = self.store.query_events(**filters, limit=cap)
        except ValueError as exc:
            return self._bad(exc)
        return self._ok(
            {
                "total": total,
                "returned": len(events),
                "truncated": total > len(events),
                "events": [event_view(e, self.limits) for e in events],
            }
        )

    def list_events(
        self, zone: str | None = None, object_type: str | None = None, limit: int | None = None
    ) -> dict:
        return self.query_events(zone=zone, object_type=object_type, limit=limit)

    def search_events(
        self,
        query: str,
        zone: str | None = None,
        object_type: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Events whose evidence matches a keyword search ("truck", "blue pickup", "hoodie")."""
        if not query or not str(query).strip():
            return {"error": "query must not be empty"}
        try:
            zone_v = _enum(Zone, zone, "zone")
            type_v = _enum(ObjectType, object_type, "object_type")
            found = self.store.query_observations(text=str(query), zone=zone_v, limit=200)
        except ValueError as exc:
            return self._bad(exc)
        matched: dict[str, list[int]] = {}
        for obs in found:
            for event_id in self.store.event_ids_for_observation(obs.observation_id):
                matched.setdefault(event_id, []).append(obs.frame_id)
        events = [
            e
            for e in (self.store.get_event(i) for i in matched)
            if e and (type_v is None or e.object_type is type_v)
        ]
        events.sort(key=lambda e: (e.first_seen, e.event_id))
        cap = self._cap(limit, self.limits.max_events, self.limits.max_events)
        views = []
        for e in events[:cap]:
            view = event_view(e, self.limits)
            view["matched_frame_refs"] = [
                ref("frame", f)
                for f in sorted(set(matched[e.event_id]))[: self.limits.max_frame_ids]
            ]
            views.append(view)
        return self._ok(
            {
                "query": clip(str(query), 80),
                "total": len(events),
                "returned": len(views),
                "truncated": len(events) > len(views),
                "events": views,
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
                "observations": [self._obs_view(o) for o in obs[:cap]],
            }
        )

    # -- entities ---------------------------------------------------------------------------
    def list_entities(self) -> dict:
        cap = self.limits.max_entities
        entities = self.store.list_entities(limit=cap)
        total = self.store.count_entities()
        return self._ok(
            {
                "total": total,
                "returned": len(entities),
                "truncated": total > len(entities),
                "entities": [entity_view(e) for e in entities],
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
        alerts = self.store.query_alerts(entity_id=ent.entity_id, limit=self.limits.max_alerts)
        return self._ok(
            {
                "entity": entity_view(ent),
                "note": "coarse attribute match; identity is not confirmed unless match_level "
                "is 'confirmed'",
                "appearances_total": len(ent.event_ids),
                "appearances": [event_view(e, self.limits) for e in appearances if e],
                "alerts": [self._alert_view(a, detail=False) for a in alerts],
            }
        )

    def entity_history(self, id: str) -> dict:
        """History for a vehicle entity (ENT-V-...), or recurrence HINTS for a person event
        (EVT-P-...): earlier events in the same zone with similar clothing. People are never
        identity-linked, so a hint is never a match."""
        ident = str(id).strip()
        if ident.upper().startswith("ENT-"):
            return self.get_entity_history(ident.upper())
        event = self.store.get_event(ident) or self.store.get_event(ident.upper())
        if event is None:
            return {"error": f"no entity or event with id '{clip(ident, 60)}'"}
        if event.object_type is not ObjectType.PERSON:
            if event.entity_id:
                return self.get_entity_history(event.entity_id)
            return {"error": f"event {event.event_id} is not linked to a vehicle entity"}
        candidates = self.store.query_events(
            object_type=ObjectType.PERSON, zone=event.zone, end=event.first_seen
        )
        hints = similar_earlier_person_events(event, candidates)
        cap = self.limits.max_events
        return self._ok(
            {
                "event": event_view(event, self.limits),
                "note": (
                    "People are never identity-linked. These are earlier person events in the "
                    "same zone with similar described clothing: a weak hint, not the same person."
                ),
                "hint_score_cap": MAX_SCORE,
                "similar_earlier_events_total": len(hints),
                "similar_earlier_events": [
                    {
                        **event_view(h.event, self.limits),
                        "hint_score": h.score,
                        "why": h.reason,
                    }
                    for h in hints[:cap]
                ],
            }
        )

    # -- alerts -----------------------------------------------------------------------------
    def list_alerts(
        self,
        rule_id: str | None = None,
        severity: str | None = None,
        zone: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> dict:
        try:
            filters = {
                "rule_id": rule_id or None,
                "severity": _enum(Severity, severity, "severity"),
                "zone": _enum(Zone, zone, "zone"),
                "status": _enum(AlertStatus, status, "status"),
            }
            cap = self._cap(limit, self.limits.max_alerts, self.limits.max_alerts)
            total = self.store.count_alerts_matching(**filters)
            alerts = self.store.query_alerts(**filters, limit=cap)
        except ValueError as exc:
            return self._bad(exc)
        return self._ok(
            {
                "total": total,
                "returned": len(alerts),
                "truncated": total > len(alerts),
                "alerts": [self._alert_view(a, detail=False) for a in alerts],
            }
        )

    def get_alert(self, alert_id: str) -> dict:
        """One alert with its lifecycle, the events (and entity) that triggered it; alerts are
        rule output."""
        alert = self.store.get_alert(str(alert_id))
        if alert is None:
            return {"error": f"no alert with id '{clip(str(alert_id), 60)}'"}
        events = [self.store.get_event(e) for e in alert.event_ids[: self.limits.max_events]]
        result = {
            "alert": self._alert_view(alert),
            "events": [event_view(e, self.limits) for e in events if e],
        }
        if alert.entity_id and (ent := self.store.get_entity(alert.entity_id)):
            result["entity"] = entity_view(ent)
        return self._ok(result)

    # -- telemetry and time ---------------------------------------------------------------------
    def telemetry_at(self, time: str) -> dict:
        """The drone's telemetry at a moment: the most recent sample at or before `time`."""
        try:
            moment = _time(time, "time")
        except ValueError as exc:
            return self._bad(exc)
        if moment is None:
            return {"error": "time is required (ISO-8601 with a UTC offset)"}
        found = self.store.telemetry_before(moment)
        if found is None:
            return {"error": "no telemetry is stored at or before that time"}
        tid, sample = found
        age = (moment - sample.timestamp).total_seconds()
        view = telemetry_view(sample, tid, age)
        view["stale"] = age > 30.0  # the project's default staleness threshold
        return self._ok({"telemetry": view, "note": "telemetry is SIMULATED in this project"})

    def get_timeline(
        self, start: str | None = None, end: str | None = None, limit: int | None = None
    ) -> dict:
        """A chronological list of what happened: events opening/closing and alert changes.

        Reads only what the answer can show: the first `cap` events and the first `cap` alert
        changes in the window are enough to know the first `cap` merged items (every item is at
        or after the start of the event/alert it belongs to), and the totals come from counts.
        """
        try:
            lo, hi = _time(start, "start"), _time(end, "end")
        except ValueError as exc:
            return self._bad(exc)
        cap = self._cap(limit, self.limits.max_timeline, self.limits.max_timeline)
        items: list[tuple[datetime, int, dict]] = []
        for ev in self.store.query_events(start=lo, end=hi, limit=cap):
            who = f"{ev.object_type.value} in {ev.zone.value}"
            items.append(
                (
                    ev.first_seen,
                    0,
                    {
                        "kind": "event_opened",
                        "summary": who,
                        "event_ref": ref("event", ev.event_id),
                    },
                )
            )
            if not ev.is_open and ev.last_seen != ev.first_seen:
                closed = f"{who} ({ev.close_reason.value}, {ev.duration_seconds:.0f}s)"
                items.append(
                    (
                        ev.last_seen,
                        3,
                        {
                            "kind": "event_closed",
                            "summary": closed,
                            "event_ref": ref("event", ev.event_id),
                        },
                    )
                )
        for row in self.store.timeline_alert_items(lo, hi, limit=cap):
            items.append(
                (
                    datetime.fromisoformat(row["effective_at"]),
                    1 if row["kind"] == "raised" else 2,
                    {
                        "kind": f"alert_{row['kind']}",
                        "summary": f"{row['title']}: {row['to_severity']}/{row['to_status']}",
                        "alert_ref": ref("alert", row["alert_id"]),
                        "actor": row["actor"],
                    },
                )
            )
        items.sort(key=lambda t: (t[0], t[1], str(t[2].get("event_ref") or t[2].get("alert_ref"))))
        total = self.store.count_timeline_event_items(
            lo, hi
        ) + self.store.count_timeline_alert_items(lo, hi)
        shown = [{"time": t.isoformat(), **body} for t, _, body in items[:cap]]
        return self._ok(
            {
                "total": total,
                "returned": len(shown),
                "truncated": total > len(shown),
                "timeline": shown,
            }
        )

    # -- evidence chain ---------------------------------------------------------------------
    def get_evidence(self, ref_id: str) -> dict:
        """The full evidence chain behind an alert, event or entity:
        alert -> rule facts -> events -> observations (frames) -> telemetry."""
        ident = str(ref_id).strip()
        kind, _, name = ident.partition(":")
        if not name:
            name, kind = ident, ""
        upper = name.upper()
        if kind == "alert" or upper.startswith("ALR-"):
            return self._alert_chain(name if kind else upper)
        if kind == "entity" or upper.startswith("ENT-"):
            return self._entity_chain(name if kind else upper)
        if kind == "event" or upper.startswith("EVT-"):
            return self.get_event_evidence(name if kind else upper)
        return {
            "error": f"cannot tell what '{clip(ident, 60)}' refers to; "
            "use alert:ID, event:ID or entity:ID"
        }

    def _alert_chain(self, alert_id: str) -> dict:
        alert = self.store.get_alert(alert_id)
        if alert is None:
            return {"error": f"no alert with id '{clip(alert_id, 60)}'"}
        chain = {"alert": self._alert_view(alert), "steps": []}
        steps = chain["steps"]
        steps.append(f"rule '{alert.rule_id}' fired: {alert.message}")
        events = []
        for event_id in alert.event_ids[: self.limits.max_events]:
            ev = self.store.get_event(event_id)
            if ev is None:
                continue
            observations = self.store.get_event_observations(event_id)
            cap = self.limits.max_evidence
            events.append(
                {
                    "event": event_view(ev, self.limits),
                    "observations_total": len(observations),
                    "observations": [self._obs_view(o) for o in observations[:cap]],
                    "truncated": len(observations) > cap,
                }
            )
        if events:
            steps.append("evidence: events -> observations -> frames (each with its telemetry)")
        chain["events"] = events
        samples = []
        for stamp in alert.telemetry_at[: self.limits.max_telemetry]:
            tid = self.store.telemetry_id_for(stamp)
            found = self.store.telemetry_before(stamp)
            if tid is not None and found is not None:
                samples.append(telemetry_view(found[1], tid))
        if samples:
            steps.append("evidence: telemetry samples that crossed the rule's thresholds")
        chain["telemetry"] = samples
        if alert.entity_id and (ent := self.store.get_entity(alert.entity_id)):
            chain["entity"] = entity_view(ent)
        return self._ok(chain)

    def _entity_chain(self, entity_id: str) -> dict:
        res = self.get_entity_history(entity_id)
        if "error" in res:
            return res
        ent = self.store.get_entity(entity_id)
        assert ent is not None
        res["link_evidence"] = [
            {
                "event_ref": ref("event", eid),
                "score": ent.link_scores.get(eid),
                "components": ent.link_components.get(eid),
            }
            for eid in ent.event_ids
            if eid in ent.link_scores
        ]
        return self._ok(res)

    # -- run --------------------------------------------------------------------------------
    def summarize_run(self) -> dict:
        return self._ok(run_facts(self.store, self.limits))


def telemetry_sample_label(sample: Telemetry) -> str:
    """One line for a telemetry sample, for answers that quote it."""
    signal = "n/a" if sample.signal_percent is None else f"{sample.signal_percent:.0f}%"
    return (
        f"battery {sample.battery_percent:.0f}%, signal {signal}, "
        f"altitude {sample.altitude_m:.0f} m,"
        f" zone {sample.zone.value}"
    )
