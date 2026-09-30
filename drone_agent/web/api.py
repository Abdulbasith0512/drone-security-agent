"""Thin, read-only adapter between the dashboard and the existing backend.

Every payload is produced by existing code (`run_facts`, `AnalystTools`, the `*_view` functions,
`AnalystSession`, `Store`, the rules config) and only reshaped for JSON. The adapter adds no
detection logic. It joins ids (for example which events a frame belongs to) and labels data; it does
not decide anything about security.

Read-only by construction: databases are opened with SQLite `mode=ro`, so a write cannot succeed.
"""

import secrets
import sqlite3
import threading
from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from drone_agent.agent.citations import extract_citations
from drone_agent.agent.evidence import (
    ToolLimits,
    alert_view,
    entity_view,
    event_view,
    observation_view,
)
from drone_agent.agent.session import AnalystSession
from drone_agent.agent.summary import run_facts
from drone_agent.agent.tools import AnalystTools
from drone_agent.models import ObjectType, Observation, Severity, Zone
from drone_agent.rules import load_rules_config
from drone_agent.store import Store, StoreError
from drone_agent.store.schema import SCHEMA_VERSION
from drone_agent.web.frames import DEFAULT_WIDTH, FrameNotFound, FrameProvider
from drone_agent.web.guard import POLICY, check_question
from drone_agent.web.registry import Run, RunRegistry

# Generous caps: the demo runs are small, and a dashboard listing should not silently cut them.
LIMITS = ToolLimits(
    default_observations=200,
    max_observations=200,
    max_events=1000,
    max_alerts=1000,
    max_entities=500,
    max_evidence=500,
    max_frame_ids=1000,
    max_description=1500,
)
MAX_QUESTION_CHARS = 500
MAX_SESSIONS = 64
ANALYST_NOTE = (
    "Answered offline by the deterministic template analyst from stored evidence: no language "
    "model, no network and no Gemini call were involved."
)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def open_readonly(path: Path) -> Store:
    """Open an existing database read-only (SQLite refuses every write on this connection)."""
    if not path.is_file():
        raise StoreError(f"database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise StoreError(f"{path} is not a readable SQLite database: {exc}") from exc
    if version != SCHEMA_VERSION:
        conn.close()
        raise StoreError(
            f"{path} has schema version {version}, this build reads {SCHEMA_VERSION}; "
            "open it once with the CLI (for example `python -m drone_agent events --db ...`) "
            "to upgrade it"
        )
    return Store(conn)


class DashboardApi:
    def __init__(self, registry: RunRegistry) -> None:
        self.registry = registry
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._frames: dict[str, FrameProvider] = {}
        for run in registry.all():
            if run.video is not None:  # fail fast: an unreadable video is a startup error
                self._frames[run.name] = FrameProvider(run.video)

    # ---- plumbing --------------------------------------------------------------------------
    def _run(self, name: str) -> Run:
        run = self.registry.get(name)
        if run is None:
            raise ApiError(404, f"unknown run '{name}'")
        return run

    @contextmanager
    def _store(self, run: Run) -> Iterator[Store]:
        with self._lock:
            try:
                store = open_readonly(run.db)
            except StoreError as exc:
                raise ApiError(500, str(exc)) from exc
            try:
                yield store
            finally:
                store.close()

    def _run_info(self, run: Run) -> dict:
        frames = self._frames.get(run.name)
        return {
            "name": run.name,
            "kind": run.kind.key,
            "label": run.kind.label,
            "detail": run.kind.detail,
            "description_origin": run.kind.description_origin,
            "has_video": frames is not None,
            "video": (
                {
                    "name": frames.meta.source_name,
                    "frame_count": frames.meta.frame_count,
                    "fps": frames.meta.fps,
                    "duration_seconds": round(frames.meta.duration_seconds, 2),
                    "width": frames.meta.width,
                    "height": frames.meta.height,
                }
                if frames
                else None
            ),
        }

    @staticmethod
    def _events_by_observation(events: list) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for ev in events:
            for obs_id in ev.observation_ids:
                mapping.setdefault(obs_id, []).append(f"event:{ev.event_id}")
        return mapping

    def _observation_json(
        self, run: Run, obs: Observation, event_map: dict[str, list[str]]
    ) -> dict:
        frames = self._frames.get(run.name)
        view = observation_view(obs, LIMITS)
        view.update(
            {
                "observation_id": obs.observation_id,
                "frame_id": obs.frame_id,
                "event_refs": event_map.get(obs.observation_id, []),
                "image": frames is not None and 0 <= obs.frame_id < frames.frame_count,
                "telemetry_simulated": (
                    {
                        "time": obs.telemetry.telemetry.timestamp.isoformat(),
                        "battery_percent": obs.telemetry.telemetry.battery_percent,
                        "altitude_m": obs.telemetry.telemetry.altitude_m,
                        "age_seconds": round(obs.telemetry.age_seconds, 1),
                        "stale": obs.telemetry.stale,
                    }
                    if obs.telemetry
                    else None
                ),
            }
        )
        return view

    # ---- runs and overview -----------------------------------------------------------------
    def runs(self) -> dict:
        out = []
        for run in self.registry.all():
            with self._store(run) as store:
                counts = {
                    "observations": store.count_observations(),
                    "events": store.count_events(),
                    "entities": store.count_entities(),
                    "alerts": store.count_alerts(),
                }
            out.append({**self._run_info(run), "counts": counts})
        return {"runs": out}

    def rules(self) -> dict:
        """The built-in rule configuration (identical to config/security_rules.yaml)."""
        config = load_rules_config()
        rules = []
        for name, value in config.model_dump(mode="json").items():
            doc = (type(getattr(config, name)).__doc__ or "").strip().split("\n\n")[0]
            settings = {k: v for k, v in value.items() if k not in ("enabled", "severity")}
            rules.append(
                {
                    "id": name,
                    "enabled": value["enabled"],
                    "severity": value["severity"],
                    "description": " ".join(doc.split()),
                    "settings": settings,
                }
            )
        return {
            "rules": rules,
            "note": "Built-in defaults. Alerts in a database come from whichever rules "
            "`analyze` was run with.",
        }

    def overview(self, name: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            facts = run_facts(store, LIMITS)
        counts = facts["counts"]
        rules = self.rules()["rules"]
        enabled = sum(1 for r in rules if r["enabled"])
        sources = [
            {"source": s, "count": n, "carried_forward": s.endswith("+carried")}
            for s, n in facts["sources"].items()
        ]
        pipeline = [
            {
                "key": "observations",
                "title": "Video / Observations",
                "count": counts["observations"],
                "caption": f"{counts['frames']} frames, {counts['detected_objects']} objects",
                "panel": "panel-video",
            },
            {
                "key": "events",
                "title": "Event memory",
                "count": counts["events"],
                "caption": "repeated detections merged in time",
                "panel": "panel-timeline",
            },
            {
                "key": "entities",
                "title": "Entity memory",
                "count": counts["entities"],
                "caption": "similar-vehicle groups, never identity",
                "panel": "panel-overview",
            },
            {
                "key": "rules",
                "title": "Security rules",
                "count": enabled,
                "caption": "deterministic; no model decides",
                "panel": "panel-alerts",
            },
            {
                "key": "alerts",
                "title": "Alerts",
                "count": counts["alerts"],
                "caption": "each with evidence ids",
                "panel": "panel-alerts",
            },
            {
                "key": "analyst",
                "title": "AI analyst",
                "count": None,
                "caption": "offline template; cites stored evidence",
                "panel": "panel-analyst",
            },
        ]
        return {
            "run": self._run_info(run),
            "counts": counts,
            "observation_span": facts["observation_span"],
            "severity_order": [s.value for s in Severity],
            "severity_breakdown": facts["alerts_by_severity"],
            "rule_breakdown": facts["alerts_by_rule"],
            "zones": facts["zones"],
            "object_counts": facts["object_counts"],
            "sources": sources,
            "limitations": facts["limitations"],
            "pipeline": pipeline,
            "suggested_questions": self._suggestions(facts),
        }

    @staticmethod
    def _suggestions(facts: dict) -> list[str]:
        """Questions the template analyst can answer for THIS run, built from the run's own data."""
        out = ["Summarize the run.", "List the alerts.", "What objects were observed?"]
        for rule in facts["alerts_by_rule"]:
            out.append(f"Why was the {rule.replace('_', ' ')} alert raised?")
        for z in facts["zones"]:
            out.append(f"What happened at the {z['zone'].replace('_', ' ')}?")
        for ent in facts["entities"]:
            colour, body = ent["attributes"].get("color"), ent["attributes"].get("body_type")
            if colour and body:
                out.append(f"Was the {colour} {body} seen more than once?")
        seen: set[str] = set()
        return [q for q in out if not (q in seen or seen.add(q))]

    # ---- events, entities, alerts ------------------------------------------------------------
    def events(self, name: str, zone: str | None = None, object_type: str | None = None) -> dict:
        run = self._run(name)
        try:
            zone_v = Zone(zone) if zone else None
            type_v = ObjectType(object_type) if object_type else None
        except ValueError:
            raise ApiError(
                400,
                f"zone must be one of {[z.value for z in Zone]}, "
                f"type one of {[t.value for t in ObjectType]}",
            ) from None
        with self._store(run) as store:
            events = store.query_events(zone=zone_v, object_type=type_v)
            alerts = store.query_alerts()
            all_events = events if not (zone_v or type_v) else store.query_events()
        alert_map: dict[str, list[str]] = {}
        for a in alerts:
            for event_id in a.event_ids:
                alert_map.setdefault(event_id, []).append(f"alert:{a.alert_id}")
        rows = []
        for ev in events:
            view = event_view(ev, LIMITS)
            view.update(
                {
                    "event_id": ev.event_id,
                    "frame_ids": ev.frame_ids,
                    "observation_ids": ev.observation_ids,
                    "alert_refs": alert_map.get(ev.event_id, []),
                }
            )
            rows.append(view)
        span = None
        if all_events:
            span = {
                "first": min(e.first_seen for e in all_events).isoformat(),
                "last": max(e.last_seen for e in all_events).isoformat(),
            }
        return {"total": len(rows), "events": rows, "span": span}

    def event_detail(self, name: str, event_id: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            event = store.get_event(event_id)
            if event is None:
                raise ApiError(404, f"no event with id '{event_id}'")
            events = store.query_events()
            event_map = self._events_by_observation(events)
            observations = store.get_event_observations(event_id)
            alerts = [a for a in store.query_alerts() if event_id in a.event_ids]
            entity = store.get_entity(event.entity_id) if event.entity_id else None
            return {
                "event": event_view(event, LIMITS),
                "frame_ids": event.frame_ids,
                "alerts": [alert_view(a, LIMITS) for a in alerts],
                "entity": entity_view(entity) if entity else None,
                "observations": [self._observation_json(run, o, event_map) for o in observations],
            }

    def entities(self, name: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            entities = store.list_entities()
        rows = []
        for ent in entities:
            view = entity_view(ent)
            view["entity_id"] = ent.entity_id
            rows.append(view)
        return {"total": len(rows), "entities": rows}

    def alerts(self, name: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            alerts = store.query_alerts()
        rows = []
        for a in alerts:
            view = alert_view(a, LIMITS)
            view.update({"alert_id": a.alert_id, "frame_ids": a.frame_ids})
            rows.append(view)
        return {"total": len(rows), "severity_order": [s.value for s in Severity], "alerts": rows}

    def alert_detail(self, name: str, alert_id: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            tools = AnalystTools(store, LIMITS)
            found = tools.get_alert(alert_id)
            if "error" in found:
                raise ApiError(404, found["error"])
            alert = store.get_alert(alert_id)
            assert alert is not None
            event_map = self._events_by_observation(store.query_events())
            evidence = []
            for event_id in alert.event_ids:
                observations = store.get_event_observations(event_id)
                evidence.append(
                    {
                        "event_ref": f"event:{event_id}",
                        "observations": [
                            self._observation_json(run, o, event_map) for o in observations
                        ],
                    }
                )
            # A backend with telemetry (safety) alerts can also return the drone samples that
            # crossed the rule's limits; an older backend has no `get_evidence`, so it is optional.
            get_evidence = getattr(tools, "get_evidence", None)
            telemetry = (
                get_evidence(f"alert:{alert_id}").get("telemetry", []) if get_evidence else []
            )
        steps = [f"rule '{alert.rule_id}' fired: {alert.message}"]
        if alert.event_ids:
            steps.append("evidence: events -> observations -> frames")
        if telemetry:
            steps.append(
                "evidence: simulated drone telemetry samples that crossed the rule's limits"
            )
        if not alert.event_ids and not telemetry:
            steps.append("no camera events or telemetry samples are linked to this alert")
        return {
            "alert": {**found["alert"], "alert_id": alert.alert_id, "frame_ids": alert.frame_ids},
            "facts": [{"name": k, "value": v} for k, v in alert.facts.items()],
            "events": found["events"],
            "entity": found.get("entity"),
            "evidence": evidence,
            "telemetry": telemetry,
            "steps": steps,
        }

    # ---- observations and frames -------------------------------------------------------------
    def observations(
        self, name: str, zone: str | None = None, query: str | None = None, limit: int = 200
    ) -> dict:
        run = self._run(name)
        try:
            zone_v = Zone(zone) if zone else None
        except ValueError:
            raise ApiError(400, f"zone must be one of {[z.value for z in Zone]}") from None
        limit = max(1, min(int(limit), 500))
        text = query.strip() if query and query.strip() else None
        with self._store(run) as store:
            total = store.count_matching_observations(zone=zone_v, text=text)
            found = store.query_observations(zone=zone_v, text=text, limit=limit)
            event_map = self._events_by_observation(store.query_events())
        return {
            "total": total,
            "returned": len(found),
            "truncated": total > len(found),
            "observations": [self._observation_json(run, o, event_map) for o in found],
        }

    def frame_jpeg(self, name: str, frame_id: int, width: int = DEFAULT_WIDTH) -> bytes:
        run = self._run(name)
        frames = self._frames.get(run.name)
        if frames is None:
            raise ApiError(404, f"run '{name}' has no registered video")
        try:
            return frames.jpeg(frame_id, width)
        except FrameNotFound as exc:
            raise ApiError(404, str(exc)) from None

    # ---- the analyst ---------------------------------------------------------------------------
    def summary(self, name: str) -> dict:
        run = self._run(name)
        with self._store(run) as store:
            answer = AnalystSession(store).summarize()
        return self._answer_json(run, "Summarize the run.", answer, None, [])

    def ask(self, name: str, question: str, session_id: str | None = None) -> dict:
        run = self._run(name)
        question = (question or "").strip()
        if not question:
            raise ApiError(400, "question must not be empty")
        if len(question) > MAX_QUESTION_CHARS:
            raise ApiError(400, f"question is longer than {MAX_QUESTION_CHARS} characters")
        known = self._sessions.get(session_id) if session_id else None
        # a session belongs to ONE run: an id from another run (or an unknown id) starts fresh
        sid = session_id if known and known["run"] == run.name else secrets.token_hex(8)

        decision = check_question(question)
        if decision is not None:  # refused before the analyst is called; not remembered
            return {
                "session_id": sid,
                "question": question,
                "refused": True,
                "text": decision.text,
                "mode": "guard",
                "label": "Refused by the wording guard (the analyst was not asked)",
                "guard": {
                    "category": decision.category,
                    "matched": decision.matched,
                    "policy": POLICY,
                },
                "citations": [],
                "verified": True,
                "notes": [],
                "origin": self._origin(run),
                "followups": [],
                "suggested_questions": self._suggestions_for(run),
            }

        with self._store(run) as store:
            session = AnalystSession(store)  # llm=None: the deterministic template analyst
            state = self._sessions.get(sid)
            if state and state["run"] == run.name:
                session.history = deque(state["history"], maxlen=session.history.maxlen)
                session.context = dict(state["context"])
            try:
                answer = session.ask(question)
            except ValueError as exc:
                raise ApiError(400, str(exc)) from None
            self._sessions[sid] = {
                "run": run.name,
                "history": list(session.history),
                "context": dict(session.context),
            }
            self._sessions.move_to_end(sid)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)
            followups = self._followups(session.context)
        return self._answer_json(run, question, answer, sid, followups)

    @staticmethod
    def _followups(context: dict) -> list[str]:
        out = []
        if "entity" in context:
            out.append("Where was it seen?")
        if "alert" in context or "event" in context:
            out.append("Show me the evidence for it.")
        if "zone" in context:
            out.append("What else happened there?")
        return out

    def _suggestions_for(self, run: Run) -> list[str]:
        with self._store(run) as store:
            return self._suggestions(run_facts(store, LIMITS))

    @staticmethod
    def _origin(run: Run) -> dict:
        return {
            "analyst": ANALYST_NOTE,
            "observations": f"{run.kind.label}: {run.kind.detail}",
        }

    def _answer_json(
        self, run: Run, question: str, answer, sid: str | None, followups: list[str]
    ) -> dict:
        return {
            "session_id": sid,
            "question": question,
            "refused": False,
            "text": answer.text,
            "mode": answer.mode,
            "label": answer.label,
            "citations": [
                {"token": c.token, "kind": c.kind, "id": c.ref}
                for c in extract_citations(answer.text)
            ],
            "verified": not answer.notes,
            "notes": answer.notes,
            "origin": self._origin(run),
            "followups": followups,
        }
