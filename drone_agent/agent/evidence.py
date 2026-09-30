"""Bounded, JSON-safe views of stored evidence. Shared by the tools, the summary and the fallback.

Every view carries citation refs ("kind:id") for exactly the ids it contains, so an answer can only
cite evidence that was really retrieved. Limits are explicit and documented; results that were cut
say so (`total`, `returned`, `truncated`). No LangChain here.

Ref kinds: frame, obs (observation), event, entity, alert, tel (a telemetry sample).
"""

from dataclasses import dataclass

from drone_agent.models import Alert, Entity, Event, Observation, Telemetry


@dataclass(frozen=True)
class ToolLimits:
    """Hard caps on what one tool call may return (defaults chosen to keep prompts small)."""

    default_observations: int = 8
    max_observations: int = 15  # search_observations
    max_events: int = 20  # list_events, summary events, entity history
    max_alerts: int = 20  # list_alerts, summary alerts
    max_entities: int = 20
    max_evidence: int = 15  # observations returned per event
    max_frame_ids: int = 10  # frame ids listed per event/alert
    max_description: int = 200  # characters kept from a description; longer ones end in "..."
    max_timeline: int = 40  # items in one timeline answer
    max_telemetry: int = 12  # telemetry samples returned per call
    max_result_chars: int = 11500  # one tool result, as JSON; lists are trimmed to fit (< 12000)


def ref(kind: str, ident: object) -> str:
    return f"{kind}:{ident}"


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def telemetry_view(sample: Telemetry, telemetry_id: int, age_seconds: float | None = None) -> dict:
    """A telemetry sample as evidence. Values are SIMULATED in this project and labelled so."""
    view = {
        "telemetry_ref": ref("tel", telemetry_id),
        "time": sample.timestamp.isoformat(),
        "zone": sample.zone.value,
        "battery_percent": sample.battery_percent,
        "signal_percent": sample.signal_percent,
        "altitude_m": sample.altitude_m,
        "latitude": sample.latitude,
        "longitude": sample.longitude,
        "simulated": True,
    }
    if age_seconds is not None:
        view["age_seconds"] = round(age_seconds, 1)
    return view


def observation_view(obs: Observation, limits: ToolLimits, telemetry_id: int | None = None) -> dict:
    view = {
        "ref": ref("obs", obs.observation_id),
        "frame_ref": ref("frame", obs.frame_id),
        "time": obs.timestamp.isoformat(),
        "zone": obs.zone.value,
        "source": obs.source,
        # How the description was obtained: scripted / canned / replayed / live (None = unknown).
        "provenance": obs.provenance.value if obs.provenance else None,
        # A carried observation repeats the last accepted model answer; it is not a new inference.
        "carried_forward": obs.source.endswith("+carried"),
        "description": clip(obs.description, limits.max_description),
        "objects": [
            {"type": o.object_type.value, "activity": o.activity, "attributes": o.attributes}
            for o in obs.detected_objects
        ],
    }
    if obs.telemetry is not None:
        view["telemetry"] = {
            "age_seconds": round(obs.telemetry.age_seconds, 1),
            "stale": obs.telemetry.stale,
            "battery_percent": obs.telemetry.telemetry.battery_percent,
            "signal_percent": obs.telemetry.telemetry.signal_percent,
            "altitude_m": obs.telemetry.telemetry.altitude_m,
        }
        if telemetry_id is not None:
            view["telemetry_ref"] = ref("tel", telemetry_id)
    return view


def event_view(ev: Event, limits: ToolLimits, detail: bool = True) -> dict:
    """An event as evidence. `detail=False` is the compact form used in the run summary: no
    description, frame list or confidence (`get_event` returns those)."""
    frames = ev.frame_ids[: limits.max_frame_ids] if detail else []
    view = {
        "ref": ref("event", ev.event_id),
        "type": ev.object_type.value,
        "zone": ev.zone.value,
        "first_seen": ev.first_seen.isoformat(),
        "last_seen": ev.last_seen.isoformat(),
        "duration_seconds": ev.duration_seconds,
        "activities": ev.activities,
        "attributes": ev.attributes,
        "observation_count": ev.evidence_count,
        "frames_total": len(ev.frame_ids),
        "entity_ref": ref("entity", ev.entity_id) if ev.entity_id else None,
        "status": "open" if ev.is_open else "closed",
    }
    if detail:
        view["description"] = clip(ev.description, limits.max_description)
        view["frame_refs"] = [ref("frame", f) for f in frames]
        view["close_reason"] = ev.close_reason.value  # why observation of it stopped
        if ev.departure_observed_at is not None:  # only set when the zone was seen without it
            view["departure_observed_at"] = ev.departure_observed_at.isoformat()
        view["confidence"] = ev.confidence
    return view


def entity_view(ent: Entity) -> dict:
    return {
        "ref": ref("entity", ent.entity_id),
        "label": ent.label,  # "similar blue pickup": wording that does not overclaim identity
        "match_level": ent.match_level.value,
        "identity_confirmed": ent.match_level.value == "confirmed",
        # A heuristic 0-1 score of the weakest link between appearances; NOT a probability.
        "likelihood_same_vehicle": ent.match_score,
        "link_scores": ent.link_scores,
        "attributes": ent.attributes,
        "first_seen": ent.first_seen.isoformat(),
        "last_seen": ent.last_seen.isoformat(),
        "appearance_count": ent.appearance_count,
        "event_refs": [ref("event", e) for e in ent.event_ids],
    }


def alert_view(
    alert: Alert,
    limits: ToolLimits,
    telemetry_ids: dict | None = None,
    detail: bool = True,
) -> dict:
    """An alert as evidence. `detail=False` is the compact form used in listings and the run
    summary: what identifies and ranks an alert, and why. The recommended action, frames, ladder,
    history and rule facts are left to `get_alert` and `get_evidence`, so a listing of many alerts
    stays within the tool-result cap."""
    if not detail:
        return {
            "ref": ref("alert", alert.alert_id),
            "rule_id": alert.rule_id,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "verdict": alert.verdict.value if alert.verdict else None,
            "title": alert.title,
            "reason": clip(alert.message, limits.max_description),
            "zone": alert.zone.value if alert.zone else None,
            "triggered_at": alert.triggered_at.isoformat(),
            "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            "event_refs": [ref("event", e) for e in alert.event_ids],
            "entity_ref": ref("entity", alert.entity_id) if alert.entity_id else None,
        }
    view = {
        "ref": ref("alert", alert.alert_id),
        "rule_id": alert.rule_id,
        "severity": alert.severity.value,  # decided by the deterministic rule, never by an LLM
        "status": alert.status.value,  # open / escalated / acknowledged / resolved
        "verdict": alert.verdict.value if alert.verdict else None,
        "confidence": alert.confidence,
        "recommended_action": alert.recommended_action,
        "title": alert.title,
        "reason": alert.message,
        "triggered_at": alert.triggered_at.isoformat(),
        "detected_at": alert.detected_at.isoformat() if alert.detected_at else None,
        "detection_latency_seconds": alert.detection_latency_seconds,
        "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
        "first_seen": alert.first_seen.isoformat(),
        "last_seen": alert.last_seen.isoformat(),
        "zone": alert.zone.value if alert.zone else None,
        "event_refs": [ref("event", e) for e in alert.event_ids],
        "entity_ref": ref("entity", alert.entity_id) if alert.entity_id else None,
        "frame_refs": [ref("frame", f) for f in alert.frame_ids[: limits.max_frame_ids]],
        "frames_total": len(alert.frame_ids),
        "rule_facts": alert.facts,
        "ladder": [
            {"level": s.label, "severity": s.severity.value, "reached_at": s.at.isoformat()}
            for s in alert.ladder
        ],
        "history": [
            {
                "kind": h.kind.value,
                "actor": h.actor,
                "recorded_at": h.at.isoformat(),
                "effective_at": h.effective_at.isoformat(),
                "to_status": h.to_status.value,
                "to_severity": h.to_severity.value,
                "reason": h.reason,
            }
            for h in alert.history
        ],
    }
    if alert.telemetry_at:
        ids = telemetry_ids or {}
        view["telemetry_refs"] = [ref("tel", ids[t]) for t in alert.telemetry_at if t in ids]
        view["telemetry_times"] = [t.isoformat() for t in alert.telemetry_at]
    return view


def collect_refs(view: object) -> set[str]:
    """Every citation ref ("kind:id") that appears in a view (a dict/list tree of plain values).

    A ref only counts when it sits under a key called ref, *_ref or *_refs, so ids that merely
    appear inside prose are never treated as retrieved evidence.
    """
    found: set[str] = set()

    def walk(node: object, key: str = "") -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, key)
        elif isinstance(node, str) and (key == "ref" or key.endswith(("_ref", "_refs"))):
            found.add(node)

    walk(view)
    return found
