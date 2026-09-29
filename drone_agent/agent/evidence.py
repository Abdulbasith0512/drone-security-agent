"""Bounded, JSON-safe views of stored evidence. Shared by the tools, the summary and the fallback.

Every view carries citation refs ("kind:id") for exactly the ids it contains, so an answer can only
cite evidence that was really retrieved. Limits are explicit and documented; results that were cut
say so (`total`, `returned`, `truncated`). No LangChain here.
"""

from dataclasses import dataclass

from drone_agent.models import Alert, Entity, Event, Observation


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


def ref(kind: str, ident: object) -> str:
    return f"{kind}:{ident}"


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(0, limit - 3)] + "..."


def observation_view(obs: Observation, limits: ToolLimits) -> dict:
    return {
        "ref": ref("obs", obs.observation_id),
        "frame_ref": ref("frame", obs.frame_id),
        "time": obs.timestamp.isoformat(),
        "zone": obs.zone.value,
        "source": obs.source,
        # A carried observation repeats the last accepted model answer; it is not a new inference.
        "carried_forward": obs.source.endswith("+carried"),
        "description": clip(obs.description, limits.max_description),
        "objects": [
            {"type": o.object_type.value, "activity": o.activity, "attributes": o.attributes}
            for o in obs.detected_objects
        ],
    }


def event_view(ev: Event, limits: ToolLimits) -> dict:
    frames = ev.frame_ids[: limits.max_frame_ids]
    return {
        "ref": ref("event", ev.event_id),
        "type": ev.object_type.value,
        "zone": ev.zone.value,
        "first_seen": ev.first_seen.isoformat(),
        "last_seen": ev.last_seen.isoformat(),
        "duration_seconds": ev.duration_seconds,
        "activities": ev.activities,
        "attributes": ev.attributes,
        "description": clip(ev.description, limits.max_description),
        "observation_count": ev.evidence_count,
        "frame_refs": [ref("frame", f) for f in frames],
        "frames_total": len(ev.frame_ids),
        "entity_ref": ref("entity", ev.entity_id) if ev.entity_id else None,
        "close_reason": ev.close_reason.value,
    }


def entity_view(ent: Entity) -> dict:
    return {
        "ref": ref("entity", ent.entity_id),
        "label": ent.label,  # "similar blue pickup": wording that does not overclaim identity
        "match_level": ent.match_level.value,
        "identity_confirmed": ent.match_level.value == "confirmed",
        "attributes": ent.attributes,
        "first_seen": ent.first_seen.isoformat(),
        "last_seen": ent.last_seen.isoformat(),
        "appearance_count": ent.appearance_count,
        "event_refs": [ref("event", e) for e in ent.event_ids],
    }


def alert_view(alert: Alert, limits: ToolLimits) -> dict:
    return {
        "ref": ref("alert", alert.alert_id),
        "rule_id": alert.rule_id,
        "severity": alert.severity.value,  # decided by the deterministic rule, never by an LLM
        "title": alert.title,
        "reason": alert.message,
        "triggered_at": alert.triggered_at.isoformat(),
        "first_seen": alert.first_seen.isoformat(),
        "last_seen": alert.last_seen.isoformat(),
        "zone": alert.zone.value if alert.zone else None,
        "event_refs": [ref("event", e) for e in alert.event_ids],
        "entity_ref": ref("entity", alert.entity_id) if alert.entity_id else None,
        "frame_refs": [ref("frame", f) for f in alert.frame_ids[: limits.max_frame_ids]],
        "frames_total": len(alert.frame_ids),
        "rule_facts": alert.facts,
    }


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
