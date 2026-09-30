"""Helpers shared by the rules: alert ids, triage wording, ladders, confidence."""

from collections.abc import Iterable
from datetime import UTC, datetime

from drone_agent.models import Event, LadderStep, Severity, Verdict


def alert_id(code: str, subject_id: str) -> str:
    """Deterministic alert identity: same rule + same event/entity -> same id."""
    return f"ALR-{code}-{subject_id}"


def compact_utc(ts: datetime) -> str:
    """'20260929T183000Z': a citation-safe, deterministic id fragment for time-based alerts."""
    return ts.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def frame_union(events: Iterable[Event]) -> list[int]:
    """Sorted, de-duplicated frame ids across events (the alert's evidence frames)."""
    return sorted({f for ev in events for f in ev.frame_ids})


def type_label(event: Event) -> str:
    """'Person', 'Vehicle' (used only in the alert wording)."""
    return event.object_type.value.capitalize()


def event_confidence(events: Iterable[Event]) -> float | None:
    """Mean perception confidence of the events behind an alert (None if none is known).

    Providers that report no confidence contribute the documented placeholder, so this is a
    traceable number, not a calibrated probability.
    """
    values = [ev.confidence for ev in events if ev.confidence is not None]
    return round(sum(values) / len(values), 4) if values else None


# Fixed triage table: a verdict is a function of severity, never chosen by a model.
_VERDICTS = {
    Severity.LOW: Verdict.SUSPICIOUS,
    Severity.MEDIUM: Verdict.SUSPICIOUS,
    Severity.HIGH: Verdict.CRITICAL,
    Severity.CRITICAL: Verdict.CRITICAL,
}


def verdict_for(severity: Severity) -> Verdict:
    return _VERDICTS[severity]


def single_step(severity: Severity, at: datetime) -> list[LadderStep]:
    """The one-rung ladder of a rule without escalation."""
    return [LadderStep(at=at, severity=severity, label=severity.value)]


# Built-in recommended actions (a rule's config may override them). They are advice to a human
# operator, written for the property owner, and are chosen by rule and severity only.
ACTIONS: dict[str, str] = {
    "loitering": (
        "Keep the drone on station and verify the person visually; notify on-site security if "
        "they remain."
    ),
    "loitering:critical": (
        "Dispatch security to the location now and keep the drone observing; note the person's "
        "description."
    ),
    "after_hours_presence": (
        "Check whether the person is authorised staff; if not, notify security and keep the "
        "drone observing."
    ),
    "repeat_vehicle": (
        "Review the appearances for a pattern (times, zones); log the vehicle description and "
        "compare it with expected visitors."
    ),
    "restricted_zone_entry": (
        "Treat as a possible intrusion: notify security immediately and keep the drone "
        "observing without approaching."
    ),
    "low_battery": "Finish the current observation, then return to the dock to recharge.",
    "low_battery:critical": "Return to the dock immediately and suspend the patrol.",
    "signal_loss": (
        "Move toward the dock to recover the link and check for interference; log the outage."
    ),
    "signal_loss:high": (
        "Assume the drone is unmanaged: hold or return to the dock on the last good position and "
        "do not resume the patrol until the link is restored."
    ),
    "altitude_anomaly": (
        "Verify the altitude sensor and return the drone to its patrol altitude before continuing."
    ),
    "altitude_anomaly:high": (
        "Bring the drone to a safe altitude immediately or return it to the dock."
    ),
    "geofence_violation": (
        "Steer the drone back inside the geofence and check GPS quality and wind."
    ),
    "geofence_violation:critical": (
        "Recall the drone to the dock now; it is far outside the permitted area."
    ),
}


def triage_fields(
    rule_id: str,
    severity: Severity,
    triggered_at: datetime,
    events: Iterable[Event],
    action_override: str | None = None,
) -> dict:
    """The lifecycle/triage fields every event-based alert carries (single-rung ladder)."""
    return {
        "confidence": event_confidence(events),
        "verdict": verdict_for(severity),
        "recommended_action": recommended_action(rule_id, severity, action_override),
        "ladder": single_step(severity, triggered_at),
    }


def recommended_action(rule_id: str, severity: Severity, override: str | None = None) -> str:
    """The wording for this rule at this severity: an explicit override, else the built-in table
    entry for (rule, severity), else the rule's default."""
    if override:
        return override
    return ACTIONS.get(f"{rule_id}:{severity.value}") or ACTIONS[rule_id]
