"""Loitering: a monitored object stays in a monitored zone for at least the zone's threshold.

The alert has an escalation ladder: it opens at the rule's severity once the event lasts
`threshold_seconds` (WARNING), and the SAME alert reaches CRITICAL once it lasts
`critical_after_seconds`, when the zone configures one.
"""

from collections.abc import Iterable
from datetime import timedelta

from drone_agent.models import Alert, AlertStatus, Event, LadderStep, Severity
from drone_agent.rules.common import (
    alert_id,
    event_confidence,
    frame_union,
    recommended_action,
    type_label,
    verdict_for,
)
from drone_agent.rules.config import LoiteringRule

RULE_ID = "loitering"
CODE = "LOITER"


def evaluate(events: Iterable[Event], cfg: LoiteringRule) -> list[Alert]:
    """One alert per qualifying event.

    Uses the event's duration (last_seen - first_seen), never a frame count, so a tolerated
    missed detection inside the event does not matter. Boundary: duration >= threshold fires.
    An event whose object was seen leaving (not merely the end of the input) resolves the alert.
    """
    if not cfg.enabled:
        return []
    alerts: list[Alert] = []
    for ev in events:
        zone_cfg = cfg.zones.get(ev.zone)
        if zone_cfg is None or ev.object_type not in cfg.object_types:
            continue
        threshold = zone_cfg.threshold_seconds
        critical_after = zone_cfg.critical_after_seconds
        duration = ev.duration_seconds
        if duration < threshold:
            continue
        ladder = [
            LadderStep(
                at=ev.first_seen + timedelta(seconds=threshold),
                severity=cfg.severity,
                label="warning" if critical_after is not None else cfg.severity.value,
            )
        ]
        escalated = (
            critical_after is not None
            and duration >= critical_after
            and cfg.severity.rank < Severity.CRITICAL.rank
        )
        if escalated:
            assert critical_after is not None
            ladder.append(
                LadderStep(
                    at=ev.first_seen + timedelta(seconds=critical_after),
                    severity=Severity.CRITICAL,
                    label="critical",
                )
            )
        severity = ladder[-1].severity
        override = cfg.critical_action if escalated else cfg.recommended_action
        who = type_label(ev)
        message = (
            f"{who} stayed in {ev.zone.value} for {duration:.0f}s, reaching the "
            f"{threshold:.0f}s loitering threshold (event {ev.event_id})."
        )
        if escalated:
            message += f" It passed the {critical_after:.0f}s critical threshold and was escalated."
        departed = ev.close_reason.object_departed
        facts: dict = {
            "zone": ev.zone.value,
            "object_type": ev.object_type.value,
            "duration_seconds": duration,
            "threshold_seconds": threshold,
            "comparison": "duration_seconds >= threshold_seconds",
            "level": ladder[-1].label,
        }
        if critical_after is not None:
            facts["critical_after_seconds"] = critical_after
        alerts.append(
            Alert(
                alert_id=alert_id(CODE, ev.event_id),
                rule_id=RULE_ID,
                severity=severity,
                title=f"{who} loitering at {ev.zone.value}",
                message=message,
                triggered_at=ladder[0].at,
                first_seen=ev.first_seen,
                last_seen=ev.last_seen,
                zone=ev.zone,
                event_ids=[ev.event_id],
                frame_ids=frame_union([ev]),
                facts=facts,
                status=AlertStatus.RESOLVED if departed else AlertStatus.OPEN,
                resolved_at=ev.last_seen if departed else None,
                confidence=event_confidence([ev]),
                verdict=verdict_for(severity),
                recommended_action=recommended_action(RULE_ID, severity, override),
                ladder=ladder,
            )
        )
    return alerts
