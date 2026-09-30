"""Restricted-zone entry: a monitored object is observed in a restricted zone."""

from collections.abc import Iterable

from drone_agent.models import Alert, Event
from drone_agent.rules.common import alert_id, frame_union, triage_fields, type_label
from drone_agent.rules.config import RestrictedZoneRule

RULE_ID = "restricted_zone_entry"
CODE = "RESTRICTED"


def evaluate(events: Iterable[Event], cfg: RestrictedZoneRule) -> list[Alert]:
    """One alert per qualifying EVENT, so a continuous stay yields one alert, not one per frame."""
    if not cfg.enabled:
        return []
    alerts: list[Alert] = []
    for ev in events:
        if ev.zone not in cfg.zones or ev.object_type not in cfg.object_types:
            continue
        who = type_label(ev)
        alerts.append(
            Alert(
                alert_id=alert_id(CODE, ev.event_id),
                rule_id=RULE_ID,
                severity=cfg.severity,
                title=f"{who} in restricted zone {ev.zone.value}",
                message=(
                    f"{who} was observed in restricted zone {ev.zone.value} for "
                    f"{ev.duration_seconds:.0f}s (event {ev.event_id})."
                ),
                triggered_at=ev.first_seen,
                first_seen=ev.first_seen,
                last_seen=ev.last_seen,
                zone=ev.zone,
                event_ids=[ev.event_id],
                frame_ids=frame_union([ev]),
                facts={
                    "zone": ev.zone.value,
                    "object_type": ev.object_type.value,
                    "duration_seconds": ev.duration_seconds,
                },
                **triage_fields(RULE_ID, cfg.severity, ev.first_seen, [ev], cfg.recommended_action),
            )
        )
    return alerts
