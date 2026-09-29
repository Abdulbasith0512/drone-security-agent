"""Loitering: a monitored object stays in a monitored zone for at least the zone's threshold."""

from collections.abc import Iterable
from datetime import timedelta

from drone_agent.models import Alert, Event
from drone_agent.rules.common import alert_id, frame_union, type_label
from drone_agent.rules.config import LoiteringRule

RULE_ID = "loitering"
CODE = "LOITER"


def evaluate(events: Iterable[Event], cfg: LoiteringRule) -> list[Alert]:
    """One alert per qualifying event.

    Uses the event's duration (last_seen - first_seen), never a frame count, so a tolerated
    missed detection inside the event does not matter. Boundary: duration >= threshold fires.
    """
    if not cfg.enabled:
        return []
    alerts: list[Alert] = []
    for ev in events:
        zone_cfg = cfg.zones.get(ev.zone)
        if zone_cfg is None or ev.object_type not in cfg.object_types:
            continue
        threshold = zone_cfg.threshold_seconds
        duration = ev.duration_seconds
        if duration < threshold:
            continue
        who = type_label(ev)
        alerts.append(
            Alert(
                alert_id=alert_id(CODE, ev.event_id),
                rule_id=RULE_ID,
                severity=cfg.severity,
                title=f"{who} loitering at {ev.zone.value}",
                message=(
                    f"{who} stayed in {ev.zone.value} for {duration:.0f}s, reaching the "
                    f"{threshold:.0f}s loitering threshold (event {ev.event_id})."
                ),
                triggered_at=ev.first_seen + timedelta(seconds=threshold),
                first_seen=ev.first_seen,
                last_seen=ev.last_seen,
                zone=ev.zone,
                event_ids=[ev.event_id],
                frame_ids=frame_union([ev]),
                facts={
                    "zone": ev.zone.value,
                    "object_type": ev.object_type.value,
                    "duration_seconds": duration,
                    "threshold_seconds": threshold,
                    "comparison": "duration_seconds >= threshold_seconds",
                },
            )
        )
    return alerts
