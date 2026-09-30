"""After-hours presence: a monitored object is present during the daily closed window."""

from collections.abc import Iterable
from datetime import datetime, time, timedelta, timezone, tzinfo

from drone_agent.models import Alert, Event
from drone_agent.rules.common import alert_id, frame_union, triage_fields, type_label
from drone_agent.rules.config import AfterHoursRule

RULE_ID = "after_hours_presence"
CODE = "AFTERHOURS"


def closed_overlap_start(
    first: datetime, last: datetime, closed_from: time, closed_until: time
) -> datetime | None:
    """Start of the first overlap between [first, last] and a daily closed window, else None.

    `first` and `last` must share one time zone (their wall-clock time is what the window is
    compared with). The window is [closed_from, closed_until); if closed_until <= closed_from it
    crosses midnight. Candidate windows start on the day before `first` up to the day of `last`,
    so an event that began before midnight is checked against the previous evening's window.
    """
    tz: tzinfo | None = first.tzinfo
    day = first.date() - timedelta(days=1)
    while day <= last.date():
        start = datetime.combine(day, closed_from, tzinfo=tz)
        end = datetime.combine(day, closed_until, tzinfo=tz)
        if closed_until <= closed_from:
            end += timedelta(days=1)
        if first < end and last >= start:
            return max(first, start)
        day += timedelta(days=1)
    return None


def _fmt(t: time) -> str:
    return t.strftime("%H:%M")


def _offset_text(ts: datetime) -> str:
    offset = ts.utcoffset() or timedelta(0)
    minutes = int(offset.total_seconds() // 60)
    sign = "+" if minutes >= 0 else "-"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"


def evaluate(events: Iterable[Event], cfg: AfterHoursRule) -> list[Alert]:
    """One alert per event that overlaps closed hours (an event is never alerted twice)."""
    if not cfg.enabled:
        return []
    zone_offset: timezone | None = cfg.offset
    alerts: list[Alert] = []
    for ev in events:
        if ev.object_type not in cfg.object_types or (cfg.zones and ev.zone not in cfg.zones):
            continue
        # Evaluate in the property's local time: the configured offset, else the event's own.
        local_first = ev.first_seen.astimezone(zone_offset) if zone_offset else ev.first_seen
        local_last = ev.last_seen.astimezone(local_first.tzinfo)
        start = closed_overlap_start(local_first, local_last, cfg.closed_from, cfg.closed_until)
        if start is None:
            continue
        who = type_label(ev)
        alerts.append(
            Alert(
                alert_id=alert_id(CODE, ev.event_id),
                rule_id=RULE_ID,
                severity=cfg.severity,
                title=f"{who} present after hours at {ev.zone.value}",
                message=(
                    f"{who} was in {ev.zone.value} at {start.strftime('%H:%M:%S')} local time "
                    f"({_offset_text(start)}), inside closed hours "
                    f"{_fmt(cfg.closed_from)}-{_fmt(cfg.closed_until)} (event {ev.event_id})."
                ),
                triggered_at=start,
                first_seen=ev.first_seen,
                last_seen=ev.last_seen,
                zone=ev.zone,
                event_ids=[ev.event_id],
                frame_ids=frame_union([ev]),
                facts={
                    "zone": ev.zone.value,
                    "object_type": ev.object_type.value,
                    "event_local_time": start.strftime("%H:%M:%S"),
                    "first_seen_local": local_first.isoformat(),
                    "last_seen_local": local_last.isoformat(),
                    "utc_offset": _offset_text(start),
                    "closed_from": _fmt(cfg.closed_from),
                    "closed_until": _fmt(cfg.closed_until),
                },
                **triage_fields(RULE_ID, cfg.severity, start, [ev], cfg.recommended_action),
            )
        )
    return alerts
