"""Drone-safety rules over telemetry: low battery, signal loss, altitude anomaly, geofence.

These need no camera: they read the drone's own telemetry. Each condition is scanned as an
EPISODE (a run of unhealthy samples). One episode is one alert with an escalation ladder, and it
resolves when the drone is healthy again. Ids are deterministic (`ALR-<CODE>-<episode start>`),
so re-evaluating a growing telemetry stream yields the same alert with more evidence.

A missing measurement is "unknown", never "healthy" or "unhealthy": a sample without
`signal_percent` says nothing about link quality (only the delivery gap between samples does).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from drone_agent.models import Alert, AlertStatus, LadderStep, Severity, Telemetry
from drone_agent.rules.common import alert_id, compact_utc, recommended_action, verdict_for
from drone_agent.rules.config import (
    AltitudeRule,
    GeofenceRule,
    LowBatteryRule,
    SignalLossRule,
)
from drone_agent.telemetry.geo import distance_m

MAX_EVIDENCE_SAMPLES = 6


@dataclass
class _Item:
    """One point on the episode's timeline: a real sample, or a synthetic gap marker."""

    at: datetime
    level: int | None  # None = healthy; 0 = first rung; 1 = second rung
    sample: Telemetry | None
    is_gap: bool = False


@dataclass
class _Episode:
    items: list[_Item] = field(default_factory=list)
    resolved_at: datetime | None = None
    resolver: Telemetry | None = None
    lead_sample: Telemetry | None = None  # last good sample before a gap

    @property
    def start(self) -> datetime:
        return self.items[0].at

    def first_reached(self) -> dict[int, datetime]:
        reached: dict[int, datetime] = {}
        for item in self.items:
            if item.level is not None:
                reached.setdefault(item.level, item.at)
        return reached

    def samples(self) -> list[Telemetry]:
        return [i.sample for i in self.items if i.sample is not None]


def _scan(
    samples: Sequence[Telemetry],
    classify: Callable[[Telemetry], int | None],
    recovered: Callable[[Telemetry], bool],
) -> list[_Episode]:
    """Episodes of consecutive unhealthy samples (with an optional hysteresis band)."""
    episodes: list[_Episode] = []
    current: _Episode | None = None
    for s in samples:
        level = classify(s)
        if current is None:
            if level is not None:
                current = _Episode(items=[_Item(s.timestamp, level, s)])
                episodes.append(current)
        elif level is not None:
            current.items.append(_Item(s.timestamp, level, s))
        elif recovered(s):
            current.resolved_at, current.resolver = s.timestamp, s
            current = None
    return episodes


def _evidence_times(ep: _Episode, worst: Telemetry | None) -> list[datetime]:
    """A bounded, chronological set of the samples that justify the alert."""
    picked: dict[datetime, None] = {}
    if ep.lead_sample is not None:
        picked[ep.lead_sample.timestamp] = None
    real = ep.samples()
    if real:
        picked[real[0].timestamp] = None
    for at in sorted(ep.first_reached().values()):
        if any(s.timestamp == at for s in real):
            picked[at] = None
    if worst is not None:
        picked[worst.timestamp] = None
    if real:
        picked[real[-1].timestamp] = None
    if ep.resolver is not None:
        picked[ep.resolver.timestamp] = None
    times = sorted(picked)
    if len(times) > MAX_EVIDENCE_SAMPLES:  # keep the ends and the most informative middle
        times = [times[0], *times[1 : MAX_EVIDENCE_SAMPLES - 2], times[-2], times[-1]][
            :MAX_EVIDENCE_SAMPLES
        ]
    return sorted(set(times))


def _ladder(ep: _Episode, rungs: Sequence[tuple[Severity, str]]) -> list[LadderStep]:
    steps: list[LadderStep] = []
    for level, at in sorted(ep.first_reached().items()):
        severity, label = rungs[level]
        if steps and severity.rank <= steps[-1].severity.rank:
            continue  # configured severities do not rise here: keep the first rung only
        steps.append(LadderStep(at=at, severity=severity, label=label))
    return steps


def _build_alert(
    *,
    rule_id: str,
    code: str,
    ep: _Episode,
    rungs: Sequence[tuple[Severity, str]],
    title: str,
    message: str,
    facts: dict,
    worst: Telemetry | None,
    action_override: str | None,
    last_seen: datetime | None = None,
) -> Alert:
    ladder = _ladder(ep, rungs)
    severity = ladder[-1].severity
    evidence = _evidence_times(ep, worst)
    first_seen = min([ep.start, *evidence])
    end = last_seen or max([i.at for i in ep.items] + evidence)
    resolved = ep.resolved_at is not None
    real = ep.samples()
    zone = (
        (ep.lead_sample or (real[0] if real else None)).zone if (ep.lead_sample or real) else None
    )
    return Alert(
        alert_id=alert_id(code, compact_utc(ep.start)),
        rule_id=rule_id,
        severity=severity,
        title=title,
        message=message,
        triggered_at=ladder[0].at,
        first_seen=first_seen,
        last_seen=max(end, ladder[0].at),
        zone=zone,
        facts=facts,
        status=AlertStatus.RESOLVED if resolved else AlertStatus.OPEN,
        resolved_at=ep.resolved_at,
        confidence=1.0,  # a direct measurement of the drone's own state, not a perception
        verdict=verdict_for(severity),
        recommended_action=recommended_action(rule_id, severity, action_override),
        ladder=ladder,
        telemetry_at=evidence,
    )


def _sorted(samples: Sequence[Telemetry]) -> list[Telemetry]:
    unique = {s.timestamp: s for s in samples}
    return [unique[t] for t in sorted(unique)]


# ---- low battery --------------------------------------------------------------------------------
RULE_LOW_BATTERY = "low_battery"


def low_battery(samples: Sequence[Telemetry], cfg: LowBatteryRule) -> list[Alert]:
    if not cfg.enabled:
        return []

    def classify(s: Telemetry) -> int | None:
        if s.battery_percent <= cfg.critical_percent:
            return 1
        if s.battery_percent <= cfg.warning_percent:
            return 0
        return None

    episodes = _scan(_sorted(samples), classify, lambda s: s.battery_percent >= cfg.recover_percent)
    alerts = []
    for ep in episodes:
        worst = min(ep.samples(), key=lambda s: (s.battery_percent, s.timestamp))
        critical = 1 in ep.first_reached()
        alerts.append(
            _build_alert(
                rule_id=RULE_LOW_BATTERY,
                code="BATTERY",
                ep=ep,
                rungs=[(cfg.severity, "warning"), (Severity.CRITICAL, "critical")],
                title=f"Battery {'critical' if critical else 'low'} ({worst.battery_percent:.0f}%)",
                message=(
                    f"Battery fell to {worst.battery_percent:.1f}%, at or below the "
                    f"{cfg.critical_percent:.0f}% critical limit."
                    if critical
                    else f"Battery fell to {worst.battery_percent:.1f}%, at or below the "
                    f"{cfg.warning_percent:.0f}% warning limit."
                ),
                facts={
                    "metric": "battery_percent",
                    "lowest_percent": worst.battery_percent,
                    "warning_percent": cfg.warning_percent,
                    "critical_percent": cfg.critical_percent,
                    "comparison": "battery_percent <= warning_percent",
                },
                worst=worst,
                action_override=cfg.critical_action if critical else cfg.recommended_action,
            )
        )
    return alerts


# ---- signal loss --------------------------------------------------------------------------------
RULE_SIGNAL_LOSS = "signal_loss"


def signal_loss(
    samples: Sequence[Telemetry], cfg: SignalLossRule, now: datetime | None = None
) -> list[Alert]:
    """Degraded link (low signal) and lost link (very low signal, or no telemetry for too long).

    `now` is the end of the data: it lets an outage that is still going on (no sample has arrived
    since) be reported. Without it only outages that already ended, or that are visible between
    two delivered samples, can be found.
    """
    if not cfg.enabled:
        return []
    ordered = _sorted(samples)
    gap = timedelta(seconds=cfg.gap_seconds)

    def level_of(s: Telemetry) -> int | None:
        if s.signal_percent is None:
            return None  # unknown is not a fault
        if s.signal_percent < cfg.lost_below_percent:
            return 1
        if s.signal_percent < cfg.degraded_below_percent:
            return 0
        return None

    episodes: list[_Episode] = []
    current: _Episode | None = None
    previous: Telemetry | None = None

    def close(at: Telemetry) -> None:
        nonlocal current
        assert current is not None
        current.resolved_at, current.resolver = at.timestamp, at
        current = None

    def open_gap(after: Telemetry) -> None:
        nonlocal current
        marker = _Item(after.timestamp + gap, 1, None, is_gap=True)
        if current is None:
            current = _Episode(items=[marker], lead_sample=after)
            episodes.append(current)
        else:
            current.items.append(marker)

    for s in ordered:
        if previous is not None and s.timestamp - previous.timestamp > gap:
            open_gap(previous)
        level = level_of(s)
        if current is not None and level is None:
            close(s)
        elif level is not None:
            if current is None:
                current = _Episode(items=[_Item(s.timestamp, level, s)])
                episodes.append(current)
            else:
                current.items.append(_Item(s.timestamp, level, s))
        previous = s
    if now is not None and previous is not None and now - previous.timestamp > gap:
        open_gap(previous)  # the outage is still going on at the end of the data

    alerts = []
    for ep in episodes:
        lost = 1 in ep.first_reached()
        via_gap = any(i.is_gap for i in ep.items)
        real = ep.samples()
        lowest = min((s.signal_percent for s in real if s.signal_percent is not None), default=None)
        ongoing = ep.resolved_at is None and now is not None
        reason = (
            f"no telemetry for more than {cfg.gap_seconds:.0f}s"
            if via_gap
            else f"signal at {lowest:.0f}%"
            if lowest is not None
            else "link quality degraded"
        )
        alerts.append(
            _build_alert(
                rule_id=RULE_SIGNAL_LOSS,
                code="SIGNAL",
                ep=ep,
                rungs=[(cfg.severity, "degraded"), (cfg.lost_severity, "lost")],
                title="Link lost" if lost else "Link degraded",
                message=(
                    f"Drone link {'lost' if lost else 'degraded'}: {reason} "
                    f"(degraded below {cfg.degraded_below_percent:.0f}%, lost below "
                    f"{cfg.lost_below_percent:.0f}% or after a {cfg.gap_seconds:.0f}s gap)."
                ),
                facts={
                    "metric": "signal_percent",
                    **({"lowest_percent": lowest} if lowest is not None else {}),
                    "degraded_below_percent": cfg.degraded_below_percent,
                    "lost_below_percent": cfg.lost_below_percent,
                    "gap_seconds": cfg.gap_seconds,
                    "delivery_gap": via_gap,
                    "comparison": "signal_percent < degraded_below_percent or telemetry gap",
                },
                worst=min(
                    real, key=lambda s: s.signal_percent if s.signal_percent is not None else 101
                )
                if real
                else None,
                action_override=cfg.lost_action if lost else cfg.recommended_action,
                last_seen=now if ongoing else None,
            )
        )
    return alerts


# ---- altitude -----------------------------------------------------------------------------------
RULE_ALTITUDE = "altitude_anomaly"


def altitude_anomaly(samples: Sequence[Telemetry], cfg: AltitudeRule) -> list[Alert]:
    if not cfg.enabled:
        return []

    def classify(s: Telemetry) -> int | None:
        if s.altitude_m < cfg.severe_min_m or s.altitude_m > cfg.severe_max_m:
            return 1
        if s.altitude_m < cfg.min_m or s.altitude_m > cfg.max_m:
            return 0
        return None

    episodes = _scan(_sorted(samples), classify, lambda s: True)
    alerts = []
    for ep in episodes:
        centre = (cfg.min_m + cfg.max_m) / 2
        worst = max(ep.samples(), key=lambda s: (abs(s.altitude_m - centre), s.timestamp))
        severe = 1 in ep.first_reached()
        alerts.append(
            _build_alert(
                rule_id=RULE_ALTITUDE,
                code="ALTITUDE",
                ep=ep,
                rungs=[(cfg.severity, "out_of_range"), (Severity.HIGH, "severe")],
                title=f"Abnormal altitude ({worst.altitude_m:.0f} m)",
                message=(
                    f"Altitude reached {worst.altitude_m:.1f} m, outside the safe patrol range of "
                    f"{cfg.min_m:.0f}-{cfg.max_m:.0f} m"
                    + (" and beyond the severe limits." if severe else ".")
                ),
                facts={
                    "metric": "altitude_m",
                    "extreme_altitude_m": worst.altitude_m,
                    "min_m": cfg.min_m,
                    "max_m": cfg.max_m,
                    "comparison": "altitude_m outside [min_m, max_m]",
                },
                worst=worst,
                action_override=cfg.severe_action if severe else cfg.recommended_action,
            )
        )
    return alerts


# ---- geofence -----------------------------------------------------------------------------------
RULE_GEOFENCE = "geofence_violation"


def geofence_violation(samples: Sequence[Telemetry], cfg: GeofenceRule) -> list[Alert]:
    if not cfg.enabled:
        return []

    def away(s: Telemetry) -> float:
        return distance_m(s.latitude, s.longitude, cfg.center_latitude, cfg.center_longitude)

    def classify(s: Telemetry) -> int | None:
        d = away(s)
        if d > cfg.radius_m * cfg.critical_factor:
            return 1
        if d > cfg.radius_m:
            return 0
        return None

    episodes = _scan(_sorted(samples), classify, lambda s: True)
    alerts = []
    for ep in episodes:
        worst = max(ep.samples(), key=lambda s: (away(s), s.timestamp))
        far = 1 in ep.first_reached()
        alerts.append(
            _build_alert(
                rule_id=RULE_GEOFENCE,
                code="GEOFENCE",
                ep=ep,
                rungs=[(cfg.severity, "outside"), (Severity.CRITICAL, "far_outside")],
                title=f"Geofence violation ({away(worst):.0f} m from centre)",
                message=(
                    f"The drone was {away(worst):.0f} m from the property centre, outside the "
                    f"{cfg.radius_m:.0f} m geofence."
                ),
                facts={
                    "metric": "distance_from_center_m",
                    "max_distance_m": round(away(worst), 1),
                    "radius_m": cfg.radius_m,
                    "comparison": "distance_from_center_m > radius_m",
                },
                worst=worst,
                action_override=cfg.critical_action if far else cfg.recommended_action,
            )
        )
    return alerts
