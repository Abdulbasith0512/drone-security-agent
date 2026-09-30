"""Alert lifecycle: OPEN -> (ESCALATED) -> ACKNOWLEDGED -> RESOLVED, with a full history.

The rule engine produces *snapshots*: what each alert should look like given all data so far
(severity reached, evidence, whether the condition ended). The manager compares a snapshot with
the alert it already holds and records the difference as transitions. Escalation raises the SAME
alert; nothing is ever deleted; a resolved alert stays resolved.

Two ways to drive it, with the same result on the final state:
* streaming (`live=True`): transitions are stamped with `now`, the moment the system noticed;
  the gap to `effective_at` is the detection latency;
* batch (`live=False`): transitions are stamped with the time the condition actually began.

Only an operator (or the auto-resolution of a state that ended) changes status; a language model
never does.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from drone_agent.models import (
    Alert,
    AlertStatus,
    AlertTransition,
    LadderStep,
    Severity,
    TransitionKind,
)
from drone_agent.rules.common import verdict_for


@dataclass
class AlertChange:
    """What one reconcile step did to one alert."""

    alert: Alert
    transitions: list[AlertTransition] = field(default_factory=list)
    is_new: bool = False


class LifecycleError(ValueError):
    """An operator action that the alert's current state does not allow."""


def _evolve(alert: Alert, **changes: object) -> Alert:
    """A validated copy (frozen models are rebuilt, so every invariant is re-checked)."""
    return Alert.model_validate({**alert.model_dump(), **changes})


class AlertManager:
    def __init__(self, alerts: Iterable[Alert] = ()) -> None:
        self._alerts: dict[str, Alert] = {a.alert_id: a for a in alerts}
        # The last snapshot each alert was reconciled with: an identical snapshot changes nothing.
        self._last_snapshot: dict[str, Alert] = {}

    # ---- reads -----------------------------------------------------------------------------
    @property
    def alerts(self) -> list[Alert]:
        return sorted(self._alerts.values(), key=lambda a: (a.triggered_at, a.alert_id))

    def get(self, alert_id: str) -> Alert | None:
        return self._alerts.get(alert_id)

    # ---- rule-driven changes ---------------------------------------------------------------
    def reconcile(
        self, snapshots: Iterable[Alert], now: datetime, *, live: bool = True
    ) -> list[AlertChange]:
        """Apply rule snapshots; returns the alerts that changed (new, escalated, resolved or
        with new evidence), in chronological order."""
        changes: list[AlertChange] = []
        for snap in sorted(snapshots, key=lambda a: (a.triggered_at, a.alert_id)):
            existing = self._alerts.get(snap.alert_id)
            if existing is not None and self._last_snapshot.get(snap.alert_id) == snap:
                continue  # the rules see exactly what they saw last time: nothing to record
            self._last_snapshot[snap.alert_id] = snap
            change = (
                self._create(snap, now, live)
                if existing is None
                else self._update(existing, snap, now, live)
            )
            if change is not None:
                self._alerts[snap.alert_id] = change.alert
                changes.append(change)
        return changes

    def _create(self, snap: Alert, now: datetime, live: bool) -> AlertChange:
        history: list[AlertTransition] = []
        previous_severity: Severity | None = None
        previous_status: AlertStatus | None = None
        for step in snap.ladder:
            first = not history
            status = AlertStatus.OPEN if first else AlertStatus.ESCALATED
            history.append(
                AlertTransition(
                    seq=len(history) + 1,
                    kind=TransitionKind.RAISED if first else TransitionKind.ESCALATED,
                    at=now if live else step.at,
                    effective_at=step.at,
                    from_status=previous_status,
                    to_status=status,
                    from_severity=previous_severity,
                    to_severity=step.severity,
                    reason=(
                        f"{snap.rule_id} condition met ({step.label})"
                        if first
                        else f"escalated to {step.severity.value}: {step.label} level reached"
                    ),
                )
            )
            previous_severity, previous_status = step.severity, status
        status = previous_status or AlertStatus.OPEN
        resolved_at = None
        if snap.status is AlertStatus.RESOLVED and snap.resolved_at is not None:
            resolved_at = snap.resolved_at
            history.append(self._resolution(history, snap.severity, status, resolved_at, now, live))
            status = AlertStatus.RESOLVED
        alert = _evolve(
            snap,
            status=status,
            history=history,
            detected_at=history[0].at,
            updated_at=history[-1].at if not live else now,
            resolved_at=resolved_at,
        )
        return AlertChange(alert, list(history), is_new=True)

    def _update(
        self, existing: Alert, snap: Alert, now: datetime, live: bool
    ) -> AlertChange | None:
        if existing.status is AlertStatus.RESOLVED:
            return None  # resolved is final: neither more evidence nor a higher rung reopens it
        severity, status = existing.severity, existing.status
        history = list(existing.history)
        added: list[AlertTransition] = []
        resolved_at = existing.resolved_at

        for step in snap.ladder:
            if step.severity.rank > severity.rank:
                added.append(
                    AlertTransition(
                        seq=len(history) + len(added) + 1,
                        kind=TransitionKind.ESCALATED,
                        at=now if live else step.at,
                        effective_at=step.at,
                        from_status=status,
                        to_status=AlertStatus.ESCALATED,
                        from_severity=severity,
                        to_severity=step.severity,
                        reason=self._escalation_reason(step),
                    )
                )
                severity, status = step.severity, AlertStatus.ESCALATED
        if snap.status is AlertStatus.RESOLVED and snap.resolved_at is not None:
            added.append(
                self._resolution([*history, *added], severity, status, snap.resolved_at, now, live)
            )
            status, resolved_at = AlertStatus.RESOLVED, snap.resolved_at

        keep_ladder = snap.severity.rank < existing.severity.rank
        ladder = existing.ladder if keep_ladder else snap.ladder
        updated = _evolve(
            existing,
            status=status,
            severity=severity,
            title=existing.title if keep_ladder else snap.title,
            message=existing.message if keep_ladder else snap.message,
            facts=existing.facts if keep_ladder else snap.facts,
            confidence=snap.confidence,
            verdict=verdict_for(severity),
            recommended_action=(
                existing.recommended_action if keep_ladder else snap.recommended_action
            ),
            ladder=ladder,
            first_seen=min(existing.first_seen, snap.first_seen),
            last_seen=max(existing.last_seen, snap.last_seen),
            zone=snap.zone or existing.zone,
            event_ids=snap.event_ids,
            frame_ids=snap.frame_ids,
            entity_id=snap.entity_id or existing.entity_id,
            telemetry_at=snap.telemetry_at or existing.telemetry_at,
            history=[*history, *added],
            resolved_at=resolved_at,
            updated_at=(added[-1].at if added else existing.updated_at),
        )
        if updated == existing:
            return None
        return AlertChange(updated, added)

    @staticmethod
    def _escalation_reason(step: LadderStep) -> str:
        return f"escalated to {step.severity.value}: {step.label} level reached"

    @staticmethod
    def _resolution(
        history: list[AlertTransition],
        severity: Severity,
        status: AlertStatus,
        resolved_at: datetime,
        now: datetime,
        live: bool,
    ) -> AlertTransition:
        return AlertTransition(
            seq=len(history) + 1,
            kind=TransitionKind.RESOLVED,
            at=now if live else resolved_at,
            effective_at=resolved_at,
            from_status=status,
            to_status=AlertStatus.RESOLVED,
            from_severity=severity,
            to_severity=severity,
            reason="the condition ended",
        )

    # ---- operator actions ------------------------------------------------------------------
    def acknowledge(
        self, alert_id: str, at: datetime, reason: str = "acknowledged by operator"
    ) -> AlertChange:
        alert = self._require(alert_id)
        if alert.status is AlertStatus.RESOLVED:
            raise LifecycleError(f"{alert_id} is already resolved")
        if alert.status is AlertStatus.ACKNOWLEDGED:
            raise LifecycleError(f"{alert_id} is already acknowledged")
        return self._operator(
            alert, AlertStatus.ACKNOWLEDGED, TransitionKind.ACKNOWLEDGED, at, reason
        )

    def resolve(
        self, alert_id: str, at: datetime, reason: str = "resolved by operator"
    ) -> AlertChange:
        alert = self._require(alert_id)
        if alert.status is AlertStatus.RESOLVED:
            raise LifecycleError(f"{alert_id} is already resolved")
        return self._operator(
            alert, AlertStatus.RESOLVED, TransitionKind.RESOLVED, at, reason, resolved_at=at
        )

    def _require(self, alert_id: str) -> Alert:
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise LifecycleError(f"no alert {alert_id}")
        return alert

    def _operator(
        self,
        alert: Alert,
        status: AlertStatus,
        kind: TransitionKind,
        at: datetime,
        reason: str,
        resolved_at: datetime | None = None,
    ) -> AlertChange:
        step = AlertTransition(
            seq=len(alert.history) + 1,
            kind=kind,
            at=at,
            effective_at=at,
            from_status=alert.status,
            to_status=status,
            from_severity=alert.severity,
            to_severity=alert.severity,
            reason=reason,
            actor="operator",
        )
        updated = _evolve(
            alert,
            status=status,
            history=[*alert.history, step],
            updated_at=at,
            resolved_at=resolved_at if resolved_at is not None else alert.resolved_at,
        )
        self._alerts[alert.alert_id] = updated
        return AlertChange(updated, [step])
