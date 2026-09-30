"""Alert lifecycle: rule snapshots -> raise / escalate / acknowledge / resolve, with history."""

from datetime import datetime, timedelta

import pytest
from conftest import IST

from drone_agent.models import (
    Alert,
    AlertStatus,
    CloseReason,
    Event,
    ObjectType,
    Severity,
    TransitionKind,
    Zone,
)
from drone_agent.rules import AlertManager, LifecycleError, RuleEngine
from drone_agent.rules.config import LoiteringRule

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)


def event(seconds: int, reason=CloseReason.OPEN, frames=None) -> Event:
    frames = frames or list(range(1, seconds // 60 + 2))
    return Event(
        event_id="EVT-P-001",
        object_type=ObjectType.PERSON,
        zone=Zone.MAIN_GATE,
        first_seen=T0,
        last_seen=T0 + timedelta(seconds=seconds),
        attributes={"clothing": "jacket"},
        observation_ids=[f"obs-{f}" for f in frames],
        frame_ids=frames,
        close_reason=reason,
        confidence=0.9,
    )


def snapshot(seconds: int, reason=CloseReason.OPEN) -> Alert:
    (alert,) = RuleEngine().run([event(seconds, reason)], [])
    return alert


def t(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def kinds(alert: Alert) -> list[TransitionKind]:
    return [h.kind for h in alert.history]


# ---- streaming: the same alert raises, escalates and resolves ---------------------------------
def test_live_alert_is_raised_escalated_and_resolved_as_time_passes():
    manager = AlertManager()
    assert manager.reconcile([], t(60)) == []  # nothing yet

    (raised,) = manager.reconcile([snapshot(120)], t(125))  # noticed 5 s after the threshold
    assert raised.is_new and raised.alert.status is AlertStatus.OPEN
    assert raised.alert.severity is Severity.MEDIUM
    assert raised.alert.detected_at == t(125) and raised.alert.triggered_at == t(120)
    assert raised.alert.detection_latency_seconds == 5
    assert kinds(raised.alert) == [TransitionKind.RAISED]

    (grown,) = manager.reconcile([snapshot(180)], t(185))  # more evidence, same severity
    assert grown.transitions == [] and grown.alert.last_seen == t(180)
    assert grown.alert.status is AlertStatus.OPEN

    (escalated,) = manager.reconcile([snapshot(300)], t(305))
    assert [x.kind for x in escalated.transitions] == [TransitionKind.ESCALATED]
    alert = escalated.alert
    assert alert.alert_id == raised.alert.alert_id  # the SAME alert, not a new one
    assert alert.status is AlertStatus.ESCALATED and alert.severity is Severity.CRITICAL
    step = alert.history[-1]
    assert (step.from_severity, step.to_severity) == (Severity.MEDIUM, Severity.CRITICAL)
    assert (step.at, step.effective_at) == (t(305), t(300))  # recorded 5 s after it was true
    assert "Dispatch security" in alert.recommended_action  # the action follows the severity

    (resolved,) = manager.reconcile([snapshot(400, CloseReason.GAP)], t(520))
    assert resolved.alert.status is AlertStatus.RESOLVED
    assert resolved.alert.resolved_at == t(400) and resolved.alert.history[-1].at == t(520)
    assert kinds(resolved.alert) == [
        TransitionKind.RAISED,
        TransitionKind.ESCALATED,
        TransitionKind.RESOLVED,
    ]
    assert [h.seq for h in resolved.alert.history] == [1, 2, 3]
    assert len(manager.alerts) == 1


def test_reapplying_the_same_snapshot_changes_nothing():
    manager = AlertManager()
    manager.reconcile([snapshot(300)], t(300))
    assert manager.reconcile([snapshot(300)], t(310)) == []
    assert manager.reconcile([snapshot(300)], t(999)) == []


def test_sampling_that_jumps_over_the_warning_still_records_both_rungs():
    manager = AlertManager()
    (change,) = manager.reconcile([snapshot(360)], t(360))  # first look is already past 300 s
    assert kinds(change.alert) == [TransitionKind.RAISED, TransitionKind.ESCALATED]
    assert [h.effective_at for h in change.alert.history] == [t(120), t(300)]
    assert change.alert.status is AlertStatus.ESCALATED
    assert change.alert.detection_latency_seconds == 240  # noticed 240 s after it was true


# ---- batch and streaming end in the same state -----------------------------------------------
def test_batch_and_streaming_agree_on_everything_except_when_things_were_recorded():
    final = snapshot(480, CloseReason.GAP)
    batch = AlertManager()
    batch.reconcile([final], t(480), live=False)

    live = AlertManager()
    for seconds, reason in (
        (60, CloseReason.OPEN),
        (130, CloseReason.OPEN),
        (310, CloseReason.OPEN),
    ):
        live.reconcile(RuleEngine().run([event(seconds, reason)], []), t(seconds + 3))
    live.reconcile([final], t(500))

    (b,), (v,) = batch.alerts, live.alerts
    fields = {
        "alert_id", "rule_id", "severity", "status", "title", "message", "triggered_at",
        "first_seen", "last_seen", "zone", "event_ids", "frame_ids", "facts", "confidence",
        "verdict", "recommended_action", "ladder", "resolved_at", "entity_id",
    }  # fmt: skip
    assert b.model_dump(include=fields) == v.model_dump(include=fields)
    assert kinds(b) == kinds(v)
    assert [h.at for h in b.history] == [t(120), t(300), t(480)]  # batch: when it became true
    assert v.detected_at > b.detected_at  # streaming: when it was noticed


# ---- operator actions ------------------------------------------------------------------------
def test_operator_acknowledgement_then_a_later_escalation_reopens_attention():
    manager = AlertManager()
    manager.reconcile([snapshot(150)], t(150))
    alert_id = snapshot(150).alert_id

    change = manager.acknowledge(alert_id, t(200))
    assert change.alert.status is AlertStatus.ACKNOWLEDGED
    step = change.alert.history[-1]
    assert step.actor == "operator" and step.kind is TransitionKind.ACKNOWLEDGED

    assert manager.reconcile([snapshot(200)], t(210)) == [] or manager.get(alert_id).status in (
        AlertStatus.ACKNOWLEDGED,
    )
    assert manager.get(alert_id).status is AlertStatus.ACKNOWLEDGED  # acknowledged stays put

    (escalated,) = manager.reconcile([snapshot(320)], t(330))
    assert escalated.alert.status is AlertStatus.ESCALATED  # it got worse after being seen
    assert kinds(escalated.alert)[-1] is TransitionKind.ESCALATED


def test_operator_can_resolve_and_a_resolved_alert_never_reopens():
    manager = AlertManager()
    manager.reconcile([snapshot(150)], t(150))
    alert_id = snapshot(150).alert_id
    resolved = manager.resolve(alert_id, t(160), "false alarm").alert
    assert resolved.status is AlertStatus.RESOLVED and resolved.resolved_at == t(160)
    assert resolved.history[-1].reason == "false alarm" and resolved.history[-1].actor == "operator"
    # the rule still sees the person, even escalated; the operator's decision stands
    assert manager.reconcile([snapshot(400)], t(410)) == [] or (
        manager.get(alert_id).status is AlertStatus.RESOLVED
    )
    assert manager.get(alert_id).status is AlertStatus.RESOLVED
    assert manager.get(alert_id).severity is Severity.MEDIUM  # not escalated after resolution


def test_illegal_operator_actions_are_rejected_clearly():
    manager = AlertManager()
    manager.reconcile([snapshot(150)], t(150))
    alert_id = snapshot(150).alert_id
    with pytest.raises(LifecycleError, match="no alert"):
        manager.acknowledge("ALR-NOPE", t(1))
    manager.acknowledge(alert_id, t(160))
    with pytest.raises(LifecycleError, match="already acknowledged"):
        manager.acknowledge(alert_id, t(161))
    manager.resolve(alert_id, t(170))
    with pytest.raises(LifecycleError, match="already resolved"):
        manager.acknowledge(alert_id, t(171))
    with pytest.raises(LifecycleError, match="already resolved"):
        manager.resolve(alert_id, t(172))


def test_alerts_absent_from_a_snapshot_are_kept_not_deleted():
    manager = AlertManager()
    manager.reconcile([snapshot(150)], t(150))
    manager.reconcile([], t(200))
    assert len(manager.alerts) == 1  # nothing is ever deleted by the manager


def test_a_disabled_or_changed_rule_never_lowers_an_existing_severity():
    manager = AlertManager()
    manager.reconcile([snapshot(400)], t(400))  # critical
    milder = RuleEngine(
        config=type(RuleEngine().config)(loitering=LoiteringRule(severity=Severity.LOW))
    ).run([event(150)], [])
    manager.reconcile(milder, t(410))
    assert manager.alerts[0].severity is Severity.CRITICAL


def test_manager_can_be_restored_from_stored_alerts(store):
    """Persisted alerts (with their history) round-trip through the store and keep working."""
    for f in range(1, 4):
        from drone_agent.models import DetectedObject, Observation

        store.insert_observation(
            Observation(
                observation_id=f"obs-{f}",
                frame_id=f,
                timestamp=t(f * 60),
                source="scripted",
                zone=Zone.MAIN_GATE,
                description="p",
                detected_objects=[DetectedObject(object_type=ObjectType.PERSON, confidence=0.9)],
            )
        )
    ev = event(180, frames=[1, 2, 3]).model_copy(
        update={"observation_ids": ["obs-1", "obs-2", "obs-3"]}
    )
    store.save_events([ev])
    manager = AlertManager()
    manager.reconcile(RuleEngine().run([ev], []), t(180))
    for change in manager.alerts:
        store.save_alert(change)
    restored = AlertManager(store.query_alerts())
    assert restored.alerts == manager.alerts
    restored.acknowledge(restored.alerts[0].alert_id, t(190))
    store.save_alert(restored.alerts[0])
    assert store.query_alerts()[0].status is AlertStatus.ACKNOWLEDGED
