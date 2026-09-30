"""Drone-safety rules over telemetry: episodes, ladders, resolution, determinism."""

from datetime import datetime, timedelta

import pytest
from conftest import IST

from drone_agent.models import AlertStatus, Severity, Telemetry, Zone
from drone_agent.rules import RuleEngine, SecurityRulesConfig
from drone_agent.rules.config import (
    AltitudeRule,
    GeofenceRule,
    LowBatteryRule,
    SignalLossRule,
)
from drone_agent.rules.safety import (
    MAX_EVIDENCE_SAMPLES,
    altitude_anomaly,
    geofence_violation,
    low_battery,
    signal_loss,
)
from drone_agent.telemetry import east_offset_degrees

T0 = datetime(2026, 9, 29, 23, 40, tzinfo=IST)


def sample(offset: int, **fields) -> Telemetry:
    data = dict(
        timestamp=T0 + timedelta(seconds=offset),
        latitude=0.0004,
        longitude=0.00045,
        altitude_m=30.0,
        zone=Zone.MAIN_GATE,
        battery_percent=90.0,
    )
    data.update(fields)
    return Telemetry(**data)


def series(values, step=10):
    """One sample every `step` seconds; `values` are battery percentages or field dicts."""
    return [
        sample(i * step, **(v if isinstance(v, dict) else {"battery_percent": v}))
        for i, v in enumerate(values)
    ]


# ---- low battery ----------------------------------------------------------------------------
def test_battery_above_the_warning_limit_is_silent():
    assert low_battery(series([90, 60, 21]), LowBatteryRule()) == []


def test_low_battery_warns_then_escalates_the_same_alert_and_stays_open():
    (alert,) = low_battery(series([30, 25, 20, 15, 9, 8]), LowBatteryRule())
    assert alert.severity is Severity.CRITICAL and alert.status is AlertStatus.OPEN
    assert [(s.label, s.severity) for s in alert.ladder] == [
        ("warning", Severity.MEDIUM),
        ("critical", Severity.CRITICAL),
    ]
    assert alert.triggered_at == T0 + timedelta(seconds=20)  # first sample at or below 20%
    assert alert.ladder[1].at == T0 + timedelta(seconds=40)  # first sample at or below 10%
    assert alert.facts["lowest_percent"] == 8
    assert alert.event_ids == [] and alert.frame_ids == [] and alert.confidence == 1.0
    assert alert.verdict.value == "critical" and "dock" in alert.recommended_action
    assert alert.alert_id == "ALR-BATTERY-20260929T181020Z"
    assert 2 <= len(alert.telemetry_at) <= MAX_EVIDENCE_SAMPLES
    assert T0 + timedelta(seconds=50) in alert.telemetry_at  # the worst sample is cited


def test_battery_jumping_straight_to_critical_has_a_single_rung():
    (alert,) = low_battery(series([50, 8, 7]), LowBatteryRule())
    assert alert.severity is Severity.CRITICAL and [s.label for s in alert.ladder] == ["critical"]


def test_battery_resolves_only_after_recovering_past_the_hysteresis_band():
    (still_open,) = low_battery(series([19, 22, 24]), LowBatteryRule())  # 22-24 < recover 25
    assert still_open.status is AlertStatus.OPEN
    (resolved,) = low_battery(series([19, 22, 26]), LowBatteryRule())
    assert resolved.status is AlertStatus.RESOLVED
    assert resolved.resolved_at == T0 + timedelta(seconds=20)


def test_a_second_low_battery_episode_is_a_second_alert_with_its_own_id():
    alerts = low_battery(series([19, 30, 90, 18]), LowBatteryRule())
    assert len(alerts) == 2 and len({a.alert_id for a in alerts}) == 2


def test_low_battery_thresholds_are_validated():
    with pytest.raises(ValueError, match="critical_percent < warning_percent"):
        LowBatteryRule(critical_percent=30, warning_percent=20)
    with pytest.raises(ValueError):
        LowBatteryRule(recover_percent=15)


# ---- signal loss ----------------------------------------------------------------------------
def test_signal_is_not_a_fault_when_the_source_reports_none():
    assert signal_loss(series([90] * 5), SignalLossRule()) == []


def test_degraded_then_lost_then_recovered_signal_is_one_escalating_alert():
    samples = series(
        [
            {"signal_percent": 90},
            {"signal_percent": 35},  # degraded (< 40)
            {"signal_percent": 30},
            {"signal_percent": 10},  # lost (< 15)
            {"signal_percent": 80},  # recovered
        ]
    )
    (alert,) = signal_loss(samples, SignalLossRule())
    assert alert.severity is Severity.HIGH and alert.status is AlertStatus.RESOLVED
    assert [(s.label, s.at) for s in alert.ladder] == [
        ("degraded", T0 + timedelta(seconds=10)),
        ("lost", T0 + timedelta(seconds=30)),
    ]
    assert alert.resolved_at == T0 + timedelta(seconds=40)
    assert alert.facts["lowest_percent"] == 10 and alert.facts["delivery_gap"] is False


def test_a_delivery_gap_is_a_lost_link_that_resolves_when_telemetry_resumes():
    samples = [sample(0), sample(10), sample(150), sample(160)]  # 140 s of silence
    (alert,) = signal_loss(samples, SignalLossRule())
    assert alert.severity is Severity.HIGH and alert.facts["delivery_gap"] is True
    assert alert.triggered_at == T0 + timedelta(seconds=40)  # last good sample + 30 s
    assert alert.first_seen == T0 + timedelta(seconds=10)  # the last good sample is evidence
    assert alert.status is AlertStatus.RESOLVED and alert.resolved_at == T0 + timedelta(seconds=150)
    assert {T0 + timedelta(seconds=10), T0 + timedelta(seconds=150)} <= set(alert.telemetry_at)
    assert alert.zone is Zone.MAIN_GATE and alert.event_ids == []


def test_gap_of_exactly_the_limit_is_not_an_outage():
    assert signal_loss([sample(0), sample(30)], SignalLossRule()) == []
    assert len(signal_loss([sample(0), sample(31)], SignalLossRule())) == 1


def test_an_outage_still_going_on_is_reported_only_when_now_is_known():
    samples = [sample(0), sample(10)]
    assert signal_loss(samples, SignalLossRule()) == []  # nothing visible in the data itself
    now = T0 + timedelta(seconds=100)
    (alert,) = signal_loss(samples, SignalLossRule(), now)
    assert alert.status is AlertStatus.OPEN and alert.last_seen == now
    assert alert.triggered_at == T0 + timedelta(seconds=40)
    assert signal_loss(samples, SignalLossRule(), T0 + timedelta(seconds=40)) == []  # not > 30 s


def test_signal_thresholds_are_validated():
    with pytest.raises(ValueError, match="lost_below_percent"):
        SignalLossRule(lost_below_percent=50, degraded_below_percent=40)
    with pytest.raises(ValueError, match="lost_severity"):
        SignalLossRule(severity=Severity.HIGH, lost_severity=Severity.LOW)


# ---- altitude -------------------------------------------------------------------------------
def alt(*values):
    return series([{"altitude_m": a} for a in values])


def test_altitude_inside_the_envelope_is_silent_and_outside_is_an_alert():
    assert altitude_anomaly(alt(5, 30, 120), AltitudeRule()) == []
    (alert,) = altitude_anomaly(alt(30, 130, 135, 30), AltitudeRule())
    assert alert.severity is Severity.MEDIUM and alert.status is AlertStatus.RESOLVED
    assert alert.facts["extreme_altitude_m"] == 135
    assert alert.resolved_at == T0 + timedelta(seconds=30)


def test_severe_altitude_escalates_to_high():
    (alert,) = altitude_anomaly(alt(30, 125, 160), AltitudeRule())
    assert alert.severity is Severity.HIGH
    assert [s.label for s in alert.ladder] == ["out_of_range", "severe"]
    (low,) = altitude_anomaly(alt(3, 1), AltitudeRule())
    assert low.severity is Severity.HIGH  # below the severe floor


def test_altitude_envelope_is_validated():
    with pytest.raises(ValueError, match="severe_min_m"):
        AltitudeRule(min_m=10, max_m=5)


# ---- geofence -------------------------------------------------------------------------------
def fenced(east_m: float, offset: int) -> Telemetry:
    return sample(offset, longitude=0.00045 + east_offset_degrees(east_m, 0.0004))


def test_geofence_violation_high_then_critical_then_resolved():
    samples = [fenced(0, 0), fenced(110, 10), fenced(160, 20), fenced(20, 30)]
    (alert,) = geofence_violation(samples, GeofenceRule())
    assert alert.severity is Severity.CRITICAL and alert.status is AlertStatus.RESOLVED
    assert [(s.label, s.severity) for s in alert.ladder] == [
        ("outside", Severity.HIGH),
        ("far_outside", Severity.CRITICAL),
    ]
    assert 159 < alert.facts["max_distance_m"] < 161 and alert.facts["radius_m"] == 100
    assert alert.triggered_at == T0 + timedelta(seconds=10)
    assert geofence_violation([fenced(99, 0), fenced(0, 10)], GeofenceRule()) == []


# ---- engine integration and determinism ----------------------------------------------------------
def test_engine_runs_safety_rules_and_disabled_rules_are_silent():
    samples = series([30, 19, 9])
    assert [a.rule_id for a in RuleEngine().run([], [], samples)] == ["low_battery"]
    off = SecurityRulesConfig(low_battery=LowBatteryRule(enabled=False))
    assert RuleEngine(off).run([], [], samples) == []
    assert RuleEngine().run([], []) == []  # the old call shape still works


def test_alert_ids_and_evidence_are_stable_as_the_telemetry_stream_grows():
    full = series([30, 25, 20, 15, 9, 8, 8])
    for cut in range(3, len(full) + 1):
        alerts = low_battery(full[:cut], LowBatteryRule())
        assert [a.alert_id for a in alerts] == ["ALR-BATTERY-20260929T181020Z"]
    assert low_battery(full, LowBatteryRule()) == low_battery(full[::-1], LowBatteryRule())


def test_duplicate_timestamps_are_collapsed():
    once = low_battery(series([30, 19, 9]), LowBatteryRule())
    twice = low_battery(series([30, 19, 9]) * 2, LowBatteryRule())
    assert once == twice
