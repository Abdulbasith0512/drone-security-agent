from datetime import UTC, datetime, timedelta, timezone

import pytest
from conftest import IST, ROOT
from pydantic import ValidationError
from test_aggregator import obs as make_obs
from test_aggregator import person as make_person
from test_linker import blue
from test_linker import event as vehicle_event

from drone_agent.models import CloseReason, Event, ObjectType, Severity, Zone
from drone_agent.pipeline import ConfigError, EntityLinker, aggregate_observations
from drone_agent.rules import (
    RuleEngine,
    SecurityRulesConfig,
    after_hours,
    load_rules_config,
    loitering,
    repeat_vehicle,
    restricted_zone,
)
from drone_agent.rules.config import (
    AfterHoursRule,
    LoiteringRule,
    RepeatVehicleRule,
    RestrictedZoneRule,
    ZoneThreshold,
)

DAY = datetime(2026, 9, 29, 0, 0, tzinfo=IST)
# The pre-M9 single-threshold rule: one alert at 300 s, no escalation. The shipped default now
# has a ladder (WARNING at 120 s, CRITICAL at 300 s); these tests pin the boundary behaviour.
SINGLE_THRESHOLD = LoiteringRule(zones={Zone.MAIN_GATE: ZoneThreshold(threshold_seconds=300)})


def at(hour: int, minute: int = 0, second: int = 0, day: int = 0, tz=IST) -> datetime:
    return datetime(2026, 9, 29 + day, hour, minute, second, tzinfo=tz)


def make_event(
    event_id="EVT-P-001",
    kind=ObjectType.PERSON,
    zone=Zone.MAIN_GATE,
    first=None,
    seconds=0,
    frames=(1,),
) -> Event:
    first = first or at(23, 41)
    return Event(
        event_id=event_id,
        object_type=kind,
        zone=zone,
        first_seen=first,
        last_seen=first + timedelta(seconds=seconds),
        observation_ids=[f"obs-{f}" for f in frames],
        frame_ids=list(frames),
        close_reason=CloseReason.GAP,
    )


# ---- loitering ------------------------------------------------------------------------------
def test_loitering_below_threshold_does_not_fire():
    assert loitering.evaluate([make_event(seconds=299)], SINGLE_THRESHOLD) == []


def test_loitering_exactly_at_threshold_fires_boundary_is_inclusive():
    (alert,) = loitering.evaluate([make_event(seconds=300)], SINGLE_THRESHOLD)
    assert alert.facts["duration_seconds"] == 300 == alert.facts["threshold_seconds"]


def test_loitering_above_threshold_fires_with_facts_and_evidence():
    ev = make_event(seconds=480, frames=(1, 2, 3, 4, 6, 7, 8, 9))
    (alert,) = loitering.evaluate([ev], SINGLE_THRESHOLD)
    assert alert.alert_id == "ALR-LOITER-EVT-P-001"
    assert alert.rule_id == "loitering" and alert.severity is Severity.MEDIUM
    assert alert.facts == {
        "zone": "main_gate",
        "object_type": "person",
        "duration_seconds": 480,
        "threshold_seconds": 300,
        "comparison": "duration_seconds >= threshold_seconds",
        "level": "medium",
    }
    assert alert.triggered_at == ev.first_seen + timedelta(
        seconds=300
    )  # when the threshold elapsed
    assert (alert.first_seen, alert.last_seen) == (ev.first_seen, ev.last_seen)
    assert alert.event_ids == ["EVT-P-001"] and alert.frame_ids == [1, 2, 3, 4, 6, 7, 8, 9]
    assert "480s" in alert.message and "300s" in alert.message


def test_loitering_uses_duration_not_frame_count_or_frame_ids():
    few_frames = make_event(seconds=600, frames=(1, 2))
    many_frames = make_event(seconds=100, frames=tuple(range(50, 80)))
    assert len(loitering.evaluate([few_frames], LoiteringRule())) == 1
    assert loitering.evaluate([many_frames], LoiteringRule()) == []
    other_ids = make_event(seconds=600, frames=(500, 900))
    assert loitering.evaluate([other_ids], LoiteringRule())[0].facts == (
        loitering.evaluate([few_frames], LoiteringRule())[0].facts
    )


def test_loitering_only_applies_to_monitored_zones_and_object_types():
    rule = LoiteringRule()
    assert loitering.evaluate([make_event(zone=Zone.WAREHOUSE, seconds=900)], rule) == []
    assert loitering.evaluate([make_event(kind=ObjectType.VEHICLE, seconds=900)], rule) == []
    custom = LoiteringRule(
        zones={Zone.WAREHOUSE: ZoneThreshold(threshold_seconds=60)},
        object_types=[ObjectType.PERSON, ObjectType.VEHICLE],
    )
    assert len(loitering.evaluate([make_event(zone=Zone.WAREHOUSE, seconds=60)], custom)) == 1
    assert loitering.evaluate([make_event(zone=Zone.MAIN_GATE, seconds=900)], custom) == []


def test_loitering_disabled_and_severity_configurable():
    assert loitering.evaluate([make_event(seconds=900)], LoiteringRule(enabled=False)) == []
    single_high = LoiteringRule(severity=Severity.HIGH, zones=SINGLE_THRESHOLD.zones)
    (alert,) = loitering.evaluate([make_event(seconds=900)], single_high)
    assert alert.severity is Severity.HIGH


def test_flicker_compressed_event_loiters_naturally():
    """The aggregator merges a missed detection into one 480 s event; the rule just sees 480 s."""
    frames = [
        make_obs(i, 60 * i, Zone.MAIN_GATE, *([] if i == 5 else [make_person()]), start=at(23, 40))
        for i in range(1, 10)
    ]
    (event,) = aggregate_observations(frames)
    assert event.frame_ids == [1, 2, 3, 4, 6, 7, 8, 9]
    (alert,) = loitering.evaluate([event], LoiteringRule())
    assert alert.facts["duration_seconds"] == 480


# ---- after hours ----------------------------------------------------------------------------
def after_hours_alerts(first, seconds=0, cfg=None, **kwargs):
    ev = make_event(first=first, seconds=seconds, **kwargs)
    return after_hours.evaluate([ev], cfg or AfterHoursRule())


@pytest.mark.parametrize(
    "when,expected",
    [
        (at(21, 59, 59), False),  # just before the window opens
        (at(22, 0, 0), True),  # exactly 22:00: window start is inclusive
        (at(23, 59, 59), True),  # before midnight
        (at(0, 0, 0, day=1), True),  # midnight
        (at(3, 0, 0, day=1), True),  # after midnight
        (at(5, 59, 59, day=1), True),  # just before the window closes
        (at(6, 0, 0, day=1), False),  # exactly 06:00: window end is exclusive
        (at(12, 0, 0), False),  # daytime
    ],
)
def test_after_hours_window_boundaries(when, expected):
    assert bool(after_hours_alerts(when)) is expected


def test_after_hours_alert_facts_and_wording():
    (alert,) = after_hours_alerts(at(0, 11, day=1), seconds=60, zone=Zone.WAREHOUSE)
    assert alert.alert_id == "ALR-AFTERHOURS-EVT-P-001"
    assert alert.triggered_at == at(0, 11, day=1)
    assert alert.facts["event_local_time"] == "00:11:00"
    assert alert.facts["closed_from"] == "22:00" and alert.facts["closed_until"] == "06:00"
    assert alert.facts["utc_offset"] == "+05:30"
    assert "inside closed hours 22:00-06:00" in alert.message


def test_event_starting_before_the_window_alerts_from_the_window_start():
    (alert,) = after_hours_alerts(at(21, 58), seconds=300)  # 21:58 -> 22:03
    assert alert.triggered_at == at(22, 0)
    assert alert.first_seen == at(21, 58)


def test_event_starting_inside_and_running_past_the_window_end_alerts_once():
    (alert,) = after_hours_alerts(at(5, 58, day=1), seconds=300)  # 05:58 -> 06:03
    assert alert.triggered_at == at(5, 58, day=1)


def test_event_that_ends_before_the_window_or_starts_after_it_does_not_alert():
    assert after_hours_alerts(at(21, 50), seconds=300) == []  # ends 21:55
    assert after_hours_alerts(at(6, 0, day=1), seconds=300) == []  # starts at 06:00


def test_event_crossing_midnight_yields_a_single_alert():
    assert len(after_hours_alerts(at(23, 59), seconds=120)) == 1


def test_daytime_closed_window_that_does_not_cross_midnight():
    cfg = AfterHoursRule(closed_from="08:00", closed_until="17:00")
    assert bool(after_hours_alerts(at(10, 0), cfg=cfg))
    assert not after_hours_alerts(at(7, 59, 59), cfg=cfg)
    assert not after_hours_alerts(at(17, 0, 0), cfg=cfg)
    assert bool(after_hours_alerts(at(8, 0, 0), cfg=cfg))


def test_after_hours_uses_the_events_own_offset_unless_a_zone_is_configured():
    utc = UTC
    same_instant_as_midnight_ist = datetime(2026, 9, 29, 18, 30, tzinfo=utc)  # 00:00 IST
    assert after_hours_alerts(same_instant_as_midnight_ist) == []  # 18:30 in its own (UTC) offset
    cfg = AfterHoursRule(utc_offset="+05:30")
    (alert,) = after_hours_alerts(same_instant_as_midnight_ist, cfg=cfg)
    assert alert.facts["event_local_time"] == "00:00:00" and alert.facts["utc_offset"] == "+05:30"
    west = AfterHoursRule(utc_offset="-08:00")
    assert after_hours_alerts(same_instant_as_midnight_ist, cfg=west) == []  # 10:30 local
    assert AfterHoursRule(utc_offset="-08:00").offset == timezone(timedelta(hours=-8))


def test_after_hours_object_type_and_zone_filters():
    assert after_hours_alerts(at(23, 0), kind=ObjectType.VEHICLE) == []  # people only by default
    both = AfterHoursRule(object_types=[ObjectType.PERSON, ObjectType.VEHICLE])
    assert len(after_hours_alerts(at(23, 0), kind=ObjectType.VEHICLE, cfg=both)) == 1
    only_gate = AfterHoursRule(zones=[Zone.MAIN_GATE])
    assert after_hours_alerts(at(23, 0), zone=Zone.WAREHOUSE, cfg=only_gate) == []
    assert len(after_hours_alerts(at(23, 0), zone=Zone.MAIN_GATE, cfg=only_gate)) == 1
    assert after_hours_alerts(at(23, 0), cfg=AfterHoursRule(enabled=False)) == []


# ---- repeat vehicle -------------------------------------------------------------------------
def repeat_alerts(events, cfg=None):
    events, entities = EntityLinker().link(events)
    return repeat_vehicle.evaluate(events, entities, cfg or RepeatVehicleRule())


def appearances(n: int, **attrs):
    return [vehicle_event(i + 1, 60 * i, blue(**attrs)) for i in range(n)]


@pytest.mark.parametrize("n", [1, 2])
def test_repeat_vehicle_below_minimum_does_not_fire(n):
    assert repeat_alerts(appearances(n)) == []


def test_repeat_vehicle_fires_once_at_the_minimum_with_careful_wording():
    (alert,) = repeat_alerts(
        [
            vehicle_event(1, 0, blue("blue")),
            vehicle_event(2, 60, blue("dark blue")),
            vehicle_event(3, 120, blue("blue")),
        ]
    )
    assert alert.alert_id == "ALR-REPEAT-ENT-V-001"
    assert alert.entity_id == "ENT-V-001"
    assert alert.event_ids == ["EVT-V-001", "EVT-V-002", "EVT-V-003"]
    assert alert.frame_ids == [1, 2, 3]
    assert "similar blue pickup was observed in 3 separate appearances" in alert.message
    assert "does not confirm the same vehicle" in alert.message
    assert "same vehicle returned" not in alert.message
    assert alert.facts["appearance_count"] == 3
    assert alert.facts["minimum_appearances"] == 3
    assert alert.facts["match_level"] == "coarse_match"
    assert alert.severity is Severity.LOW


def test_repeat_vehicle_above_minimum_is_still_one_alert_triggered_when_count_was_reached():
    events = appearances(5)
    (alert,) = repeat_alerts(events)
    assert alert.facts["appearance_count"] == 5
    assert alert.triggered_at == events[2].first_seen  # the 3rd appearance reached the minimum
    assert alert.first_seen == events[0].first_seen and alert.last_seen == events[-1].last_seen


def test_repeat_vehicle_minimum_is_configurable():
    assert len(repeat_alerts(appearances(2), RepeatVehicleRule(minimum_appearances=2))) == 1
    assert repeat_alerts(appearances(3), RepeatVehicleRule(minimum_appearances=4)) == []
    assert repeat_alerts(appearances(3), RepeatVehicleRule(enabled=False)) == []


def test_confirmed_plate_match_is_worded_as_confirmed():
    events = [vehicle_event(i, 60 * i, blue(plate="AB12CD")) for i in (1, 2, 3)]
    (alert,) = repeat_alerts(events)
    assert alert.facts["match_level"] == "confirmed"
    assert "confirmed blue pickup (plate AB12CD)" in alert.message
    assert "coarse" not in alert.message


def test_a_different_vehicle_does_not_count_towards_the_repeat():
    events = [
        *appearances(3),
        vehicle_event(4, 500, {"color": "white", "body_type": "sedan"}),
        vehicle_event(5, 600, {"color": "white", "body_type": "sedan"}),
    ]
    (alert,) = repeat_alerts(events)
    assert alert.entity_id == "ENT-V-001"  # only the pickup; the sedan (2) stays below minimum


# ---- restricted zone ------------------------------------------------------------------------
def test_allowed_zone_does_not_fire():
    assert restricted_zone.evaluate([make_event(zone=Zone.MAIN_GATE)], RestrictedZoneRule()) == []


def test_restricted_zone_multi_frame_event_gives_exactly_one_alert():
    ev = make_event("EVT-P-003", zone=Zone.RESTRICTED_ZONE, seconds=120, frames=(22, 23, 24))
    (alert,) = restricted_zone.evaluate([ev], RestrictedZoneRule())
    assert alert.alert_id == "ALR-RESTRICTED-EVT-P-003"
    assert alert.severity is Severity.HIGH
    assert alert.frame_ids == [22, 23, 24]
    assert alert.facts == {
        "zone": "restricted_zone",
        "object_type": "person",
        "duration_seconds": 120,
    }
    assert alert.triggered_at == ev.first_seen


def test_restricted_zone_vehicles_and_other_types_follow_config():
    vehicle = make_event("EVT-V-001", ObjectType.VEHICLE, Zone.RESTRICTED_ZONE)
    other = make_event("EVT-O-001", ObjectType.OTHER, Zone.RESTRICTED_ZONE)
    assert len(restricted_zone.evaluate([vehicle, other], RestrictedZoneRule())) == 1
    people_only = RestrictedZoneRule(object_types=[ObjectType.PERSON])
    assert restricted_zone.evaluate([vehicle], people_only) == []
    warehouse = RestrictedZoneRule(zones=[Zone.WAREHOUSE])
    assert restricted_zone.evaluate([make_event(zone=Zone.WAREHOUSE)], warehouse)
    assert restricted_zone.evaluate([make_event(zone=Zone.RESTRICTED_ZONE)], warehouse) == []
    assert restricted_zone.evaluate([vehicle], RestrictedZoneRule(enabled=False)) == []


# ---- engine: determinism and overlap policy -------------------------------------------------
def test_one_event_can_trigger_several_rules_and_each_alert_is_distinct():
    ev = make_event("EVT-P-003", zone=Zone.RESTRICTED_ZONE, first=at(23, 41), seconds=600)
    cfg = SecurityRulesConfig(
        loitering=LoiteringRule(zones={Zone.RESTRICTED_ZONE: ZoneThreshold(threshold_seconds=300)})
    )
    alerts = RuleEngine(cfg).run([ev], [])
    # ordered by triggered_at (23:41, 23:41, then loitering at 23:46), ties by rule order
    assert [a.rule_id for a in alerts] == [
        "after_hours_presence",
        "restricted_zone_entry",
        "loitering",
    ]
    assert len({a.alert_id for a in alerts}) == 3
    assert {a.event_ids[0] for a in alerts} == {"EVT-P-003"}  # different risks, same event


def test_engine_output_is_deterministic_and_never_duplicates_a_rule_subject_pair():
    ev = make_event(seconds=600)
    once = RuleEngine().run([ev], [])
    twice = RuleEngine().run([ev, ev], [])  # the same event listed twice
    assert once == twice == RuleEngine().run([ev], [])
    assert [a.alert_id for a in once] == ["ALR-AFTERHOURS-EVT-P-001", "ALR-LOITER-EVT-P-001"]


def test_engine_with_all_rules_disabled_returns_nothing():
    cfg = SecurityRulesConfig(
        loitering=LoiteringRule(enabled=False),
        after_hours_presence=AfterHoursRule(enabled=False),
        repeat_vehicle=RepeatVehicleRule(enabled=False),
        restricted_zone_entry=RestrictedZoneRule(enabled=False),
    )
    assert RuleEngine(cfg).run([make_event(seconds=900)], []) == []


# ---- configuration --------------------------------------------------------------------------
def test_shipped_rules_yaml_equals_built_in_defaults():
    shipped = load_rules_config(ROOT / "config" / "security_rules.yaml")
    assert shipped == SecurityRulesConfig() == load_rules_config()
    gate = shipped.loitering.zones[Zone.MAIN_GATE]
    assert (gate.threshold_seconds, gate.critical_after_seconds) == (120, 300)
    assert (
        shipped.after_hours_presence.closed_from.hour,
        shipped.after_hours_presence.closed_until.hour,
    ) == (22, 6)
    assert shipped.repeat_vehicle.minimum_appearances == 3


def write(tmp_path, text):
    path = tmp_path / "rules.yaml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "yaml_text,fragment",
    [
        ("loitering: {zones: {main_gate: {threshold_seconds: 0}}}", "threshold_seconds"),
        ("loitering: {zones: {moon_base: {threshold_seconds: 60}}}", "moon_base"),
        ("loitering: {zones: {}}", "no zones"),
        ("loitering: {severity: urgent}", "severity"),
        ("loitering: {typo_key: 1}", "typo_key"),
        ("after_hours_presence: {closed_from: 22:00}", "quoted"),
        ('after_hours_presence: {closed_from: "22:00", closed_until: "22:00"}', "differ"),
        ('after_hours_presence: {closed_from: "25:99"}', "closed_from"),
        ('after_hours_presence: {utc_offset: "IST"}', "utc_offset"),
        ("after_hours_presence: {object_types: []}", "object_types"),
        ("repeat_vehicle: {minimum_appearances: 1}", "minimum_appearances"),
        ("restricted_zone_entry: {zones: []}", "no zones"),
        ("some_new_rule: {enabled: true}", "some_new_rule"),
    ],
)
def test_invalid_rules_config_fails_clearly(tmp_path, yaml_text, fragment):
    with pytest.raises(ConfigError, match=fragment):
        load_rules_config(write(tmp_path, yaml_text + "\n"))


def test_missing_or_malformed_rules_file_fails_clearly(tmp_path):
    with pytest.raises(ConfigError):
        load_rules_config(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError):
        load_rules_config(write(tmp_path, "loitering: [unclosed"))


def test_disabled_rule_may_have_nothing_to_monitor_and_partial_files_use_defaults(tmp_path):
    cfg = load_rules_config(write(tmp_path, "loitering: {enabled: false, zones: {}}\n"))
    assert not cfg.loitering.enabled and cfg.repeat_vehicle == RepeatVehicleRule()
    assert load_rules_config(write(tmp_path, "")) == SecurityRulesConfig()


def test_rule_models_reject_direct_misuse():
    with pytest.raises(ValidationError):
        AfterHoursRule(closed_from=1320)
    with pytest.raises(ValidationError):
        RepeatVehicleRule(minimum_appearances=0)


# ---- escalation ladder, triage fields and lifecycle inputs (M9) --------------------------------
def gate_event(seconds: int, reason=CloseReason.GAP, departed_after: int | None = None) -> Event:
    """`departed_after`: seconds after last_seen at which the zone was first seen without it."""
    event = make_event(seconds=seconds)
    departed = (
        None if departed_after is None else event.last_seen + timedelta(seconds=departed_after)
    )
    return event.model_copy(
        update={"close_reason": reason, "confidence": 0.92, "departure_observed_at": departed}
    )


def test_default_loitering_ladder_warns_at_120s_and_escalates_at_300s():
    rule = LoiteringRule()
    assert loitering.evaluate([gate_event(119)], rule) == []
    (warn,) = loitering.evaluate([gate_event(120)], rule)
    assert warn.severity is Severity.MEDIUM and [s.label for s in warn.ladder] == ["warning"]
    (still,) = loitering.evaluate([gate_event(299)], rule)
    assert still.severity is Severity.MEDIUM and len(still.ladder) == 1
    (crit,) = loitering.evaluate([gate_event(300)], rule)
    assert crit.severity is Severity.CRITICAL
    assert [(s.label, s.severity) for s in crit.ladder] == [
        ("warning", Severity.MEDIUM),
        ("critical", Severity.CRITICAL),
    ]
    # one alert that escalates: the id never changes and triggered_at stays at the first rung
    assert warn.alert_id == still.alert_id == crit.alert_id
    assert warn.triggered_at == crit.triggered_at == crit.first_seen + timedelta(seconds=120)
    assert crit.ladder[1].at == crit.first_seen + timedelta(seconds=300)


def test_loitering_alert_resolves_only_when_the_object_was_seen_to_be_gone():
    # the observation coverage ended, which says nothing about whether the person left
    for reason in (
        CloseReason.GAP,
        CloseReason.ZONE_CHANGE,
        CloseReason.SCENARIO_END,
        CloseReason.OPEN,
        CloseReason.MISSED_FRAMES,  # without a recorded departure (for example an older database)
    ):
        (alert,) = loitering.evaluate([gate_event(200, reason)], LoiteringRule())
        assert alert.status.value == "open" and alert.resolved_at is None, reason
    # the drone kept looking at the same zone and the person was not there: resolved at that moment
    event = gate_event(200, CloseReason.MISSED_FRAMES, departed_after=60)
    (alert,) = loitering.evaluate([event], LoiteringRule())
    assert alert.status.value == "resolved"
    assert (
        alert.resolved_at == event.last_seen + timedelta(seconds=60) == event.departure_observed_at
    )


def test_alerts_carry_confidence_verdict_and_a_recommended_action():
    (alert,) = loitering.evaluate([gate_event(400)], LoiteringRule())
    assert alert.confidence == 0.92
    assert alert.verdict.value == "critical" and "Dispatch security" in alert.recommended_action
    (warning,) = loitering.evaluate([gate_event(150)], LoiteringRule())
    assert warning.verdict.value == "suspicious"
    assert "verify the person" in warning.recommended_action
    custom = LoiteringRule(recommended_action="Call the guard", critical_action="Lock the gate")
    assert loitering.evaluate([gate_event(150)], custom)[0].recommended_action == "Call the guard"
    assert loitering.evaluate([gate_event(400)], custom)[0].recommended_action == "Lock the gate"


def test_every_event_rule_fills_verdict_action_confidence_and_a_ladder():
    event = make_event("EVT-P-003", zone=Zone.RESTRICTED_ZONE, first=at(23, 41), seconds=600)
    alerts = RuleEngine().run([event.model_copy(update={"confidence": 0.8})], [])
    assert {a.rule_id for a in alerts} == {"after_hours_presence", "restricted_zone_entry"}
    for alert in alerts:
        assert alert.verdict is not None and alert.recommended_action
        assert alert.confidence == 0.8
        assert [s.severity for s in alert.ladder] == [alert.severity]
        assert alert.status.value == "open"


def test_ladder_configuration_is_validated():
    with pytest.raises(ValidationError, match="critical_after_seconds"):
        ZoneThreshold(threshold_seconds=300, critical_after_seconds=300)
    with pytest.raises(ValidationError, match="critical_after_seconds"):
        ZoneThreshold(threshold_seconds=300, critical_after_seconds=60)
    only_warning = LoiteringRule(zones={Zone.MAIN_GATE: ZoneThreshold(threshold_seconds=60)})
    assert loitering.evaluate([gate_event(9000)], only_warning)[0].severity is Severity.MEDIUM


def test_a_critical_base_severity_has_no_second_rung():
    rule = LoiteringRule(severity=Severity.CRITICAL)
    (alert,) = loitering.evaluate([gate_event(900)], rule)
    assert alert.severity is Severity.CRITICAL and len(alert.ladder) == 1
