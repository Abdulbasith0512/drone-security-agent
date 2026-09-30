"""Ground-truth scenarios and the scorer: perfect labels score perfectly, wrong labels do not."""

import pytest
from conftest import SCENARIO_DIR
from pydantic import ValidationError
from rich.console import Console

from drone_agent.eval.ground_truth import (
    _percentile,
    aggregate,
    render_scores,
    run_scenario_scored,
    scenario_paths,
    score_store,
)
from drone_agent.expectations import (
    Expectations,
    ExpectedAlert,
    ExpectedEntity,
    ExpectedEvent,
    ExpectedStep,
    ForbiddenAlert,
)
from drone_agent.models import AlertStatus, ObjectType, Severity, Zone
from drone_agent.runtime import PatrolEngine
from drone_agent.scenario import load_scenario
from drone_agent.store import Store

NAMES = [
    "loiter_midnight",
    "repeat_pickup",
    "benign_delivery",
    "intruder_garage",
    "low_battery",
    "flicker_noise",
    "geofence_violation",
    "signal_loss",
    "security_events",
    "normal_day",
]


def scenario(name):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


def with_expect(name, **changes):
    sc = scenario(name)
    return sc.model_copy(update={"expect": sc.expect.model_copy(update=changes)})


def scored(sc):
    return run_scenario_scored(sc)[0]


# ---- the shipped scenarios ---------------------------------------------------------------------
def test_all_ten_scenarios_ship_with_ground_truth():
    assert {p.stem for p in scenario_paths()} == set(NAMES)
    assert len(NAMES) == 10


@pytest.mark.parametrize("name", NAMES)
def test_each_scenario_matches_its_author_written_ground_truth_exactly(name):
    score = scored(scenario(name))
    assert score.exact, score.to_dict()
    assert score.fp == score.fn == score.forbidden_hits == 0


def test_the_geometry_and_timing_worked_out_in_advance_are_reproduced():
    """The geofence and signal-fade numbers were computed by hand from the anomaly parameters."""
    (geo,) = scored(scenario("geofence_violation")).matches
    assert [s.at for s in geo.actual.ladder] == [
        scenario("geofence_violation").start + __import__("datetime").timedelta(seconds=t)
        for t in (310, 390)
    ]
    signal = scored(scenario("signal_loss"))
    assert [[s.severity.value for s in m.actual.ladder] for m in signal.matches] == [
        ["medium"],
        ["medium", "high"],
        ["high"],
    ]


def test_the_control_scenarios_raise_nothing_and_flicker_stays_one_alert():
    for name in ("benign_delivery", "normal_day"):
        with Store.open(":memory:") as store:
            PatrolEngine(scenario(name), store).run()
            assert store.query_alerts() == []
    with Store.open(":memory:") as store:
        PatrolEngine(scenario("flicker_noise"), store).run()
        (alert,) = store.query_alerts()
        assert alert.rule_id == "loitering" and alert.status is AlertStatus.ESCALATED
        assert len(store.query_events(object_type=ObjectType.PERSON)) == 1


# ---- the scorer really discriminates -------------------------------------------------------------
def test_a_wrong_expected_severity_is_reported_as_a_severity_error_not_a_pass():
    sc = scenario("loiter_midnight")
    alerts = list(sc.expect.alerts)
    alerts[0] = alerts[0].model_copy(
        update={"steps": [ExpectedStep(severity=Severity.MEDIUM, at_s=180)]}
    )
    score = scored(
        sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": alerts})})
    )
    assert not score.exact and score.tp == 2  # still found, but on the wrong ladder
    bad = next(m for m in score.matches if m.expected.rule_id == "loitering")
    assert not bad.severity_path_ok and "severity path" in bad.issues[0]


def test_rung_times_outside_the_tolerance_are_flagged_and_inside_are_not():
    sc = scenario("loiter_midnight")

    def shifted(seconds):
        alerts = [
            a.model_copy(
                update={
                    "steps": [
                        ExpectedStep(severity=s.severity, at_s=s.at_s + seconds) for s in a.steps
                    ]
                }
            )
            if a.rule_id == "loitering"
            else a
            for a in sc.expect.alerts
        ]
        return scored(
            sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": alerts})})
        )

    assert shifted(3).exact  # inside the 5 s tolerance
    late = shifted(30)
    loiter = next(m for m in late.matches if m.expected.rule_id == "loitering")
    assert not loiter.steps_timely and "rung times" in loiter.issues[0]


def test_an_alert_expected_but_never_raised_is_a_false_negative():
    sc = scenario("normal_day")
    missing = ExpectedAlert(
        rule_id="loitering", steps=[ExpectedStep(severity=Severity.MEDIUM, at_s=100)]
    )
    score = scored(sc.model_copy(update={"expect": Expectations(alerts=[missing])}))
    assert (score.tp, score.fp, score.fn) == (0, 0, 1) and score.recall == 0.0
    assert score.precision is None  # 0/0 is N/A, never an invented 0 or 1
    assert score.matches[0].issues == ["expected alert was never raised"]


def test_an_alert_no_label_explains_is_a_false_positive_and_a_forbidden_one_is_counted():
    sc = scenario("loiter_midnight")
    trimmed = sc.expect.model_copy(
        update={"alerts": [a for a in sc.expect.alerts if a.rule_id != "after_hours_presence"]}
    )
    score = scored(sc.model_copy(update={"expect": trimmed}))
    assert (score.tp, score.fp, score.fn) == (1, 1, 0) and score.precision == 0.5
    assert [a.rule_id for a in score.unexpected] == ["after_hours_presence"]
    assert score.forbidden_hits == 0
    forbidden = trimmed.model_copy(
        update={"forbidden": [ForbiddenAlert(rule_id="after_hours_presence")]}
    )
    assert scored(sc.model_copy(update={"expect": forbidden})).forbidden_hits == 1
    quiet = trimmed.model_copy(update={"unscored_rules": ["after_hours_presence"]})
    assert scored(sc.model_copy(update={"expect": quiet})).fp == 0  # explicitly unscored


def test_a_wrong_final_status_or_zone_is_caught():
    sc = scenario("loiter_midnight")
    alerts = [
        a.model_copy(update={"final_status": AlertStatus.OPEN}) if a.rule_id == "loitering" else a
        for a in sc.expect.alerts
    ]
    score = scored(
        sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": alerts})})
    )
    loiter = next(m for m in score.matches if m.expected.rule_id == "loitering")
    assert loiter.final_status_ok is False and not score.exact

    elsewhere = [
        a.model_copy(update={"zone": Zone.WAREHOUSE}) if a.rule_id == "loitering" else a
        for a in sc.expect.alerts
    ]
    other = scored(
        sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": elsewhere})})
    )
    assert other.fn == 1 and other.fp == 1  # the alert exists, in the wrong zone: not a match


def test_a_tight_detection_deadline_is_reported_as_late():
    sc = scenario("signal_loss")  # the gap alert is raised 10 s after its condition
    alerts = [a.model_copy(update={"detect_within_s": 5}) for a in sc.expect.alerts]
    score = scored(
        sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": alerts})})
    )
    assert any(m.on_time is False and "after the condition" in m.issues[-1] for m in score.matches)
    assert not score.exact


def test_events_and_entities_that_are_expected_but_absent_are_reported():
    sc = scenario("repeat_pickup")
    events = [
        *sc.expect.events,
        ExpectedEvent(object_type=ObjectType.PERSON, zone=Zone.GARAGE, min_duration_s=10),
    ]
    entities = [*sc.expect.entities, ExpectedEntity(attributes={"color": "green"}, appearances=2)]
    score = scored(
        sc.model_copy(
            update={"expect": sc.expect.model_copy(update={"events": events, "entities": entities})}
        )
    )
    assert score.events_found == 4 and score.events_expected == 5
    assert score.entities_found == 2 and score.entities_expected == 3
    assert "person in garage" in score.events_missing[0] and score.entities_missing


def test_greedy_matching_pairs_each_expected_alert_with_the_nearest_condition_time():
    sc = scenario("signal_loss")
    reversed_labels = sc.expect.model_copy(update={"alerts": list(reversed(sc.expect.alerts))})
    assert scored(sc.model_copy(update={"expect": reversed_labels})).exact  # order is irrelevant


def test_scoring_needs_an_expect_block():
    plain = scenario("normal_day").model_copy(update={"expect": None})
    with Store.open(":memory:") as store, pytest.raises(ValueError, match="no expect"):
        score_store(plain, store)


# ---- aggregation ---------------------------------------------------------------------------------
def test_aggregate_micro_metrics_are_computed_from_pooled_counts():
    perfect = scored(scenario("loiter_midnight"))
    trimmed = scenario("loiter_midnight")
    trimmed = trimmed.model_copy(
        update={
            "expect": trimmed.expect.model_copy(
                update={
                    "alerts": [a for a in trimmed.expect.alerts if a.rule_id == "loitering"],
                }
            )
        }
    )
    extra = scored(trimmed)  # one true positive and one unexplained alert
    agg = aggregate([perfect, extra])
    assert (agg["tp"], agg["fp"], agg["fn"]) == (3, 1, 0)
    assert agg["precision"] == 0.75 and agg["recall"] == 1.0
    assert agg["f1"] == pytest.approx(2 * 0.75 / 1.75, abs=1e-4)
    assert agg["exact_scenarios"] == 1 and agg["per_rule"]["after_hours_presence"]["fp"] == 1


def test_aggregate_of_nothing_reports_na_not_zero():
    agg = aggregate([])
    assert agg["precision"] is None and agg["recall"] is None and agg["f1"] is None
    assert agg["detection_latency_s"]["n"] == 0 and agg["detection_latency_s"]["mean"] is None


def test_percentile_picks_the_ranked_value():
    assert _percentile([], 0.95) is None
    assert _percentile([1.0], 0.95) == 1.0
    assert _percentile([0.0, 0.0, 10.0, 10.0], 0.5) in (0.0, 10.0)
    assert _percentile(list(map(float, range(101))), 0.95) == 95.0


def test_latency_is_measured_from_the_condition_not_from_the_first_frame():
    agg = aggregate([scored(scenario("signal_loss"))])
    lat = agg["detection_latency_s"]
    assert lat["n"] == 3 and lat["max"] == 10.0  # the delivery gap is seen on the next 10 s tick
    assert lat["median"] == 0.0


# ---- expectation model ---------------------------------------------------------------------------
def test_expected_steps_must_rise_in_severity_and_time():
    with pytest.raises(ValidationError, match="rise"):
        ExpectedAlert(
            rule_id="x",
            steps=[
                ExpectedStep(severity=Severity.HIGH, at_s=10),
                ExpectedStep(severity=Severity.MEDIUM, at_s=20),
            ],
        )
    with pytest.raises(ValidationError, match="rise"):
        ExpectedAlert(
            rule_id="x",
            steps=[
                ExpectedStep(severity=Severity.LOW, at_s=30),
                ExpectedStep(severity=Severity.HIGH, at_s=20),
            ],
        )
    with pytest.raises(ValidationError):
        ExpectedEvent(
            object_type=ObjectType.PERSON, zone=Zone.MAIN_GATE, min_duration_s=9, max_duration_s=3
        )
    alert = ExpectedAlert(rule_id="x", steps=[ExpectedStep(severity=Severity.LOW, at_s=5)])
    assert alert.final_severity is Severity.LOW and alert.condition_at_s == 5


# ---- rendering -----------------------------------------------------------------------------------
def test_render_scores_states_plainly_whether_the_run_matches_and_what_is_unexpected():
    from io import StringIO

    out = StringIO()
    render_scores(Console(file=out, width=160), scored(scenario("loiter_midnight")))
    text = out.getvalue()
    assert "MATCHES ground truth" in text and "loitering" in text and "author-written" in text
    sc = scenario("loiter_midnight")
    trimmed = sc.model_copy(
        update={
            "expect": sc.expect.model_copy(
                update={"alerts": [a for a in sc.expect.alerts if a.rule_id == "loitering"]}
            )
        }
    )
    out = StringIO()
    render_scores(Console(file=out, width=160), scored(trimmed))
    assert "DIFFERS from ground truth" in out.getvalue() and "unexpected alert" in out.getvalue()
