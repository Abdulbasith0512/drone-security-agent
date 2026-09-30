"""T05: the evaluation's precision / recall / F1 / exactness must be mathematically right and must
say what they measure.

Three strengths of "matched" are kept apart and tested separately:

* SUBSET (`label_satisfied`): everything the labels ask for happened; surplus events and entities
  are allowed. Alerts are never subset: an unlabelled alert is a false positive.
* EXHAUSTIVE (`Expectations.exhaustive`): the event and entity labels are declared complete, so a
  surplus one fails the scenario.
* EXACT (`exact`): nothing at all beyond the labels was produced (set equality).
"""

from datetime import datetime, timedelta

import pytest
from conftest import IST, SCENARIO_DIR

from drone_agent.eval import rules_exp
from drone_agent.eval.base import Confusion, count_confusion, f1_score, ratio
from drone_agent.eval.ground_truth import (
    ScenarioScore,
    _match_alerts,
    _match_events,
    _max_matching,
    aggregate,
    run_scenario_scored,
)
from drone_agent.expectations import (
    Expectations,
    ExpectedAlert,
    ExpectedEvent,
    ExpectedStep,
)
from drone_agent.models import (
    Alert,
    CloseReason,
    Event,
    LadderStep,
    ObjectType,
    Severity,
    Zone,
)
from drone_agent.scenario import load_scenario


def scenario(name):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


def with_expect(name, **changes):
    sc = scenario(name)
    return sc.model_copy(update={"expect": sc.expect.model_copy(update=changes)})


def scored(sc):
    return run_scenario_scored(sc)[0]


def label(rule, at_s=100, zone=None):
    return ExpectedAlert(
        rule_id=rule, zone=zone, steps=[ExpectedStep(severity=Severity.MEDIUM, at_s=at_s)]
    )


# ---- the formula -----------------------------------------------------------------------------
# (name, tp, fp, fn, precision, recall, f1)
CASES = [
    ("perfect match", 3, 0, 0, 1.0, 1.0, 1.0),
    ("one false positive", 3, 1, 0, 0.75, 1.0, 6 / 7),
    ("one false negative", 3, 0, 1, 1.0, 0.75, 6 / 7),
    ("false positive and false negative", 3, 1, 1, 0.75, 0.75, 0.75),
    ("expected nothing, got nothing", 0, 0, 0, None, None, None),
    ("expected some, got nothing", 0, 0, 3, None, 0.0, 0.0),
    ("expected nothing, got some", 0, 3, 0, 0.0, None, 0.0),
    ("everything wrong", 0, 2, 2, 0.0, 0.0, 0.0),
]


@pytest.mark.parametrize(
    ("name", "tp", "fp", "fn", "p", "r", "f1"), CASES, ids=[c[0] for c in CASES]
)
def test_precision_recall_and_f1_for_every_edge_case(name, tp, fp, fn, p, r, f1):
    c = Confusion(tp=tp, fp=fp, fn=fn)
    assert c.precision == p and c.recall == r, name
    assert (f1_score(tp, fp, fn) is None) == (f1 is None), name
    if f1 is not None:
        assert f1_score(tp, fp, fn) == pytest.approx(f1)
        assert c.f1 == pytest.approx(f1)
    score = ScenarioScore("x", tp=tp, fp=fp, fn=fn)
    assert (score.precision, score.recall) == (p, r)
    assert (score.f1 is None) == (f1 is None)


def test_f1_is_2tp_over_2tp_plus_fp_plus_fn_and_equals_the_harmonic_mean():
    for tp in range(0, 6):
        for fp in range(0, 6):
            for fn in range(0, 6):
                value = f1_score(tp, fp, fn)
                if 2 * tp + fp + fn == 0:
                    assert value is None
                    continue
                assert value == pytest.approx(2 * tp / (2 * tp + fp + fn))
                p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
                if p is not None and r is not None and p + r > 0:  # harmonic mean where defined
                    assert value == pytest.approx(2 * p * r / (p + r))
                else:
                    assert tp == 0  # the only way precision and recall cannot give it is TP = 0
                    assert value == 0.0 if fp + fn else value is None


def test_a_confusion_cell_counts_duplicates_as_false_positives_and_keeps_true_negatives_apart():
    assert count_confusion(1, 3) == Confusion(tp=1, fp=2)  # duplicates of one expected alert
    assert count_confusion(2, 1) == Confusion(tp=1, fn=1)
    assert count_confusion(0, 2) == Confusion(fp=2)
    assert count_confusion(3, 0) == Confusion(fn=3)
    empty = count_confusion(0, 0)
    assert empty == Confusion(tn=1) and empty.f1 is None  # a true negative is not a true positive
    assert (Confusion(tp=1, fp=2) + Confusion(tn=1, fn=1)) == Confusion(1, 2, 1, 1)


def test_the_aggregate_pools_counts_instead_of_averaging_per_scenario_scores():
    perfect = ScenarioScore("a", tp=1)  # F1 1.0
    bad = ScenarioScore("b", tp=0, fp=1, fn=1)  # F1 0.0
    agg = aggregate([perfect, bad])
    assert (agg["tp"], agg["fp"], agg["fn"]) == (1, 1, 1)
    assert agg["f1"] == 0.5  # 2 / (2 + 1 + 1), not the mean of 1.0 and 0.0 (also 0.5) ...
    big = ScenarioScore("c", tp=9, fp=0, fn=0)
    assert aggregate([big, bad])["f1"] == pytest.approx(2 * 9 / (2 * 9 + 1 + 1))  # ... but this is
    assert aggregate([big, bad])["f1"] != pytest.approx((1.0 + 0.0) / 2)
    assert aggregate([])["f1"] is None and aggregate([])["precision"] is None


# ---- the scorer on real runs: alerts ----------------------------------------------------------
def test_perfect_match():
    score = scored(scenario("security_events"))
    assert (score.tp, score.fp, score.fn) == (7, 0, 0) and score.f1 == 1.0
    assert score.label_satisfied and score.exact
    assert all(m.fully_correct for m in score.matches)


def test_one_false_positive_an_alert_no_label_explains():
    sc = scenario("loiter_midnight")
    kept = [a for a in sc.expect.alerts if a.rule_id != "after_hours_presence"]
    score = scored(sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": kept})}))
    assert (score.tp, score.fp, score.fn) == (1, 1, 0)
    assert score.precision == 0.5 and score.recall == 1.0 and score.f1 == pytest.approx(2 / 3)
    assert not score.label_satisfied and not score.exact


def test_one_false_negative_an_expected_alert_never_raised():
    sc = scenario("normal_day")
    phantom = Expectations(alerts=[label("loitering")])
    score = scored(sc.model_copy(update={"expect": phantom}))
    assert (score.tp, score.fp, score.fn) == (0, 0, 1)
    assert score.recall == 0.0 and score.precision is None and score.f1 == 0.0


def test_false_positive_and_false_negative_together():
    sc = scenario("loiter_midnight")
    swapped = [a for a in sc.expect.alerts if a.rule_id != "after_hours_presence"]
    swapped.append(label("geofence_violation"))  # never raised; after-hours is now unexplained
    score = scored(
        sc.model_copy(update={"expect": sc.expect.model_copy(update={"alerts": swapped})})
    )
    assert (score.tp, score.fp, score.fn) == (1, 1, 1) and score.f1 == pytest.approx(2 / 4)


def test_expected_nothing_and_got_nothing_is_undefined_not_perfect_and_not_zero():
    score = scored(scenario("normal_day").model_copy(update={"expect": Expectations()}))
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    assert score.precision is None and score.recall is None and score.f1 is None
    assert score.label_satisfied  # nothing was asked for and no alert was raised


def test_expected_nothing_but_alerts_were_raised():
    sc = scenario("security_events")
    score = scored(sc.model_copy(update={"expect": Expectations()}))
    assert (score.tp, score.fp, score.fn) == (0, 7, 0)
    assert score.precision == 0.0 and score.recall is None and score.f1 == 0.0
    assert not score.label_satisfied


def test_duplicate_alerts_for_one_label_are_false_positives_not_extra_true_positives():
    sc = scenario("security_events")  # three after-hours alerts; label only one of them
    everything_else = sorted({a.rule_id for a in sc.expect.alerts} - {"after_hours_presence"})
    expect = sc.expect.model_copy(
        update={
            "alerts": [label("after_hours_presence", at_s=60)],
            "unscored_rules": everything_else + ["signal_loss"],
        }
    )
    score = scored(sc.model_copy(update={"expect": expect}))
    assert (score.tp, score.fp, score.fn) == (1, 2, 0)  # one-to-one: the other two are duplicates


# ---- subset, exhaustive and exact ---------------------------------------------------------------
def test_a_subset_match_is_not_reported_as_exact():
    score = scored(scenario("normal_day"))  # labels 1 of its 5 events and none of its 2 entities
    assert score.label_satisfied and not score.exact
    assert (score.events_extra, score.entities_extra) == (4, 2)
    assert (score.events_found, score.events_expected) == (1, 1)


def test_dropping_a_label_from_a_complete_fixture_turns_exact_into_subset():
    sc = scenario("security_events")
    assert scored(sc).exact  # 7 labelled events of 7, 2 entities of 2
    fewer = sc.expect.model_copy(update={"events": sc.expect.events[:-1]})
    score = scored(sc.model_copy(update={"expect": fewer}))
    assert score.label_satisfied and not score.exact and score.events_extra == 1


def test_a_declared_exhaustive_fixture_rejects_surplus_output():
    sc = scenario("security_events")
    fewer = sc.expect.model_copy(update={"events": sc.expect.events[:-1], "exhaustive": True})
    score = scored(sc.model_copy(update={"expect": fewer}))
    assert score.events_extra == 1
    assert not score.label_satisfied and not score.exact  # the surplus event is an error here
    complete = sc.expect.model_copy(update={"exhaustive": True})  # labels really are complete
    ok = scored(sc.model_copy(update={"expect": complete}))
    assert ok.label_satisfied and ok.exact and ok.exhaustive
    # the same surplus under subset labels is tolerated (and still not exact)
    subset = scored(sc.model_copy(update={"expect": sc.expect.model_copy(update={"events": []})}))
    assert subset.label_satisfied and not subset.exact and subset.events_extra == 7


def test_the_aggregate_keeps_the_three_counts_apart():
    scores = [scored(scenario(n)) for n in ("security_events", "normal_day", "low_battery")]
    agg = aggregate(scores)
    assert (agg["scenarios"], agg["label_satisfied_scenarios"], agg["exact_scenarios"]) == (3, 3, 2)
    assert (agg["events_extra"], agg["entities_extra"], agg["exhaustive_scenarios"]) == (4, 2, 0)


# ---- matching must not miss what some assignment could match -------------------------------------
def event(seconds: int, eid: str) -> Event:
    start = datetime(2026, 9, 29, 12, 0, tzinfo=IST)
    return Event(
        event_id=eid,
        object_type=ObjectType.PERSON,
        zone=Zone.MAIN_GATE,
        first_seen=start,
        last_seen=start + timedelta(seconds=seconds),
        observation_ids=["o1"],
        frame_ids=[1],
        close_reason=CloseReason.SCENARIO_END,
    )


def test_event_matching_is_optimal_not_first_come_first_served():
    anything = ExpectedEvent(object_type=ObjectType.PERSON, zone=Zone.MAIN_GATE)
    long_one = ExpectedEvent(object_type=ObjectType.PERSON, zone=Zone.MAIN_GATE, min_duration_s=400)
    events = [event(500, "EVT-P-001"), event(10, "EVT-P-002")]  # the long event comes first
    # greedy: `anything` takes EVT-P-001 and `long_one` is wrongly reported missing
    found, missing = _match_events([anything, long_one], events)
    assert found == 2 and missing == []


def test_alert_matching_is_optimal_not_closest_first():
    sc = scenario("loiter_midnight")
    at = lambda s: sc.start + timedelta(seconds=s)  # noqa: E731

    def alert(aid, zone, offset):
        return Alert(
            alert_id=aid,
            rule_id="loitering",
            severity=Severity.MEDIUM,
            title="t",
            message="m",
            triggered_at=at(offset),
            first_seen=at(offset),
            last_seen=at(offset),
            zone=zone,
            event_ids=["EVT-P-001"],
            frame_ids=[1],
            ladder=[LadderStep(at=at(offset), severity=Severity.MEDIUM, label="warning")],
        )

    a_gate, a_drive = alert("A1", Zone.MAIN_GATE, 100), alert("A2", Zone.DRIVEWAY, 150)
    anywhere = label("loitering", at_s=100)  # closest to the gate alert, but could take either
    gate_only = label("loitering", at_s=105, zone=Zone.MAIN_GATE)  # can ONLY take the gate alert
    matches, leftovers = _match_alerts(sc, [anywhere, gate_only], [a_gate, a_drive])
    assert [m.found for m in matches] == [True, True] and leftovers == []
    assert matches[1].actual.alert_id == "A1" and matches[0].actual.alert_id == "A2"


def test_max_matching_prefers_the_listed_order_when_nothing_conflicts():
    assert _max_matching([[0, 1], [1, 0]], [0, 1]) == {0: 0, 1: 1}
    assert _max_matching([[0], [0]], [0, 1]) == {0: 0}  # one item cannot satisfy two labels
    assert _max_matching([[], [0]], [0, 1]) == {1: 0}


# ---- what "22 of 22" in the rules matrix means ---------------------------------------------------
def test_the_rules_matrix_compares_counts_per_rule_and_reports_f1():
    matrix = rules_exp.build_matrix()
    assert len(matrix) == 22
    results = [rules_exp.evaluate_scenario(sc) for sc in matrix]
    assert all(r["counts_match"] for r in results)  # the "22 of 22"
    per_rule = rules_exp.run().measured["per_rule"]
    for rule, cell in per_rule.items():
        assert cell["f1"] == pytest.approx(
            2 * cell["tp"] / (2 * cell["tp"] + cell["fp"] + cell["fn"])
        )
        assert set(cell) >= {"tp", "fp", "tn", "fn", "precision", "recall", "f1"}, rule


def test_a_count_match_does_not_see_which_event_raised_the_alert_and_the_docs_say_so():
    text = rules_exp.__doc__
    assert "NUMBER of alerts" in text and "WHICH event" in text and "severity" in text
    assert "drone-safety rules cannot fire" in text


def fake_pipeline(monkeypatch, alerts):
    monkeypatch.setattr(rules_exp, "run_pipeline", lambda observations: (None, None, alerts))


def matrix_alert(rule):
    sc = scenario("loiter_midnight")
    return Alert(
        alert_id=f"ALR-{rule}-1",
        rule_id=rule,
        severity=Severity.MEDIUM,
        title="t",
        message="m",
        triggered_at=sc.start,
        first_seen=sc.start,
        last_seen=sc.start,
        zone=Zone.MAIN_GATE,
        event_ids=["EVT-P-001"],
        frame_ids=[1],
    )


def test_a_surplus_alert_in_one_cell_cannot_cancel_a_missing_alert_in_another(monkeypatch):
    sc = next(s for s in rules_exp.build_matrix() if s.name == "loiter_exact_threshold")
    fake_pipeline(monkeypatch, [matrix_alert("after_hours_presence")])  # wrong rule raised it
    result = rules_exp.evaluate_scenario(sc)
    assert result["expected"]["loitering"] == 1 and result["actual"]["loitering"] == 0
    assert result["actual"]["after_hours_presence"] == 1
    assert not result["counts_match"]  # total counts agree (1 and 1); the cells do not
    assert (
        result["cells"]["loitering"]["fn"] == 1
        and result["cells"]["after_hours_presence"]["fp"] == 1
    )


def test_an_alert_from_a_rule_the_matrix_does_not_list_is_not_ignored(monkeypatch):
    sc = next(s for s in rules_exp.build_matrix() if s.name == "loiter_below_threshold")
    fake_pipeline(monkeypatch, [matrix_alert("geofence_violation")])
    result = rules_exp.evaluate_scenario(sc)
    assert result["unlisted_rule_alerts"] == ["geofence_violation"] and not result["counts_match"]


# ---- the citation / factual evaluation labels ----------------------------------------
def test_the_citation_suite_accuracy_is_recomputed_from_its_own_rows():
    from drone_agent.eval import grounding_exp

    exp = grounding_exp.run_adversarial()
    rows = exp.rows
    correct = sum(1 for r in rows if r["correct"])
    assert exp.measured["accuracy_on_this_fixture_set"] == pytest.approx(correct / len(rows))
    false_accepts = [r for r in rows if r["expected"] == "reject" and r["actual"] == "accept"]
    false_rejects = [r for r in rows if r["expected"] == "accept" and r["actual"] == "reject"]
    assert len(false_accepts) + len(false_rejects) == len(rows) - correct


def test_a_probe_case_with_no_structured_claims_is_not_rejected_by_the_checker():
    """The hand-structured checker is vacuously satisfied by an empty claim list; the experiment
    must therefore decide such cases by the gate alone, and count them as 'not expressible'."""
    from drone_agent.eval import claims as cl
    from drone_agent.eval import grounding_exp

    assert cl.all_supported([]) is True  # vacuous: a caller must not read this as "verified"
    m = grounding_exp.run_semantic_probe().measured
    checker = m["checker"]
    inexpressible = checker["inexpressible_unsupported_cases"]
    assert checker["unsupported_accepted_gate_plus_checker"] == inexpressible  # the checker adds
    assert (
        checker["expressible_caught"] == checker["expressible_unsupported_cases"]
    )  # nothing there
    assert m["production_gate"]["unsupported_accepted"] <= m["unsupported_accepted_by_gate"]
