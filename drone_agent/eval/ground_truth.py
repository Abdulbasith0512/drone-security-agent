"""Ground-truth evaluation: score a run against the scenario's `expect:` block.

For every scenario the author wrote down (before scoring) which events must exist, which alerts
must be raised with which severity ladder and when their condition first holds, and which alerts
must not appear. A run is scored by matching what the system produced to that truth:

* an expected alert that was produced is a TRUE POSITIVE; one that was not is a FALSE NEGATIVE;
  a produced alert that no label explains (or a forbidden one) is a FALSE POSITIVE;
* precision, recall and F1 follow from those counts (F1 = 2TP / (2TP + FP + FN); N/A only when
  a denominator is zero). They measure PRESENCE: a matched alert is a true positive even if its
  ladder, timing or status is wrong; those are scored separately (`fully_correct`);
* labels come in two strengths. Alerts are exhaustive: an unlabelled alert is a false positive.
  Events and entities are SUBSET labels unless the scenario sets `exhaustive`: they must exist,
  and surplus ones are counted (`events_extra`, `entities_extra`) but are not an error. A
  scenario is `label_satisfied` when everything labelled happened and no alert was unexplained
  (and, if declared exhaustive, nothing else was produced); it is `exact` only when nothing at
  all beyond the labels was produced, so a subset match with surplus output is not exact;
* for matched alerts we also score the severity ladder (right rungs?), the rung timing (within
  `step_tolerance_s` of the expected condition time?), the final lifecycle status, and the
  alert-DETECTION LATENCY: simulated seconds between the moment the condition became true and the
  moment the system raised the alert.

Matching is one-to-one and maximum-cardinality (it never leaves an expected item unmatched when
some assignment could match it), preferring the smallest time difference between the alert's
condition time and the expected one, among alerts of the same rule (and zone, when the label
names one). A duplicate alert for one label is a false positive.

What this is and is not: the labels were written by the same person who wrote the rules, and the
scripted detections are perfect. Agreement therefore checks that the implementation matches the
author's intent (thresholds, ladders, timing, lifecycle); it does not estimate real-world
performance. The noise experiment (eval/noise_exp.py) is what shows the scores moving.
"""

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from drone_agent.agents.perception import PerceptionAgent
from drone_agent.eval.base import CONTROLLED_QUALIFICATION, ExperimentResult, f1_score, fmt, ratio
from drone_agent.expectations import ExpectedAlert, ExpectedEntity, ExpectedEvent
from drone_agent.models import Alert, Entity, Event
from drone_agent.pipeline.config import NormalizationSettings
from drone_agent.pipeline.normalize import normalize_attribute
from drone_agent.runtime import EngineOptions, PatrolEngine, RunSummary
from drone_agent.scenario import Scenario, load_scenario
from drone_agent.store import Store

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"
_NORMALIZATION = NormalizationSettings()


# ---- result types -----------------------------------------------------------------------------
@dataclass
class AlertMatch:
    expected: ExpectedAlert
    actual: Alert | None
    severity_path_ok: bool = False
    steps_timely: bool = False
    final_status_ok: bool | None = None
    latency_s: float | None = None  # raise latency: condition true -> alert raised
    on_time: bool | None = None
    escalation_latencies_s: list[float] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.actual is not None

    @property
    def fully_correct(self) -> bool:
        """Found AND right rung severities, rung times, detection deadline and final status."""
        return (
            self.found
            and self.severity_path_ok
            and self.steps_timely
            and bool(self.on_time)
            and self.final_status_ok is not False
        )

    def to_dict(self) -> dict:
        return {
            "rule_id": self.expected.rule_id,
            "zone": self.expected.zone.value if self.expected.zone else None,
            "expected_path": [s.severity.value for s in self.expected.steps],
            "expected_condition_s": self.expected.condition_at_s,
            "found": self.found,
            "alert_id": self.actual.alert_id if self.actual else None,
            "actual_path": (
                [s.severity.value for s in self.actual.ladder] if self.actual else None
            ),
            "severity_path_ok": self.severity_path_ok,
            "steps_timely": self.steps_timely,
            "final_status_ok": self.final_status_ok,
            "fully_correct": self.fully_correct,
            "latency_s": self.latency_s,
            "on_time": self.on_time,
            "escalation_latencies_s": self.escalation_latencies_s,
            "issues": self.issues,
        }


@dataclass
class ScenarioScore:
    scenario: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    forbidden_hits: int = 0
    matches: list[AlertMatch] = field(default_factory=list)
    unexpected: list[Alert] = field(default_factory=list)
    events_expected: int = 0
    events_found: int = 0
    events_missing: list[str] = field(default_factory=list)
    entities_expected: int = 0
    entities_found: int = 0
    entities_missing: list[str] = field(default_factory=list)
    # what the run produced beyond the labelled events / entities (see the module docstring)
    events_extra: int = 0
    entities_extra: int = 0
    exhaustive: bool = False  # the scenario declares its event/entity labels complete

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        return f1_score(self.tp, self.fp, self.fn)

    @property
    def label_satisfied(self) -> bool:
        """Everything the labels ask for happened and no alert was unexplained.

        Events and entities are checked as a SUBSET (surplus ones do not matter) unless the
        scenario declares them exhaustive. This used to be reported as "exact".
        """
        return (
            self.fp == 0
            and self.fn == 0
            and self.forbidden_hits == 0
            and all(m.fully_correct for m in self.matches)
            and self.events_found == self.events_expected
            and self.entities_found == self.entities_expected
            and not (self.exhaustive and self.surplus)
        )

    @property
    def surplus(self) -> int:
        """Events and entities the run produced that no label accounts for."""
        return self.events_extra + self.entities_extra

    @property
    def exact(self) -> bool:
        """Label-satisfied AND nothing beyond the labels was produced: set equality for events,
        entities and alerts. A subset match with surplus events or entities is NOT exact."""
        return self.label_satisfied and self.surplus == 0

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "expected_alerts": len(self.matches),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": _round(self.precision),
            "recall": _round(self.recall),
            "f1": _round(self.f1),
            "forbidden_hits": self.forbidden_hits,
            "events": f"{self.events_found}/{self.events_expected}",
            "entities": f"{self.entities_found}/{self.entities_expected}",
            "events_extra": self.events_extra,
            "entities_extra": self.entities_extra,
            "exhaustive": self.exhaustive,
            "label_satisfied": self.label_satisfied,
            "exact": self.exact,
            "matches": [m.to_dict() for m in self.matches],
            "unexpected_alerts": [a.alert_id for a in self.unexpected],
            "events_missing": self.events_missing,
            "entities_missing": self.entities_missing,
        }


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


# ---- matching ---------------------------------------------------------------------------------
def _offset(scenario: Scenario, ts) -> float:
    return (ts - scenario.start).total_seconds()


def _max_matching(candidates: list[list[int]], order: list[int]) -> dict[int, int]:
    """A maximum one-to-one assignment of labels to produced items (augmenting paths).

    `candidates[i]` lists the produced items label `i` may take, best first; labels are seated in
    `order`. The preferred pair wins unless that would leave another label with no match at all,
    so a label is never counted as missed while some assignment could have matched it (a plain
    greedy pass can do exactly that when labels overlap). Returns label index -> item index.
    """
    owner: dict[int, int] = {}  # item -> label currently holding it

    def seat(label: int, seen: set[int]) -> bool:
        for item in candidates[label]:
            if item in seen:
                continue
            seen.add(item)
            if item not in owner or seat(owner[item], seen):
                owner[item] = label
                return True
        return False

    for label in order:
        seat(label, set())
    return {label: item for item, label in owner.items()}


def _match_alerts(
    scenario: Scenario, expected: list[ExpectedAlert], actual: list[Alert]
) -> tuple[list[AlertMatch], list[Alert]]:
    """One-to-one assignment, preferring the smallest condition-time difference within a rule.

    A second alert for the same label is left over, and so a false positive."""
    candidates: list[list[int]] = []
    closest: list[float] = []
    for exp in expected:
        options = sorted(
            (abs(_offset(scenario, alert.triggered_at) - exp.condition_at_s), j)
            for j, alert in enumerate(actual)
            if alert.rule_id == exp.rule_id and (exp.zone is None or alert.zone is exp.zone)
        )
        candidates.append([j for _, j in options])
        closest.append(options[0][0] if options else float("inf"))
    order = sorted(range(len(expected)), key=lambda i: (closest[i], i))
    chosen = _max_matching(candidates, order)
    used = set(chosen.values())
    matches = [
        _score_pair(scenario, exp, actual[chosen[i]] if i in chosen else None)
        for i, exp in enumerate(expected)
    ]
    leftovers = [a for j, a in enumerate(actual) if j not in used]
    return matches, leftovers


def _score_pair(scenario: Scenario, exp: ExpectedAlert, alert: Alert | None) -> AlertMatch:
    match = AlertMatch(exp, alert)
    if alert is None:
        match.issues.append("expected alert was never raised")
        return match
    tolerance = scenario.expect.step_tolerance_s  # type: ignore[union-attr]
    path = [s.severity for s in alert.ladder] or [alert.severity]
    want = [s.severity for s in exp.steps]
    match.severity_path_ok = path == want
    if not match.severity_path_ok:
        match.issues.append(
            f"severity path {[s.value for s in path]} instead of {[s.value for s in want]}"
        )
    if match.severity_path_ok:
        offsets = [_offset(scenario, s.at) for s in alert.ladder]
        late = [
            (round(got, 1), s.at_s)
            for got, s in zip(offsets, exp.steps, strict=True)
            if abs(got - s.at_s) > tolerance
        ]
        match.steps_timely = not late
        if late:
            match.issues.append(f"rung times {late} differ from expected by more than {tolerance}s")
    if exp.final_status is not None:
        match.final_status_ok = alert.status is exp.final_status
        if not match.final_status_ok:
            match.issues.append(f"final status {alert.status.value}, expected {exp.final_status}")
    raised = [h for h in alert.history if h.kind.value == "raised"]
    escalated = [h for h in alert.history if h.kind.value == "escalated"]
    if raised:
        match.latency_s = max(0.0, (raised[0].at - raised[0].effective_at).total_seconds())
        match.on_time = match.latency_s <= exp.detect_within_s
        if not match.on_time:
            match.issues.append(
                f"raised {match.latency_s:.0f}s after the condition (allowed {exp.detect_within_s}s)"
            )
    match.escalation_latencies_s = [
        max(0.0, (h.at - h.effective_at).total_seconds()) for h in escalated
    ]
    return match


def _attributes_match(actual: dict[str, str], wanted: dict[str, str]) -> bool:
    return all(actual.get(k, "").strip().lower() == v.strip().lower() for k, v in wanted.items())


def _match_events(expected: list[ExpectedEvent], events: list[Event]) -> tuple[int, list[str]]:
    """Labels matched one-to-one (maximum matching) to events; returns (matched, missing labels)."""
    candidates = [
        [
            j
            for j, e in enumerate(events)
            if e.object_type is exp.object_type
            and e.zone is exp.zone
            and _attributes_match(e.attributes, exp.attributes)
            and e.duration_seconds >= exp.min_duration_s - 1e-9
            and (exp.max_duration_s is None or e.duration_seconds <= exp.max_duration_s + 1e-9)
        ]
        for exp in expected
    ]
    chosen = _max_matching(candidates, list(range(len(expected))))
    missing: list[str] = []
    for i, exp in enumerate(expected):
        if i not in chosen:
            label = f"{exp.object_type.value} in {exp.zone.value}"
            if exp.attributes:
                label += f" {exp.attributes}"
            missing.append(f"{label}, {exp.min_duration_s:g}-{exp.max_duration_s or 'any'}s")
    return len(chosen), missing


def _match_entities(
    expected: list[ExpectedEntity], entities: list[Entity]
) -> tuple[int, list[str]]:
    """Labels matched one-to-one (maximum matching) to entities; (matched, missing labels)."""
    candidates = []
    for exp in expected:
        wanted = {k: normalize_attribute(k, v, _NORMALIZATION) for k, v in exp.attributes.items()}
        candidates.append(
            [
                j
                for j, e in enumerate(entities)
                if all(e.attributes.get(k) == v for k, v in wanted.items())
                and e.appearance_count == exp.appearances
            ]
        )
    chosen = _max_matching(candidates, list(range(len(expected))))
    missing = [
        f"{exp.attributes} x{exp.appearances}" for i, exp in enumerate(expected) if i not in chosen
    ]
    return len(chosen), missing


def score_store(scenario: Scenario, store: Store) -> ScenarioScore:
    """Score the alerts/events/entities in `store` against the scenario's expectations."""
    if scenario.expect is None:
        raise ValueError(f"scenario {scenario.name} has no expect: block to score against")
    expect = scenario.expect
    actual = store.query_alerts()
    matches, leftovers = _match_alerts(scenario, expect.alerts, actual)
    score = ScenarioScore(scenario.name, matches=matches)
    score.tp = sum(1 for m in matches if m.found)
    score.fn = sum(1 for m in matches if not m.found)
    for alert in leftovers:
        if alert.rule_id in expect.unscored_rules:
            continue
        score.fp += 1
        score.unexpected.append(alert)
        if any(
            f.rule_id == alert.rule_id and (f.zone is None or f.zone is alert.zone)
            for f in expect.forbidden
        ):
            score.forbidden_hits += 1
    events, entities = store.query_events(), store.list_entities()
    score.events_expected = len(expect.events)
    score.events_found, score.events_missing = _match_events(expect.events, events)
    score.entities_expected = len(expect.entities)
    score.entities_found, score.entities_missing = _match_entities(expect.entities, entities)
    score.events_extra = len(events) - score.events_found  # matching is one-to-one
    score.entities_extra = len(entities) - score.entities_found
    score.exhaustive = expect.exhaustive
    return score


# ---- running scenarios -------------------------------------------------------------------------
def run_scenario_scored(
    scenario: Scenario,
    *,
    options: EngineOptions | None = None,
    perception: PerceptionAgent | None = None,
) -> tuple[ScenarioScore, RunSummary]:
    """Run one scenario headless (no pacing, in memory) and score it."""
    with Store.open(":memory:") as store:
        summary = PatrolEngine(scenario, store, options=options, perception=perception).run()
        return score_store(scenario, store), summary


def scenario_paths() -> list[Path]:
    """Every shipped scenario that carries ground truth."""
    paths = []
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        if load_scenario(path).expect is not None:
            paths.append(path)
    return paths


# ---- aggregation --------------------------------------------------------------------------------
def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def aggregate(scores: list[ScenarioScore]) -> dict:
    tp, fp, fn = (sum(getattr(s, k) for s in scores) for k in ("tp", "fp", "fn"))
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = f1_score(tp, fp, fn)
    found = [m for s in scores for m in s.matches if m.found]
    latencies = [m.latency_s for m in found if m.latency_s is not None]
    escalations = [x for m in found for x in m.escalation_latencies_s]
    per_rule: dict[str, dict] = {}
    for s in scores:
        for m in s.matches:
            row = per_rule.setdefault(m.expected.rule_id, {"tp": 0, "fp": 0, "fn": 0})
            row["tp" if m.found else "fn"] += 1
        for a in s.unexpected:
            per_rule.setdefault(a.rule_id, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
    for row in per_rule.values():
        row["precision"] = _round(ratio(row["tp"], row["tp"] + row["fp"]))
        row["recall"] = _round(ratio(row["tp"], row["tp"] + row["fn"]))
    events_expected = sum(s.events_expected for s in scores)
    entities_expected = sum(s.entities_expected for s in scores)
    return {
        "scenarios": len(scores),
        "label_satisfied_scenarios": sum(1 for s in scores if s.label_satisfied),
        "exact_scenarios": sum(1 for s in scores if s.exact),
        "exhaustive_scenarios": sum(1 for s in scores if s.exhaustive),
        "events_extra": sum(s.events_extra for s in scores),
        "entities_extra": sum(s.entities_extra for s in scores),
        "alerts_expected": tp + fn,
        "alerts_fully_correct": sum(1 for m in found if m.fully_correct),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "forbidden_hits": sum(s.forbidden_hits for s in scores),
        "severity_path_accuracy": _round(ratio(sum(m.severity_path_ok for m in found), len(found))),
        "step_timing_accuracy": _round(ratio(sum(m.steps_timely for m in found), len(found))),
        "final_status_accuracy": _round(
            ratio(
                sum(1 for m in found if m.final_status_ok),
                sum(1 for m in found if m.final_status_ok is not None),
            )
        ),
        "on_time_rate": _round(ratio(sum(1 for m in found if m.on_time), len(found))),
        "events_recall": _round(ratio(sum(s.events_found for s in scores), events_expected)),
        "entities_recall": _round(ratio(sum(s.entities_found for s in scores), entities_expected)),
        "detection_latency_s": {
            "n": len(latencies),
            "mean": _round(statistics.fmean(latencies), 2) if latencies else None,
            "median": _round(statistics.median(latencies), 2) if latencies else None,
            "p95": _round(_percentile(latencies, 0.95), 2),
            "max": _round(max(latencies), 2) if latencies else None,
        },
        "escalation_latency_s": {
            "n": len(escalations),
            "mean": _round(statistics.fmean(escalations), 2) if escalations else None,
            "max": _round(max(escalations), 2) if escalations else None,
        },
        "per_rule": dict(sorted(per_rule.items())),
    }


def run() -> ExperimentResult:
    """Every shipped scenario with ground truth, run headless and scored."""
    scores: list[ScenarioScore] = []
    calls: dict[str, dict] = {}
    for path in scenario_paths():
        scenario = load_scenario(path)
        score, summary = run_scenario_scored(scenario)
        scores.append(score)
        calls[scenario.name] = {
            "frames": summary.frames,
            "model_calls": summary.model_calls,
            "avoided_pct": summary.avoided_pct,
        }
    agg = aggregate(scores)
    rows = [
        {
            "experiment": "ground_truth",
            "scenario": s.scenario,
            "expected_alerts": len(s.matches),
            "tp": s.tp,
            "fp": s.fp,
            "fn": s.fn,
            "precision": _round(s.precision),
            "recall": _round(s.recall),
            "f1": _round(s.f1),
            "forbidden_hits": s.forbidden_hits,
            "events": f"{s.events_found}/{s.events_expected}",
            "events_extra": s.events_extra,
            "entities_extra": s.entities_extra,
            "label_satisfied": s.label_satisfied,
            "exact": s.exact,
        }
        for s in scores
    ]
    lat = agg["detection_latency_s"]
    interpretation = [
        f"{agg['label_satisfied_scenarios']} of {agg['scenarios']} scenarios satisfied their "
        f"ground-truth labels, and {agg['exact_scenarios']} of {agg['scenarios']} matched them "
        f"exactly (nothing produced beyond the labels; across all scenarios "
        f"{agg['events_extra']} event(s) and {agg['entities_extra']} entity(ies) were not "
        f"labelled, and {agg['exhaustive_scenarios']} scenario(s) declare their event and entity "
        f"labels exhaustive). Alert presence, micro-averaged over {agg['tp'] + agg['fn']} "
        f"expected alerts: precision {fmt(agg['precision'])}, recall {fmt(agg['recall'])}, "
        f"F1 {fmt(agg['f1'])} ({agg['tp']} TP, {agg['fp']} FP, {agg['fn']} FN; F1 = 2TP / "
        f"(2TP + FP + FN)); forbidden alerts raised: {agg['forbidden_hits']}. "
        f"{agg['alerts_fully_correct']} of {agg['alerts_expected']} expected alerts were also "
        "fully correct (ladder, rung timing, detection deadline, final status).",
        f"Of the matched alerts, the severity ladder was right in "
        f"{fmt(agg['severity_path_accuracy'])} and every rung landed within the tolerance in "
        f"{fmt(agg['step_timing_accuracy'])}; the final lifecycle status was right in "
        f"{fmt(agg['final_status_accuracy'])}.",
        f"Alert-detection latency (simulated seconds from the condition becoming true to the alert "
        f"being raised): median {fmt(lat['median'], 1)} s, mean {fmt(lat['mean'], 1)} s, "
        f"p95 {fmt(lat['p95'], 1)} s, max {fmt(lat['max'], 1)} s over {lat['n']} alerts. It is "
        "bounded by frame spacing (60 s) for camera alerts and by the 10 s telemetry tick for "
        "safety alerts.",
        f"Events recalled {fmt(agg['events_recall'])}, vehicle entities {fmt(agg['entities_recall'])}.",
    ]
    return ExperimentResult(
        key="ground_truth",
        title="Ground-truth scenarios: precision, recall, F1, severity and detection latency",
        research_question="RQ8: does a streaming run reproduce the events, alerts, severity ladders "
        "and timing that each scenario's author wrote down in advance?",
        definition=f"{len(scores)} scenario files with machine-readable `expect:` blocks, each run "
        "headless through the streaming engine with the default rules and the adaptive sampler on, "
        "then matched one-to-one to its labels (drone_agent/eval/ground_truth.py).",
        measured={
            "aggregate": agg,
            "per_scenario": [s.to_dict() for s in scores],
            "model_calls": calls,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            CONTROLLED_QUALIFICATION,
            "The labels were written by the same author as the rules, so agreement mainly shows "
            "that the implementation matches that author's intent; it is not independent evidence.",
            "Scripted detections are perfect; upstream perception errors are the subject of the "
            "noise experiment, not of this one.",
            "Latency is measured in simulated time on scripted data with 60 s frame spacing.",
        ],
    )


# ---- rendering -----------------------------------------------------------------------------------
def render_scores(console: Console, score: ScenarioScore) -> None:
    """A short table for one scenario, used by `run` after a patrol."""
    table = Table(title=f"Ground truth: {score.scenario}")
    for col in ("Expected alert", "Found", "Ladder", "Timing", "Status", "Latency"):
        table.add_column(col, overflow="fold")
    for m in score.matches:
        exp = m.expected
        table.add_row(
            Text(f"{exp.rule_id} ({' > '.join(s.severity.value for s in exp.steps)})"),
            Text("yes" if m.found else "MISSING"),
            Text("ok" if m.severity_path_ok else ("-" if not m.found else "WRONG")),
            Text("ok" if m.steps_timely else ("-" if not m.found else "OFF")),
            Text("-" if m.final_status_ok is None else ("ok" if m.final_status_ok else "WRONG")),
            Text("-" if m.latency_s is None else f"{m.latency_s:.0f}s"),
        )
    console.print(table)
    verdict = (
        f"precision {fmt(score.precision)}, recall {fmt(score.recall)}, F1 {fmt(score.f1)} "
        f"(TP {score.tp}, FP {score.fp}, FN {score.fn}); events {score.events_found}/"
        f"{score.events_expected}; forbidden alerts raised {score.forbidden_hits}"
    )
    if score.label_satisfied and score.surplus:
        verdict += (
            f"; the labels are a subset of the output: {score.events_extra} unlabelled event(s), "
            f"{score.entities_extra} unlabelled entity(ies)"
        )
    console.print(
        Text(
            ("MATCHES ground truth: " if score.label_satisfied else "DIFFERS from ground truth: ")
            + verdict
        )
    )
    for alert in score.unexpected:
        console.print(Text(f"  unexpected alert: {alert.alert_id} ({alert.title})", style="yellow"))
    console.print(
        Text("Labels are author-written; see docs/evaluation.md for what this proves.", style="dim")
    )
