"""Ground-truth evaluation: score a run against the scenario's `expect:` block.

For every scenario the author wrote down (before scoring) which events must exist, which alerts
must be raised with which severity ladder and when their condition first holds, and which alerts
must not appear. A run is scored by matching what the system produced to that truth:

* an expected alert that was produced is a TRUE POSITIVE; one that was not is a FALSE NEGATIVE;
  a produced alert that no label explains (or a forbidden one) is a FALSE POSITIVE;
* precision, recall and F1 follow from those counts (N/A when a denominator is zero);
* for matched alerts we also score the severity ladder (right rungs?), the rung timing (within
  `step_tolerance_s` of the expected condition time?), the final lifecycle status, and the
  alert-DETECTION LATENCY: simulated seconds between the moment the condition became true and the
  moment the system raised the alert.

Matching is one-to-one and greedy on the smallest time difference between the alert's condition
time and the expected one, among alerts of the same rule (and zone, when the label names one).

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
from drone_agent.eval.base import CONTROLLED_QUALIFICATION, ExperimentResult, fmt, ratio
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

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def exact(self) -> bool:
        """Everything the label asked for happened, exactly, and nothing else was raised."""
        found = [m for m in self.matches if m.found]
        return (
            self.fp == 0
            and self.fn == 0
            and self.forbidden_hits == 0
            and all(m.severity_path_ok and m.steps_timely and m.on_time for m in found)
            and all(m.final_status_ok is not False for m in found)
            and self.events_found == self.events_expected
            and self.entities_found == self.entities_expected
        )

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


def _match_alerts(
    scenario: Scenario, expected: list[ExpectedAlert], actual: list[Alert]
) -> tuple[list[AlertMatch], list[Alert]]:
    """One-to-one greedy assignment by smallest condition-time difference within a rule."""
    pairs = []
    for i, exp in enumerate(expected):
        for j, alert in enumerate(actual):
            if alert.rule_id != exp.rule_id:
                continue
            if exp.zone is not None and alert.zone is not exp.zone:
                continue
            pairs.append((abs(_offset(scenario, alert.triggered_at) - exp.condition_at_s), i, j))
    chosen: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in sorted(pairs):
        if i not in chosen and j not in used:
            chosen[i] = j
            used.add(j)
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
    used: set[str] = set()
    missing: list[str] = []
    found = 0
    for exp in expected:
        hit = next(
            (
                e
                for e in events
                if e.event_id not in used
                and e.object_type is exp.object_type
                and e.zone is exp.zone
                and _attributes_match(e.attributes, exp.attributes)
                and e.duration_seconds >= exp.min_duration_s - 1e-9
                and (exp.max_duration_s is None or e.duration_seconds <= exp.max_duration_s + 1e-9)
            ),
            None,
        )
        if hit is None:
            label = f"{exp.object_type.value} in {exp.zone.value}"
            if exp.attributes:
                label += f" {exp.attributes}"
            missing.append(f"{label}, {exp.min_duration_s:g}-{exp.max_duration_s or 'any'}s")
        else:
            used.add(hit.event_id)
            found += 1
    return found, missing


def _match_entities(
    expected: list[ExpectedEntity], entities: list[Entity]
) -> tuple[int, list[str]]:
    used: set[str] = set()
    missing: list[str] = []
    found = 0
    for exp in expected:
        wanted = {k: normalize_attribute(k, v, _NORMALIZATION) for k, v in exp.attributes.items()}
        hit = next(
            (
                e
                for e in entities
                if e.entity_id not in used
                and all(e.attributes.get(k) == v for k, v in wanted.items())
                and e.appearance_count == exp.appearances
            ),
            None,
        )
        if hit is None:
            missing.append(f"{exp.attributes} x{exp.appearances}")
        else:
            used.add(hit.entity_id)
            found += 1
    return found, missing


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
    score.events_expected = len(expect.events)
    score.events_found, score.events_missing = _match_events(expect.events, store.query_events())
    score.entities_expected = len(expect.entities)
    score.entities_found, score.entities_missing = _match_entities(
        expect.entities, store.list_entities()
    )
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
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
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
        "exact_scenarios": sum(1 for s in scores if s.exact),
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
            "exact": s.exact,
        }
        for s in scores
    ]
    lat = agg["detection_latency_s"]
    interpretation = [
        f"{agg['exact_scenarios']} of {agg['scenarios']} scenarios matched their ground truth "
        f"exactly. Micro-averaged over {agg['tp'] + agg['fn']} expected alerts: "
        f"precision {fmt(agg['precision'])}, recall {fmt(agg['recall'])}, F1 {fmt(agg['f1'])} "
        f"({agg['tp']} TP, {agg['fp']} FP, {agg['fn']} FN); forbidden alerts raised: "
        f"{agg['forbidden_hits']}.",
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
        research_question="RQ7: does a streaming run reproduce the events, alerts, severity ladders "
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
    console.print(
        Text(("MATCHES ground truth: " if score.exact else "DIFFERS from ground truth: ") + verdict)
    )
    for alert in score.unexpected:
        console.print(Text(f"  unexpected alert: {alert.alert_id} ({alert.title})", style="yellow"))
    console.print(
        Text("Labels are author-written; see docs/evaluation.md for what this proves.", style="dim")
    )
