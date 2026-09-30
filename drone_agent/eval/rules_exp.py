"""Experiment 5 (RQ5): rule correctness on a labelled scenario matrix (controlled fixtures only).

Each scenario is authored with expected alert counts per rule. Every (scenario, rule) pair is one
cell. Counts: matched alerts are TP, surplus alerts FP, missing alerts FN, and a cell where both
expectation and result are empty is one TN. A zero denominator is reported as N/A. These numbers
describe agreement with hand-labelled synthetic fixtures; they are NOT real-world precision or
recall. No LLM is involved.
"""

from dataclasses import dataclass, field
from datetime import datetime

from drone_agent.eval.base import (
    CONTROLLED_QUALIFICATION,
    IST,
    Confusion,
    ExperimentResult,
    Timeline,
    count_confusion,
    det,
    fmt,
)
from drone_agent.eval.harness import run_pipeline, scenario_observations
from drone_agent.models import Observation, Zone

RULES = ("loitering", "after_hours_presence", "repeat_vehicle", "restricted_zone_entry")
PERSON = ("person", "standing")


def at(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2026, 9, 29, hour, minute, second, tzinfo=IST)


def person(clothing: str = "dark jacket"):
    return det("person", "standing", clothing=clothing)


def pickup(color: str = "blue", **extra: str):
    return det("vehicle", "parked", color=color, body_type="pickup", **extra)


@dataclass
class Scenario:
    name: str
    purpose: str
    observations: list[Observation]
    expected: dict[str, int] = field(default_factory=dict)  # alerts per rule (default 0)


def _minutes(n: int) -> list[int]:
    return [m * 60 for m in range(n)]


def build_matrix() -> list[Scenario]:
    s: list[Scenario] = []

    def add(name, purpose, timeline: Timeline, **expected: int) -> None:
        s.append(Scenario(name, purpose, timeline.observations, expected))

    # --- loitering (main_gate, persons, >= 120 s warning, escalating at 300 s; daytime so
    # after-hours stays quiet). Counts only: severity progression is scored by the ground-truth
    # scenarios (eval/ground_truth.py).
    add(
        "loiter_exact_threshold",
        "person at the gate for exactly 120 s",
        Timeline(at(12)).run(Zone.MAIN_GATE, [0, 60, 120], person()),
        loitering=1,
    )
    add(
        "loiter_below_threshold",
        "person at the gate for 119 s",
        Timeline(at(12)).run(Zone.MAIN_GATE, [0, 60, 119], person()),
    )
    add(
        "loiter_past_critical_threshold",
        "person at the gate for 300 s (one alert that escalates, not two alerts)",
        Timeline(at(12)).run(Zone.MAIN_GATE, [m * 60 for m in range(6)], person()),
        loitering=1,
    )
    add(
        "loiter_other_zone",
        "person stands 10 minutes in the driveway (loitering is gate-only)",
        Timeline(at(12)).run(Zone.DRIVEWAY, _minutes(11), person()),
    )
    add(
        "loiter_vehicle_at_gate",
        "pickup parked at the gate 10 minutes (rule is for persons)",
        Timeline(at(12)).run(Zone.MAIN_GATE, _minutes(11), pickup()),
    )
    tl = Timeline(at(12))
    for m in range(9):
        tl.add(m * 60, Zone.MAIN_GATE, *([] if m == 4 else [person()]))
    add(
        "loiter_with_one_missed_detection",
        "8 minutes at the gate, one frame undetected",
        tl,
        loitering=1,
    )
    # --- after-hours (closed 22:00-06:00, start inclusive, end exclusive) ---------------------
    add(
        "afterhours_person_late",
        "person at 23:00 for one minute",
        Timeline(at(23)).run(Zone.DRIVEWAY, [0, 60], person()),
        after_hours_presence=1,
    )
    add(
        "afterhours_start_boundary",
        "person at exactly 22:00:00",
        Timeline(at(22)).run(Zone.DRIVEWAY, [0, 60], person()),
        after_hours_presence=1,
    )
    add(
        "afterhours_end_boundary",
        "person at exactly 06:00:00 (window end is outside)",
        Timeline(at(6)).run(Zone.DRIVEWAY, [0, 60], person()),
    )
    add(
        "afterhours_daytime",
        "person at 14:00",
        Timeline(at(14)).run(Zone.DRIVEWAY, [0, 60], person()),
    )
    add(
        "afterhours_crosses_midnight",
        "one person event from 23:58 to 00:03",
        Timeline(at(23, 58)).run(Zone.WAREHOUSE, [m * 60 for m in range(6)], person()),
        after_hours_presence=1,
    )
    add(
        "afterhours_vehicle_only",
        "pickup at 23:00 (rule targets persons)",
        Timeline(at(23)).run(Zone.PARKING_AREA, [0, 60], pickup()),
    )

    # --- repeat vehicle (>= 3 separate appearances of a similar vehicle) ----------------------
    def appearances(count: int, colors: list[str], **extra: str) -> Timeline:
        tl = Timeline(at(12))
        for i in range(count):
            zone = (Zone.PARKING_AREA, Zone.DRIVEWAY, Zone.MAIN_GATE)[i % 3]
            tl.run(zone, [i * 1200, i * 1200 + 60], pickup(colors[i], **extra))
        return tl

    add(
        "repeat_three_similar",
        "similar blue pickup three times (one described 'dark blue')",
        appearances(3, ["blue", "dark blue", "blue"]),
        repeat_vehicle=1,
    )
    add("repeat_only_two", "similar pickup twice", appearances(2, ["blue", "blue"]))
    add(
        "repeat_different_vehicles",
        "blue, red and white pickups",
        appearances(3, ["blue", "red", "white"]),
    )
    tl = Timeline(at(12))
    for i in range(3):
        zone = (Zone.PARKING_AREA, Zone.DRIVEWAY, Zone.MAIN_GATE)[i]
        tl.run(zone, [i * 1200, i * 1200 + 60], det("vehicle", "parked", color="blue"))
    add("repeat_missing_body_type", "three blue vehicles with no body type (evidence missing)", tl)
    # --- restricted zone (persons and vehicles, any hour) --------------------------------------
    add(
        "restricted_person_day",
        "person in the restricted zone at noon",
        Timeline(at(12)).run(Zone.RESTRICTED_ZONE, [0, 60], person()),
        restricted_zone_entry=1,
    )
    add(
        "restricted_vehicle_day",
        "pickup in the restricted zone at noon",
        Timeline(at(12)).run(Zone.RESTRICTED_ZONE, [0, 60], pickup()),
        restricted_zone_entry=1,
    )
    add(
        "restricted_warehouse_person",
        "person in the warehouse (not restricted) at noon",
        Timeline(at(12)).run(Zone.WAREHOUSE, [0, 60], person()),
    )
    # --- benign and combined -------------------------------------------------------------------
    tl = Timeline(at(10))
    tl.run(Zone.DRIVEWAY, [0, 60], person("blue shirt"))
    tl.run(
        Zone.PARKING_AREA, [300, 360], det("vehicle", "parked", color="white", body_type="sedan")
    )
    add("benign_daytime", "a person walks by and a sedan parks, mid-morning", tl)
    add(
        "night_restricted_person",
        "person in the restricted zone at 23:30",
        Timeline(at(23, 30)).run(Zone.RESTRICTED_ZONE, [0, 60], person()),
        after_hours_presence=1,
        restricted_zone_entry=1,
    )
    add(
        "night_gate_loiterer",
        "person at the gate 8 minutes at 23:41",
        Timeline(at(23, 41)).run(Zone.MAIN_GATE, _minutes(9), person()),
        loitering=1,
        after_hours_presence=1,
    )
    return s


def evaluate_scenario(sc: Scenario) -> dict:
    _, _, alerts = run_pipeline(sc.observations)
    actual = {r: sum(1 for a in alerts if a.rule_id == r) for r in RULES}
    expected = {r: sc.expected.get(r, 0) for r in RULES}
    cells = {r: count_confusion(expected[r], actual[r]) for r in RULES}
    return {
        "scenario": sc.name,
        "purpose": sc.purpose,
        "expected": expected,
        "actual": actual,
        "exact_match": expected == actual,
        "cells": {r: c.to_dict() for r, c in cells.items()},
    }


def rule_confusions(results: list[dict]) -> dict[str, Confusion]:
    totals = {r: Confusion() for r in RULES}
    for res in results:
        for r in RULES:
            c = res["cells"][r]
            totals[r] += Confusion(c["tp"], c["fp"], c["tn"], c["fn"])
    return totals


def run() -> ExperimentResult:
    matrix = build_matrix()
    results = [evaluate_scenario(sc) for sc in matrix]
    totals = rule_confusions(results)
    overall = sum(totals.values(), Confusion())
    per_rule = {r: c.to_dict() for r, c in totals.items()}
    dev = {}
    for name, expected in (
        (
            "security_events",
            {
                "loitering": 1,
                "after_hours_presence": 3,
                "repeat_vehicle": 1,
                "restricted_zone_entry": 1,
            },
        ),
        ("normal_day", {}),
    ):
        _, _, alerts = run_pipeline(scenario_observations(name))
        actual = {r: sum(1 for a in alerts if a.rule_id == r) for r in RULES}
        dev[name] = {"expected": {r: expected.get(r, 0) for r in RULES}, "actual": actual}
    mismatches = [r["scenario"] for r in results if not r["exact_match"]]
    rows = [
        {
            "experiment": "rules",
            "rule": r,
            "tp": c["tp"],
            "fp": c["fp"],
            "tn": c["tn"],
            "fn": c["fn"],
            "precision": c["precision"],
            "recall": c["recall"],
        }
        for r, c in per_rule.items()
    ]
    interpretation = [
        f"On the {len(matrix)} labelled scenarios ({len(matrix) * len(RULES)} scenario-rule cells) "
        f"the rules matched the labels exactly in {len(matrix) - len(mismatches)} scenarios; "
        f"mismatching scenarios: {mismatches or 'none'}.",
        "Per rule (TP/FP/TN/FN, precision, recall): "
        + "; ".join(
            f"{r} {c['tp']}/{c['fp']}/{c['tn']}/{c['fn']} P={fmt(c['precision'])} "
            f"R={fmt(c['recall'])}"
            for r, c in per_rule.items()
        )
        + ". N/A means a zero denominator.",
        "Rules are deterministic thresholds over events, so they need no LLM; agreement with the "
        "labels mainly confirms the implemented boundaries (inclusive 300 s, closed-window start "
        "inclusive and end exclusive, midnight crossing, missing-evidence non-link).",
        CONTROLLED_QUALIFICATION,
    ]
    return ExperimentResult(
        key="rules",
        title="Rule evaluation on a labelled scenario matrix",
        research_question="RQ5: do the deterministic rules produce the expected alerts on "
        "controlled labelled scenarios without an LLM?",
        definition=f"{len(matrix)} authored scenarios with expected alert counts per rule, run "
        "through the production aggregator, linker and rule engine (default configs). "
        "security_events and normal_day are reported separately as development fixtures.",
        measured={
            "scenarios": results,
            "per_rule": per_rule,
            "overall": overall.to_dict(),
            "development_fixtures": dev,
            "qualification": CONTROLLED_QUALIFICATION,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            CONTROLLED_QUALIFICATION,
            "The labels were written by the same author as the rules, so agreement is not "
            "independent validation; it checks the documented boundaries, not usefulness.",
            "Scripted detections are perfect; upstream perception errors are not measured here.",
        ],
    )
