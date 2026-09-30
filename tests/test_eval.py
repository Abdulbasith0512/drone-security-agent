"""Evaluation harness: determinism, metric math, each experiment's mechanics, serialization.

These tests check that the harness measures what it claims and that its results are reproducible.
They never assert timing thresholds and never touch the network.
"""

import ast
import csv
import json
import socket
from io import StringIO

import numpy as np
import pytest
from conftest import ROOT
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.eval import claims as cl
from drone_agent.eval import gate_exp, grounding_exp, pipeline_exp, rules_exp, scale
from drone_agent.eval.base import Confusion, count_confusion, fmt, median, pct, ratio
from drone_agent.eval.runner import (
    ABLATION_COLUMNS,
    deterministic_view,
    run_all,
    write_results,
)


@pytest.fixture(scope="module")
def two_runs():
    return run_all(), run_all()


@pytest.fixture(scope="module")
def results(two_runs):
    return two_runs[0]["experiments"]


# ---- determinism, serialization, no network -------------------------------------------------
def test_the_functional_results_are_identical_between_runs(two_runs):
    first, second = two_runs
    assert json.dumps(deterministic_view(first), sort_keys=True) == json.dumps(
        deterministic_view(second), sort_keys=True
    )


def test_every_experiment_separates_measured_interpretation_and_limitations(results):
    assert set(results) == {
        "gate", "compression", "flicker", "entity_memory", "rules", "citation_adversarial",
        "semantic_probe",
    }  # fmt: skip
    for exp in results.values():
        assert exp["measured"] and exp["interpretation"] and exp["limitations"]
        assert exp["definition"] and exp["research_question"].startswith("RQ")


def test_results_serialize_to_json_and_csv(two_runs, tmp_path):
    paths = write_results(two_runs[0], tmp_path / "eval")
    loaded = json.loads(paths[0].read_text(encoding="utf-8"))
    assert loaded["experiments"]["gate"]["measured"]["default_threshold"] == 6.0
    assert "qualifications" in loaded["meta"] and loaded["ablation_table"]
    with paths[1].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0].keys() >= {"experiment"} and {r["experiment"] for r in rows} >= {"gate", "rules"}
    with paths[2].open(encoding="utf-8", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert tuple(table[0].keys())[: len(ABLATION_COLUMNS)] == ABLATION_COLUMNS
    assert len(table) == len(two_runs[0]["ablation_table"])


def test_the_harness_needs_no_network_and_no_api_key(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the evaluation must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    for name in ("GEMINI_API_KEY", "ANALYST_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)
    assert run_all()["experiments"]["rules"]["measured"]["overall"]["tp"] > 0


def test_the_eval_package_does_not_import_the_llm_framework():
    for path in (ROOT / "drone_agent" / "eval").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(n.startswith(("langchain", "langgraph")) for n in names), path.name


# ---- metric math ----------------------------------------------------------------------------
def test_ratio_pct_and_formatting_report_na_for_zero_denominators():
    assert ratio(1, 0) is None and pct(1, 0) is None
    assert ratio(1, 4) == 0.25 and pct(1, 4) == 25.0
    assert fmt(None) == "N/A" and fmt(3) == "3" and fmt(0.5) == "0.500"
    assert median([3, 1, 2]) == 2


def test_confusion_metrics_and_zero_denominators():
    c = Confusion(tp=3, fp=1, tn=5, fn=1)
    assert (c.precision, c.recall) == (0.75, 0.75)
    empty = Confusion(tn=4)  # nothing expected, nothing produced
    assert empty.precision is None and empty.recall is None
    assert empty.to_dict()["precision"] is None
    no_alerts_but_expected = Confusion(fn=2)
    assert no_alerts_but_expected.precision is None and no_alerts_but_expected.recall == 0.0
    assert (Confusion(tp=1) + Confusion(fp=2, tn=1)) == Confusion(1, 2, 1, 0)


def test_count_confusion_cells():
    assert count_confusion(0, 0) == Confusion(tn=1)
    assert count_confusion(1, 1) == Confusion(tp=1)
    assert count_confusion(1, 0) == Confusion(fn=1)
    assert count_confusion(0, 2) == Confusion(fp=2)
    assert count_confusion(2, 3) == Confusion(tp=2, fp=1)


# ---- experiment 1: gate ---------------------------------------------------------------------
def test_gate_fixture_is_deterministic():
    assert np.array_equal(gate_exp.render(7), gate_exp.render(7))
    assert not np.array_equal(gate_exp.render(4), gate_exp.render(5))
    assert gate_exp.frame_scores() == gate_exp.frame_scores()


def test_gate_sweep_mechanics(results):
    sweep = results["gate"]["measured"]["sweep"]
    accepted = [s["accepted"] for s in sweep]
    assert sweep[0]["threshold"] == 0.0 and sweep[0]["accepted"] == gate_exp.CANDIDATES
    assert accepted == sorted(accepted, reverse=True)  # a higher threshold never accepts more
    recalls = [s["meaningful_recall"] for s in sweep]
    assert recalls == sorted(recalls, reverse=True)
    assert sweep[0]["meaningful_recall"] == 1.0 and sweep[0]["call_reduction_pct"] == 0.0
    default = next(s for s in sweep if s["threshold"] == 6.0)
    assert default["accepted"] + default["skipped"] == gate_exp.CANDIDATES
    thresholds = [s["threshold"] for s in sweep]
    assert 0.0 in thresholds and 6.0 in thresholds and max(thresholds) > 6.0


def test_gate_shows_pixel_change_is_not_semantic_importance(results):
    m = results["gate"]["measured"]
    px = m["pixel_vs_semantic"]
    assert px["small_change_scores_below_irrelevant_illumination"] is True
    assert (
        "small object appears"
        in next(s for s in m["sweep"] if s["threshold"] == 6.0)["missed_names"]
    )
    static_max = m["static_frame_score"]["max"]
    assert static_max < min(t["score_vs_previous_frame"] for t in m["transition_scores"])


# ---- experiments 2-4 ------------------------------------------------------------------------
def test_compression_keeps_every_detection_as_evidence(results):
    fixtures = {f["fixture"]: f for f in results["compression"]["measured"]["fixtures"]}
    sec = fixtures["security_events"]
    assert (sec["observations"], sec["detected_objects"], sec["events"]) == (27, 20, 7)
    dwell = fixtures["long_dwell_controlled"]
    assert (dwell["detected_objects"], dwell["events"], dwell["compression_ratio"]) == (90, 2, 45.0)
    for f in fixtures.values():
        assert f["all_detections_have_evidence"] and f["event_spans_match_evidence"]


def test_flicker_ablation_changes_events_and_the_loitering_alert(results):
    episode = results["flicker"]["measured"]["loitering_episode"]
    default, ablated = episode["max_missed_frames=2"], episode["max_missed_frames=0"]
    assert (default["person_events"], default["loitering_alerts"]) == (1, 1)
    assert (ablated["person_events"], ablated["loitering_alerts"]) == (2, 0)
    assert ablated["durations_s"] == [180.0, 180.0]


def test_flicker_gap_sweep_shows_the_tolerance_boundary():
    def summary(missed, setting):
        return pipeline_exp._flicker_summary(pipeline_exp.gap_fixture(missed), setting)

    assert summary(0, 0)["person_events"] == 1  # no gap: nothing to tolerate
    assert [summary(k, 2)["person_events"] for k in range(5)] == [1, 1, 1, 2, 2]
    assert [summary(k, 0)["person_events"] for k in range(5)] == [1, 2, 2, 2, 2]


def test_entity_memory_ablation_and_controls(results):
    m = results["entity_memory"]["measured"]
    assert (m["baseline"]["vehicle_entities"], m["baseline"]["repeat_vehicle_alerts"]) == (2, 1)
    assert (m["ablated"]["vehicle_entities"], m["ablated"]["repeat_vehicle_alerts"]) == (0, 0)
    assert m["baseline"]["vehicle_events"] == m["ablated"]["vehicle_events"] == 4
    assert {e["entity_id"] for e in m["baseline"]["pickup_events"]} == {"ENT-V-001"}
    assert {e["entity_id"] for e in m["ablated"]["pickup_events"]} == {None}


def test_negative_controls_follow_the_documented_contract():
    by_case = {c["case"]: c for c in pipeline_exp.negative_controls()}
    assert all(c["contract_respected"] for c in by_case.values())
    unlinked = (
        "different colour",
        "different body type",
        "missing body type",
        "conflicting explicit plates",
        "overlapping in time (cannot be one vehicle)",
    )
    for name in unlinked:
        assert by_case[name]["actual_linked"] is False and not by_case[name]["false_link"]
    same_plate = by_case["same plate, different colour (explicit plate evidence)"]
    assert same_plate["actual_linked"] and same_plate["match_levels"] == ["confirmed"]
    false_link = by_case["two real pickups, same coarse attributes (false-link limitation)"]
    assert false_link["actual_linked"] and false_link["false_link"] is True  # reported limitation


# ---- experiment 5: rules --------------------------------------------------------------------
def test_rule_matrix_is_well_formed_and_scored():
    matrix = rules_exp.build_matrix()
    assert len(matrix) >= 20 and len({s.name for s in matrix}) == len(matrix)
    for sc in matrix:
        assert set(sc.expected) <= set(rules_exp.RULES)
    for rule in rules_exp.RULES:  # every rule has positive AND negative scenarios
        assert any(s.expected.get(rule) for s in matrix)
        assert any(not s.expected.get(rule) for s in matrix)


def test_rule_confusion_totals_are_consistent(results):
    m = results["rules"]["measured"]
    n_scenarios = len(m["scenarios"])
    for rule, c in m["per_rule"].items():
        assert c["tp"] + c["fp"] + c["tn"] + c["fn"] >= n_scenarios  # every cell is counted
        expected_total = sum(s["expected"][rule] for s in m["scenarios"])
        assert c["tp"] + c["fn"] == expected_total
        actual_total = sum(s["actual"][rule] for s in m["scenarios"])
        assert c["tp"] + c["fp"] == actual_total
    assert "controlled" in m["qualification"].lower() and "real-world" in m["qualification"]
    dev = m["development_fixtures"]
    assert dev["security_events"]["expected"] == dev["security_events"]["actual"]
    assert dev["normal_day"]["actual"] == dev["normal_day"]["expected"]


def test_a_rule_with_no_alerts_reports_na_not_a_number():
    scenario = rules_exp.Scenario("nothing", "empty", rules_exp.build_matrix()[3].observations, {})
    res = rules_exp.evaluate_scenario(scenario)
    totals = rules_exp.rule_confusions([res])
    assert totals["loitering"].precision is None and totals["loitering"].recall is None


# ---- experiments 6-8 ------------------------------------------------------------------------
def test_adversarial_suite_covers_every_required_category(results):
    cats = set(results["citation_adversarial"]["measured"]["by_category"])
    assert cats >= {
        "valid_retrieved", "nonexistent", "real_not_retrieved", "malformed", "no_citation",
        "identity_claim", "intent_claim", "negated_identity", "uncertainty_vehicle",
        "insufficient_evidence",
    }  # fmt: skip


def test_adversarial_results_are_internally_consistent(results):
    exp = results["citation_adversarial"]
    m = exp["measured"]
    wrong = [r["case"] for r in exp["rows"] if not r["correct"]]
    assert m["cases"] - m["correct"] == len(wrong)
    assert sorted(wrong) == sorted(m["false_accept_cases"] + m["false_reject_cases"])
    for row in exp["rows"]:
        if row["category"] in {"nonexistent", "real_not_retrieved", "no_citation", "intent_claim"}:
            assert row["actual"] == "reject", row["case"]
        if row["category"] in {"valid_retrieved", "insufficient_evidence", "uncertainty_vehicle"}:
            assert row["actual"] == "accept", row["case"]
    assert m["no_validator_accepts"] == m["cases"]


def test_gate_decision_helper_matches_the_real_session():
    with grounding_exp.evidence_store() as store:
        ok, _ = grounding_exp.gate_accepts(
            store, "Loitering [alert:ALR-LOITER-EVT-P-001].", [grounding_exp.LOITER]
        )
        assert ok
        ok, reasons = grounding_exp.gate_accepts(store, "Loitering [event:EVT-Z-9].", [])
        assert not ok and "do not exist" in reasons[0]


def test_semantic_probe_records_true_and_false_claims_separately(results):
    exp = results["semantic_probe"]
    m = exp["measured"]
    controls = [r for r in exp["rows"] if r["claims_supported_by_evidence"]]
    unsupported = [r for r in exp["rows"] if not r["claims_supported_by_evidence"]]
    assert len(controls) == m["control_cases"] and len(unsupported) == m["unsupported_cases"]
    assert m["unsupported_accepted_by_gate"] == sum(1 for r in unsupported if r["gate_accepts"])
    assert m["controls_accepted_by_gate"] == m["control_cases"]  # true claims pass too
    assert "hallucination" in exp["interpretation"][-1].lower()  # scope statement present
    assert all(p.supported is False or p.category == "control_true" for p in grounding_exp.probes())


def test_every_probe_uses_only_existing_retrieved_wellformed_citations():
    with grounding_exp.evidence_store() as store:
        from drone_agent.agent.citations import extract_citations, validate_answer
        from drone_agent.agent.tools import AnalystTools

        for probe in grounding_exp.probes():
            tools = AnalystTools(store)
            for name, args in probe.prep:
                getattr(tools, name)(**args)
            report = validate_answer(store, probe.text, retrieved=tools.retrieved)
            assert extract_citations(probe.text), probe.name
            assert not (report.invalid or report.not_retrieved or report.malformed), probe.name


def test_claim_checker_semantics_and_limits(results):
    ck = results["semantic_probe"]["measured"]["checker"]
    assert ck["controls_rejected_by_checker"] == 0
    assert ck["expressible_caught"] == ck["expressible_unsupported_cases"]
    assert ck["unsupported_accepted_gate_plus_checker"] == ck["inexpressible_unsupported_cases"]
    assert ck["unsupported_accepted_gate_alone"] > ck["unsupported_accepted_gate_plus_checker"]
    assert set(ck["supported_claim_classes"]) == set(cl.SUPPORTED_CLASSES)


def test_claim_checker_unit_verdicts():
    with grounding_exp.evidence_store() as store:
        C = cl.Claim

        def status(kind, subject, value):
            return cl.check_claim(store, C(kind, subject, value)).status

        assert status("event_zone", "EVT-V-001", "parking_area") == "supported"
        assert status("event_zone", "EVT-V-001", "main_gate") == "contradicted"
        assert status("event_type", "EVT-P-001", "person") == "supported"
        assert status("event_activity", "EVT-P-001", "pacing") == "supported"
        assert status("event_activity", "EVT-P-001", "running") == "contradicted"
        assert status("event_duration_s", "EVT-P-001", 480) == "supported"
        assert status("event_duration_s", "EVT-P-001", 480.5) == "supported"  # tolerance 1 s
        assert status("event_duration_s", "EVT-P-001", 900) == "contradicted"
        assert status("event_start_hhmm", "EVT-P-001", "23:41") == "supported"
        assert status("alert_severity", "ALR-REPEAT-ENT-V-001", "low") == "supported"
        assert status("alert_severity", "ALR-REPEAT-ENT-V-001", "high") == "contradicted"
        assert status("alert_rule", "ALR-LOITER-EVT-P-001", "loitering") == "supported"
        assert status("entity_appearance_count", "ENT-V-001", 3) == "supported"
        assert status("entity_appearance_count", "ENT-V-001", 5) == "contradicted"
        assert status("event_zone", "EVT-NOPE", "main_gate") == "unknown_subject"
        assert status("identity", "ENT-V-001", "same") == "unsupported_class"
        verdicts = cl.check_claims(store, [C("event_zone", "EVT-V-001", "parking_area")])
        assert cl.all_supported(verdicts) and not cl.all_supported(
            verdicts + [cl.check_claim(store, C("event_zone", "EVT-V-001", "x"))]
        )


# ---- scale ----------------------------------------------------------------------------------
def test_generated_observations_are_deterministic_and_chronological():
    a, b = scale.generate_observations(200), scale.generate_observations(200)
    assert [o.model_dump_json() for o in a] == [o.model_dump_json() for o in b]
    assert len(a) == 200
    assert all(x.timestamp < y.timestamp for x, y in zip(a, a[1:], strict=False))
    assert scale.generate_observations(200, seed=1) != a


def test_scale_benchmark_smoke_reports_structure_without_asserting_timings():
    exp = scale.run((30, 60), repeats=1, query_repeats=2)
    sizes = exp.measured["sizes"]
    assert [s["observations"] for s in sizes] == [30, 60]
    for s in sizes:
        assert {"event_aggregation", "entity_linking", "rule_evaluation"} <= set(s["median_ms"])
        assert s["events"] > 0
    assert {"python", "platform", "processor", "sqlite_version"} <= set(exp.measured["environment"])
    assert "not a universal" in exp.measured["environment"]["note"]
    assert set(exp.measured["scaling_exponent_two_largest_sizes"]) == set(sizes[0]["median_ms"])


def test_scaling_exponent_math():
    fake = [
        {"observations": 100, "median_ms": {"a": 1.0, "b": 1.0, "c": 0.0}},
        {"observations": 1000, "median_ms": {"a": 10.0, "b": 100.0, "c": 1.0}},
    ]
    exps = scale.scaling_exponents(fake)
    assert exps["a"] == 1.0 and exps["b"] == 2.0 and exps["c"] is None
    assert scale.scaling_exponents(fake[:1]) == {}


# ---- CLI ------------------------------------------------------------------------------------
def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


def test_evaluate_command_writes_results_and_prints_the_ablation_table(tmp_path):
    code, text = run_cli("evaluate", "--output", tmp_path / "eval")
    assert code == 0, text
    assert "Ablation table" in text and "Frame gate" in text and "Entity memory" in text
    assert "Controlled synthetic fixtures only" in text
    assert (tmp_path / "eval" / "results.json").is_file()
    assert (tmp_path / "eval" / "results.csv").is_file()
    assert (
        json.loads((tmp_path / "eval" / "results.json").read_text())["meta"]["scale_included"]
        is False
    )


def test_evaluate_command_scale_flag_and_argument_errors(tmp_path):
    args = ["evaluate", "--output", tmp_path / "s", "--scale", "--scale-sizes", "20,40"]
    code, text = run_cli(*args, "--repeats", "1")
    assert code == 0, text
    data = json.loads((tmp_path / "s" / "results.json").read_text())
    assert data["meta"]["scale_included"] and "scale" in data["experiments"]
    code, text = run_cli("evaluate", "--output", tmp_path / "bad", "--scale-sizes", "3")
    assert code == 2 and "scale-sizes" in text
