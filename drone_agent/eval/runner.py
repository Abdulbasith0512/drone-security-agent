"""Run every experiment, derive the ablation table, and write machine-readable results.

`run_all` is the single entry point (used by `python -m drone_agent evaluate`). Nothing is typed in
by hand: the tables and prose are built from values the experiments computed.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from drone_agent.eval import (
    gate_exp,
    ground_truth,
    grounding_exp,
    noise_exp,
    pipeline_exp,
    rules_exp,
    sampler_exp,
    scale,
)
from drone_agent.eval.base import CONTROLLED_QUALIFICATION, ExperimentResult

HARNESS_VERSION = "2"
QUALIFICATIONS = (
    CONTROLLED_QUALIFICATION,
    "Scripted detections do not measure vision-model perception accuracy.",
    "No live model or network call is made by the evaluation.",
    "Timing results (scale) are measurements on one machine, not universal claims.",
)

ABLATION_COLUMNS = (
    "component",
    "metric",
    "baseline_setting",
    "baseline",
    "ablated_setting",
    "ablated",
    "effect",
    "interpretation",
)


def _signed(delta: float | int) -> str:
    return f"{delta:+d}" if isinstance(delta, int) else f"{delta:+.1f}"


def build_ablation_table(results: dict[str, ExperimentResult]) -> list[dict]:
    rows: list[dict] = []
    gate = results["gate"].measured
    default = next(s for s in gate["sweep"] if s["threshold"] == gate["default_threshold"])
    off = next(s for s in gate["sweep"] if s["threshold"] == 0.0)
    rows.append(
        {
            "component": "Frame gate",
            "metric": f"vision-model calls per {gate['candidates']} candidate frames (proxy)",
            "baseline_setting": f"gate threshold {gate['default_threshold']:g} (default)",
            "baseline": default["accepted"],
            "ablated_setting": "gate off (threshold 0)",
            "ablated": off["accepted"],
            "effect": f"{_signed(default['accepted'] - off['accepted'])} calls "
            f"({default['call_reduction_pct']}% fewer)",
            "interpretation": "fewer calls, paid for by the transitions listed in the next row",
        }
    )
    rows.append(
        {
            "component": "Frame gate",
            "metric": "meaningful scene transitions detected",
            "baseline_setting": f"gate threshold {gate['default_threshold']:g} (default)",
            "baseline": f"{default['meaningful_detected']}/{default['meaningful_transitions']}",
            "ablated_setting": "gate off (threshold 0)",
            "ablated": f"{off['meaningful_detected']}/{off['meaningful_transitions']}",
            "effect": f"{_signed(default['meaningful_detected'] - off['meaningful_detected'])} "
            "transitions",
            "interpretation": "missed: "
            + (", ".join(default["missed_names"]) or "none")
            + "; pixel change is not semantic importance",
        }
    )
    comp = next(
        f for f in results["compression"].measured["fixtures"] if f["fixture"] == "security_events"
    )
    rows.append(
        {
            "component": "Temporal aggregation",
            "metric": "evidence units (security_events)",
            "baseline_setting": "events (aggregation on)",
            "baseline": comp["events"],
            "ablated_setting": "raw frame-level detections",
            "ablated": comp["detected_objects"],
            "effect": f"{_signed(comp['events'] - comp['detected_objects'])} units "
            f"(ratio {comp['compression_ratio']})",
            "interpretation": "repeated detections become bounded events; every detection stays "
            f"linked as evidence ({comp['all_detections_have_evidence']})",
        }
    )
    fl = results["flicker"].measured["loitering_episode"]
    base_key = f"max_missed_frames={results['flicker'].measured['settings'][1]}"
    for metric, field in (
        ("events from one loitering episode", "person_events"),
        ("loitering alerts from that episode", "loitering_alerts"),
    ):
        rows.append(
            {
                "component": "Flicker tolerance",
                "metric": metric,
                "baseline_setting": base_key + " (default)",
                "baseline": fl[base_key][field],
                "ablated_setting": "max_missed_frames=0",
                "ablated": fl["max_missed_frames=0"][field],
                "effect": _signed(fl["max_missed_frames=0"][field] - fl[base_key][field]),
                "interpretation": "one undetected frame splits the episode into fragments that "
                "each alert at the warning level instead of one escalated alert"
                if field == "loitering_alerts"
                else "one undetected frame splits one continuous episode",
            }
        )
    top = {
        key: (fl[key]["loitering_severities"] or ["none"])[-1]
        for key in (base_key, "max_missed_frames=0")
    }
    rows.append(
        {
            "component": "Flicker tolerance",
            "metric": "highest loitering severity from that episode",
            "baseline_setting": base_key + " (default)",
            "baseline": top[base_key],
            "ablated_setting": "max_missed_frames=0",
            "ablated": top["max_missed_frames=0"],
            "effect": f"{top[base_key]} -> {top['max_missed_frames=0']}",
            "interpretation": "the split fragments never reach the critical threshold, so the "
            "escalation is lost",
        }
    )
    sp = results["sampler"].measured["policies"]
    adaptive = sp["adaptive (pixel + telemetry + event watch + refresh)"]
    pixel = sp["pixel gate, threshold 6 (previous default)"]
    periodic = sp["pixel gate + refresh every 15 frames"]
    frames = results["sampler"].measured["frames_per_stream"]
    window = results["sampler"].measured["detect_window_frames"]
    rows.append(
        {
            "component": "Adaptive sampling",
            "metric": f"meaningful changes detected within {window} frames (generated streams)",
            "baseline_setting": "adaptive (pixel + telemetry + event watch + refresh)",
            "baseline": adaptive["recall"],
            "ablated_setting": "pixel gate only (previous default)",
            "ablated": pixel["recall"],
            "effect": f"{pixel['recall'] - adaptive['recall']:+.3f}",
            "interpretation": f"the pixel gate misses {pixel['missed_small']} small and "
            f"{pixel['missed_medium']} medium changes that the adaptive sampler mostly catches",
        }
    )
    rows.append(
        {
            "component": "Adaptive sampling",
            "metric": f"model calls per {frames}-frame stream",
            "baseline_setting": "adaptive",
            "baseline": adaptive["calls_mean"],
            "ablated_setting": "pixel gate only (previous default)",
            "ablated": pixel["calls_mean"],
            "effect": f"{pixel['calls_mean'] - adaptive['calls_mean']:+.1f} calls",
            "interpretation": "recall is bought with calls: never quote the saving without the "
            "miss rate in the row above",
        }
    )
    rows.append(
        {
            "component": "Adaptive sampling",
            "metric": "model calls per detected change",
            "baseline_setting": "adaptive",
            "baseline": adaptive["calls_per_detected"],
            "ablated_setting": "pixel gate + refresh every 15 frames",
            "ablated": periodic["calls_per_detected"],
            "effect": f"{periodic['calls_per_detected'] - adaptive['calls_per_detected']:+.2f}",
            "interpretation": "against the fair cheap baseline (a fixed refresh), not a straw man",
        }
    )
    gt = results["ground_truth"].measured["aggregate"]
    rows.append(
        {
            "component": "Streaming alerts with ground truth",
            "metric": "alert F1 (perfect scripted perception)",
            "baseline_setting": f"{gt['scenarios']} scenarios, {gt['tp'] + gt['fn']} expected alerts",
            "baseline": gt["f1"],
            "ablated_setting": "n/a",
            "ablated": "n/a",
            "effect": f"latency median {gt['detection_latency_s']['median']} s, "
            f"max {gt['detection_latency_s']['max']} s",
            "interpretation": "1.0 by construction (author-written labels, perfect input); the "
            "noise sweep shows how it degrades",
        }
    )
    em = results["entity_memory"].measured
    for metric, field in (
        ("repeat-vehicle alerts (security_events)", "repeat_vehicle_alerts"),
        ("multi-appearance vehicle entities", "multi_appearance_entities"),
    ):
        rows.append(
            {
                "component": "Entity memory",
                "metric": metric,
                "baseline_setting": "coarse linking on (default)",
                "baseline": em["baseline"][field],
                "ablated_setting": "each vehicle event independent",
                "ablated": em["ablated"][field],
                "effect": _signed(em["ablated"][field] - em["baseline"][field]),
                "interpretation": "linking is what lets the repeat rule and 'seen again' "
                f"questions work; false links exist ({len(em['false_links'])} in the controls)",
            }
        )
    ca = results["citation_adversarial"].measured
    rows.append(
        {
            "component": "Citation validation",
            "metric": f"invalid outputs accepted (of {ca['expected_reject_cases']} adversarial)",
            "baseline_setting": "citation gate on",
            "baseline": ca["invalid_outputs_accepted"],
            "ablated_setting": "no gate",
            "ablated": ca["no_validator_accepts"] - ca["expected_accept_cases"],
            "effect": _signed(
                ca["invalid_outputs_accepted"]
                - (ca["no_validator_accepts"] - ca["expected_accept_cases"])
            ),
            "interpretation": f"gate decision accuracy on this fixture set: "
            f"{ca['accuracy_on_this_fixture_set']}; misses: {ca['false_accept_cases'] or 'none'}",
        }
    )
    sp = results["semantic_probe"].measured
    ck = sp["checker"]
    pg = sp["production_gate"]
    rows.append(
        {
            "component": "Factual-claim validation (production)",
            "metric": f"semantically unsupported answers accepted (of {sp['unsupported_cases']})",
            "baseline_setting": "citation gate + facts read from the answer text (production)",
            "baseline": pg["unsupported_accepted"],
            "ablated_setting": "citation gate alone (before factual validation)",
            "ablated": ck["unsupported_accepted_gate_alone"],
            "effect": _signed(pg["unsupported_accepted"] - ck["unsupported_accepted_gate_alone"]),
            "interpretation": "checks structured facts only (zone, time, duration, severity, "
            f"status, rule, activity, counts); true controls rejected: {pg['controls_rejected']}; "
            "remaining passes: " + ", ".join(pg["still_accepted_categories"]),
        }
    )
    rows.append(
        {
            "component": "Claim-consistency prototype (eval-only, hand-structured claims)",
            "metric": f"semantically unsupported answers accepted (of {sp['unsupported_cases']})",
            "baseline_setting": "citation gate + structured claim checker",
            "baseline": ck["unsupported_accepted_gate_plus_checker"],
            "ablated_setting": "citation gate alone (before factual validation)",
            "ablated": ck["unsupported_accepted_gate_alone"],
            "effect": _signed(
                ck["unsupported_accepted_gate_plus_checker"] - ck["unsupported_accepted_gate_alone"]
            ),
            "interpretation": "covers only structured claim classes; remaining passes: "
            + ", ".join(ck["still_accepted_categories"]),
        }
    )
    return rows


def run_all(
    scale_enabled: bool = False,
    scale_sizes: tuple[int, ...] = scale.DEFAULT_SIZES,
    repeats: int = 5,
    noise_enabled: bool = False,
    noise_seeds: int = noise_exp.DEFAULT_SEEDS,
) -> dict:
    experiments: list[ExperimentResult] = [
        gate_exp.run(),
        pipeline_exp.run_compression(),
        pipeline_exp.run_flicker(),
        pipeline_exp.run_entity_memory(),
        rules_exp.run(),
        grounding_exp.run_adversarial(),
        grounding_exp.run_semantic_probe(),
        ground_truth.run(),
        sampler_exp.run(),
    ]
    if noise_enabled:
        experiments.append(noise_exp.run(seeds=noise_seeds))
    by_key = {e.key: e for e in experiments}
    if scale_enabled:
        experiments.append(scale.run(scale_sizes, repeats=repeats))
    return {
        "meta": {
            "harness_version": HARNESS_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "scale_included": scale_enabled,
            "noise_included": noise_enabled,
            "qualifications": list(QUALIFICATIONS),
        },
        "experiments": {e.key: e.to_dict() for e in experiments},
        "ablation_table": build_ablation_table(by_key),
    }


def deterministic_view(results: dict) -> dict:
    """The parts of a run that must be identical between runs (no timestamp, no timings)."""
    return {
        "meta": {k: v for k, v in results["meta"].items() if k != "generated_at_utc"},
        "experiments": {k: v for k, v in results["experiments"].items() if k != "scale"},
        "ablation_table": results["ablation_table"],
    }


def _write_csv(path: Path, rows: list[dict], first: tuple[str, ...] = ()) -> None:
    fields = list(first)
    for row in rows:
        fields += [k for k in row if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("N/A" if v is None else v) for k, v in row.items()})


def _md_table(rows: list[dict], columns: list[str] | None = None) -> list[str]:
    columns = columns or list(dict.fromkeys(k for r in rows for k in r if k != "experiment"))
    if not rows:
        return []

    def cell(value: object) -> str:
        return ("N/A" if value is None else str(value)).replace("|", "/").replace("\n", " ")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(cell(r.get(c, "")) for c in columns) + " |" for r in rows]
    return lines


def render_markdown(results: dict) -> str:
    """A generated report: every table and sentence is built from the computed results."""
    meta = results["meta"]
    out = ["# Evaluation results (generated)", ""]
    out += [
        f"Generated by `python -m drone_agent evaluate` (harness v{meta['harness_version']}).",
        "",
    ]
    out += [f"- {q}" for q in meta["qualifications"]]
    out += ["", "## Ablation table", ""]
    out += _md_table(results["ablation_table"], list(ABLATION_COLUMNS)) + [""]
    for exp in results["experiments"].values():
        out += [f"## {exp['title']}", "", f"**Research question.** {exp['research_question']}", ""]
        out += [f"**Setup.** {exp['definition']}", "", "**Measured result**", ""]
        out += _md_table(exp["rows"]) + ["", "**Interpretation**", ""]
        out += [f"- {line}" for line in exp["interpretation"]]
        out += ["", "**Limitations**", ""] + [f"- {line}" for line in exp["limitations"]] + [""]
    scale_exp = results["experiments"].get("scale")
    if scale_exp:
        env = scale_exp["measured"]["environment"]
        out += ["**Environment (scale benchmark)**", ""]
        out += [f"- {k}: {v}" for k, v in env.items()] + [""]
    return "\n".join(out) + "\n"


def write_results(results: dict, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out / "results.json", out / "results.csv"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [row for exp in results["experiments"].values() for row in exp["rows"]]
    _write_csv(csv_path, rows, first=("experiment",))
    ablation_csv = out / "ablation_table.csv"
    _write_csv(ablation_csv, results["ablation_table"], first=ABLATION_COLUMNS)
    summary = out / "summary.md"
    summary.write_text(render_markdown(results), encoding="utf-8")
    return [json_path, csv_path, ablation_csv, summary]
