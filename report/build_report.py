"""Build the final report: report/report_template.html -> report/report.html -> PDF (headless Edge).

Every table and chart is generated from docs/evaluation.md (the committed evaluation snapshot); the
functional results are cross-checked against a fresh run of the deterministic, offline harness. The
real-video numbers are computed by replaying the bundled clip and cassette (demo_assets/real_clip/) and
asserted against docs/real_video_experiment.md. After the PDF is written it is verified (page count, no
embedded files, no confidential file). No API call and no network access.

Run:  python report/make_diagrams.py && python report/build_report.py [--no-pdf]
"""

import argparse
import base64
import re
import subprocess
import sys
import tempfile
from datetime import date
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
PDF_NAME = "Drone_Security_Analyst_Report_Abdul_Basith_Syed.pdf"  # must not contain "assignment"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

sys.path.insert(0, str(ROOT))


# ---- parsing the committed evaluation snapshot ------------------------------------------------
def parse_tables(text: str) -> dict[str, list[list[dict]]]:
    """'### Title' -> list of markdown tables, each a list of row dicts."""
    out: dict[str, list[list[dict]]] = {}
    lines = text.split("\n")
    title, i = "", 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            title = line[4:].strip()
        if (
            line.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\|[-| ]+\|\s*$", lines[i + 1])
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            j, rows = i + 2, []
            while j < len(lines) and lines[j].startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                rows.append(dict(zip(header, cells, strict=False)))
                j += 1
            out.setdefault(title, []).append(rows)
            i = j
            continue
        i += 1
    return out


def first(tables, title):
    return tables[title][0]


def num(value):
    return None if value in ("N/A", "", None) else float(value)


# ---- HTML helpers -----------------------------------------------------------------------------
def table(header, rows, caption=None, cls="", align=None):
    align = align or {}
    head = "".join(
        f'<th class="{align.get(h, "")}">{escape(str(h).replace("_", " "))}</th>' for h in header
    )
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{align.get(h, "")}">{escape(str(row.get(h, "")))}</td>' for h in header
        )
        body += f"<tr>{cells}</tr>"
    cap = f"<caption>{escape(caption)}</caption>" if caption else ""
    cls = f"{cls} keep" if len(rows) <= 12 else cls  # a short table stays on one page
    return f'<table class="data {cls}">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def bars(title, labels, values, ymax, color, w=380, h=190, fmt="{:g}"):
    left, bottom, top = 40, h - 42, 38
    plot_h, n = bottom - top, len(values)
    gap = (w - left - 10) / n
    out = [f'<text x="{w / 2}" y="16" text-anchor="middle" class="ct">{escape(title)}</text>']
    for k in range(0, 5):
        yv = ymax * k / 4
        y = bottom - plot_h * k / 4
        out.append(f'<line x1="{left}" x2="{w - 6}" y1="{y}" y2="{y}" class="gl"/>')
        out.append(f'<text x="{left - 5}" y="{y + 4}" text-anchor="end" class="ax">{yv:g}</text>')
    for i, (lab, val) in enumerate(zip(labels, values, strict=True)):
        x = left + i * gap + gap * 0.15
        bw = gap * 0.7
        bh = plot_h * val / ymax
        out.append(
            f'<rect x="{x:.1f}" y="{bottom - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}"/>'
        )
        out.append(
            f'<text x="{x + bw / 2:.1f}" y="{bottom - bh - 4:.1f}" text-anchor="middle" class="ax">{fmt.format(val)}</text>'
        )
        out.append(
            f'<text x="{x + bw / 2:.1f}" y="{bottom + 14}" text-anchor="middle" class="ax">{escape(lab)}</text>'
        )
    out.append(
        f'<text x="{w / 2}" y="{h - 4}" text-anchor="middle" class="ax">gate threshold</text>'
    )
    return f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="0 0 {w} {h}">{"".join(out)}</svg>'


CHART_CSS = (
    "<style>.ct{font:600 12px 'Segoe UI',Arial;fill:#1c2430}.ax{font:10.5px 'Segoe UI',Arial;fill:#4a5568}"
    ".gl{stroke:#dde2e8;stroke-width:1}.lg{font:11px 'Segoe UI',Arial;fill:#1c2430}</style>"
)


def gate_chart(rows):
    labels = [f"{float(r['threshold']):g}" for r in rows]
    accepted = [float(r["accepted"]) for r in rows]
    detected = [float(r["meaningful_detected"]) for r in rows]
    a = bars("Frames sent to the model (of 40)", labels, accepted, 40, "#5b8bd0")
    b = bars("Meaningful changes detected (of 4)", labels, detected, 4, "#3f8f5b")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 780 190" role="img" aria-label="Gate ablation">{CHART_CSS}'
        f'{a}<g transform="translate(400,0)">{b}</g></svg>'
    )


def scale_chart(pivot, sizes, series):
    import math

    w, h, left, right, top, bottom = 780, 300, 62, 170, 24, 46
    xs = [math.log10(s) for s in sizes]
    values = [v for name in series for v in pivot[name]]
    lo, hi = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))

    def px(s):
        return left + (math.log10(s) - xs[0]) / (xs[-1] - xs[0]) * (w - left - right)

    def py(v):
        return h - bottom - (math.log10(v) - lo) / (hi - lo) * (h - bottom - top)

    out = [CHART_CSS]
    for d in range(lo, hi + 1):
        y = py(10**d)
        out.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" class="gl"/>')
        out.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" class="ax">{10.0**d:g}</text>'
        )
    for s in sizes:
        out.append(
            f'<text x="{px(s):.1f}" y="{h - bottom + 16}" text-anchor="middle" class="ax">{s:,}</text>'
        )
    out.append(
        f'<text x="{(left + w - right) / 2}" y="{h - 8}" text-anchor="middle" class="ax">observations (log scale)</text>'
    )
    out.append(
        f'<text x="14" y="{(top + h - bottom) / 2}" transform="rotate(-90 14 {(top + h - bottom) / 2})" text-anchor="middle" class="ax">median time, ms (log scale)</text>'
    )
    colors = ["#5b8bd0", "#c25a1f", "#3f8f5b", "#7255b0", "#b8891a"]
    for k, name in enumerate(series):
        pts = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(sizes, pivot[name], strict=True))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[k]}" stroke-width="2"/>')
        for s, v in zip(sizes, pivot[name], strict=True):
            out.append(f'<circle cx="{px(s):.1f}" cy="{py(v):.1f}" r="3" fill="{colors[k]}"/>')
        ly = top + 8 + k * 18
        out.append(
            f'<line x1="{w - right + 14}" x2="{w - right + 34}" y1="{ly}" y2="{ly}" stroke="{colors[k]}" stroke-width="2"/>'
        )
        out.append(f'<text x="{w - right + 40}" y="{ly + 4}" class="lg">{escape(name)}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="Scale benchmark">{"".join(out)}</svg>'


# ---- assemble values --------------------------------------------------------------------------
def build_context():
    ev_text = (ROOT / "docs/evaluation.md").read_text(encoding="utf-8")
    rv_text = (ROOT / "docs/real_video_experiment.md").read_text(encoding="utf-8")
    T = parse_tables(ev_text)
    ctx: dict[str, str] = {}
    V: dict[str, str] = {}

    # cross-check the committed snapshot against a fresh deterministic, offline run
    from drone_agent.eval.runner import run_all

    fresh = run_all()["experiments"]
    gate = first(T, "Frame-gate threshold ablation")
    fresh_gate = fresh["gate"]["measured"]
    assert [int(r["accepted"]) for r in gate] == [s["accepted"] for s in fresh_gate["sweep"]], (
        "gate snapshot differs from a fresh run"
    )
    rules = first(T, "Rule evaluation on a labelled scenario matrix")
    fresh_rules = fresh["rules"]["measured"]["per_rule"]
    for r in rules:
        assert (
            int(r["tp"]) == fresh_rules[r["rule"]]["tp"]
            and int(r["fn"]) == fresh_rules[r["rule"]]["fn"]
        ), "rules snapshot differs"
    adv = first(T, "Citation-gate adversarial suite")
    assert (
        sum(1 for r in adv if r["correct"] == "True")
        == fresh["citation_adversarial"]["measured"]["correct"]
    ), "adversarial snapshot differs"

    # -- gate
    default = next(r for r in gate if float(r["threshold"]) == 6.0)
    off = next(r for r in gate if float(r["threshold"]) == 0.0)
    low = next(r for r in gate if float(r["threshold"]) == 0.5)
    V |= {
        "gate_default_accepted": default["accepted"],
        "gate_default_pct": default["call_reduction_pct"],
        "gate_default_detected": default["meaningful_detected"],
        "gate_off_accepted": off["accepted"],
        "gate_low_accepted": low["accepted"],
        "gate_low_pct": low["call_reduction_pct"],
        "gate_low_detected": low["meaningful_detected"],
        "gate_irrelevant": default["irrelevant_accepted"],
    }
    ms = fresh_gate["transition_scores"]
    score = {t["name"]: t["score_vs_previous_frame"] for t in ms}
    V |= {
        "noise_max": f"{fresh_gate['static_frame_score']['max']:.3f}",
        "s_small": f"{score['small object appears']:.3f}",
        "s_medium": f"{score['medium object appears']:.3f}",
        "s_p8": f"{score['illumination step +8']:.3f}",
        "s_p25": f"{score['illumination step +25']:.3f}",
        "s_large_in": f"{score['large object appears']:.3f}",
        "s_large_out": f"{score['large object leaves']:.3f}",
    }
    ctx["TBL_GATE"] = table(
        [
            "threshold",
            "accepted",
            "skipped",
            "call_reduction_pct",
            "meaningful_detected",
            "meaningful_missed",
            "meaningful_recall",
            "irrelevant_accepted",
        ],
        gate,
        "Frame-gate threshold sweep (controlled synthetic fixture: 40 candidate frames, 4 meaningful and 2 illumination-only changes)",
        "num",
    )
    ctx["CHART_GATE"] = gate_chart(gate)
    ctx["TBL_GATE_SCORES"] = table(
        ["change", "meaningful", "score vs previous frame"],
        [
            {
                "change": t["name"],
                "meaningful": "yes" if t["meaningful"] else "no (illumination only)",
                "score vs previous frame": f"{t['score_vs_previous_frame']:.3f}",
            }
            for t in ms
        ],
        "Gate score of each known change (a static frame scores at most " + V["noise_max"] + ")",
        "num",
    )
    # -- compression
    comp = first(T, "Temporal compression: observations to events")
    ctx["TBL_COMP"] = table(
        ["fixture", "observations", "detected_objects", "events", "compression_ratio"],
        comp,
        "Observations to events (controlled scripted and generated fixtures)",
        "num",
    )
    sec = next(r for r in comp if r["fixture"] == "security_events")
    dwell = next(r for r in comp if r["fixture"] == "long_dwell_controlled")
    V |= {
        "sec_obs": sec["observations"],
        "sec_det": sec["detected_objects"],
        "sec_events": sec["events"],
        "sec_ratio": f"{float(sec['compression_ratio']):.2f}",
        "dwell_det": dwell["detected_objects"],
        "dwell_events": dwell["events"],
        "dwell_ratio": f"{float(dwell['compression_ratio']):g}",
    }
    # -- flicker
    fl = first(T, "Flicker-tolerance ablation (max_missed_frames)")
    ctx["TBL_FLICKER"] = table(
        [
            "fixture",
            "max_missed_frames",
            "person_events",
            "durations_s",
            "loitering_alerts",
            "loitering_severities",
        ],
        fl,
        "Gap sweep: one person seen for 4 frames, a gap of N undetected frames, then 4 more frames (controlled fixture)",
        "num",
    )
    ep = fresh["flicker"]["measured"]["loitering_episode"]
    V |= {
        "fl_def_events": ep["max_missed_frames=2"]["person_events"],
        "fl_def_alerts": ep["max_missed_frames=2"]["loitering_alerts"],
        "fl_off_events": ep["max_missed_frames=0"]["person_events"],
        "fl_off_alerts": ep["max_missed_frames=0"]["loitering_alerts"],
        "fl_def_dur": ep["max_missed_frames=2"]["durations_s"][0],
        "fl_off_dur": " + ".join(f"{d:g}" for d in ep["max_missed_frames=0"]["durations_s"]),
    }
    # -- entity
    ent = first(T, "Coarse vehicle entity memory ablation and negative controls")
    settings = [r for r in ent if r["setting"] in ("linking on", "linking off")]
    controls = [r for r in ent if r["setting"].startswith("control: ")]
    ctx["TBL_ENTITY_SETTINGS"] = table(
        [
            "setting",
            "vehicle_events",
            "vehicle_entities",
            "multi_appearance_entities",
            "repeat_vehicle_alerts",
        ],
        settings,
        "security_events with and without coarse entity linking (controlled scripted scenario)",
        "num",
    )
    ctx["TBL_ENTITY_CONTROLS"] = table(
        ["control", "linked", "documented contract respected", "false link"],
        [
            {
                "control": r["setting"][len("control: ") :],
                "linked": r["actual_linked"],
                "documented contract respected": r["contract_respected"],
                "false link": r["false_link"],
            }
            for r in controls
        ],
        "Vehicle-linking controls (controlled synthetic sightings)",
    )
    V |= {
        "ent_ctrl_total": str(len(controls)),
        "ent_ctrl_ok": str(sum(1 for r in controls if r["contract_respected"] == "True")),
        "ent_false": str(sum(1 for r in controls if r["false_link"] == "True")),
        "ent_base_entities": settings[0]["vehicle_entities"],
        "ent_base_alerts": settings[0]["repeat_vehicle_alerts"],
        "ent_off_alerts": settings[1]["repeat_vehicle_alerts"],
    }
    # -- rules
    ctx["TBL_RULES"] = table(
        ["rule", "tp", "fp", "tn", "fn", "precision", "recall", "f1"],
        rules,
        "Rule agreement with hand-written labels (controlled synthetic scenarios only)",
        "num",
    )
    n_scen = len(fresh["rules"]["measured"]["scenarios"])
    V["rule_scenarios"] = str(n_scen)
    # -- test inventory, counted from the collected suite rather than typed by hand
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    test_ids = [line for line in collected if "::" in line]
    V["n_tests"] = str(len(test_ids))
    V["n_test_modules"] = str(len({line.split("::")[0] for line in test_ids}))
    by_module: dict[str, int] = {}
    for line in test_ids:
        stem = Path(line.split("::")[0]).stem
        by_module[stem] = by_module.get(stem, 0) + 1
    groups = [
        (
            "Domain and simulation",
            [
                "test_models",
                "test_clock",
                "test_telemetry_simulator",
                "test_sync",
                "test_scenario",
                "test_simulate",
            ],
            "strict immutable models, timezone-aware time, seeded determinism, midnight crossing, dropouts and the stale-telemetry boundary, scenario validation",
        ),
        (
            "Storage and indexing",
            ["test_store", "test_store_v4", "test_store_v5", "test_ingest"],
            "schema and foreign keys, exact round trips, timezone-safe ordering, filters, FTS5 ranking and safe search text, atomic ingest, in-place migrations, CLI queries",
        ),
        (
            "Events, entities, observed departure",
            [
                "test_aggregator",
                "test_context_engine",
                "test_build_events",
                "test_departure",
                "test_linker",
                "test_linker_scoring",
                "test_recurrence",
            ],
            "evidence and duration, missed-frame tolerance, gap and zone splits, incremental equals batch, a close reason is not a departure, coarse linking, plates, people never linked, idempotent rebuilds",
        ),
        (
            "Rules, safety and lifecycle",
            ["test_rules", "test_analyze", "test_safety_rules", "test_lifecycle"],
            "inclusive thresholds, after-hours window edges, repeat vehicle, restricted zone, battery, signal, altitude, geofence, escalation of the same alert, operator actions, illegal transitions refused",
        ),
        (
            "Progressive runtime",
            ["test_runtime", "test_sampling", "test_perception", "test_cli_run"],
            "tick ordering and pacing, alerts persisted mid-run, streaming equals batch, sampling reasons, perception timeout and fallback, the run command",
        ),
        (
            "Video, providers, resilience",
            [
                "test_video",
                "test_cassette",
                "test_video_ingest",
                "test_gemini",
                "test_video_live",
                "test_video_resilience",
                "test_real_clip_evidence",
            ],
            "unusable files, sampling, gate, carry bound, cassette record/replay/miss/mismatch, strict provider output, Gemini key handling and retries (SDK mocked), progressive equals batch, classified failures, timeouts",
        ),
        (
            "Analyst and grounding",
            [
                "test_agent_core",
                "test_agent_llm",
                "test_agent_truncation",
                "test_analyst_m12",
                "test_factcheck",
                "test_tool_contract",
            ],
            "bounded read-only tools, citation validation, the factual-claim check, retry then fallback, sessions, one tool-name contract, LangChain adapter with a fake chat model, truncation notices",
        ),
        (
            "Evaluation harness",
            [
                "test_ground_truth",
                "test_eval_metrics",
                "test_sampler_noise_eval",
                "test_eval",
                "test_scale_bench",
            ],
            "determinism, metric maths and zero denominators, label-satisfied versus exact, one-to-one matching, experiment mechanics, no network or key",
        ),
        (
            "Dashboard server, hosting, real demo",
            [
                "test_web_api",
                "test_web_bundle",
                "test_web_guard",
                "test_web_server",
                "test_web_public",
                "test_web_real_demo",
            ],
            "read-only API, localhost guard, static bundle, frame images, the question guard, the opt-in public hostname, the bundled clip and cassette checked against their manifest and replayed with no key and no network",
        ),
        (
            "Dashboard UI",
            ["test_web_ui", "test_web_ui_browser"],
            "required structure, no inline styles, WCAG AA contrast, no hardcoded results; a real headless browser compares the page with the API and checks interactions, errors, keyboard use and five viewport sizes",
        ),
        (
            "Repository rules",
            ["test_scaffold", "test_repo_layout"],
            "core packages never import LangChain, repository layout",
        ),
    ]
    covered = sorted(m for _, mods, _ in groups for m in mods)
    assert covered == sorted(by_module), (
        f"test modules missing from the report's groups: {set(covered) ^ set(by_module)}"
    )
    ctx["TBL_TESTS"] = table(
        ["Area", "Modules", "Tests", "What is verified"],
        [
            {
                "Area": name,
                "Modules": len(mods),
                "Tests": sum(by_module[m] for m in mods),
                "What is verified": what,
            }
            for name, mods, what in groups
        ],
        f"Test organisation, counted from the collected suite ({len(test_ids)} tests in {len(by_module)} modules)",
        "",
        {"Modules": "num", "Tests": "num"},
    )
    V["n_ui_tests"] = str(by_module["test_web_ui"] + by_module["test_web_ui_browser"])
    V["n_browser_tests"] = str(by_module["test_web_ui_browser"])
    # -- streaming ground truth and adaptive sampling (tables parsed from the committed snapshot,
    # cross-checked against the fresh run so a stale snapshot cannot be rendered)
    gt = first(T, "Ground-truth scenarios: precision, recall, F1, severity and detection latency")
    fresh_gt = fresh["ground_truth"]["measured"]["aggregate"]
    assert sum(int(r["tp"]) for r in gt) == fresh_gt["tp"], "ground-truth snapshot differs"
    assert sum(int(r["fp"]) + int(r["fn"]) for r in gt) == fresh_gt["fp"] + fresh_gt["fn"], (
        "ground-truth snapshot differs"
    )
    V["gt_scenarios"] = str(len(gt))
    V["gt_tp"] = str(fresh_gt["tp"])
    V["gt_exact"] = str(sum(1 for r in gt if r["exact"] == "True"))
    V["gt_satisfied"] = str(sum(1 for r in gt if r["label_satisfied"] == "True"))
    V["gt_alerts"] = str(sum(int(r["expected_alerts"]) for r in gt))
    ctx["TBL_GT"] = table(
        [
            "scenario",
            "expected_alerts",
            "tp",
            "fp",
            "fn",
            "f1",
            "forbidden_hits",
            "events",
            "events_extra",
            "label_satisfied",
            "exact",
        ],
        gt,
        "Streaming run against author-written labels (controlled synthetic scenarios only)",
        "num",
    )
    samp = first(T, "Adaptive sampling versus a pixel gate: calls saved AND changes missed")
    fresh_samp = fresh["sampler"]["measured"]["policies"]
    assert len(samp) == len(fresh_samp), "sampler snapshot differs"
    ctx["TBL_SAMPLER"] = table(
        [
            "policy",
            "calls_mean",
            "avoided_pct_mean",
            "recall",
            "calls_per_detected",
            "missed_small",
        ],
        samp,
        "Model calls against changes detected (8 generated 600-frame streams)",
        "num",
    )
    # -- adversarial
    cats: dict[str, list] = {}
    for r in adv:
        cats.setdefault(r["category"], []).append(r)
    ctx["TBL_ADV"] = table(
        ["category", "cases", "expected", "correct decisions"],
        [
            {
                "category": c,
                "cases": len(v),
                "expected": v[0]["expected"],
                "correct decisions": sum(1 for r in v if r["correct"] == "True"),
            }
            for c, v in cats.items()
        ],
        "Citation-gate decisions on the authored adversarial suite",
        "num",
    )
    V |= {
        "adv_cases": str(len(adv)),
        "adv_correct": str(sum(1 for r in adv if r["correct"] == "True")),
        "adv_reject": str(sum(1 for r in adv if r["expected"] == "reject")),
        "adv_accept": str(sum(1 for r in adv if r["expected"] == "accept")),
    }
    # -- semantic probe
    pr = first(T, "Citation-valid / semantically-unsupported probe (+ claim-evidence checker)")
    pcats: dict[str, list] = {}
    for r in pr:
        pcats.setdefault(r["category"], []).append(r)
    rows = []
    for c, v in pcats.items():
        rows.append(
            {
                "category": c,
                "answers": len(v),
                "claims true": "yes" if v[0]["claims_supported_by_evidence"] == "True" else "no",
                "accepted by citation check alone": sum(
                    1 for r in v if r["gate_accepts"] == "True"
                ),
                "accepted by production gate": sum(
                    1 for r in v if r["production_gate_accepts"] == "True"
                ),
            }
        )
    ctx["TBL_PROBE"] = table(
        [
            "category",
            "answers",
            "claims true",
            "accepted by citation check alone",
            "accepted by production gate",
        ],
        rows,
        "Citation-valid answers, by category (authored from the evidence model; production gate = citation check + factual-claim check)",
        "num",
    )
    unsup = [r for r in pr if r["claims_supported_by_evidence"] != "True"]
    ctrl = [r for r in pr if r["claims_supported_by_evidence"] == "True"]
    expressible = [r for r in unsup if r["structured_claims"] != "0"]
    V |= {
        "probe_unsup": str(len(unsup)),
        "probe_gate_acc": str(sum(1 for r in unsup if r["gate_accepts"] == "True")),
        "probe_ctrl": str(len(ctrl)),
        "probe_ctrl_acc": str(sum(1 for r in ctrl if r["gate_accepts"] == "True")),
        "probe_expr": str(len(expressible)),
        "probe_caught": str(
            sum(1 for r in expressible if r["gate_plus_checker_accepts"] != "True")
        ),
        "probe_still": str(sum(1 for r in unsup if r["gate_plus_checker_accepts"] == "True")),
        "probe_prod_acc": str(sum(1 for r in unsup if r["production_gate_accepts"] == "True")),
        "probe_prod_rej": str(sum(1 for r in unsup if r["production_gate_accepts"] != "True")),
        "probe_ctrl_prod_acc": str(sum(1 for r in ctrl if r["production_gate_accepts"] == "True")),
        "probe_expr_prod_rej": str(
            sum(1 for r in expressible if r["production_gate_accepts"] != "True")
        ),
    }
    # which unsupported answers the production gate still accepts, and what it rejects (from the table)
    accepted = [r for r in unsup if r["production_gate_accepts"] == "True"]
    rejected = [r for r in unsup if r["production_gate_accepts"] != "True"]
    rej_by_cat: dict[str, int] = {}
    for r in rejected:
        rej_by_cat[r["category"]] = rej_by_cat.get(r["category"], 0) + 1
    ctx["LIST_PROBE_ACCEPTED"] = (
        "<ul>"
        + "".join(
            f"<li><code>{escape(r['category'])}</code>: {escape(r['case'])}</li>" for r in accepted
        )
        + "</ul>"
    )
    ctx["LIST_PROBE_REJECTED"] = (
        "<ul>"
        + "".join(f"<li><code>{escape(c)}</code>: {n}</li>" for c, n in rej_by_cat.items())
        + "</ul>"
    )
    # -- ablation
    ab = first(T, "Ablation table")
    ctx["TBL_ABLATION"] = table(
        [
            "component",
            "metric",
            "baseline_setting",
            "baseline",
            "ablated_setting",
            "ablated",
            "effect",
        ],
        ab,
        "Ablation summary (every value computed by the harness)",
        "abl",
    )
    # -- scale
    sc = first(T, "Scale benchmark (generated observations)")
    sizes = sorted({int(r["observations"]) for r in sc})
    stages = list(dict.fromkeys(r["stage"] for r in sc))
    pivot = {
        s: [
            float(next(r["ms"] for r in sc if r["stage"] == s and int(r["observations"]) == n))
            for n in sizes
        ]
        for s in stages
    }
    ctx["TBL_SCALE"] = table(
        ["stage"] + [f"{n:,}" for n in sizes],
        [
            {"stage": s, **{f"{n:,}": f"{v:g}" for n, v in zip(sizes, pivot[s], strict=True)}}
            for s in stages
        ],
        "Median time in milliseconds by number of generated observations (SQLite ingest and derived-table writes are single runs)",
        "num",
    )
    ctx["CHART_SCALE"] = scale_chart(
        pivot,
        sizes,
        [
            "event_aggregation",
            "entity_linking",
            "rule_evaluation",
            "query_list_alerts",
            "sqlite_ingest",
        ],
    )
    m = re.search(
        r"event_aggregation ([0-9.]+), entity_linking ([0-9.]+), rule_evaluation ([0-9.]+)", ev_text
    )
    V |= {"exp_agg": m.group(1), "exp_link": m.group(2), "exp_rules": m.group(3)}
    env = dict(
        re.findall(r"^- (python|platform|processor|cpu_count|sqlite_version): (.+)$", ev_text, re.M)
    )
    ctx["SCALE_ENV"] = table(
        ["property", "value"],
        [{"property": k, "value": v} for k, v in env.items()],
        "Measurement environment (this machine only)",
    )
    V |= {
        "p50k_agg": f"{pivot['event_aggregation'][-1]:g}",
        "p50k_link": f"{pivot['entity_linking'][-1]:g}",
        "p50k_rules": f"{pivot['rule_evaluation'][-1]:g}",
        "p50k_fts": f"{pivot['query_fts_keyword_top10'][-1]:g}",
        "p100_agg": f"{pivot['event_aggregation'][0]:g}",
    }

    # -- real video: replay the BUNDLED clip and cassette (no model, no network) and assert that the
    # committed documents say exactly what the replay produces
    from drone_agent.store import Store
    from drone_agent.web.bundle import build_real_run, load_real_assets

    manifest = load_real_assets()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        run = build_real_run(tmp)
        with Store.open(run.db) as store:
            obs = store.query_observations()
            events = store.query_events()
            entities = store.list_entities()
            n_alerts = store.count_alerts()
    carried = sum(1 for o in obs if o.source.endswith("+carried"))
    objs = [d.object_type.value for o in obs for d in o.detected_objects]
    single = sum(1 for e in events if len(e.frame_ids) == 1)
    clip = manifest["clip"]
    V |= {
        "rv_bytes": f"{clip['size_bytes']:,}",
        "rv_frames": str(clip["frame_count"]),
        "rv_fps": f"{clip['fps']:g}",
        "rv_seconds": f"{clip['duration_seconds']:g}",
        "rv_size": f"{clip['width']}&times;{clip['height']}",
        "rv_sha": clip["sha256"][:16],
        "rv_model": manifest["cassette"]["provider"],
        "rv_model_name": manifest["cassette"]["provider"].split(":")[-1].replace("-", " ").title(),
        "rv_entries": str(manifest["cassette"]["entries"]),
        "rv_obs": str(len(obs)),
        "rv_replayed": str(len(obs) - carried),
        "rv_carried": str(carried),
        "rv_objects": str(len(objs)),
        "rv_obj_person": str(objs.count("person")),
        "rv_obj_vehicle": str(objs.count("vehicle")),
        "rv_events": str(len(events)),
        "rv_ev_person": str(sum(1 for e in events if e.object_type.value == "person")),
        "rv_ev_vehicle": str(sum(1 for e in events if e.object_type.value == "vehicle")),
        "rv_single": str(single),
        "rv_entities": str(len(entities)),
        "rv_alerts": str(n_alerts),
        "rv_max_event": f"{max(e.duration_seconds for e in events):g}",
        "rv_start": manifest["replay"]["start"],
    }
    assert (V["rv_obs"], V["rv_replayed"], V["rv_carried"], V["rv_objects"]) == (
        "9",
        "8",
        "1",
        "33",
    )
    assert (V["rv_events"], V["rv_entities"], V["rv_alerts"]) == ("28", "12", "0")
    for needle in [
        "10,797,538",
        "617 frames",
        "24.68 s",
        "d7e3ceeba55902be",
        f"{V['rv_events']} ({V['rv_ev_person']} person, {V['rv_ev_vehicle']} vehicle; {V['rv_single']} single-frame)",
        f"{V['rv_entities']} (1 with several appearances)",
        "9 (8 replayed, 1 carried forward at frame 525)",
        f"33 ({V['rv_obj_person']} person, {V['rv_obj_vehicle']} vehicle)",
        "**25** (13 person, 12 vehicle)",  # the first recording, section 5 of the document
        "**11** coarse",
        "licensed for redistribution",
    ]:
        assert needle in rv_text, f"real-video document no longer contains: {needle}"
    # the README and the dashboard document must state the same replay result
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for needle in ["10,797,538 bytes", "28 events, 12 entities, 0 alerts", "1301 tests"]:
        assert needle in readme, f"README no longer contains: {needle}"
    assert V["n_tests"] == "1301", f"the documents say 1301 tests, the suite has {V['n_tests']}"

    # -- the shipped scripted scenarios: counts computed by running the existing CLI commands
    from drone_agent.web.bundle import _cli

    def scripted(name: str) -> dict:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = str(Path(tmp) / f"{name}.db")
            _cli("ingest", str(ROOT / "scenarios" / f"{name}.yaml"), "--db", db)
            _cli("build-events", "--db", db)
            _cli("analyze", "--db", db)
            with Store.open(db) as store:
                return {
                    "obs": store.count_observations(),
                    "objects": sum(len(o.detected_objects) for o in store.query_observations()),
                    "events": store.count_events(),
                    "entities": store.count_entities(),
                    "alerts": store.count_alerts(),
                }

    sec_counts, normal_counts = scripted("security_events"), scripted("normal_day")
    assert (str(sec_counts["obs"]), str(sec_counts["objects"]), str(sec_counts["events"])) == (
        V["sec_obs"],
        V["sec_det"],
        V["sec_events"],
    ), "the compression table and the scenario replay disagree"
    V |= {
        "sec_entities": str(sec_counts["entities"]),
        "sec_alerts": str(sec_counts["alerts"]),
        "normal_alerts": str(normal_counts["alerts"]),
    }
    assert (V["sec_alerts"], V["sec_entities"], V["normal_alerts"]) == ("7", "2", "0")

    # -- the committed scale report (previously measured, one machine): quoted, never re-run here
    bench = (ROOT / "bench" / "scale_report.md").read_text(encoding="utf-8")

    def bench_row(label: str) -> list[str]:
        m = re.search(rf"^\| {re.escape(label)} \| ([^|]+) \| ([^|]+) \|", bench, re.M)
        assert m, f"bench/scale_report.md has no row {label!r}"
        return [c.strip() for c in m.groups()]

    summ = re.search(r"tool_summarize_run \| ([0-9.]+) / ([0-9.]+) \| ([0-9.]+) / ([0-9.]+)", bench)
    link = re.search(r"- seconds\.link: ([0-9.]+)", bench)
    machine = re.search(r"^Machine: (.+)$", bench, re.M)
    assert summ and link and machine, "bench/scale_report.md changed shape"
    V |= {
        "bench_tput_10k": bench_row("ingest throughput (obs/s)")[0],
        "bench_tput_100k": bench_row("ingest throughput (obs/s)")[1],
        "bench_db_100k": bench_row("DB size (MiB)")[1],
        "bench_mem_100k": bench_row("peak memory (MiB)")[1],
        "bench_sum_p50_100k": summ.group(3),
        "bench_sum_p95_100k": summ.group(4),
        "bench_link_exp": link.group(1),
        "bench_machine": machine.group(1).split(". measurements")[0].rstrip("."),
    }

    # -- cover facts and the dashboard figure
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    V["built_on"] = head or "unknown"
    changed = [
        ln
        for ln in subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.splitlines()
        if not ln.endswith("plan.md")
    ]
    V["built_state"] = (
        f"plus {len(changed)} uncommitted file changes" if changed else "working tree clean"
    )
    V["build_date"] = f"{date.today().day} {date.today():%B %Y}"
    png = (REPORT / "diagrams" / "dashboard_scripted.png").read_bytes()
    ctx["IMG_DASHBOARD"] = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    return ctx, V


def render(template: str, ctx: dict, V: dict) -> str:
    for key, value in ctx.items():
        template = template.replace("{{" + key + "}}", value)
    for key, value in V.items():
        template = template.replace("{{V." + key + "}}", str(value))
    for name in ("architecture", "analyst", "datamodel"):
        svg = (REPORT / "diagrams" / f"{name}.svg").read_text(encoding="utf-8")
        template = template.replace("{{SVG_" + name.upper() + "}}", svg)
    left = sorted(set(re.findall(r"\{\{[^}]+\}\}", template)))
    assert not left, f"unfilled placeholders: {left}"
    return template


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    profile = Path(__import__("tempfile").mkdtemp(prefix="edge-report-"))
    cmd = [
        EDGE,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        f"--user-data-dir={profile}",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    assert pdf_path.exists() and pdf_path.stat().st_size > 10_000, "PDF was not produced"


def verify_pdf(html: str, pdf: Path) -> dict:
    """Checks on the produced file: it is a PDF, how many pages it has, nothing is embedded in it, and
    the confidential assignment document is neither tracked, referenced nor named."""
    data = pdf.read_bytes()
    assert data.startswith(b"%PDF-"), "not a PDF"
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data)]
    pages = max(counts) if counts else len(re.findall(rb"/Type\s*/Page\b", data))
    assert pages >= 10, f"suspiciously short report: {pages} pages"
    for marker in (b"/EmbeddedFile", b"/Filespec", b"/FileAttachment"):
        assert marker not in data, f"the PDF embeds a file ({marker.decode()})"
    assert "assignment" not in pdf.name.lower(), (
        "the report file name must not mention the assignment"
    )
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True
    ).stdout.lower()
    assert not [f for f in tracked.splitlines() if "assignment" in f and f.endswith(".pdf")], (
        "the confidential assignment PDF is tracked by git"
    )
    low = html.lower()
    for tag in ("<embed", "<object", "<iframe", "<link ", 'src="http', "src='http"):
        assert tag not in low, f"the report HTML loads or embeds something external: {tag}"
    assert "assignment.pdf" not in low and "assignment .pdf" not in low
    return {"pages": pages, "bytes": len(data)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    ctx, V = build_context()
    html = render((REPORT / "report_template.html").read_text(encoding="utf-8"), ctx, V)
    out = REPORT / "report.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out, len(html), "characters")
    if not args.no_pdf:
        pdf = REPORT / PDF_NAME
        to_pdf(out, pdf)
        print("wrote", pdf, pdf.stat().st_size, "bytes")
        print("verified:", verify_pdf(html, pdf))


if __name__ == "__main__":
    main()
