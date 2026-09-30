"""Build the final report: report/report_template.html -> report/report.html -> PDF (headless Edge).

Every table and chart is generated from docs/evaluation.md (the committed evaluation snapshot); the
functional results are cross-checked against a fresh run of the deterministic, offline harness. Real-video
numbers are asserted against docs/real_video_experiment.md. No API call and no network access.

Run:  python report/make_diagrams.py && python report/build_report.py [--no-pdf]
"""

import argparse
import re
import subprocess
import sys
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
    head = "".join(f'<th class="{align.get(h, "")}">{escape(str(h))}</th>' for h in header)
    body = ""
    for row in rows:
        cells = "".join(
            f'<td class="{align.get(h, "")}">{escape(str(row.get(h, "")))}</td>' for h in header
        )
        body += f"<tr>{cells}</tr>"
    cap = f"<caption>{escape(caption)}</caption>" if caption else ""
    return f'<table class="data {cls}">{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def bars(title, labels, values, ymax, color, w=380, h=190, fmt="{:g}"):
    left, bottom, top = 40, h - 42, 26
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
        ["fixture", "max_missed_frames", "person_events", "durations_s", "loitering_alerts"],
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
        "ent_base_entities": settings[0]["vehicle_entities"],
        "ent_base_alerts": settings[0]["repeat_vehicle_alerts"],
        "ent_off_alerts": settings[1]["repeat_vehicle_alerts"],
    }
    # -- rules
    ctx["TBL_RULES"] = table(
        ["rule", "tp", "fp", "tn", "fn", "precision", "recall"],
        rules,
        "Rule agreement with hand-written labels (controlled synthetic scenarios only)",
        "num",
    )
    n_scen = len(fresh["rules"]["measured"]["scenarios"])
    V["rule_scenarios"] = str(n_scen)
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
                "accepted by citation gate": sum(1 for r in v if r["gate_accepts"] == "True"),
                "accepted by gate + structured checker": sum(
                    1 for r in v if r["gate_plus_checker_accepts"] == "True"
                ),
            }
        )
    ctx["TBL_PROBE"] = table(
        [
            "category",
            "answers",
            "claims true",
            "accepted by citation gate",
            "accepted by gate + structured checker",
        ],
        rows,
        "Citation-valid answers, by category (authored from the evidence model)",
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
    }
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

    # -- real video: numbers asserted against the committed document
    for needle in [
        "10,797,538",
        "617 frames",
        "24.68 s",
        "8 accepted, 1 skipped",
        "8 API calls, 0 retries, 76.1 s",
        "9 observations",
        "33 detected objects",
        "(17 person, 16 vehicle)",
        "**25** (13 person, 12 vehicle)",
        "**11** coarse",
        "**0**",
        "1.32",
        "d7e3ceeba55902be",
    ]:
        assert needle in rv_text, f"real-video document no longer contains: {needle}"
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


if __name__ == "__main__":
    main()
