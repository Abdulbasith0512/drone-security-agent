"""Draw the report's three diagrams as static SVG files (no dependencies, deterministic).

Run:  python report/make_diagrams.py     (writes report/diagrams/*.svg)
"""

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parent / "diagrams"

STYLE = """
<style>
 text{font-family:'Segoe UI',Arial,sans-serif;font-size:12.5px;fill:#1c2430}
 .t{font-weight:600}
 .s{font-size:11px;fill:#4a5568}
 .lbl{font-size:11px;fill:#2d3748}
 .in{fill:#e8f1fb;stroke:#4a78b5}
 .pr{fill:#e6f4ea;stroke:#3f8f5b}
 .st{fill:#fff6dc;stroke:#b8891a}
 .an{fill:#efeafa;stroke:#7255b0}
 .gt{fill:#fde9df;stroke:#c25a1f}
 .box rect,.box polygon{stroke-width:1.3}
 .grp{fill:#fafafa;stroke:#9aa5b1;stroke-dasharray:5 4;stroke-width:1.2}
 .edge{fill:none;stroke:#3b4655;stroke-width:1.4}
 .dash{fill:none;stroke:#3b4655;stroke-width:1.2;stroke-dasharray:4 3}
</style>
<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#3b4655"/></marker></defs>
"""


def lines_svg(cx, cy, lines, bold_first=False):
    n = len(lines)
    y0 = cy - (n - 1) * 8 + 4
    parts = []
    for i, line in enumerate(lines):
        cls = ' class="t"' if (bold_first and i == 0) else ""
        parts.append(f'<tspan x="{cx}" y="{y0 + i * 16}"{cls}>{escape(line)}</tspan>')
    return f'<text text-anchor="middle">{"".join(parts)}</text>'


def box(x, y, w, h, lines, cls="pr", bold_first=False):
    return (
        f'<g class="box"><rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        f"{lines_svg(x + w / 2, y + h / 2, lines, bold_first)}</g>"
    )


def diamond(cx, cy, w, h, lines, cls="gt"):
    pts = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
    return f'<g class="box"><polygon class="{cls}" points="{pts}"/>{lines_svg(cx, cy, lines)}</g>'


def edge(points, label=None, lx=None, ly=None, dashed=False, both=False):
    d = "M" + " L".join(f"{x},{y}" for x, y in points)
    cls = "dash" if dashed else "edge"
    start = ' marker-start="url(#ah)"' if both else ""
    out = f'<path class="{cls}" d="{d}" marker-end="url(#ah)"{start}/>'
    if label:
        out += f'<text class="lbl" x="{lx}" y="{ly}" text-anchor="middle">{escape(label)}</text>'
    return out


def group(x, y, w, h, title, left=False):
    tx, anchor = (x + 10, "start") if left else (x + w - 10, "end")
    return (
        f'<rect class="grp" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
        f'<text class="s" x="{tx}" y="{y + 15}" text-anchor="{anchor}">{escape(title)}</text>'
    )


def svg(width, height, body, title):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(title)}">{STYLE}{body}</svg>'
    )


def architecture() -> str:
    b = []
    b.append(
        group(
            4,
            4,
            792,
            250,
            "Inputs and perception (the patrol engine and ingest-video --live process them frame by frame)",
        )
    )
    # scripted path
    b.append(box(16, 30, 128, 44, ["Scripted scenario", "(text frames, YAML)"], "in"))
    b.append(box(176, 30, 128, 44, ["ScriptedProvider"], "pr"))
    b.append(edge([(144, 52), (176, 52)]))
    # video path
    b.append(box(16, 100, 128, 44, ["Video file", "(decoded with OpenCV)"], "in"))
    b.append(box(176, 100, 116, 44, ["Deterministic", "sampler"], "pr"))
    b.append(box(322, 100, 116, 44, ["Frame-change gate", "(pixel difference)"], "gt"))
    b.append(
        box(468, 96, 148, 52, ["Vision provider:", "Gemini (optional)", "or cassette replay"], "pr")
    )
    b.append(edge([(144, 122), (176, 122)]))
    b.append(edge([(292, 122), (322, 122)]))
    b.append(edge([(438, 122), (468, 122)], "accepted", 453, 90))
    # telemetry path
    b.append(box(16, 180, 128, 44, ["Simulated telemetry", "and timeline"], "in"))
    b.append(box(176, 180, 150, 44, ["Nearest-prior", "telemetry association"], "pr"))
    b.append(edge([(144, 202), (176, 202)]))
    # observation
    b.append(box(650, 100, 140, 70, ["Observation", "+ detected objects"], "pr", True))
    b.append(edge([(304, 52), (632, 52), (632, 112), (650, 112)]))
    b.append(edge([(616, 130), (650, 130)]))
    b.append(
        edge(
            [(380, 144), (380, 170), (628, 170), (628, 150), (650, 150)],
            "skipped: carried forward",
            505,
            166,
        )
    )
    b.append(edge([(326, 202), (642, 202), (642, 160), (650, 160)]))
    # memory
    b.append(
        group(
            4,
            270,
            792,
            138,
            "Shared consumer (ObservationPipeline): SQLite + FTS5 index, then events, entities, rules, alerts",
        )
    )
    b.append(box(16, 300, 130, 70, ["frames, telemetry,", "observations,", "objects + FTS5"], "st"))
    b.append(box(168, 306, 104, 58, ["Event", "aggregator"], "pr"))
    b.append(box(294, 306, 84, 58, ["events"], "st"))
    b.append(box(400, 306, 96, 58, ["Entity linker", "(vehicles)"], "pr"))
    b.append(box(518, 306, 90, 58, ["entities"], "st"))
    b.append(box(630, 306, 84, 58, ["Rule engine", "(8 rules)"], "pr"))
    b.append(box(724, 306, 66, 58, ["alerts"], "st"))
    b.append(edge([(720, 170), (720, 250), (81, 250), (81, 300)]))
    b.append(edge([(146, 335), (168, 335)]))
    b.append(edge([(272, 335), (294, 335)]))
    b.append(edge([(378, 335), (400, 335)]))
    b.append(edge([(496, 335), (518, 335)]))
    b.append(edge([(608, 335), (630, 335)]))
    b.append(edge([(714, 335), (724, 335)]))
    b.append(edge([(336, 364), (336, 390), (672, 390), (672, 364)]))
    # analyst
    b.append(group(4, 424, 792, 132, "Analyst (read-only access to the stored evidence)"))
    b.append(box(16, 452, 176, 60, ["Bounded read-only tools", "(13, no SQL access)"], "an"))
    b.append(box(224, 452, 206, 60, ["Offline template analyst", "or LangChain + Gemini"], "an"))
    b.append(box(462, 452, 132, 60, ["Citation gate +", "factual-claim check"], "gt"))
    b.append(box(626, 452, 164, 60, ["Cited answer", "or run summary"], "an"))
    b.append(edge([(192, 482), (224, 482)]))
    b.append(edge([(430, 482), (462, 482)]))
    b.append(edge([(594, 482), (626, 482)]))
    b.append(edge([(104, 408), (104, 452)], "queries", 132, 434))
    # dashboard
    b.append(
        group(
            4,
            570,
            792,
            96,
            "Dashboard (read-only): opens each run's database read-only, no authentication",
            left=True,
        )
    )
    b.append(
        box(
            16, 596, 236, 56, ["Server (drone_agent/web)", "fixed routes, Host/Origin checks"], "pr"
        )
    )
    b.append(box(280, 596, 220, 56, ["Frontend (frontend/)", "plain HTML, CSS and JS"], "in"))
    b.append(
        box(
            528,
            596,
            262,
            56,
            ["Three runs: bundled real clip", "(recorded replay), scripted, synthetic"],
            "st",
        )
    )
    b.append(edge([(252, 624), (280, 624)]))
    b.append(edge([(500, 624), (528, 624)]))
    b.append(edge([(708, 512), (708, 596)], "answers", 738, 566))
    return svg(800, 676, "".join(b), "System architecture")


def analyst() -> str:
    b = []
    b.append(box(10, 78, 86, 44, ["Question"], "in"))
    b.append(box(120, 78, 130, 44, ["AnalystSession", "(last 6 turns)"], "an"))
    b.append(diamond(318, 100, 104, 70, ["LLM", "configured?"], "gt"))
    b.append(
        box(392, 68, 150, 64, ["LangChain tool loop", "(at most 6 rounds,", "12 tool calls)"], "an")
    )
    b.append(box(392, 8, 150, 36, ["13 read-only tools"], "pr"))
    b.append(diamond(612, 100, 112, 74, ["Citation gate", "+ fact check"], "gt"))
    b.append(box(690, 78, 100, 44, ["Answer shown"], "in"))
    b.append(box(268, 184, 170, 50, ["Deterministic template", "answer (labelled)"], "pr"))
    b.append(box(466, 168, 116, 44, ["One correction", "retry"], "an"))
    b.append(edge([(96, 100), (120, 100)]))
    b.append(edge([(250, 100), (266, 100)]))
    b.append(edge([(370, 100), (392, 100)], "yes", 381, 92))
    b.append(edge([(467, 44), (467, 68)], both=True))
    b.append(edge([(542, 100), (556, 100)]))
    b.append(edge([(668, 100), (690, 100)], "pass", 679, 92))
    b.append(edge([(318, 135), (318, 184)], "no", 330, 160))
    b.append(edge([(596, 136), (596, 152), (530, 152), (530, 168)], "fail (first time)", 602, 150))
    b.append(edge([(466, 190), (440, 190), (440, 132)]))
    b.append(edge([(640, 130), (640, 226), (438, 226)], "fail again or model error", 610, 240))
    b.append(edge([(353, 234), (353, 262), (740, 262), (740, 122)]))
    return svg(800, 272, "".join(b), "Analyst path with citation gate")


def datamodel() -> str:
    b = []

    def table(x, y, w, name, cols, cls="st"):
        h = 30 + 16 * len(cols)
        out = (
            f'<g class="box"><rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        )
        out += f'<text class="t" x="{x + w / 2}" y="{y + 19}" text-anchor="middle">{escape(name)}</text>'
        for i, c in enumerate(cols):
            out += f'<text class="s" x="{x + 10}" y="{y + 38 + i * 16}">{escape(c)}</text>'
        return out + "</g>", h

    t, _ = table(
        16,
        16,
        200,
        "telemetry",
        ["sample time, position, altitude", "zone, battery (simulated)"],
        "in",
    )
    b.append(t)
    t, _ = table(16, 132, 200, "frames", ["frame_id, timestamp, zone", "source"], "st")
    b.append(t)
    t, _ = table(
        300,
        70,
        210,
        "observations",
        ["one per frame", "description, zone, source", "telemetry age and stale flag"],
        "st",
    )
    b.append(t)
    t, _ = table(
        580, 16, 204, "detected_objects", ["type, activity", "attributes (JSON), confidence"], "st"
    )
    b.append(t)
    t, _ = table(
        580, 116, 204, "observations_fts (FTS5)", ["description, objects text, zone"], "st"
    )
    b.append(t)
    t, _ = table(
        300,
        250,
        210,
        "events",
        [
            "type, zone, first/last seen",
            "attributes, activities",
            "close reason",
            "departure_observed_at (v6)",
        ],
        "pr",
    )
    b.append(t)
    t, _ = table(
        16, 260, 200, "entities", ["coarse attributes", "match level (coarse/confirmed)"], "pr"
    )
    b.append(t)
    t, _ = table(
        580,
        250,
        204,
        "alerts",
        ["rule, severity, status, reason", "ladder, action, entity (optional)"],
        "gt",
    )
    b.append(t)
    b.append(edge([(216, 46), (300, 96)], "nearest prior", 250, 58))
    b.append(edge([(216, 160), (300, 130)], "1 : 1", 246, 156))
    b.append(edge([(510, 100), (580, 50)], "1 : n", 550, 68))
    b.append(edge([(510, 140), (580, 140)], "indexed by", 546, 132))
    b.append(edge([(405, 216), (405, 250)], both=True))
    b.append(text_label(413, 236, "event_observations (n : m)"))
    b.append(edge([(300, 290), (216, 290)], both=True))
    b.append(text_label(258, 282, "entity_events"))
    b.append(edge([(510, 290), (580, 290)], both=True))
    b.append(text_label(545, 282, "alert_events"))
    b.append(edge([(580, 326), (540, 372), (116, 372), (116, 340)], dashed=True))
    b.append(text_label(340, 366, "alerts may cite an entity"))
    return svg(800, 392, "".join(b), "Data model")


def text_label(x, y, s):
    return f'<text class="lbl" x="{x}" y="{y}" text-anchor="middle">{escape(s)}</text>'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("architecture", architecture),
        ("analyst", analyst),
        ("datamodel", datamodel),
    ):
        (OUT / f"{name}.svg").write_text(fn(), encoding="utf-8")
        print("wrote", OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
