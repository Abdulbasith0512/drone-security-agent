"""Large-scale benchmark: 10K / 100K / 1M generated observations through the REAL code paths.

What is real: ingestion goes through `Store.insert_observation`, events through the streaming
`EventAggregator`, entities through `EntityLinker`, alerts through `RuleEngine`, persistence
through the store, and every query goes through the store or the analyst tools the agent uses.
What is synthetic: the observations (generated, chronological, with a fleet of vehicles that
reappear) and the scene-score stream used for the sampler estimate. No model or network is used.

Each size runs in its OWN process so peak memory is that size's own. Wall-clock numbers are
measurements on the machine that ran them: not portable claims. Counts (events, alerts, entities,
sampler calls) are deterministic for a given (size, seed).

    python -m drone_agent scale --sizes 10000,100000,1000000 --output bench
"""

import argparse
import json
import math
import os
import platform
import random
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from itertools import islice
from pathlib import Path

from drone_agent.agent.tools import AnalystTools
from drone_agent.agents.sampling import AdaptiveSampler
from drone_agent.eval.base import IST, det
from drone_agent.models import Observation, Telemetry, Zone
from drone_agent.pipeline import EntityLinker
from drone_agent.pipeline.aggregator import EventAggregator
from drone_agent.rules import RuleEngine
from drone_agent.store import Store

DEFAULT_SIZES = (10_000, 100_000, 1_000_000)
SEED = 42
BATCH = 10_000  # observations per ingest transaction
TELEMETRY_EVERY = 10  # one telemetry sample per this many frames
FRAME_INTERVAL_S = 30
COLORS = ("blue", "red", "white", "black", "silver", "green")
BODIES = ("pickup", "sedan", "van", "suv")
CLOTHING = ("dark jacket", "red hoodie", "blue shirt", "grey coat", "yellow vest")
ZONES = tuple(Zone)
RESULT_MARK = "SCALE_RESULT_JSON:"
_START = datetime(2026, 9, 29, 8, 0, tzinfo=IST)


# ---- generation --------------------------------------------------------------------------------
class Stream:
    """Chronological observations, produced lazily (a million are never held at once).

    Empty stretches, people and vehicles that dwell for a few frames in a zone. Vehicles come from
    a fleet that grows with the run; about half of the sightings show a plate. Alongside each
    observation it tracks whether the scene really changed (zone or object set), which feeds the
    simulated sampler score. Fully determined by (n, seed).
    """

    def __init__(self, n: int, seed: int = SEED) -> None:
        self.n = n
        self.rng = random.Random(seed)
        self.noise = random.Random(seed + 1)  # sampler score noise, kept off the main stream
        fleet = max(20, n // 200)
        self.fleet = [
            (f"FL{i:05d}", self.rng.choice(COLORS), self.rng.choice(BODIES)) for i in range(fleet)
        ]
        self.changed: list[bool] = []  # per frame: did the scene change (ground truth for scores)
        self.scores: list[float] = []

    def __iter__(self) -> Iterator[Observation]:
        frame, previous = 0, None
        while frame < self.n:
            zone = self.rng.choice(ZONES)
            kind = self.rng.choices(("empty", "person", "vehicle"), weights=(3, 4, 3))[0]
            length = self.rng.randint(2, 8)
            obj = None
            if kind == "person":
                obj = det("person", "standing", clothing=self.rng.choice(CLOTHING))
            elif kind == "vehicle":
                plate, color, body = self.rng.choice(self.fleet)
                attrs = {"color": color, "body_type": body}
                if self.rng.random() < 0.5:
                    attrs["plate"] = plate
                obj = det("vehicle", "parked", **attrs)
            signature = (zone, kind, tuple(sorted(obj.attributes.items())) if obj else ())
            for _ in range(min(length, self.n - frame)):
                frame += 1
                changed = signature != previous
                previous = signature
                self.changed.append(changed)
                # a real pixel score is high on change and small noise otherwise
                self.scores.append(100.0 if changed else self.noise.random() * 3.0)
                yield Observation(
                    observation_id=f"obs-{frame}",
                    frame_id=frame,
                    timestamp=_START + timedelta(seconds=(frame - 1) * FRAME_INTERVAL_S),
                    source="scripted",
                    zone=zone,
                    description=f"generated frame {frame}",
                    detected_objects=[obj] if obj else [],
                )


def _telemetry(obs: Observation) -> Telemetry:
    return Telemetry(
        timestamp=obs.timestamp,
        latitude=0.0001,
        longitude=0.0001,
        altitude_m=30.0,
        zone=obs.zone,
        battery_percent=max(10.0, 100.0 - (obs.frame_id / 5000.0) % 90.0),
        signal_percent=90.0,
    )


# ---- measurement helpers -----------------------------------------------------------------------
def peak_memory_mb() -> float | None:
    """Peak resident memory of THIS process in MiB (None where the OS gives no way to read it)."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(Counters),
                wintypes.DWORD,
            ]
            ok = psapi.GetProcessMemoryInfo(
                kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
            )
            return round(counters.PeakWorkingSetSize / 2**20, 1) if ok else None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # never fail a benchmark for a missing OS facility
        return None


def percentiles(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)

    def at(q: float) -> float:
        return round(ordered[min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)], 3)

    return {
        "n": len(ordered),
        "p50_ms": at(0.5),
        "p95_ms": at(0.95),
        "max_ms": round(ordered[-1], 3),
    }


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
        "note": "measurements on this machine only; not a universal performance claim",
    }


# ---- one size, in this process ----------------------------------------------------------------
def measure_size(n: int, workdir: Path, samples: int = 60, heavy_samples: int = 8) -> dict:
    """Ingest, derive and query `n` generated observations. Returns the machine-readable report."""
    db_path = workdir / f"scale_{n}.db"
    for stale in workdir.glob(f"scale_{n}.db*"):  # every size starts from an empty database
        stale.unlink()
    stream = Stream(n)
    aggregator, sampler = EventAggregator(), AdaptiveSampler()
    events: list = []
    t = dict.fromkeys(("generate", "ingest_sqlite", "telemetry", "aggregate", "sampler"), 0.0)
    detections = 0
    with Store.open(db_path) as store:
        it = iter(stream)
        while True:
            started = time.perf_counter()
            batch = list(islice(it, BATCH))
            t["generate"] += time.perf_counter() - started
            if not batch:
                break
            started = time.perf_counter()
            with store.transaction():
                for obs in batch:
                    store.insert_observation(obs)
            t["ingest_sqlite"] += time.perf_counter() - started
            started = time.perf_counter()
            with store.transaction():
                for obs in batch:
                    if obs.frame_id % TELEMETRY_EVERY == 0:
                        store.insert_telemetry(_telemetry(obs))
            t["telemetry"] += time.perf_counter() - started
            started = time.perf_counter()
            for obs in batch:
                detections += len(obs.detected_objects)
                events.extend(aggregator.process(obs))
            t["aggregate"] += time.perf_counter() - started
            started = time.perf_counter()
            base = batch[0].frame_id - 1
            for i, obs in enumerate(batch):
                sampler.decide(
                    now=obs.timestamp,
                    zone=obs.zone,
                    scene_score=stream.scores[base + i],
                    open_events=1 if obs.detected_objects else 0,
                )
            t["sampler"] += time.perf_counter() - started
        started = time.perf_counter()
        events.extend(aggregator.flush())
        t["aggregate"] += time.perf_counter() - started

        started = time.perf_counter()
        events_linked, entities = EntityLinker().link(events)
        t["link"] = time.perf_counter() - started
        started = time.perf_counter()
        alerts = RuleEngine().run(events_linked, entities)
        t["rules"] = time.perf_counter() - started
        started = time.perf_counter()
        store.replace_derived(events_linked, entities)
        store.replace_alerts(alerts)
        t["persist_derived"] = time.perf_counter() - started
        store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        queries = measure_queries(store, n, samples, heavy_samples)
        peak = peak_memory_mb()
    size_bytes = sum(p.stat().st_size for p in workdir.glob(f"scale_{n}.db*"))
    changes = sum(stream.changed)
    caught = sampler.stats.by_reason
    return {
        "observations": n,
        "detected_objects": detections,
        "events": len(events_linked),
        "entities": len(entities),
        "alerts": len(alerts),
        "seconds": {k: round(v, 3) for k, v in t.items()},
        "throughput_obs_per_s": {
            "ingest_sqlite": round(n / t["ingest_sqlite"]),
            "aggregate": round(n / t["aggregate"]),
        },
        "db_size_mb": round(size_bytes / 2**20, 2),
        "db_bytes_per_observation": round(size_bytes / n),
        "peak_memory_mb": peak,
        "frames_to_events": {
            "frames": n,
            "events": len(events_linked),
            "ratio": round(n / max(1, len(events_linked)), 2),
        },
        "sampler_simulation": {
            "frames": n,
            "vlm_calls": sampler.stats.analyzed,
            "calls_avoided_pct": sampler.stats.avoided_pct,
            "scene_changes": changes,
            "by_reason": dict(sorted(caught.items())),
            "note": "SIMULATED score stream (generator ground truth), not measured perception",
        },
        "queries": queries,
    }


def measure_queries(store: Store, n: int, samples: int, heavy_samples: int) -> dict:
    """Latency percentiles for the operations the agent and CLI actually perform."""
    rng = random.Random(SEED + 7)
    tools = AnalystTools(store)
    span_s = n * FRAME_INTERVAL_S

    def when() -> str:
        return (_START + timedelta(seconds=rng.uniform(0, span_s))).isoformat()

    def hour_window() -> tuple[str, str]:
        start = _START + timedelta(seconds=rng.uniform(0, max(1, span_s - 3600)))
        return start.isoformat(), (start + timedelta(hours=1)).isoformat()

    def zone() -> str:
        return rng.choice(ZONES).value

    def keyword() -> str:
        return rng.choice(["blue pickup", "red sedan", "dark jacket", "person", "white van"])

    def q_events_zone() -> object:
        return tools.query_events(zone=zone())

    def q_events_window() -> object:
        a, b = hour_window()
        return tools.query_events(start=a, end=b)

    def q_search_events() -> object:
        return tools.search_events(keyword())

    def q_alerts_rule() -> object:
        return tools.list_alerts(rule_id="loitering")

    def q_alerts_all() -> object:
        return tools.list_alerts()

    def q_frame() -> object:
        return tools.get_frame(rng.randint(1, n))

    def q_telemetry() -> object:
        return tools.telemetry_at(when())

    def q_timeline() -> object:
        a, b = hour_window()
        return tools.get_timeline(start=a, end=b)

    def q_summary() -> object:
        return tools.summarize_run()

    def s_fts() -> object:
        return store.query_observations(text=keyword(), limit=10)

    def s_filter() -> object:
        a, b = hour_window()
        return store.query_observations(
            zone=Zone(zone()),
            start=datetime.fromisoformat(a),
            end=datetime.fromisoformat(b),
            limit=50,
        )

    def s_count() -> object:
        return store.count_matching_observations(zone=Zone(zone()))

    plan = [
        ("store_fts_keyword_top10", s_fts, samples),
        ("store_filter_zone_window", s_filter, samples),
        ("store_count_zone", s_count, samples),
        ("tool_get_frame", q_frame, samples),
        ("tool_telemetry_at", q_telemetry, samples),
        ("tool_query_events_zone", q_events_zone, heavy_samples),
        ("tool_query_events_window", q_events_window, samples),
        ("tool_search_events", q_search_events, heavy_samples),
        ("tool_list_alerts_rule", q_alerts_rule, heavy_samples),
        ("tool_list_alerts_all", q_alerts_all, heavy_samples),
        ("tool_get_timeline_hour", q_timeline, heavy_samples),
        ("tool_summarize_run", q_summary, heavy_samples),
    ]
    out = {}
    for name, fn, count in plan:
        times = []
        for _ in range(count):
            started = time.perf_counter()
            fn()
            times.append((time.perf_counter() - started) * 1000)
        out[name] = percentiles(times)
    return out


# ---- the parent: one process per size, then the artefacts ---------------------------------------
def run_benchmark(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    output: Path | None = None,
    samples: int = 60,
    heavy_samples: int = 8,
    log: Callable[[str], None] = lambda _msg: None,
) -> dict:
    reports = []
    with tempfile.TemporaryDirectory(prefix="scale_bench_") as tmp:
        for n in sizes:
            log(f"size {n:,}: running in a fresh process ...")
            started = time.perf_counter()
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "drone_agent.eval.scale_bench",
                    "--worker",
                    str(n),
                    "--workdir",
                    tmp,
                    "--samples",
                    str(samples),
                    "--heavy-samples",
                    str(heavy_samples),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "GEMINI_API_KEY": ""},  # a benchmark never needs a key
            )
            if proc.returncode != 0:
                raise RuntimeError(f"size {n} failed:\n{proc.stderr[-2000:]}")
            line = next(
                ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(RESULT_MARK)
            )
            reports.append(json.loads(line[len(RESULT_MARK) :]))
            log(f"size {n:,}: done in {time.perf_counter() - started:.1f}s")
    result = {
        "harness": "scale-bench-1",
        "environment": environment(),
        "sizes": reports,
        "scaling_exponents": scaling_exponents(reports),
        "qualifications": [
            "Generated observations with a small attribute vocabulary and a plate fleet; not real "
            "footage, and no vision model is involved (its latency would dominate a deployment).",
            "One machine, one run per size; timings vary run to run. Counts are deterministic.",
            "The sampler figures come from a SIMULATED scene-score stream: they estimate call "
            "savings if scores looked like this, not measured perception.",
            "SQLite on a local disk, one writer; no concurrent access, no network filesystem.",
        ],
    }
    if output is not None:
        write_artifacts(result, Path(output))
    return result


def scaling_exponents(reports: list[dict]) -> dict:
    """k in time ~ size^k between the two largest sizes (k=1 is linear)."""
    if len(reports) < 2:
        return {}
    a, b = reports[-2], reports[-1]
    ratio = math.log(b["observations"] / a["observations"])
    out: dict[str, float | None] = {}
    for stage, value in b["seconds"].items():
        prev = a["seconds"].get(stage)
        out[f"seconds.{stage}"] = (
            round(math.log(value / prev) / ratio, 2) if prev and value and prev > 0 else None
        )
    for name, stats in b["queries"].items():
        prev = a["queries"][name]["p50_ms"]
        out[f"query.{name}.p50"] = (
            round(math.log(stats["p50_ms"] / prev) / ratio, 2)
            if prev > 0 and stats["p50_ms"] > 0
            else None
        )
    return out


# ---- artefacts: JSON, markdown, SVG -------------------------------------------------------------
def write_artifacts(result: dict, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        output / "scale_results.json",
        output / "scale_report.md",
        output / "scale_chart.svg",
    ]
    paths[0].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths[1].write_text(render_markdown(result), encoding="utf-8")
    paths[2].write_text(render_svg(result), encoding="utf-8")
    return paths


def render_markdown(result: dict) -> str:
    sizes = result["sizes"]
    head = " | ".join(f"{s['observations']:,}" for s in sizes)
    sep = "|---|" + "---:|" * len(sizes)
    env = result["environment"]
    lines = [
        "# Scale benchmark",
        "",
        f"Machine: {env['platform']}, Python {env['python']}, {env['cpu_count']} logical CPUs, "
        f"SQLite {env['sqlite_version']}. {env['note']}.",
        "",
        "## Pipeline (real code paths, generated observations)",
        "",
        f"| | {head} |",
        sep,
    ]

    def row(label: str, get: Callable[[dict], object]) -> None:
        lines.append(f"| {label} | " + " | ".join(str(get(s)) for s in sizes) + " |")

    row("events", lambda s: f"{s['events']:,}")
    row("entities", lambda s: f"{s['entities']:,}")
    row("alerts", lambda s: f"{s['alerts']:,}")
    row("frames per event", lambda s: s["frames_to_events"]["ratio"])
    for stage in ("ingest_sqlite", "aggregate", "link", "rules", "persist_derived"):
        row(f"{stage} (s)", lambda s, k=stage: s["seconds"][k])
    row("ingest throughput (obs/s)", lambda s: f"{s['throughput_obs_per_s']['ingest_sqlite']:,}")
    row("DB size (MiB)", lambda s: s["db_size_mb"])
    row("DB bytes / observation", lambda s: s["db_bytes_per_observation"])
    row("peak memory (MiB)", lambda s: s["peak_memory_mb"])
    lines += ["", "## Query latency (ms; p50 / p95)", "", f"| operation | {head} |", sep]
    for name in sizes[0]["queries"]:
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{s['queries'][name]['p50_ms']} / {s['queries'][name]['p95_ms']}" for s in sizes
            )
            + " |"
        )
    lines += ["", "## VLM calls saved (SIMULATED score stream)", "", f"| | {head} |", sep]
    row("frames", lambda s: f"{s['sampler_simulation']['frames']:,}")
    row("model calls", lambda s: f"{s['sampler_simulation']['vlm_calls']:,}")
    row("calls avoided (%)", lambda s: s["sampler_simulation"]["calls_avoided_pct"])
    row("real scene changes", lambda s: f"{s['sampler_simulation']['scene_changes']:,}")
    if result["scaling_exponents"]:
        lines += [
            "",
            "## Empirical scaling exponent between the two largest sizes (time ~ size^k)",
            "",
            "k = 1 is linear; well above 1 is superlinear.",
            "",
        ]
        lines += [f"- {k}: {v}" for k, v in result["scaling_exponents"].items() if v is not None]
    lines += ["", "## Qualifications", ""] + [f"- {q}" for q in result["qualifications"]]
    return "\n".join(lines) + "\n"


_COLORS = ("#2f6fed", "#d9534f", "#2e9d5b", "#b8860b", "#8e44ad", "#16a3b8", "#555555")


def _log_axis(values: list[float], lo: float, hi: float, size: float, base: float) -> list[float]:
    span = math.log10(hi) - math.log10(lo) or 1.0
    return [base + size * (math.log10(max(v, lo)) - math.log10(lo)) / span for v in values]


def _panel(
    title: str, x_label: str, series: dict[str, list[tuple[float, float]]], y_label: str, top: int
) -> list[str]:
    left, width, height = 70, 520, 250
    xs = [x for pts in series.values() for x, _ in pts]
    ys = [max(y, 1e-3) for pts in series.values() for _, y in pts]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = 10 ** math.floor(math.log10(min(ys))), 10 ** math.ceil(math.log10(max(ys)))
    if y_lo == y_hi:
        y_hi = y_lo * 10
    out = [
        f'<text x="{left}" y="{top - 12}" class="t">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" class="box"/>',
    ]
    for exp in range(int(math.log10(y_lo)), int(math.log10(y_hi)) + 1):
        y = top + height - _log_axis([10.0**exp], y_lo, y_hi, height, 0)[0]
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + width}" y2="{y:.1f}" class="grid"/>'
        )
        out.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" class="ax" text-anchor="end">{10.0**exp:g}</text>'
        )
    for x in sorted(set(xs)):
        px = _log_axis([x], x_lo, x_hi, width, left)[0]
        out.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + height}" class="grid"/>'
        )
        out.append(
            f'<text x="{px:.1f}" y="{top + height + 16}" class="ax" text-anchor="middle">{x:,.0f}</text>'
        )
    out.append(
        f'<text x="{left + width / 2}" y="{top + height + 34}" class="ax" text-anchor="middle">{x_label}</text>'
    )
    out.append(
        f'<text x="14" y="{top + height / 2}" class="ax" transform="rotate(-90 14 {top + height / 2})" text-anchor="middle">{y_label}</text>'
    )
    for i, (name, pts) in enumerate(series.items()):
        color = _COLORS[i % len(_COLORS)]
        px = _log_axis([x for x, _ in pts], x_lo, x_hi, width, left)
        py = [
            top + height - v
            for v in _log_axis([max(y, 1e-3) for _, y in pts], y_lo, y_hi, height, 0)
        ]
        out.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
            + " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py, strict=True))
            + '"/>'
        )
        for a, b in zip(px, py, strict=True):
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="3" fill="{color}"/>')
        out.append(
            f'<rect x="{left + width + 12}" y="{top + 8 + 16 * i}" width="10" height="10" fill="{color}"/>'
        )
        out.append(
            f'<text x="{left + width + 28}" y="{top + 17 + 16 * i}" class="ax">{name}</text>'
        )
    return out


def render_svg(result: dict) -> str:
    sizes = result["sizes"]
    stage = {
        k: [(s["observations"], s["seconds"][k]) for s in sizes]
        for k in ("ingest_sqlite", "aggregate", "link", "rules", "persist_derived")
    }
    tools = {
        k.removeprefix("tool_"): [(s["observations"], s["queries"][k]["p95_ms"]) for s in sizes]
        for k in sizes[0]["queries"]
        if k.startswith("tool_")
    }
    body = [
        '<rect width="860" height="735" class="bg"/>',
        '<text x="70" y="22" class="h">Scale benchmark: generated observations, real code paths</text>',
    ]
    body += _panel("Pipeline stage time", "observations", stage, "seconds (log)", 60)
    body += _panel("Analyst-tool query latency, p95", "observations", tools, "ms (log)", 400)
    body.append(
        '<text x="70" y="722" class="ax">Measured on one machine; generated data; '
        "no vision model. See scale_report.md.</text>"
    )
    style = (
        "<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#222}.h{font-size:15px;font-weight:600}"
        ".t{font-size:13px;font-weight:600}.ax{font-size:11px}.box{fill:#fff;stroke:#999}"
        ".bg{fill:#fff}.grid{stroke:#e2e2e2}@media (prefers-color-scheme:dark){.bg{fill:#151515}text{fill:#ddd}"
        ".box{fill:#1e1e1e;stroke:#666}.grid{stroke:#333}}</style>"
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 735" width="860" height="735">'
        + style
        + "".join(body)
        + "</svg>\n"
    )


# ---- entry point for the worker process ----------------------------------------------------------
def _worker(args: argparse.Namespace) -> int:
    report = measure_size(args.worker, Path(args.workdir), args.samples, args.heavy_samples)
    print(RESULT_MARK + json.dumps(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drone_agent.eval.scale_bench")
    parser.add_argument("--worker", type=int, required=True, help="run one size in this process")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--heavy-samples", type=int, default=8)
    return _worker(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
