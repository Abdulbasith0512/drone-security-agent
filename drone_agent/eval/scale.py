"""Scale benchmark: deterministic generated observations, timed stage by stage.

Measures wall-clock time on THIS machine (median of repeated runs). It makes no network or API
calls and its numbers are not portable claims. Timing values vary run to run; everything else in
the report (sizes, event counts, alert counts) is deterministic.
"""

import math
import os
import platform
import random
import sqlite3
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from drone_agent.eval.base import IST, ExperimentResult, Timeline, det, median
from drone_agent.models import ObjectType, Zone
from drone_agent.pipeline import EntityLinker, aggregate_observations
from drone_agent.rules import RuleEngine
from drone_agent.store import Store

DEFAULT_SIZES = (100, 1000, 10000)
COLORS = ("blue", "red", "white", "black", "silver", "green")
BODIES = ("pickup", "sedan", "van", "suv")
CLOTHING = ("dark jacket", "red hoodie", "blue shirt", "grey coat", "yellow vest")
ZONES = tuple(Zone)
START = datetime(2026, 9, 29, 8, 0, tzinfo=IST)
FRAME_INTERVAL_S = 30


def generate_observations(n: int, seed: int = 42) -> list:
    """`n` chronological single-frame observations: empty stretches, people and vehicles that
    dwell for a few frames in a zone. Fully determined by (n, seed)."""
    rng = random.Random(seed)
    tl = Timeline(START)
    step = 0
    while step < n:
        zone = rng.choice(ZONES)
        kind = rng.choices(("empty", "person", "vehicle"), weights=(3, 4, 3))[0]
        length = rng.randint(2, 8)
        obj = None
        if kind == "person":
            obj = det("person", "standing", clothing=rng.choice(CLOTHING))
        elif kind == "vehicle":
            obj = det("vehicle", "parked", color=rng.choice(COLORS), body_type=rng.choice(BODIES))
        for _ in range(min(length, n - step)):
            tl.add(step * FRAME_INTERVAL_S, zone, *([obj] if obj else []))
            step += 1
    return tl.observations


def _time_ms(fn: Callable[[], object], repeats: int) -> tuple[float, object]:
    samples, result = [], None
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        samples.append((time.perf_counter() - started) * 1000)
    return round(median(samples), 3), result


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "sqlite_version": sqlite3.sqlite_version,
        "note": "measurements on this machine only; not a universal performance claim",
    }


def benchmark_size(n: int, repeats: int = 5, query_repeats: int = 20) -> dict:
    observations = generate_observations(n)
    detections = sum(len(o.detected_objects) for o in observations)
    agg_ms, events = _time_ms(lambda: aggregate_observations(observations), repeats)
    link_ms, linked = _time_ms(lambda: EntityLinker().link(events), repeats)
    events_linked, entities = linked
    rules_ms, alerts = _time_ms(lambda: RuleEngine().run(events_linked, entities), repeats)
    with tempfile.TemporaryDirectory() as tmp, Store.open(Path(tmp) / "scale.db") as store:
        started = time.perf_counter()
        with store.transaction():
            for obs in observations:
                store.insert_observation(obs)
        ingest_ms = round((time.perf_counter() - started) * 1000, 3)
        started = time.perf_counter()
        store.replace_derived(events_linked, entities)
        store.replace_alerts(alerts)
        persist_ms = round((time.perf_counter() - started) * 1000, 3)
        window_end = START.replace(hour=12)
        queries = {
            "fts_keyword_top10": lambda: store.query_observations(text="blue pickup", limit=10),
            "filter_zone_type_window": lambda: store.query_observations(
                zone=Zone.MAIN_GATE,
                object_type=ObjectType.PERSON,
                start=START,
                end=window_end,
                limit=50,
            ),
            "count_matching_zone": lambda: store.count_matching_observations(zone=Zone.DRIVEWAY),
            "list_events_by_zone": lambda: store.query_events(zone=Zone.MAIN_GATE),
            "list_alerts": lambda: store.query_alerts(),
        }
        query_ms = {name: _time_ms(fn, query_repeats)[0] for name, fn in queries.items()}
    return {
        "observations": n,
        "detected_objects": detections,
        "events": len(events),
        "entities": len(entities),
        "alerts": len(alerts),
        "repeats": repeats,
        "query_repeats": query_repeats,
        "median_ms": {
            "event_aggregation": agg_ms,
            "entity_linking": link_ms,
            "rule_evaluation": rules_ms,
            **{f"query_{k}": v for k, v in query_ms.items()},
        },
        "single_run_ms": {"sqlite_ingest": ingest_ms, "sqlite_persist_derived": persist_ms},
        "aggregation_us_per_observation": round(agg_ms * 1000 / n, 3),
    }


def scaling_exponents(results: list[dict]) -> dict[str, float | None]:
    """k in time ~ size^k between the two largest sizes (None if not measurable)."""
    if len(results) < 2:
        return {}
    a, b = results[-2], results[-1]
    out: dict[str, float | None] = {}
    for stage in b["median_ms"]:
        t0, t1 = a["median_ms"][stage], b["median_ms"][stage]
        if t0 <= 0 or t1 <= 0 or b["observations"] == a["observations"]:
            out[stage] = None
        else:
            out[stage] = round(
                math.log(t1 / t0) / math.log(b["observations"] / a["observations"]), 3
            )
    return out


def run(
    sizes: tuple[int, ...] = DEFAULT_SIZES, repeats: int = 5, query_repeats: int = 20
) -> ExperimentResult:
    results = [benchmark_size(n, repeats, query_repeats) for n in sizes]
    rows = []
    for r in results:
        for stage, ms in {**r["median_ms"], **r["single_run_ms"]}.items():
            rows.append(
                {"experiment": "scale", "observations": r["observations"], "stage": stage, "ms": ms}
            )
    first, last = results[0], results[-1]
    exponents = scaling_exponents(results)
    superlinear = sorted(k for k, v in exponents.items() if v is not None and v > 1.3)
    growth = last["median_ms"]["event_aggregation"] / max(
        first["median_ms"]["event_aggregation"], 1e-9
    )
    interpretation = [
        f"Event aggregation took a median {first['median_ms']['event_aggregation']} ms for "
        f"{first['observations']} observations and {last['median_ms']['event_aggregation']} ms for "
        f"{last['observations']} ({growth:.1f}x time for "
        f"{last['observations'] / first['observations']:.0f}x data).",
        "Empirical scaling exponent between the two largest sizes (time ~ size^k; k=1 is linear): "
        + ", ".join(f"{k} {v:.2f}" for k, v in exponents.items() if v is not None)
        + (f". Superlinear here (k>1.3): {superlinear}." if superlinear else ". None above 1.3."),
        "Times are medians of repeated in-process runs on this machine; the SQLite ingest and "
        "derived-table writes are single runs.",
    ]
    return ExperimentResult(
        key="scale",
        title="Scale benchmark (generated observations)",
        research_question="Assignment scalability question: how do processing and query costs "
        "grow with the number of observations?",
        definition=f"Deterministic generated observations at sizes {list(sizes)}; each pipeline "
        f"stage timed {repeats} times (median), SQLite queries {query_repeats} times (median).",
        measured={
            "environment": environment(),
            "sizes": results,
            "scaling_exponent_two_largest_sizes": exponents,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            "One machine, one run session; no claim about other hardware, larger data, "
            "concurrent access or the vision model (its latency dominates a real deployment).",
            "Synthetic observations with a small attribute vocabulary; linking cost depends on "
            "the number of distinct vehicle attribute combinations.",
            "In-memory stages exclude reading rows back from SQLite.",
        ],
    )
