"""The large-scale benchmark: deterministic counts, honest artefacts, no model or network.

Only small sizes run here; the 10K / 100K / 1M run is opt-in (`python -m drone_agent scale`)."""

import json

import pytest

from drone_agent.eval import scale_bench
from drone_agent.eval.scale_bench import (
    Stream,
    measure_size,
    peak_memory_mb,
    percentiles,
    render_markdown,
    render_svg,
    run_benchmark,
    scaling_exponents,
)


def test_the_stream_is_lazy_chronological_and_deterministic():
    first, second = list(Stream(400)), list(Stream(400))
    assert [o.observation_id for o in first] == [o.observation_id for o in second]
    assert [o.frame_id for o in first] == list(range(1, 401))
    stamps = [o.timestamp for o in first]
    assert stamps == sorted(stamps) and len(set(stamps)) == 400
    assert Stream(400, seed=7).__iter__().__next__().zone is not None


def test_the_stream_marks_real_scene_changes_for_the_simulated_score():
    stream = Stream(600)
    list(stream)
    assert len(stream.changed) == len(stream.scores) == 600
    assert stream.changed[0] is True and 0 < sum(stream.changed) < 600
    assert all(s == 100.0 for s, c in zip(stream.scores, stream.changed, strict=True) if c)
    assert all(s < 3.0 for s, c in zip(stream.scores, stream.changed, strict=True) if not c)


def test_percentiles_use_nearest_rank():
    stats = percentiles([float(i) for i in range(1, 101)])
    assert stats == {"n": 100, "p50_ms": 50.0, "p95_ms": 95.0, "max_ms": 100.0}
    assert percentiles([2.0]) == {"n": 1, "p50_ms": 2.0, "p95_ms": 2.0, "max_ms": 2.0}


def test_peak_memory_is_a_positive_number_or_honestly_unknown():
    value = peak_memory_mb()
    assert value is None or value > 1


def test_scaling_exponent_is_the_log_log_slope_between_the_two_largest_sizes():
    def report(n, seconds, p50):
        return {
            "observations": n,
            "seconds": {"ingest_sqlite": seconds},
            "queries": {"tool_x": {"p50_ms": p50}},
        }

    out = scaling_exponents([report(10, 1, 1), report(100, 10, 1), report(1000, 1000, 10)])
    assert out["seconds.ingest_sqlite"] == 2.0  # 100x for 10x data
    assert out["query.tool_x.p50"] == 1.0
    assert scaling_exponents([report(10, 1, 1)]) == {}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    return measure_size(800, tmp_path_factory.mktemp("sb"), samples=3, heavy_samples=1)


def test_a_size_report_measures_every_stage_and_query(small):
    assert small["observations"] == 800
    for stage in ("ingest_sqlite", "aggregate", "link", "rules", "persist_derived"):
        assert small["seconds"][stage] > 0
    assert small["events"] > 0 and small["alerts"] > 0 and small["entities"] > 0
    assert small["db_size_mb"] > 0 and small["throughput_obs_per_s"]["ingest_sqlite"] > 0
    assert small["frames_to_events"]["ratio"] == round(800 / small["events"], 2)
    names = set(small["queries"])
    assert {"tool_get_frame", "tool_summarize_run", "store_fts_keyword_top10"} <= names
    assert all(q["p50_ms"] <= q["p95_ms"] <= q["max_ms"] for q in small["queries"].values())


def test_counts_are_deterministic_and_timings_are_not_part_of_them(small, tmp_path):
    again = measure_size(800, tmp_path, samples=2, heavy_samples=1)
    for key in ("events", "entities", "alerts", "detected_objects"):
        assert again[key] == small[key]
    assert again["sampler_simulation"]["vlm_calls"] == small["sampler_simulation"]["vlm_calls"]


def test_the_sampler_estimate_is_labelled_simulated_and_saves_calls(small):
    sim = small["sampler_simulation"]
    assert "SIMULATED" in sim["note"]
    assert sim["vlm_calls"] < sim["frames"] and 0 < sim["calls_avoided_pct"] < 100
    assert sim["by_reason"].get("scene_change", 0) + sim["by_reason"].get("zone_change", 0) > 0


def test_running_a_size_twice_in_one_folder_starts_from_an_empty_database(tmp_path):
    first = measure_size(300, tmp_path, samples=1, heavy_samples=1)
    second = measure_size(300, tmp_path, samples=1, heavy_samples=1)  # no duplicate-key error
    assert first["events"] == second["events"]


def test_the_whole_benchmark_writes_json_markdown_and_svg(tmp_path):
    result = run_benchmark((300, 600), tmp_path / "out", samples=2, heavy_samples=1)
    assert [s["observations"] for s in result["sizes"]] == [300, 600]
    data = json.loads((tmp_path / "out" / "scale_results.json").read_text())
    assert data["harness"] == "scale-bench-1" and len(data["sizes"]) == 2
    report = (tmp_path / "out" / "scale_report.md").read_text()
    assert "## Qualifications" in report and "SIMULATED" in report and "| 300 | 600 |" in report
    svg = (tmp_path / "out" / "scale_chart.svg").read_text()
    assert svg.startswith("<svg") and "polyline" in svg and "prefers-color-scheme" in svg
    assert render_markdown(result) == report and render_svg(result) == svg


def test_a_failing_worker_is_reported_not_swallowed(tmp_path, monkeypatch):
    class Failed:
        returncode, stdout, stderr = 1, "", "Traceback: boom"

    monkeypatch.setattr(scale_bench.subprocess, "run", lambda *a, **k: Failed())
    with pytest.raises(RuntimeError, match="(?s)size 300 failed.*boom"):
        run_benchmark((300,), tmp_path)


def test_the_worker_never_sees_an_api_key(tmp_path, monkeypatch):
    seen = {}

    class Failed:
        returncode, stdout, stderr = 1, "", ""

    def fake_run(*args, **kwargs):
        seen.update(kwargs["env"])
        return Failed()

    monkeypatch.setenv("GEMINI_API_KEY", "must-not-reach-the-worker")
    monkeypatch.setattr(scale_bench.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        run_benchmark((300,), tmp_path)
    assert seen["GEMINI_API_KEY"] == ""
