"""Streaming runtime: ticks, pacing, and a live run that ends where the batch pipeline ends."""

import io
from datetime import timedelta

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.agents import PerceptionAgent, SamplerConfig
from drone_agent.ingest import ingest_result
from drone_agent.models import AlertStatus, Severity, TransitionKind
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.runtime import EngineOptions, Pacer, PatrolEngine, RunEventKind, build_ticks
from drone_agent.runtime.scene import scene_score, signature
from drone_agent.runtime.view import ASCII, LiveView, LogView, glyphs_for
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store
from drone_agent.vision import ScriptedProvider

SECURITY = SCENARIO_DIR / "security_events.yaml"
NORMAL = SCENARIO_DIR / "normal_day.yaml"

# What a final alert must agree on, whichever way it was produced (history timestamps differ:
# a live run stamps when it noticed, batch stamps when the condition began).
ALERT_FIELDS = {
    "alert_id", "rule_id", "severity", "status", "title", "message", "triggered_at", "first_seen",
    "last_seen", "zone", "event_ids", "frame_ids", "facts", "confidence", "verdict",
    "recommended_action", "ladder", "resolved_at", "entity_id", "telemetry_at",
}  # fmt: skip


def run_engine(scenario, store, **kw):
    events = []
    engine = PatrolEngine(scenario, store, listener=events.append, **kw)
    return engine, engine.run(), events


def batch(scenario, store):
    ingest_result(store, run_scenario(scenario))
    build_events(store)
    analyze(store)


def final_alerts(store):
    return [a.model_dump(include=ALERT_FIELDS) for a in store.query_alerts()]


# ---- ticks ----------------------------------------------------------------------------------
def test_ticks_are_in_time_order_with_telemetry_before_frames_at_the_same_instant():
    scenario = load_scenario(NORMAL)
    ticks = build_ticks(scenario)
    assert [t.offset_s for t in ticks] == sorted(t.offset_s for t in ticks)
    same = {}
    for t in ticks:
        same.setdefault(t.offset_s, []).append(t.kind)
    assert all(kinds == sorted(kinds, key=lambda k: k != "telemetry") for kinds in same.values())
    assert sum(t.kind == "frame" for t in ticks) == len(scenario.frames)


def test_a_dropout_keeps_its_scheduled_ticks_with_nothing_delivered():
    scenario = load_scenario(SECURITY)  # telemetry silent for offsets 1570-1690
    telemetry = [t for t in build_ticks(scenario) if t.kind == "telemetry"]
    silent = [t for t in telemetry if t.sample is None]
    assert [t.offset_s for t in silent] == [float(o) for o in range(1570, 1691, 10)]
    assert (
        len(telemetry) == scenario.telemetry.duration_s // scenario.telemetry.sample_interval_s + 1
    )


# ---- pacing ---------------------------------------------------------------------------------
def test_speed_zero_never_sleeps_and_speed_divides_simulated_time():
    slept = []
    Pacer(0, sleep=slept.append).wait(600)
    Pacer(None, sleep=slept.append).wait(600)
    assert slept == []
    p = Pacer(5, max_wait_s=10, sleep=slept.append)
    p.wait(20)
    p.wait(0)
    p.wait(-5)
    assert slept == [4.0] and p.waited_s == 4.0


def test_long_quiet_stretches_are_capped_so_a_demo_stays_watchable():
    slept = []
    Pacer(5, max_wait_s=0.6, sleep=slept.append).wait(3000)
    assert slept == [0.6]
    with pytest.raises(ValueError):
        Pacer(-1)
    with pytest.raises(ValueError):
        Pacer(5, max_wait_s=-1)


def test_engine_paces_between_ticks_but_computes_the_same_result():
    scenario = load_scenario(NORMAL)
    slept = []
    with Store.open(":memory:") as a, Store.open(":memory:") as b:
        _, fast, _ = run_engine(scenario, a)
        _, paced, _ = run_engine(scenario, b, pacer=Pacer(60, 0.5, sleep=slept.append))
        assert slept and max(slept) <= 0.5
        assert final_alerts(a) == final_alerts(b)
        assert fast.to_dict()["events"] == paced.to_dict()["events"]


# ---- scene signature ------------------------------------------------------------------------
def test_scene_score_is_zero_for_identical_scenes_and_100_for_disjoint_ones():
    frames = load_scenario(SECURITY).frames
    a, b, blue = signature(frames[0]), signature(frames[1]), signature(frames[9])
    assert scene_score(None, a) == 100.0
    assert scene_score(a, a) == 0.0
    assert scene_score(a, blue) == 100.0  # a person vs a pickup: nothing shared
    assert 0 <= scene_score(a, b) <= 100
    assert scene_score(signature(frames[4]), signature(frames[4])) == 0.0  # both empty


# ---- a live run ends where the batch pipeline ends ---------------------------------------------
@pytest.mark.parametrize("path", [SECURITY, NORMAL], ids=["security_events", "normal_day"])
def test_streaming_without_sampling_equals_the_batch_pipeline_exactly(path):
    scenario = load_scenario(path)
    with Store.open(":memory:") as live, Store.open(":memory:") as ref:
        run_engine(scenario, live, options=EngineOptions(sampler=SamplerConfig(enabled=False)))
        batch(scenario, ref)
        assert live.query_events() == ref.query_events()
        assert live.list_entities() == ref.list_entities()
        assert final_alerts(live) == final_alerts(ref)
        assert live.query_observations() == ref.query_observations()
        assert live.list_telemetry() == ref.list_telemetry()


@pytest.mark.parametrize("path", [SECURITY, NORMAL], ids=["security_events", "normal_day"])
def test_with_adaptive_sampling_the_alerts_are_still_the_same(path):
    """Skipping calls must not change what is decided on these scenarios."""
    scenario = load_scenario(path)
    with Store.open(":memory:") as live, Store.open(":memory:") as ref:
        _, summary, _ = run_engine(scenario, live)
        batch(scenario, ref)
        # Identical decisions. Only `confidence` may differ slightly: a carried frame repeats the
        # last real answer's confidence instead of being a fresh look (the price of fewer calls).
        drop = {"confidence"}
        got, want = final_alerts(live), final_alerts(ref)
        assert [{k: v for k, v in a.items() if k not in drop} for a in got] == [
            {k: v for k, v in a.items() if k not in drop} for a in want
        ]
        for a, b in zip(got, want, strict=True):
            assert abs((a["confidence"] or 0) - (b["confidence"] or 0)) < 0.02
        assert [
            (e.event_id, e.first_seen, e.last_seen, e.close_reason) for e in live.query_events()
        ] == [(e.event_id, e.first_seen, e.last_seen, e.close_reason) for e in ref.query_events()]
        assert summary.model_calls + summary.skipped + summary.dropped == summary.frames


def test_two_runs_of_the_same_scenario_are_identical():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as a, Store.open(":memory:") as b:
        run_engine(scenario, a)
        run_engine(scenario, b)
        assert a.query_alerts() == b.query_alerts()  # including full history
        assert a.query_events() == b.query_events()
        assert a.query_observations() == b.query_observations()


# ---- what a live viewer sees ---------------------------------------------------------------------
def test_the_loitering_alert_is_raised_then_escalated_then_resolved_in_time_order():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        seen = []

        def listener(event):
            if (
                event.kind is RunEventKind.ALERT
                and event.data["alert_id"] == "ALR-LOITER-EVT-P-001"
                and event.data["transition"] != "evidence"  # only lifecycle changes
            ):
                stored = store.get_event("EVT-P-001")
                seen.append(
                    (event.data["transition"], event.data["severity"], event.at, stored.is_open)
                )

        PatrolEngine(scenario, store, listener=listener).run()
    assert [(s[0], s[1]) for s in seen] == [
        ("raised", "medium"),
        ("escalated", "critical"),
        ("resolved", "critical"),
    ]
    assert seen[0][2] < seen[1][2] < seen[2][2]
    assert seen[0][3] and seen[1][3] and not seen[2][3]  # open while it grew, closed at the end
    warning_at = scenario.start + timedelta(seconds=60 + 120)
    assert seen[0][2] == warning_at  # noticed on the first frame after the 120 s threshold


def test_every_event_is_labelled_simulated_scripted_or_carried():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        _, _, events = run_engine(scenario, store)
    telemetry = [e for e in events if e.kind is RunEventKind.TELEMETRY]
    assert telemetry and {e.provenance for e in telemetry} == {"SIMULATED"}
    labels = {e.provenance for e in events if e.kind is RunEventKind.OBSERVATION}
    assert labels == {"SCRIPTED", "SCRIPTED+CARRIED"}
    assert events[0].kind is RunEventKind.STARTED and events[-1].kind is RunEventKind.FINISHED
    times = [e.offset_s for e in events]
    assert times == sorted(times)


def test_a_telemetry_outage_is_noticed_at_the_first_silent_tick_not_when_data_resumes():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        _, summary, events = run_engine(scenario, store)
    latency = {k: v for k, v in summary.detection_latency_s.items() if "SIGNAL" in k}
    (alert_id,) = latency
    assert latency[alert_id] == 10  # last good sample + 30 s of grace, seen on the next 10 s tick
    raised = next(
        e for e in events if e.kind is RunEventKind.ALERT and e.data["alert_id"] == alert_id
    )
    assert raised.offset_s < 1690  # raised while the link was still down


def test_summary_counts_add_up_and_are_persisted_with_the_scenario():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        _, summary, _ = run_engine(scenario, store)
        assert summary.frames == 27 and summary.observations == 27 and summary.dropped == 0
        assert summary.model_calls + summary.skipped == summary.frames
        assert summary.telemetry_scheduled == 301 and summary.telemetry_delivered == 288
        assert summary.alerts == 7 and sum(summary.alerts_by_status.values()) == 7
        assert sum(summary.provenance.values()) == summary.observations
        (patrol,) = store.list_agent_runs("patrol")
        assert patrol["status"] == "finished" and patrol["metrics"]["alerts"] == 7
        assert patrol["trace"] and {"frame_id", "reason", "analyzed"} <= set(patrol["trace"][0])
        assert store.get_scenario("security_events")["start"].startswith("2026-09-29T23:40")


# ---- failure handling ----------------------------------------------------------------------------
class FlakyProvider:
    """Scripted answers, except for chosen frame ids where it fails."""

    name = "flaky"

    def __init__(self, scenario, fail_frames):
        self.inner = ScriptedProvider(scenario)
        self.fail_frames = set(fail_frames)

    def analyze(self, frame):
        if frame.frame_id in self.fail_frames:
            raise RuntimeError("simulated vision outage")
        return self.inner.analyze(frame)


def test_a_failing_frame_is_dropped_and_the_patrol_continues():
    scenario = load_scenario(SECURITY)
    perception = PerceptionAgent([FlakyProvider(scenario, {3})])
    with Store.open(":memory:") as store:
        _, summary, events = run_engine(
            scenario,
            store,
            perception=perception,
            options=EngineOptions(sampler=SamplerConfig(enabled=False)),
        )
        assert summary.dropped == 1 and summary.observations == summary.frames - 1
        assert store.get_observation_by_frame(3) is None
        assert summary.perception["failed"] == 1
        assert any(e.kind is RunEventKind.FRAME and e.data["dropped"] for e in events)
        assert store.count_alerts() >= 1  # the rest of the patrol was still analysed


def test_a_fallback_provider_keeps_every_frame_and_records_the_switch():
    scenario = load_scenario(SECURITY)
    perception = PerceptionAgent([FlakyProvider(scenario, {2, 3}), ScriptedProvider(scenario)])
    with Store.open(":memory:") as store:
        _, summary, _ = run_engine(
            scenario,
            store,
            perception=perception,
            options=EngineOptions(sampler=SamplerConfig(enabled=False)),
        )
        assert summary.dropped == 0 and summary.perception["fallbacks"] == 2


def test_when_the_first_analysis_fails_the_next_frame_analyses_instead_of_carrying_nothing():
    scenario = load_scenario(NORMAL)
    first = scenario.frames[0].frame_id
    perception = PerceptionAgent([FlakyProvider(scenario, {first})])
    with Store.open(":memory:") as store:
        _, summary, events = run_engine(scenario, store, perception=perception)
        frames = [e for e in events if e.kind is RunEventKind.FRAME]
        assert frames[0].data["dropped"] is True
        assert frames[1].data["analyzed"] and frames[1].data["reason"] in {
            "no_reference",
            "scene_change",
            "zone_change",
        }
        assert summary.observations == summary.frames - 1


# ---- views ------------------------------------------------------------------------------------
def console_for(encoding: str) -> tuple[Console, io.TextIOWrapper]:
    raw = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict", write_through=True)
    return Console(file=raw, width=120, force_terminal=False, color_system=None), raw


def test_the_log_view_never_crashes_on_a_legacy_windows_code_page():
    console, raw = console_for("cp1252")
    assert glyphs_for(console) is ASCII
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        PatrolEngine(
            scenario, store, listener=LogView(console)
        ).run()  # would raise if it emitted ↳
    raw.flush()
    text = raw.buffer.getvalue().decode("cp1252")
    assert "ALERT ESCALATED" in text and "PATROL FINISHED" in text and text.isascii()


def test_unicode_terminals_get_the_nicer_symbols():
    console, raw = console_for("utf-8")
    scenario = load_scenario(NORMAL)
    with Store.open(":memory:") as store:
        PatrolEngine(scenario, store, listener=LogView(console)).run()
    raw.flush()
    assert "↳" in raw.buffer.getvalue().decode("utf-8")


def test_the_live_dashboard_renders_every_panel_from_a_real_run():
    console, raw = console_for("utf-8")
    view = LiveView(console, speed=5)
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        PatrolEngine(scenario, store, listener=view).run()
    console.print(view.render())
    raw.flush()
    text = raw.buffer.getvalue().decode("utf-8")
    for expected in (
        "DRONE SECURITY PATROL",
        "SIMULATED",
        "Live timeline",
        "Alerts",
        "Perception & sampling",
    ):
        assert expected in text
    assert view.finished and view.frames == 27 and view.alerts


def test_alert_status_marks_in_the_summary_match_the_store():
    scenario = load_scenario(SECURITY)
    with Store.open(":memory:") as store:
        _, summary, _ = run_engine(scenario, store)
        by_status = {}
        for a in store.query_alerts():
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
        assert by_status == summary.alerts_by_status
        loiter = store.get_alert("ALR-LOITER-EVT-P-001")
        assert loiter.status is AlertStatus.RESOLVED and loiter.severity is Severity.CRITICAL
        assert [h.kind for h in loiter.history] == [
            TransitionKind.RAISED,
            TransitionKind.ESCALATED,
            TransitionKind.RESOLVED,
        ]
