"""V01: a video flows through the SAME downstream pipeline as a scripted patrol, progressively.

    video frame -> sampler -> gate -> provider -> Observation -> persist -> events/entities
                -> rules -> alerts -> analyst-visible evidence          (per frame, before EOF)

The deterministic demo video and its canned cassette exercise the mechanics (WHEN things happen).
They say nothing about how accurate a real vision model's descriptions are.

What the demo video does, at 2 s sampling (10 frames): empty yard at 0 and 2 s, a person from 4 s
(after-hours when the clock says 23:50), a blue pickup from 12 s, an empty yard again from 16 s.
"""

import inspect
from datetime import datetime
from io import StringIO

import pytest
from conftest import IST, SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.agent.tools import AnalystTools
from drone_agent.ingest import ingest_observations
from drone_agent.models import ObjectType, Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.runtime import ObservationPipeline, PatrolEngine
from drone_agent.runtime import consumer as consumer_module
from drone_agent.runtime import engine as engine_module
from drone_agent.runtime import video as live_module
from drone_agent.runtime.video import run_video_live
from drone_agent.scenario import load_scenario
from drone_agent.store import Store
from drone_agent.telemetry import PatrolLeg, TelemetryConfig
from drone_agent.video import (
    FailurePolicy,
    SkippedPolicy,
    VideoIngestConfig,
    VideoSource,
    VideoTimeline,
    run_video_ingest,
    single_zone_timeline,
)
from drone_agent.vision import Cassette, CassetteDescriber, DescriberProvider

NIGHT = datetime(2026, 9, 29, 23, 50, tzinfo=IST)
DAY = datetime(2026, 9, 29, 14, 0, tzinfo=IST)
ALERT_ID = "ALR-AFTERHOURS-EVT-P-001"
# what a final alert must agree on whichever way it was produced (see tests/test_runtime.py)
ALERT_FIELDS = {
    "alert_id", "rule_id", "severity", "status", "title", "message", "triggered_at", "first_seen",
    "last_seen", "zone", "event_ids", "frame_ids", "facts", "confidence", "verdict",
    "recommended_action", "ladder", "resolved_at", "entity_id", "telemetry_at",
}  # fmt: skip


def provider_for(cassette_path):
    return DescriberProvider(CassetteDescriber(Cassette.load(cassette_path)))


def live(demo_video, cassette, store, start=NIGHT, provider=None, **kw):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(start, Zone.MAIN_GATE, source.metadata.duration_seconds)
    return run_video_live(
        source, provider or provider_for(cassette), timeline, store, **kw
    ), timeline


def final_alerts(store):
    return [a.model_dump(include=ALERT_FIELDS) for a in store.query_alerts()]


class Watching:
    """Wraps a provider and records what the STORE holds at the moment each frame is analysed,
    which is independent evidence that earlier frames were already committed."""

    def __init__(self, inner, store):
        self.inner, self.store, self.name = inner, store, inner.name
        self.snapshots: dict[float, tuple[int, int, list[str]]] = {}

    def analyze(self, frame):
        self.snapshots[frame.offset_seconds] = (
            self.store.count_observations(),
            self.store.count_events(),
            [a.alert_id for a in self.store.query_alerts()],
        )
        return self.inner.analyze(frame)


# ---- progressive: before EOF ------------
def test_observation_n_is_persisted_before_the_video_completes(demo_video, demo_cassette_path):
    with Store.open(":memory:") as store:
        watcher = Watching(provider_for(demo_cassette_path), store)
        seen_in_store: list[int] = []
        result, _ = live(
            demo_video, None, store, provider=watcher,
            on_step=lambda step: seen_in_store.append(store.count_observations()),
        )  # fmt: skip
    # after each step the store holds exactly as many observations as frames processed so far
    assert seen_in_store == list(range(1, 11))
    assert [s.observations for s in result.steps] == list(range(1, 11))
    # when frame 120 (12 s) was handed to the model, frames 0..100 (six) were already stored
    assert watcher.snapshots[12.0][0] == 6
    assert watcher.snapshots[16.0][0] == 8  # and the video was still not finished


def test_events_are_created_and_extended_before_eof(demo_video, demo_cassette_path):
    with Store.open(":memory:") as store:
        counts, open_flags = [], []

        def probe(step):
            ev = store.get_event("EVT-P-001")
            counts.append(None if ev is None else ev.evidence_count)
            open_flags.append(None if ev is None else ev.is_open)

        result, _ = live(demo_video, demo_cassette_path, store, on_step=probe)
        # the person event appeared at 4 s and GREW with each frame while the person stayed
        assert counts == [None, None, 1, 2, 3, 4, 4, 4, 4, 4]
        # it closed (the same zone was seen empty) two steps before the video ended
        assert open_flags[:8] == [None, None, True, True, True, True, True, True]
        assert open_flags[8:] == [False, False]
        assert store.get_event("EVT-P-001").close_reason.value == "missed_frames"
        # the pickup event was still OPEN at the last frame, and closes only at the end of input
        assert result.steps[-1].open_events == 1
        pickup = store.get_event("EVT-V-001")
        assert pickup.close_reason.value == "scenario_end" and pickup.departure_observed_at is None


def test_an_alert_is_raised_while_the_video_is_still_being_processed(
    demo_video, demo_cassette_path
):
    with Store.open(":memory:") as store:
        tools = AnalystTools(store)
        at_alert: dict = {}

        def probe(step):
            if step.new_alerts and not at_alert:  # the moment it appears: ask the analyst tools
                at_alert["alerts"] = [a["ref"] for a in tools.list_alerts()["alerts"]]
                at_alert["evidence"] = tools.get_evidence(f"alert:{ALERT_ID}")
                at_alert["observations"] = store.count_observations()

        result, _ = live(demo_video, demo_cassette_path, store, on_step=probe)
    first = result.first_alert_step
    assert first is not None and first.new_alerts == (ALERT_ID,)
    assert (first.frame_index, first.offset_s) == (40, 4.0)
    remaining = len(result.steps) - result.steps.index(first) - 1
    assert remaining == 7  # seven sampled frames were still to come when the alert existed
    # the analyst-visible evidence existed at that moment, not just after the video
    assert at_alert["alerts"] == [f"alert:{ALERT_ID}"] and at_alert["observations"] == 3
    assert "error" not in at_alert["evidence"] and at_alert["evidence"]


def test_the_timeline_of_the_demo_is_what_the_docs_will_quote(demo_video, demo_cassette_path):
    with Store.open(":memory:") as store:
        result, _ = live(demo_video, demo_cassette_path, store)
    rows = [(s.offset_s, s.action, s.observations, s.events, s.alerts) for s in result.steps]
    assert rows == [
        (0.0, "analyzed", 1, 0, 0),
        (2.0, "carried", 2, 0, 0),
        (4.0, "analyzed", 3, 1, 1),  # person seen: event opened, alert raised
        (6.0, "carried", 4, 1, 1),
        (8.0, "carried", 5, 1, 1),
        (10.0, "carried", 6, 1, 1),
        (12.0, "analyzed", 7, 2, 1),  # pickup seen: second event
        (14.0, "carried", 8, 2, 1),
        (16.0, "analyzed", 9, 2, 1),  # yard empty again: person event closes
        (18.0, "carried", 10, 2, 1),
    ]


# ---- consistency with the batch path ------------
def test_the_final_progressive_result_equals_the_batch_result(demo_video, demo_cassette_path):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    with Store.open(":memory:") as batch, Store.open(":memory:") as progressive:
        ingested = run_video_ingest(source, provider_for(demo_cassette_path), timeline)
        ingest_observations(batch, "demo.avi", ingested.telemetry, ingested.observations)
        build_events(batch)
        analyze(batch)
        live(demo_video, demo_cassette_path, progressive)
        assert progressive.query_observations() == batch.query_observations()
        assert progressive.count_telemetry() == batch.count_telemetry()
        assert progressive.query_events() == batch.query_events()
        assert progressive.list_entities() == batch.list_entities()
        assert final_alerts(progressive) == final_alerts(batch)
        assert [a.alert_id for a in progressive.query_alerts()] == [ALERT_ID]


def test_the_live_result_has_the_same_shape_as_the_batch_result(demo_video, demo_cassette_path):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    batch = run_video_ingest(source, provider_for(demo_cassette_path), timeline)
    with Store.open(":memory:") as store:
        result, _ = live(demo_video, demo_cassette_path, store)
    assert result.result.observations == batch.observations
    assert result.result.candidates == batch.candidates
    assert (result.result.gate_accepted, result.result.carried) == (4, 6)


# ---- carried / skipped frames keep their provenance ------------
def test_carried_frames_are_stored_and_announced_as_carried(demo_video, demo_cassette_path):
    announced = []
    with Store.open(":memory:") as store:
        result, _ = live(demo_video, demo_cassette_path, store, listener=announced.append)
        stored = {o.frame_id: o for o in store.query_observations()}
    assert stored[60].source == "cassette+carried" and stored[40].source == "cassette"
    assert stored[60].description == stored[40].description  # propagation, not a new inference
    assert stored[60].provenance == stored[40].provenance
    labels = result.pipeline.provenance
    assert labels == {"CANNED": 4, "CANNED+CARRIED": 6}  # the cassette is canned, never "live"
    observed = [e for e in announced if e.kind.value == "observation"]
    assert [e.provenance for e in observed].count("CANNED+CARRIED") == 6
    assert sum(1 for e in observed if e.data["carried"]) == 6


def test_the_drop_policy_stores_nothing_for_skipped_frames(demo_video, demo_cassette_path):
    with Store.open(":memory:") as store:
        result, _ = live(
            demo_video, demo_cassette_path, store,
            config=VideoIngestConfig(skipped=SkippedPolicy.DROP),
        )  # fmt: skip
        assert store.count_observations() == 4
        assert not any(o.source.endswith("+carried") for o in store.query_observations())
    assert [s.action for s in result.steps].count("dropped") == 6
    assert result.steps[-1].observations == 4  # steps that dropped a frame add no observation


# ---- nothing to alert about ------------
def test_a_video_with_no_security_condition_raises_no_alert(demo_video, demo_cassette_path):
    with Store.open(":memory:") as store:
        result, _ = live(demo_video, demo_cassette_path, store, start=DAY)
        assert store.count_events() == 2  # a person and a pickup were still seen and logged
        assert store.count_alerts() == 0
    assert all(s.alerts == 0 for s in result.steps) and result.first_alert_step is None


# ---- the drone-safety rules run on the same stream ------------
def test_a_telemetry_alert_is_raised_while_frames_are_still_being_processed(
    demo_video, demo_cassette_path
):
    timeline = VideoTimeline(
        start=DAY,
        telemetry=TelemetryConfig(
            sample_interval_s=2,
            battery_start_percent=19.0,  # already below the 20% warning: a low-battery alert
            battery_drain_percent_per_min=0.0,
            patrol=[PatrolLeg(zone=Zone.MAIN_GATE, duration_s=21)],
        ),
    )
    with Store.open(":memory:") as store:
        result = run_video_live(
            VideoSource(demo_video), provider_for(demo_cassette_path), timeline, store
        )
        assert [a.rule_id for a in store.query_alerts()] == ["low_battery"]
    first = result.first_alert_step
    assert (
        first is not None and first.offset_s <= 2.0
    )  # no camera event needed, and well before EOF
    assert len(result.steps) - result.steps.index(first) - 1 >= 8


# ---- one shared downstream consumer ------------
def test_the_video_path_and_the_scripted_path_share_one_observation_consumer(
    demo_video, demo_cassette_path, monkeypatch
):
    consumed: list[tuple[str, str]] = []
    original = ObservationPipeline.consume

    def counting(self, obs, at, offset_s):
        consumed.append((type(self).__module__, obs.source))
        return original(self, obs, at, offset_s)

    monkeypatch.setattr(ObservationPipeline, "consume", counting)
    with Store.open(":memory:") as store:
        PatrolEngine(load_scenario(SCENARIO_DIR / "normal_day.yaml"), store).run()
    scripted = list(consumed)
    consumed.clear()
    with Store.open(":memory:") as store:
        live(demo_video, demo_cassette_path, store)
    video = list(consumed)
    assert scripted and video  # the SAME method handled both kinds of observation
    assert {m for m, _ in scripted + video} == {consumer_module.__name__}
    assert {s.split("+")[0] for _, s in scripted} == {"scripted"}
    assert {s.split("+")[0] for _, s in video} == {"cassette"}  # the source label survives


def test_neither_producer_has_its_own_event_or_rule_code():
    for module in (engine_module, live_module):
        source = inspect.getsource(module)
        for name in ("EventAggregator", "EntityLinker", "RuleEngine", "ContextEngine"):
            assert name not in source, f"{module.__name__} must not implement {name} itself"
    # the single place that wires observation -> events -> rules -> alerts
    wiring = inspect.getsource(consumer_module)
    assert "ContextEngine" in wiring and "SecurityReasoner" in wiring
    with Store.open(":memory:") as store:
        pipe = PatrolEngine(load_scenario(SCENARIO_DIR / "normal_day.yaml"), store).pipeline
        assert isinstance(pipe, ObservationPipeline)
        assert pipe.context is not None and pipe.reasoner is not None


# ---- the command line ------------
def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


def test_ingest_video_live_persists_events_and_alerts_without_further_commands(
    demo_video, demo_cassette_path, tmp_path
):
    db = tmp_path / "live.db"
    args = ("ingest-video", demo_video, "--db", db, "--provider", "cassette")
    code, text = run_cli(*args, "--cassette", demo_cassette_path, "--sample-every", 2, "--live")
    assert code == 0, text
    assert "ALERT RAISED" in text and "LIVE" in text and "7 sampled frame(s) to come" in text
    assert text.index("ALERT RAISED") < text.index("FRAME 180")  # printed before the last frame
    with Store.open(db) as store:  # no build-events, no analyze
        assert store.count_events() == 2 and store.count_alerts() == 1
        assert store.query_observations(object_type=ObjectType.PERSON)
    again, message = run_cli(*args, "--cassette", demo_cassette_path, "--sample-every", 2, "--live")
    assert again == 1 and "already holds" in message
    with Store.open(db) as store:
        assert store.count_observations() == 10 and store.count_alerts() == 1  # unchanged


def test_the_default_ingest_video_is_still_the_batch_path(demo_video, demo_cassette_path, tmp_path):
    db = tmp_path / "batch.db"
    code, text = run_cli(
        "ingest-video", demo_video, "--db", db, "--provider", "cassette",
        "--cassette", demo_cassette_path, "--sample-every", 2,
    )  # fmt: skip
    assert code == 0 and "LIVE" not in text and "ALERT RAISED" not in text
    with Store.open(db) as store:
        assert store.count_observations() == 10
        assert store.count_events() == 0 and store.count_alerts() == 0  # needs build-events


# ---- aborting on a provider failure is now an explicit choice (V02 continues by default) ------
def test_the_abort_policy_keeps_what_was_already_committed(demo_video, demo_cassette_path):
    class FailsAt16:
        name = "flaky"

        def __init__(self, inner):
            self.inner = inner

        def analyze(self, frame):
            if frame.offset_seconds >= 16:
                raise RuntimeError("model outage")
            return self.inner.analyze(frame)

    with Store.open(":memory:") as store:
        with pytest.raises(RuntimeError, match="model outage"):
            live(
                demo_video, None, store, provider=FailsAt16(provider_for(demo_cassette_path)),
                on_failure=FailurePolicy.RAISE,
            )  # fmt: skip
        # the earlier frames, the person event and its alert are committed: nothing was lost
        assert store.count_observations() == 8 and store.count_alerts() == 1
        assert store.get_event("EVT-P-001") is not None
