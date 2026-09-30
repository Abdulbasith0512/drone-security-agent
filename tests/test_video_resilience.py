"""V02: bounded carry and provider-failure resilience for the video path.

Two promises are tested.

1. A scene is never carried forever. After `max_carry_seconds` (and optionally
   `max_carried_frames`) the next frame goes to the provider whatever the pixels say, and the
   result is a FRESH observation, not a carried one. Carried frames stay labelled `+carried`.
2. A provider failure never produces an observation. The frame gets an explicit "failed" outcome,
   nothing already committed is touched, telemetry keeps flowing, stale observations are not
   carried over the failed frame, and the next frame is a fresh attempt.

Deterministic stub providers and cassettes are used throughout; nothing here says how accurate a
real vision model is.
"""

import time
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import pytest
from conftest import IST, SCENARIO_DIR, write_video
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.agents import SamplerConfig
from drone_agent.ingest import ingest_observations
from drone_agent.models import Observation, Provenance, Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.runtime import EngineOptions, PatrolEngine
from drone_agent.runtime.video import run_video_live
from drone_agent.scenario import Scenario, load_scenario
from drone_agent.store import Store
from drone_agent.telemetry import PatrolLeg, TelemetryConfig
from drone_agent.video import (
    DEFAULT_MAX_CARRY_SECONDS,
    FailurePolicy,
    SamplingConfig,
    VideoIngestConfig,
    VideoSource,
    VideoTimeline,
    run_video_ingest,
    single_zone_timeline,
)
from drone_agent.video.ingest import classify_failure
from drone_agent.vision import Cassette, CassetteDescriber, CassetteMiss, DescriberProvider
from drone_agent.vision.contract import ProviderOutputError

NIGHT = datetime(2026, 9, 29, 23, 50, tzinfo=IST)
DAY = datetime(2026, 9, 29, 14, 0, tzinfo=IST)


# ---- helpers ----------
class Stub:
    """A deterministic provider: describes a frame from its pixels and counts its calls."""

    name = "stub"

    def __init__(self):
        self.calls: list[float] = []

    def analyze(self, frame) -> Observation:
        self.calls.append(frame.offset_seconds)
        marker = bool(frame.image is not None and frame.image[0, 0, 0] > 150)
        return Observation(
            observation_id=f"obs-{frame.frame_id}",
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            source="stub",
            zone=frame.zone,
            description="marker present" if marker else "nothing to see",
            provenance=Provenance.CANNED,
        )


def grey(value=100, marker=False):
    image = np.full((90, 160, 3), value, np.uint8)
    if marker:
        image[0:2, 0:2] = 255  # a tiny patch: far below the gate's 6.0 mean-difference threshold
    return image


def static_video(path, seconds, marker_from=None):
    frames = [grey(marker=marker_from is not None and t >= marker_from) for t in range(seconds + 1)]
    return write_video(path, frames, fps=1.0)


def run(video, *, every=2.0, provider=None, timeline=None, **config):
    source = VideoSource(video)
    timeline = timeline or single_zone_timeline(
        DAY, Zone.MAIN_GATE, source.metadata.duration_seconds
    )
    cfg = VideoIngestConfig(sampling=SamplingConfig(every_seconds=every), **config)
    return run_video_ingest(source, provider or Stub(), timeline, cfg)


def ages(result):
    """Video seconds since the last fresh (analysed) frame, for every carried candidate."""
    fresh, out = None, []
    for c in result.candidates:
        if c.action == "analyzed":
            fresh = c.offset_seconds
        elif c.action == "carried":
            out.append(c.offset_seconds - fresh)
    return out


def cassette_provider(path):
    return DescriberProvider(CassetteDescriber(Cassette.load(path)))


class Flaky:
    """Fails with a chosen exception at chosen video offsets, else answers like the cassette."""

    name = "flaky"

    def __init__(self, inner, failures: dict[float, Exception]):
        self.inner, self.failures, self.seen = inner, failures, []

    def analyze(self, frame):
        self.seen.append(frame.offset_seconds)
        if frame.offset_seconds in self.failures:
            raise self.failures[frame.offset_seconds]
        return self.inner.analyze(frame)


# ---- the carry bound ----------
def test_the_default_bound_is_the_adaptive_samplers_policy_and_is_validated():
    assert DEFAULT_MAX_CARRY_SECONDS == SamplerConfig().max_skip_seconds == 300.0
    assert VideoIngestConfig().max_carry_seconds == 300.0
    for bad in ({"max_carry_seconds": 0}, {"max_carried_frames": 0}):
        with pytest.raises(ValueError):
            VideoIngestConfig(**bad)


def test_with_the_bound_switched_off_a_static_scene_is_carried_for_ever(tmp_path):
    """The behaviour before V02, kept reachable only on purpose: one call for a whole minute."""
    video = static_video(tmp_path / "static.avi", 60)
    stub = Stub()
    result = run(video, provider=stub, max_carry_seconds=None)
    assert stub.calls == [0.0] and result.carried == 30 and result.forced_refreshes == 0
    assert max(ages(result)) == 60.0  # the last carried frame repeats a minute-old answer


def test_a_long_run_of_unchanged_frames_cannot_become_unbounded_stale_perception(tmp_path):
    video = static_video(tmp_path / "long.avi", 700)
    stub = Stub()
    result = run(video, every=10.0, provider=stub)  # the default bound, 300 s
    assert stub.calls == [0.0, 300.0, 600.0]  # a fresh look at least every 300 s
    assert result.forced_refreshes == 2 and result.failed == 0
    assert all(age < DEFAULT_MAX_CARRY_SECONDS for age in ages(result))
    assert result.carried == 71 - 3  # 71 sampled frames: everything else is carried


def test_a_frame_count_bound_limits_how_many_frames_one_answer_can_cover(tmp_path):
    video = static_video(tmp_path / "s.avi", 60)
    stub = Stub()
    result = run(video, provider=stub, max_carry_seconds=None, max_carried_frames=4)
    assert stub.calls == [float(t) for t in range(0, 62, 10)][: len(stub.calls)]  # every 5th frame
    longest = run_lengths([c.action for c in result.candidates], "carried")
    assert longest == 4 and result.forced_refreshes == len(stub.calls) - 1


def run_lengths(actions, which):
    best = current = 0
    for action in actions:
        current = current + 1 if action == which else 0
        best = max(best, current)
    return best


def test_a_forced_refresh_is_a_fresh_perception_and_carried_frames_stay_labelled(tmp_path):
    video = static_video(tmp_path / "s.avi", 30)
    source = VideoSource(video)
    timeline = single_zone_timeline(DAY, Zone.MAIN_GATE, source.metadata.duration_seconds)
    cfg = VideoIngestConfig(sampling=SamplingConfig(every_seconds=2.0), max_carry_seconds=10.0)
    result = run_video_ingest(source, Stub(), timeline, cfg)
    by_offset = {c.offset_seconds: c for c in result.candidates}
    refreshed = by_offset[10.0]
    assert (refreshed.action, refreshed.gate_reason) == ("analyzed", "forced_refresh")
    assert refreshed.gate_score == 0.0  # the pixels had not changed; the BOUND forced the call
    stored = {o.frame_id: o for o in result.observations}
    assert stored[10].source == "stub" and not stored[10].source.endswith("+carried")
    assert stored[8].source == "stub+carried" and stored[8].provenance is Provenance.CANNED
    assert {o.source for o in result.observations} == {"stub", "stub+carried"}


def test_a_change_too_small_for_the_gate_is_noticed_within_the_bound(tmp_path):
    video = static_video(tmp_path / "small.avi", 60, marker_from=25)  # a 2x2 patch appears at 25 s
    unbounded = run(video, provider=Stub(), max_carry_seconds=None)
    assert {o.description for o in unbounded.observations} == {"nothing to see"}  # never noticed
    bounded = run(video, provider=Stub(), max_carry_seconds=10.0)
    first_seen = next(o for o in bounded.observations if o.description == "marker present")
    assert 25 <= first_seen.frame_id <= 25 + 10  # noticed within one bound of the change
    assert first_seen.source == "stub"  # by a fresh perception, not a carry


def test_a_large_change_overrides_the_carry_without_waiting_for_the_bound(demo_video):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    result = run_video_ingest(source, Stub(), timeline)  # default 300 s bound, 20 s video
    changed = [c for c in result.candidates if c.gate_reason == "changed"]
    assert [c.offset_seconds for c in changed] == [4.0, 12.0, 16.0]  # pixel changes, bound unused
    assert result.forced_refreshes == 0


def test_a_new_zone_is_never_carried_into_and_starts_fresh(tmp_path):
    video = static_video(tmp_path / "z.avi", 20)
    timeline = VideoTimeline(
        start=DAY,
        telemetry=TelemetryConfig(
            patrol=[
                PatrolLeg(zone=Zone.MAIN_GATE, duration_s=11),
                PatrolLeg(zone=Zone.DRIVEWAY, duration_s=11),
            ]
        ),
    )
    stub = Stub()
    result = run(video, provider=stub, timeline=timeline, max_carry_seconds=None)
    second_zone = [c for c in result.candidates if c.zone == "driveway"]
    assert (second_zone[0].action, second_zone[0].gate_reason) == ("analyzed", "first")
    assert all(o.zone is Zone.DRIVEWAY for o in result.observations if o.frame_id >= 12)


# ---- the scripted path keeps T01's change semantics inside the same bound ----------
def scripted(frames):
    return Scenario.model_validate(
        {
            "name": "bounded",
            "start": "2026-09-29T12:00:00+05:30",
            "telemetry": {
                "seed": 1,
                "sample_interval_s": 10,
                "patrol": [{"zone": "driveway", "duration_s": 300}],
            },
            "frames": [
                {
                    "frame_id": i,
                    "timestamp_offset": 10 * i,
                    "zone": "driveway",
                    "description": description,
                    "objects": objects,
                }
                for i, (description, objects) in enumerate(frames, start=1)
            ],
        }
    )


def person(activity="standing"):
    return {
        "type": "person",
        "activity": activity,
        "attributes": {"clothing": "red"},
        "confidence": 0.9,
    }


def test_scripted_unchanged_frames_are_also_refreshed_at_the_sampler_bound():
    scenario = scripted([("Driveway clear", [])] * 25)
    options = EngineOptions(sampler=SamplerConfig(max_skip_seconds=60, watch_interval_s=1000))
    with Store.open(":memory:") as store:
        summary = PatrolEngine(scenario, store, options=options).run()
        sources = [o.source for o in store.query_observations()]
    assert summary.model_calls == 5  # frames 1, 7, 13, 19, 25: never more than 60 s without a call
    assert summary.sampler["by_reason"]["forced_refresh"] == 4
    assert sources.count("scripted") == 5 and sources.count("scripted+carried") == 20


@pytest.mark.parametrize(
    ("change", "what"),
    [
        (("Person walking", [person("walking")]), "activity"),
        (("Two people standing", [person(), person()]), "object count"),
        (("A person is at the gate", [person()]), "description"),
    ],
)
def test_scripted_changes_override_carry_and_restart_the_bound(change, what):
    base = ("Person standing", [person()])
    frames = [base, base, change, *[change] * 12]
    options = EngineOptions(sampler=SamplerConfig(max_skip_seconds=60, watch_interval_s=1000))
    with Store.open(":memory:") as store:
        summary = PatrolEngine(scripted(frames), store, options=options).run()
        stored = {o.frame_id: o for o in store.query_observations()}
    assert stored[3].source == "scripted" and not stored[3].source.endswith("+carried"), what
    # frame 3 was analysed early (the change), so the bound restarted there: forced refresh at 9
    assert stored[9].source == "scripted" and stored[8].source == "scripted+carried"
    assert summary.sampler["by_reason"].get("scene_change", 0) >= 1


# ---- provider failure ----------
KINDS = [
    (ProviderOutputError("objects[0].type is not a string"), "malformed_output"),
    (CassetteMiss("no recording for frame 40"), "replay_miss"),
    (TimeoutError("read timed out"), "timeout"),
    (RuntimeError("503 service unavailable"), "provider_error"),
]


@pytest.mark.parametrize(("exc", "kind"), KINDS, ids=[k for _, k in KINDS])
def test_a_failed_frame_is_an_explicit_outcome_with_a_classified_reason(
    demo_video, demo_cassette_path, exc, kind
):
    assert classify_failure(exc) == kind
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = Flaky(cassette_provider(demo_cassette_path), {4.0: exc})
    with Store.open(":memory:") as store:
        result = run_video_live(source, provider, timeline, store)
        assert store.get_observation_by_frame(40) is None  # nothing was stored for it
    failed = result.failures
    assert [(s.offset_s, s.action) for s in failed] == [(4.0, "failed")]
    assert failed[0].error.startswith(f"{kind}: ") and type(exc).__name__ in failed[0].error


def test_nothing_is_fabricated_and_no_alert_comes_from_a_failed_perception(
    demo_video, demo_cassette_path
):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = Flaky(
        cassette_provider(demo_cassette_path),
        {4.0: RuntimeError("down"), 6.0: RuntimeError("down")},
    )
    with Store.open(":memory:") as store:
        result = run_video_live(source, provider, timeline, store)
        person = store.get_event("EVT-P-001")
        alert = store.get_alert("ALR-AFTERHOURS-EVT-P-001")
    by_offset = {s.offset_s: s for s in result.steps}
    assert by_offset[4.0].alerts == by_offset[6.0].alerts == 0  # no alert from a failed frame
    assert by_offset[4.0].events == by_offset[6.0].events == 0  # and no event either
    # the person is first OBSERVED at 8 s, so that is when the event and the alert begin
    assert by_offset[8.0].new_alerts == ("ALR-AFTERHOURS-EVT-P-001",)
    assert person.first_seen == NIGHT + timedelta(seconds=8) == alert.first_seen
    observed_frames = {o.frame_id for o in result.result.observations}
    assert 40 not in observed_frames and 60 not in observed_frames


def test_the_frame_after_a_failure_is_a_fresh_attempt_never_a_stale_carry(
    demo_video, demo_cassette_path
):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = Flaky(cassette_provider(demo_cassette_path), {4.0: RuntimeError("blip")})
    with Store.open(":memory:") as store:
        result = run_video_live(source, provider, timeline, store)
        recovered = store.get_observation_by_frame(60)
    actions = {s.offset_s: s.action for s in result.steps}
    assert actions[2.0] == "carried" and actions[4.0] == "failed"
    assert actions[6.0] == "analyzed"  # NOT carried from the empty-yard frame at 0 s
    assert recovered.source == "cassette" and "person" in recovered.description.lower()
    assert len(result.steps) == 10 and len(result.failures) == 1  # the run recovered and finished


def test_what_was_committed_before_a_failure_is_untouched(demo_video, demo_cassette_path):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = Flaky(cassette_provider(demo_cassette_path), {12.0: RuntimeError("down")})
    snapshots = {}
    with Store.open(":memory:") as store:

        def probe(step):
            snapshots[step.offset_s] = (
                store.count_observations(),
                store.get_event("EVT-P-001"),
                store.get_alert("ALR-AFTERHOURS-EVT-P-001"),
            )

        run_video_live(source, provider, timeline, store, on_step=probe)
    before, at_failure = snapshots[10.0], snapshots[12.0]
    assert at_failure[0] == before[0]  # the failed frame added no observation
    assert at_failure[2].status is before[2].status and at_failure[2].severity is before[2].severity
    assert (
        at_failure[1].evidence_count == before[1].evidence_count
    )  # the event is exactly as it was
    assert at_failure[1].is_open  # and it was NOT closed or resolved because a frame was missing


def test_a_failed_frame_is_a_gap_in_coverage_not_a_departure(demo_video, demo_cassette_path):
    """T02 still holds: only a same-zone observation WITHOUT the person counts as departure."""
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    failures = {
        t: RuntimeError("down") for t in (12.0, 16.0)
    }  # the frames that would show it empty
    provider = Flaky(cassette_provider(demo_cassette_path), failures)
    with Store.open(":memory:") as store:
        run_video_live(source, provider, timeline, store)
        person = store.get_event("EVT-P-001")
    assert person.departure_observed_at is None and person.close_reason.value != "missed_frames"


def test_telemetry_safety_keeps_working_while_every_frame_fails(demo_video):
    timeline = VideoTimeline(
        start=DAY,
        telemetry=TelemetryConfig(
            sample_interval_s=2,
            battery_start_percent=19.0,
            battery_drain_percent_per_min=0.0,
            patrol=[PatrolLeg(zone=Zone.MAIN_GATE, duration_s=21)],
        ),
    )

    class AlwaysDown:
        name = "down"

        def analyze(self, frame):
            raise RuntimeError("model outage")

    with Store.open(":memory:") as store:
        result = run_video_live(VideoSource(demo_video), AlwaysDown(), timeline, store)
        assert [a.rule_id for a in store.query_alerts()] == ["low_battery"]
        assert store.count_observations() == 0 and store.count_events() == 0
    assert len(result.failures) == 10 and result.first_alert_step is not None


def test_the_abort_policy_raises_and_leaves_committed_state_intact(demo_video, demo_cassette_path):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = Flaky(cassette_provider(demo_cassette_path), {16.0: RuntimeError("model outage")})
    with Store.open(":memory:") as store:
        with pytest.raises(RuntimeError, match="model outage"):
            run_video_live(source, provider, timeline, store, on_failure=FailurePolicy.RAISE)
        assert store.count_observations() == 8 and store.count_alerts() == 1


def test_a_real_timeout_becomes_a_failed_frame_and_the_run_moves_on(demo_video, demo_cassette_path):
    class SlowAt4(Flaky):
        def analyze(self, frame):
            if frame.offset_seconds == 4.0:
                time.sleep(0.4)
            return super().analyze(frame)

    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    provider = SlowAt4(cassette_provider(demo_cassette_path), {})
    with Store.open(":memory:") as store:
        result = run_video_live(source, provider, timeline, store, call_timeout_s=0.1)
        with pytest.raises(ValueError):
            run_video_live(source, provider, timeline, store, call_timeout_s=0)
    assert [(s.offset_s, s.action) for s in result.failures] == [(4.0, "failed")]
    assert result.failures[0].error.startswith("timeout: ")
    assert result.steps[5].action == "carried" and len(result.steps) == 10


def test_the_batch_default_still_raises_and_continue_records_failures(
    demo_video, demo_cassette_path
):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)

    def flaky():
        return Flaky(cassette_provider(demo_cassette_path), {4.0: RuntimeError("down")})

    with pytest.raises(RuntimeError, match="down"):  # unchanged default: abort, nothing stored
        run_video_ingest(source, flaky(), timeline)
    cfg = VideoIngestConfig(on_failure=FailurePolicy.CONTINUE)
    result = run_video_ingest(source, flaky(), timeline, cfg)
    assert result.failed == 1 and result.provider_calls == 5  # 4 fresh answers + the failed attempt
    assert 40 not in {o.frame_id for o in result.observations}


# ---- live and batch agree ----------
@pytest.mark.parametrize("bound", [None, 4.0, 6.0])
def test_live_and_batch_agree_under_a_carry_bound(demo_video, demo_cassette_path, bound):
    config = VideoIngestConfig(max_carry_seconds=bound)
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    with Store.open(":memory:") as batch, Store.open(":memory:") as progressive:
        ingested = run_video_ingest(source, cassette_provider(demo_cassette_path), timeline, config)
        ingest_observations(batch, "demo.avi", ingested.telemetry, ingested.observations)
        build_events(batch)
        analyze(batch)
        run_video_live(
            source, cassette_provider(demo_cassette_path), timeline, progressive, config=config
        )
        assert progressive.query_observations() == batch.query_observations()
        assert progressive.query_events() == batch.query_events()
        assert [a.alert_id for a in progressive.query_alerts()] == [
            a.alert_id for a in batch.query_alerts()
        ]


def test_the_default_bound_leaves_every_shipped_short_video_unchanged(
    demo_video, demo_cassette_path
):
    source = VideoSource(demo_video)
    timeline = single_zone_timeline(NIGHT, Zone.MAIN_GATE, source.metadata.duration_seconds)
    bounded = run_video_ingest(source, cassette_provider(demo_cassette_path), timeline)
    unbounded = run_video_ingest(
        source,
        cassette_provider(demo_cassette_path),
        timeline,
        VideoIngestConfig(max_carry_seconds=None),
    )
    assert bounded.observations == unbounded.observations
    assert (bounded.provider_calls, bounded.carried, bounded.forced_refreshes) == (4, 6, 0)


def test_scripted_scenarios_are_untouched_by_video_resilience():
    """The video changes live in the video stream; every shipped scenario behaves as before."""
    for path in sorted(SCENARIO_DIR.glob("*.yaml")):
        scenario = load_scenario(path)
        with Store.open(":memory:") as store:
            summary = PatrolEngine(scenario, store).run()
        assert summary.dropped == 0, path.name


# ---- the command line ----------
def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


def test_cli_reports_forced_refreshes_and_failures(demo_video, demo_cassette_path, tmp_path):
    base = ("ingest-video", demo_video, "--provider", "cassette", "--cassette", demo_cassette_path)
    code, text = run_cli(*base, "--db", tmp_path / "a.db", "--sample-every", 2, "--live",
                         "--max-carry-seconds", 4)  # fmt: skip
    assert code == 0 and "Forced refreshes" in text and "analysed forced_refresh" in text
    # sending every 0.5 s frame asks the cassette for frames it never recorded: replay misses
    code, text = run_cli(*base, "--db", tmp_path / "b.db", "--sample-every", 0.5, "--live",
                         "--no-gate")  # fmt: skip
    assert code == 3 and "PERCEPTION FAILURES" in text and "replay_miss" in text
    with Store.open(tmp_path / "b.db") as store:
        assert store.count_observations() > 0  # the frames that could be answered were kept
    code, text = run_cli(*base, "--db", tmp_path / "c.db", "--sample-every", 0.5, "--live",
                         "--no-gate", "--fail-fast")  # fmt: skip
    assert code == 2  # aborted at the first failure


def test_timeout_and_fail_fast_are_live_only_options(demo_video, demo_cassette_path, tmp_path):
    base = ("ingest-video", demo_video, "--provider", "cassette", "--cassette", demo_cassette_path)
    for extra in (("--call-timeout", 1), ("--fail-fast",)):
        code, text = run_cli(*base, "--db", tmp_path / "x.db", *extra)
        assert code == 2 and "--live runs only" in text
    assert not (tmp_path / "x.db").exists()  # refused before anything was opened or stored
