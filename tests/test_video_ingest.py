import json
from datetime import datetime
from io import StringIO

import pytest
from conftest import IST, SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.ingest import ingest_observations, ingest_result
from drone_agent.models import ObjectType, Observation, Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store
from drone_agent.video import (
    GateConfig,
    SkippedPolicy,
    TimelineError,
    VideoIngestConfig,
    VideoSource,
    load_timeline,
    run_video_ingest,
    single_zone_timeline,
)
from drone_agent.vision import (
    Cassette,
    CassetteDescriber,
    CassetteMiss,
    DescriberProvider,
    ScriptedProvider,
)

START = datetime(2026, 9, 29, 23, 50, tzinfo=IST)


def provider_for(cassette_path) -> tuple[DescriberProvider, CassetteDescriber]:
    describer = CassetteDescriber(Cassette.load(cassette_path))
    return DescriberProvider(describer), describer


def run(demo_video, demo_cassette_path, **config):
    source = VideoSource(demo_video)
    provider, describer = provider_for(demo_cassette_path)
    timeline = single_zone_timeline(START, Zone.MAIN_GATE, source.metadata.duration_seconds)
    result = run_video_ingest(source, provider, timeline, VideoIngestConfig(**config))
    return result, describer


def run_cli(*argv) -> tuple[int, str]:
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=240))
    return code, out.getvalue()


# ---- pipeline -------------------------------------------------------------------------------
def test_video_becomes_ordinary_validated_observations(demo_video, demo_cassette_path):
    result, describer = run(demo_video, demo_cassette_path)
    assert result.metadata.duration_seconds == 20.0 and result.sampling.planned == 10
    assert (result.gate_accepted, result.gate_skipped) == (4, 6)
    assert (result.provider_calls, result.carried) == (4, 6) and describer.replayed == 4
    assert len(result.observations) == 10
    assert all(isinstance(o, Observation) for o in result.observations)
    first = result.observations[0]
    assert (first.frame_id, first.zone, first.source) == (0, Zone.MAIN_GATE, "cassette")
    assert first.timestamp == START and first.telemetry is not None and not first.telemetry.stale
    # frame index 40 is at 4.0 s: the person scene
    person = next(o for o in result.observations if o.frame_id == 40)
    assert person.timestamp == START.replace(minute=50, second=4) and person.description
    assert person.detected_objects[0].object_type is ObjectType.PERSON
    assert person.source == "cassette" and person.observation_id == "obs-40"


def test_gate_skipped_frames_are_carried_forward_and_labelled(demo_video, demo_cassette_path):
    result, _ = run(demo_video, demo_cassette_path)
    by_frame = {o.frame_id: o for o in result.observations}
    carried = by_frame[60]  # 6.0 s: identical to the person frame analysed at 4.0 s
    assert carried.source == "cassette+carried"
    assert carried.description == by_frame[40].description
    assert carried.detected_objects == by_frame[40].detected_objects
    assert carried.timestamp > by_frame[40].timestamp
    actions = {c.frame_index: c.action for c in result.candidates}
    assert [i for i, a in actions.items() if a == "analyzed"] == [0, 40, 120, 160]


def test_drop_policy_produces_only_the_analysed_frames(demo_video, demo_cassette_path):
    result, _ = run(demo_video, demo_cassette_path, skipped=SkippedPolicy.DROP)
    assert [o.frame_id for o in result.observations] == [0, 40, 120, 160]
    assert result.carried == 0 and result.gate_skipped == 6


def test_disabled_gate_analyses_every_sample(demo_video, demo_cassette_path):
    result, describer = run(demo_video, demo_cassette_path, gate=GateConfig(enabled=False))
    assert (result.gate_accepted, result.gate_skipped, describer.replayed) == (10, 0, 10)
    assert len(result.observations) == 10 and result.carried == 0


def test_a_zone_change_resets_the_gate_and_never_carries_objects_across_zones(
    demo_video, demo_cassette_path, tmp_path
):
    timeline_file = tmp_path / "timeline.yaml"
    timeline_file.write_text(
        'start: "2026-09-29T23:50:00+05:30"\n'
        "telemetry:\n  patrol:\n"
        "    - {zone: main_gate, duration_s: 10}\n    - {zone: driveway, duration_s: 15}\n",
        encoding="utf-8",
    )
    source = VideoSource(demo_video)
    provider, _ = provider_for(demo_cassette_path)
    result = run_video_ingest(source, provider, load_timeline(timeline_file), VideoIngestConfig())
    at_10s = next(c for c in result.candidates if c.frame_index == 100)
    # the person scene is identical to the previous sample, but the drone is in a new zone
    assert (at_10s.zone, at_10s.gate_reason, at_10s.action) == ("driveway", "first", "analyzed")
    zones = {o.frame_id: o.zone for o in result.observations}
    assert zones[80] is Zone.MAIN_GATE and zones[100] is Zone.DRIVEWAY
    assert result.gate_accepted == 5  # 0, 40, 100 (zone change), 120, 160


def test_timeline_too_short_for_the_video_is_rejected(demo_video, demo_cassette_path):
    source = VideoSource(demo_video)
    provider, _ = provider_for(demo_cassette_path)
    short = single_zone_timeline(START, Zone.MAIN_GATE, 5.0)
    with pytest.raises(TimelineError):
        run_video_ingest(source, provider, short, VideoIngestConfig())


def test_a_cassette_miss_aborts_the_run(demo_video, demo_cassette_path):
    from drone_agent.video import SamplingConfig

    # The cassette only has 1 s frames. With the gate on, the extra half-second samples are
    # identical to accepted frames and are never sent to the provider, so nothing is missing...
    result, _ = run(demo_video, demo_cassette_path, sampling=SamplingConfig(every_seconds=0.5))
    assert result.gate_accepted == 4
    # ...but with the gate off every sample is sent, and the first one without an entry fails.
    with pytest.raises(CassetteMiss, match="frame index 5"):
        run(
            demo_video,
            demo_cassette_path,
            sampling=SamplingConfig(every_seconds=0.5),
            gate=GateConfig(enabled=False),
        )


# ---- into the existing pipeline -------------------------------------------------------------
def test_video_observations_go_through_the_unchanged_store(store, demo_video, demo_cassette_path):
    result, _ = run(demo_video, demo_cassette_path)
    summary = ingest_observations(store, "demo.avi", result.telemetry, result.observations)
    assert (summary.observations, summary.scenario) == (10, "demo.avi")
    # SIMULATED telemetry: a 21 s patrol sampled every 10 s (offsets 0, 10, 20)
    assert summary.telemetry_samples == len(result.telemetry) == 3
    assert summary.objects == sum(len(o.detected_objects) for o in result.observations)
    stored = store.query_observations()
    assert stored == result.observations  # rebuilt from SQLite, telemetry association included
    assert all(type(o) is Observation for o in stored)
    assert all(Observation.model_validate_json(o.model_dump_json()) == o for o in stored)
    assert [o.frame_id for o in store.query_observations(text="blue pickup")] == [120, 140]
    assert store._conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_events_entities_and_alerts_are_built_from_video_observations_unchanged(
    store, demo_video, demo_cassette_path
):
    result, _ = run(demo_video, demo_cassette_path)
    ingest_observations(store, "demo.avi", result.telemetry, result.observations)
    summary = build_events(store)
    assert (summary.events, summary.person_events, summary.vehicle_events) == (2, 1, 1)
    person = store.query_events(object_type=ObjectType.PERSON)[0]
    assert person.frame_ids == [40, 60, 80, 100] and person.duration_seconds == 6.0
    vehicle = store.query_events(object_type=ObjectType.VEHICLE)[0]
    assert vehicle.attributes == {"color": "blue", "body_type": "pickup"}
    assert store.get_entity(vehicle.entity_id).label == "similar blue pickup"
    alerts = analyze(store)
    # 23:50 local is inside the 22:00-06:00 closed window, so the person raises exactly one alert
    assert alerts.by_rule == {"after_hours_presence": 1}


def test_dropping_skipped_frames_shortens_the_person_event(store, demo_video, demo_cassette_path):
    result, _ = run(demo_video, demo_cassette_path, skipped=SkippedPolicy.DROP)
    ingest_observations(store, "demo.avi", result.telemetry, result.observations)
    build_events(store)
    person = store.query_events(object_type=ObjectType.PERSON)[0]
    assert person.frame_ids == [40] and person.duration_seconds == 0  # why carrying is the default


def test_scripted_and_video_observations_are_the_same_model(demo_video, demo_cassette_path):
    scripted = run_scenario(load_scenario(SCENARIO_DIR / "normal_day.yaml")).observations[0]
    video, _ = run(demo_video, demo_cassette_path)
    assert type(scripted) is type(video.observations[0]) is Observation
    assert isinstance(ScriptedProvider(load_scenario(SCENARIO_DIR / "normal_day.yaml")).name, str)
    assert set(scripted.model_dump()) == set(video.observations[0].model_dump())


# ---- M1-M4 regression -----------------------------------------------------------------------
def test_scripted_pipeline_still_produces_the_m4_facts(tmp_path):
    with Store.open(tmp_path / "sec.db") as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        built = build_events(s)
        analysis = analyze(s)
    assert (built.events, built.entities, built.multi_appearance_entities) == (7, 2, 1)
    assert analysis.alerts == 6 and built.detections == 20
    with Store.open(tmp_path / "day.db") as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "normal_day.yaml")))
        build_events(s)
        assert analyze(s).alerts == 0


# ---- CLI ------------------------------------------------------------------------------------
@pytest.fixture
def cli_files(tmp_path):
    code, text = run_cli("make-demo-video", "--out-dir", tmp_path / "demo")
    assert code == 0
    return tmp_path / "demo" / "demo.avi", tmp_path / "demo" / "demo.cassette.json", text


def test_make_demo_video_writes_a_video_and_a_cassette_and_labels_the_source(cli_files):
    video, cassette, text = cli_files
    assert video.is_file() and cassette.is_file()
    assert "synthetic-demo" in text and "not a vision model" in text
    assert json.loads(cassette.read_text())["provider"].startswith("synthetic-demo")


def test_cli_ingest_video_prints_the_pipeline_summary_and_queries_work(cli_files, tmp_path):
    video, cassette, _ = cli_files
    db = tmp_path / "video.db"
    code, text = run_cli(
        "ingest-video", video, "--db", db, "--provider", "cassette",
        "--cassette", cassette, "--sample-every", 2, "--verbose",
    )  # fmt: skip
    assert code == 0
    for expected in (
        "demo.avi", "20.00s", "Original frames", "200", "Sample candidates", "10 every 2s",
        "Frame gate accepted", "Frame gate skipped", "carried 6", "Provider calls",
        "cassette replays 4", "Observations written", "Objects written",
        "SIMULATED telemetry samples", "not from the video", "Sample candidates",
        "changed", "similar",
    ):  # fmt: skip
        assert expected in text
    code, text = run_cli("query", "--db", db, "--search", "blue pickup")
    assert code == 0 and "2 observation(s)" in text and "A blue pickup truck is parked" in text
    code, text = run_cli("query", "--db", db, "--object", "person")
    assert "4 observation(s)" in text


def test_cli_ingest_video_options_change_the_result(cli_files, tmp_path):
    video, cassette, _ = cli_files
    base = ["ingest-video", video, "--cassette", cassette, "--sample-every", 2]
    code, text = run_cli(*base, "--db", tmp_path / "a.db", "--no-gate")
    assert code == 0 and "gate disabled" in text and "cassette replays 10" in text
    code, text = run_cli(*base, "--db", tmp_path / "b.db", "--skipped", "drop")
    assert code == 0 and "policy: drop; carried 0" in text
    code, text = run_cli(*base, "--db", tmp_path / "c.db", "--gate-threshold", 100)
    assert code == 0 and "threshold 100" in text  # nothing changes enough: only the first passes


def test_cli_ingest_video_downstream_commands_work_on_the_video_database(cli_files, tmp_path):
    video, cassette, _ = cli_files
    db = tmp_path / "v.db"
    assert run_cli("ingest-video", video, "--db", db, "--cassette", cassette)[0] == 0
    assert run_cli("build-events", "--db", db)[0] == 0
    code, text = run_cli("events", "--db", db)
    assert code == 0 and "2 event(s)" in text
    assert run_cli("analyze", "--db", db)[0] == 0
    code, text = run_cli("alerts", "--db", db)
    assert "1 alert(s)" in text and "Person present after hours at main_gate" in text


@pytest.mark.parametrize(
    "extra,fragment",
    [
        ([], "needs --cassette"),
        (["--cassette", "missing.json"], "cannot read cassette"),
        (["--cassette", "CASSETTE", "--sample-every", 0.5, "--no-gate"], "no recorded response"),
        (["--cassette", "CASSETTE", "--sample-every", 0], "every_seconds"),
        (["--cassette", "CASSETTE", "--gate-threshold", 999], "threshold"),
        (
            [
                "--cassette",
                "CASSETTE",
                "--zone",
                "warehouse",
                "--start",
                "2026-09-29T00:00:00+00:00",
            ],
            None,
        ),
    ],
)
def test_cli_ingest_video_failures_are_clear_and_leave_no_database(
    cli_files, tmp_path, extra, fragment
):
    video, cassette, _ = cli_files
    db = tmp_path / "fail.db"
    args = [str(cassette) if a == "CASSETTE" else a for a in extra]
    code, text = run_cli("ingest-video", video, "--db", db, *args)
    if fragment is None:  # a valid variation: different simulated zone and start
        assert code == 0
    else:
        assert code == 2 and fragment in text
        assert not db.exists()  # the store is only opened after the whole pipeline succeeded


def test_cli_rejects_bad_videos_cassettes_and_repeat_ingests(cli_files, tmp_path):
    video, cassette, _ = cli_files
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"not a video" * 50)
    code, text = run_cli("ingest-video", bad, "--db", tmp_path / "x.db", "--cassette", cassette)
    assert code == 2 and "cannot open video" in text
    code, text = run_cli(
        "ingest-video", tmp_path / "nope.avi", "--db", tmp_path / "x.db", "--cassette", cassette
    )
    assert code == 2 and "not found" in text

    import numpy as np
    from conftest import write_video

    other = write_video(tmp_path / "other.avi", [np.full((36, 64, 3), 9, np.uint8)] * 30)
    code, text = run_cli("ingest-video", other, "--db", tmp_path / "y.db", "--cassette", cassette)
    assert code == 2 and "different video" in text

    db = tmp_path / "twice.db"
    assert run_cli("ingest-video", video, "--db", db, "--cassette", cassette)[0] == 0
    code, text = run_cli("ingest-video", video, "--db", db, "--cassette", cassette)
    assert code == 1 and "already" in text and "nothing" in text


def test_cli_rejects_a_malformed_gate_resize_and_a_malformed_cassette(cli_files, tmp_path):
    video, cassette, _ = cli_files
    with pytest.raises(SystemExit):
        run_cli("ingest-video", video, "--db", tmp_path / "z.db", "--gate-resize", "big")
    broken = tmp_path / "broken.json"
    data = json.loads(cassette.read_text())
    data["entries"][0]["output"]["objects"] = [{"type": "truck"}]
    broken.write_text(json.dumps(data), encoding="utf-8")
    code, text = run_cli("ingest-video", video, "--db", tmp_path / "z.db", "--cassette", broken)
    assert code == 2 and "invalid cassette" in text
