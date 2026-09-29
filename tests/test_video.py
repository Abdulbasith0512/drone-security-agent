from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pytest
from conftest import IST, write_video
from pydantic import ValidationError

from drone_agent.models import Zone
from drone_agent.video import (
    FrameGate,
    GateConfig,
    SamplingConfig,
    SamplingReport,
    TimelineError,
    VideoError,
    VideoMetadata,
    VideoSource,
    load_timeline,
    plan_samples,
    sample_frames,
    single_zone_timeline,
)


def solid(value: int, size=(64, 36)) -> np.ndarray:
    return np.full((size[1], size[0], 3), value, np.uint8)


def meta(fps: float, frames: int) -> VideoMetadata:
    return VideoMetadata(
        source_name="x.avi",
        sha256="0" * 64,
        fps=fps,
        frame_count=frames,
        width=64,
        height=36,
        duration_seconds=frames / fps,
    )


def collect(source, config):
    report = SamplingReport(
        source.metadata.frame_count, source.metadata.duration_seconds, source.metadata.fps
    )
    frames = list(sample_frames(source, config, report))
    return frames, report


# ---- VideoSource ----------------------------------------------------------------------------
def test_metadata_of_the_demo_video(demo_video):
    m = VideoSource(demo_video).metadata
    assert (m.source_name, m.fps, m.frame_count, (m.width, m.height)) == (
        "demo.avi",
        10.0,
        200,
        (320, 180),
    )
    assert m.duration_seconds == 20.0
    assert len(m.sha256) == 64 and m.sha256 == VideoSource(demo_video).metadata.sha256


def test_missing_and_empty_and_directory_paths_fail_clearly(tmp_path):
    with pytest.raises(VideoError, match="not found"):
        VideoSource(tmp_path / "nope.avi")
    empty = tmp_path / "empty.avi"
    empty.write_bytes(b"")
    with pytest.raises(VideoError, match="empty"):
        VideoSource(empty)
    with pytest.raises(VideoError, match="not found"):
        VideoSource(tmp_path)  # a directory is not a video file


def test_corrupt_and_truncated_videos_fail_clearly(tmp_path, demo_video):
    garbage = tmp_path / "garbage.avi"
    garbage.write_bytes(b"this is definitely not a video" * 100)
    with pytest.raises(VideoError):
        VideoSource(garbage)
    truncated = tmp_path / "truncated.avi"
    truncated.write_bytes(demo_video.read_bytes()[:300])
    with pytest.raises(VideoError):
        VideoSource(truncated)


# ---- sampling plan --------------------------------------------------------------------------
def test_sampling_is_deterministic_and_time_based(demo_video):
    source = VideoSource(demo_video)
    plan = plan_samples(source.metadata, SamplingConfig(every_seconds=2))
    assert plan == plan_samples(source.metadata, SamplingConfig(every_seconds=2))
    assert [i for i, _ in plan] == list(range(0, 200, 20))
    assert [t for _, t in plan] == [float(t) for t in range(0, 20, 2)]

    first, second = (
        collect(source, SamplingConfig(every_seconds=2))[0],
        collect(source, SamplingConfig(every_seconds=2))[0],
    )
    assert [(f.frame_index, f.offset_seconds, f.frame_key, f.content_hash) for f in first] == [
        (f.frame_index, f.offset_seconds, f.frame_key, f.content_hash) for f in second
    ]


def test_sampled_frames_carry_identity_offsets_and_pixels(demo_video):
    source = VideoSource(demo_video)
    frames, report = collect(source, SamplingConfig(every_seconds=5))
    assert [f.frame_index for f in frames] == [0, 50, 100, 150]
    assert [f.offset_seconds for f in frames] == [0.0, 5.0, 10.0, 15.0]
    prefix = source.metadata.sha256[:16]
    assert [f.frame_key for f in frames] == [f"{prefix}:{i}" for i in (0, 50, 100, 150)]
    assert all(f.frame_id == f.frame_index and f.source == "demo.avi" for f in frames)
    assert all((f.width, f.height) == (320, 180) and f.image.shape == (180, 320, 3) for f in frames)
    frames[0].release()
    assert frames[0].image is None  # explicit lifetime: pixels are dropped on release
    assert (report.planned, report.decoded, report.failed, report.beyond_end) == (4, 4, 0, 0)
    assert (report.original_frames, report.duration_seconds) == (200, 20.0)


def test_short_video_yields_a_single_sample(tmp_path):
    video = write_video(tmp_path / "short.avi", [solid(50), solid(60), solid(70)], fps=10.0)
    source = VideoSource(video)
    frames, report = collect(source, SamplingConfig(every_seconds=2))
    assert [f.frame_index for f in frames] == [0] and report.planned == 1 == report.decoded


def test_non_integer_fps_video_uses_real_frame_times(tmp_path):
    video = write_video(tmp_path / "f.avi", [solid(i * 10) for i in range(25)], fps=12.5)
    source = VideoSource(video)
    assert source.metadata.fps == 12.5 and source.metadata.duration_seconds == 2.0
    frames, _ = collect(source, SamplingConfig(every_seconds=0.4))
    assert [f.frame_index for f in frames] == [0, 5, 10, 15, 20]
    assert [f.offset_seconds for f in frames] == [0.0, 0.4, 0.8, 1.2, 1.6]


def test_plan_for_ntsc_style_fps_rounds_consistently():
    plan = plan_samples(meta(29.97, 300), SamplingConfig(every_seconds=1))
    assert [i for i, _ in plan][:5] == [0, 30, 60, 90, 120]
    assert plan[1][1] == pytest.approx(30 / 29.97)  # the recorded offset is the frame's own time


def test_plan_stops_at_end_of_video_and_respects_max_samples():
    assert [i for i, _ in plan_samples(meta(10, 200), SamplingConfig(every_seconds=3))] == [
        0, 30, 60, 90, 120, 150, 180,
    ]  # fmt: skip
    assert len(plan_samples(meta(10, 200), SamplingConfig(every_seconds=2, max_samples=3))) == 3


def test_intervals_shorter_than_a_frame_do_not_repeat_frames():
    plan = plan_samples(meta(10, 50), SamplingConfig(every_seconds=0.01))
    assert [i for i, _ in plan] == list(range(50))


def test_sampling_config_validation():
    for bad in ({"every_seconds": 0}, {"every_seconds": -1}, {"max_samples": 0}):
        with pytest.raises(ValidationError):
            SamplingConfig(**bad)


def fake_capture(source, total_grabs, fail_retrieve=()):
    class FakeCapture:
        def __init__(self):
            self.n = 0

        def grab(self):
            self.n += 1
            return self.n <= total_grabs

        def retrieve(self):
            if self.n - 1 in fail_retrieve:
                return False, None
            return True, solid(80)

    @contextmanager
    def capture():
        yield FakeCapture()

    source.capture = capture


def test_decode_failures_and_early_end_of_stream_are_reported(demo_video):
    source = VideoSource(demo_video)
    fake_capture(source, total_grabs=95, fail_retrieve={40})
    frames, report = collect(source, SamplingConfig(every_seconds=2))
    assert [f.frame_index for f in frames] == [0, 20, 60, 80]  # 40 failed to decode
    assert (report.planned, report.decoded, report.failed) == (10, 4, 1)
    assert report.beyond_end == 5  # 100..180 were never reached


# ---- frame-change gate ----------------------------------------------------------------------
def test_first_frame_always_passes_and_identical_frames_are_skipped():
    gate = FrameGate()
    first = gate.check(solid(100))
    assert (first.accepted, first.score, first.reason) == (True, None, "first")
    same = gate.check(solid(100))
    assert (same.accepted, same.score, same.reason) == (False, 0.0, "similar")


def test_changed_frames_are_accepted_with_their_score():
    gate = FrameGate()
    gate.check(solid(100))
    changed = gate.check(solid(160))
    assert (changed.accepted, changed.reason, changed.score) == (True, "changed", 60.0)


def test_gate_compares_with_the_last_accepted_frame_not_the_last_seen_one():
    gate = FrameGate(GateConfig(threshold=6))
    assert gate.check(solid(100)).accepted
    assert not gate.check(solid(104)).accepted  # 4 < 6 vs accepted 100
    slow_drift = gate.check(
        solid(108)
    )  # only 4 from the previous frame, but 8 from the accepted one
    assert slow_drift.accepted and slow_drift.score == 8.0


def test_gate_threshold_boundary_is_inclusive_and_zero_accepts_everything():
    gate = FrameGate(GateConfig(threshold=6))
    gate.check(solid(100))
    assert gate.check(solid(106)).accepted  # score == threshold
    gate2 = FrameGate(GateConfig(threshold=6))
    gate2.check(solid(100))
    assert not gate2.check(solid(105)).accepted  # just below
    everything = FrameGate(GateConfig(threshold=0))
    everything.check(solid(100))
    assert everything.check(solid(100)).accepted


def test_disabled_gate_accepts_every_frame_and_reset_forgets_the_reference():
    off = FrameGate(GateConfig(enabled=False))
    decisions = [off.check(solid(100)) for _ in range(3)]
    assert all(d.accepted and d.score is None and d.reason == "gate_disabled" for d in decisions)
    gate = FrameGate()
    gate.check(solid(100))
    gate.reset()
    assert gate.check(solid(100)).reason == "first"


def test_gate_config_validation():
    for bad in (
        {"threshold": -1},
        {"threshold": 300},
        {"resize_width": 2},
        {"resize_height": 5000},
    ):
        with pytest.raises(ValidationError):
            GateConfig(**bad)


def test_gate_on_the_synthetic_video_accepts_exactly_the_scene_changes(demo_video):
    source = VideoSource(demo_video)
    gate = FrameGate()
    seen = []
    for frame in sample_frames(source, SamplingConfig(every_seconds=2)):
        d = gate.check(frame.image)
        seen.append((frame.frame_index, d.accepted, d.reason))
        frame.release()
    assert [i for i, ok, _ in seen if ok] == [0, 40, 120, 160]  # empty, person, vehicle, empty
    assert [r for _, ok, r in seen if not ok] == ["similar"] * 6
    scores = []
    gate = FrameGate()
    for frame in sample_frames(source, SamplingConfig(every_seconds=2)):
        d = gate.check(frame.image)
        if d.score is not None:
            scores.append(d.score)
        frame.release()
    assert max(s for s in scores if s < 6) == 0.0 and min(s for s in scores if s >= 6) > 10


# ---- simulated timeline ---------------------------------------------------------------------
def test_single_zone_timeline_covers_the_video_and_maps_offsets_to_time_and_zone():
    start = datetime(2026, 9, 29, 23, 59, 30, tzinfo=IST)
    timeline = single_zone_timeline(start, Zone.WAREHOUSE, video_duration_seconds=20.0)
    timeline.check_covers(19.9)
    assert timeline.telemetry.zone_at(19.9) is Zone.WAREHOUSE
    later = timeline.clock.at(60)  # crosses midnight on the simulated clock
    assert (later.day, later.hour, later.minute, later.second) == (30, 0, 0, 30)


def test_timeline_shorter_than_the_video_is_rejected():
    timeline = single_zone_timeline(datetime(2026, 9, 29, tzinfo=IST), Zone.MAIN_GATE, 10.0)
    with pytest.raises(TimelineError, match="patrol lasts"):
        timeline.check_covers(500)


def test_timeline_file_loading_and_errors(tmp_path):
    good = tmp_path / "t.yaml"
    good.write_text(
        'start: "2026-09-29T23:40:00+05:30"\n'
        "telemetry:\n  seed: 3\n  patrol:\n"
        "    - {zone: main_gate, duration_s: 10}\n    - {zone: driveway, duration_s: 12}\n",
        encoding="utf-8",
    )
    timeline = load_timeline(good)
    assert timeline.telemetry.zone_at(10) is Zone.DRIVEWAY and timeline.telemetry.duration_s == 22
    for text in ("not: [valid", "- a\n- list\n", "start: nope\ntelemetry: {patrol: []}\n"):
        bad = tmp_path / "bad.yaml"
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(TimelineError):
            load_timeline(bad)
    with pytest.raises(TimelineError):
        load_timeline(tmp_path / "missing.yaml")


# ---- VideoSource validation with a stubbed OpenCV capture -------------------------------------
def stub_capture(monkeypatch, *, opened=True, fps=10.0, frames=5, first_frame_ok=True):
    """Make cv2.VideoCapture report the given properties, as a container that opens but is odd."""

    class Stub:
        def isOpened(self):
            return opened

        def get(self, prop):
            import cv2

            return {cv2.CAP_PROP_FPS: fps, cv2.CAP_PROP_FRAME_COUNT: frames}[prop]

        def read(self):
            return (True, solid(50)) if first_frame_ok else (False, None)

        def release(self):
            pass

    monkeypatch.setattr("drone_agent.video.source.cv2.VideoCapture", lambda path: Stub())


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"opened": False}, "cannot open"),
        ({"fps": 0.0}, "unusable FPS"),
        ({"fps": float("nan")}, "unusable FPS"),
        ({"fps": 5000.0}, "unusable FPS"),
        ({"frames": 0}, "no frames"),
        ({"frames": -1}, "no frames"),
        ({"first_frame_ok": False}, "cannot decode the first frame"),
    ],
)
def test_video_that_opens_but_is_unusable_is_rejected(tmp_path, monkeypatch, kwargs, fragment):
    dummy = tmp_path / "odd.avi"
    dummy.write_bytes(b"placeholder bytes, OpenCV is stubbed")
    stub_capture(monkeypatch, **kwargs)
    with pytest.raises(VideoError, match=fragment):
        VideoSource(dummy)


def test_a_well_behaved_stubbed_capture_is_accepted(tmp_path, monkeypatch):
    dummy = tmp_path / "ok.avi"
    dummy.write_bytes(b"placeholder bytes, OpenCV is stubbed")
    stub_capture(monkeypatch)
    m = VideoSource(dummy).metadata
    assert (m.fps, m.frame_count, m.width, m.height, m.duration_seconds) == (10.0, 5, 64, 36, 0.5)
