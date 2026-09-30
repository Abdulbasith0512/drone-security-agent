"""Adaptive VLM sampling: every skip and every call has a reason, and staleness is bounded."""

from datetime import datetime, timedelta

import pytest
from conftest import IST

from drone_agent.agents import AdaptiveSampler, SamplerConfig, SamplingReason
from drone_agent.models import Telemetry, Zone
from drone_agent.telemetry import east_offset_degrees

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)
R = SamplingReason


def at(seconds: int) -> datetime:
    return T0 + timedelta(seconds=seconds)


def tele(seconds: int, **fields) -> Telemetry:
    data = dict(
        timestamp=at(seconds),
        latitude=0.0004,
        longitude=0.00045,
        altitude_m=30.0,
        zone=Zone.MAIN_GATE,
        battery_percent=90.0,
        signal_percent=95.0,
    )
    data.update(fields)
    return Telemetry(**data)


def decide(sampler, seconds, zone=Zone.MAIN_GATE, score=0.0, telemetry=None, open_events=0):
    return sampler.decide(
        now=at(seconds), zone=zone, scene_score=score, telemetry=telemetry, open_events=open_events
    )


def test_first_frame_is_always_analysed_then_identical_frames_are_skipped():
    s = AdaptiveSampler()
    assert decide(s, 0).reason is R.FIRST
    skipped = decide(s, 10)
    assert not skipped.analyze and skipped.reason is R.SIMILAR
    assert s.stats.frames == 2 and s.stats.analyzed == 1 and s.stats.skipped == 1


def test_a_zone_change_is_a_new_scene_and_always_analysed():
    s = AdaptiveSampler()
    decide(s, 0)
    d = decide(s, 10, zone=Zone.DRIVEWAY)
    assert d.analyze and d.reason is R.ZONE_CHANGE and "main_gate -> driveway" in d.detail


def test_scene_change_at_or_above_the_threshold_triggers_a_call():
    s = AdaptiveSampler(SamplerConfig(scene_threshold=6))
    decide(s, 0)
    assert decide(s, 10, score=5.99).reason is R.SIMILAR
    d = decide(s, 20, score=6.0)
    assert d.analyze and d.reason is R.SCENE_CHANGE


@pytest.mark.parametrize(
    "fields,fragment",
    [
        ({"battery_percent": 87.0}, "battery"),
        ({"altitude_m": 36.0}, "altitude"),
        ({"longitude": 0.00045 + east_offset_degrees(20, 0.0004)}, "moved"),
        ({"signal_percent": 30.0}, "signal degraded"),
    ],
)
def test_material_telemetry_change_triggers_a_call_even_when_the_scene_looks_the_same(
    fields, fragment
):
    s = AdaptiveSampler()
    decide(s, 0, telemetry=tele(0))
    quiet = decide(s, 10, telemetry=tele(10, battery_percent=89.5))  # 0.5% is not material
    assert quiet.reason is R.SIMILAR
    d = decide(s, 20, telemetry=tele(20, **fields))
    assert d.analyze and d.reason is R.TELEMETRY_CHANGE and fragment in d.detail


def test_signal_recovery_is_a_telemetry_change_too():
    s = AdaptiveSampler()
    decide(s, 0, telemetry=tele(0, signal_percent=20))
    d = decide(s, 10, telemetry=tele(10, signal_percent=90))
    assert d.reason is R.TELEMETRY_CHANGE and "recovered" in d.detail


def test_telemetry_is_compared_with_the_state_at_the_last_call_not_the_last_frame():
    """Slow drift adds up: 1%/frame never trips a per-frame test but must trip this one."""
    s = AdaptiveSampler()
    decide(s, 0, telemetry=tele(0, battery_percent=90))
    reasons = [
        decide(s, 10 * i, telemetry=tele(10 * i, battery_percent=90 - i)).reason
        for i in range(1, 5)
    ]
    assert reasons == [R.SIMILAR, R.TELEMETRY_CHANGE, R.SIMILAR, R.TELEMETRY_CHANGE]


def test_an_open_event_forces_a_fresh_look_every_watch_interval():
    s = AdaptiveSampler(SamplerConfig(watch_interval_s=120))
    decide(s, 0, open_events=1)
    assert decide(s, 60, open_events=1).reason is R.SIMILAR
    d = decide(s, 120, open_events=1)
    assert d.analyze and d.reason is R.EVENT_WATCH
    assert decide(s, 180, open_events=0).reason is R.SIMILAR  # nothing open: no watching needed


def test_no_call_can_be_skipped_forever_the_forced_refresh():
    s = AdaptiveSampler(SamplerConfig(max_skip_seconds=300))
    decide(s, 0)
    assert decide(s, 299).reason is R.SIMILAR
    d = decide(s, 300)
    assert d.analyze and d.reason is R.FORCED_REFRESH
    assert decide(s, 301).reason is R.SIMILAR  # the clock restarts at the refresh


def test_a_disabled_sampler_analyses_everything():
    s = AdaptiveSampler(SamplerConfig(enabled=False))
    assert {decide(s, i * 10).reason for i in range(5)} == {R.DISABLED}
    assert s.stats.analyzed == 5 and s.stats.avoided_pct == 0


def test_stats_break_down_calls_saved_by_reason():
    s = AdaptiveSampler()
    decide(s, 0)
    for i in range(1, 5):
        decide(s, i * 10)
    decide(s, 60, zone=Zone.DRIVEWAY)
    stats = s.stats.to_dict()
    assert stats["frames"] == 6 and stats["model_calls"] == 2 and stats["skipped"] == 4
    assert stats["avoided_pct"] == pytest.approx(66.7, abs=0.05)
    assert stats["by_reason"] == {"first": 1, "similar": 4, "zone_change": 1}


def test_missing_telemetry_never_blocks_or_triggers():
    s = AdaptiveSampler()
    decide(s, 0)
    assert decide(s, 10, telemetry=None).reason is R.SIMILAR
    assert decide(s, 20, telemetry=tele(20)).reason is R.SIMILAR  # no baseline yet: nothing to diff


def test_config_is_validated():
    with pytest.raises(ValueError):
        SamplerConfig(watch_interval_s=0)
    with pytest.raises(ValueError):
        SamplerConfig(scene_threshold=-1)
