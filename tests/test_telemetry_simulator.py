from datetime import datetime

import pytest
from conftest import IST
from pydantic import ValidationError

from drone_agent.clock import SimulationClock
from drone_agent.models import Zone
from drone_agent.telemetry import ZONE_CENTERS, TelemetryConfig, simulate_telemetry

CLOCK = SimulationClock(datetime(2026, 9, 29, 23, 55, tzinfo=IST))


def _config(**overrides) -> TelemetryConfig:
    data = dict(
        seed=3,
        sample_interval_s=10,
        patrol=[
            {"zone": "main_gate", "duration_s": 300},
            {"zone": "restricted_zone", "duration_s": 300},
        ],
    )
    data.update(overrides)
    return TelemetryConfig(**data)


def test_same_seed_gives_identical_output_and_different_seed_differs():
    assert simulate_telemetry(CLOCK, _config()) == simulate_telemetry(CLOCK, _config())
    assert simulate_telemetry(CLOCK, _config()) != simulate_telemetry(CLOCK, _config(seed=4))


def test_samples_cover_patrol_at_fixed_interval_and_cross_midnight():
    samples = simulate_telemetry(CLOCK, _config())
    assert len(samples) == 61  # 0..600 s inclusive, every 10 s
    gaps = {
        (b.timestamp - a.timestamp).total_seconds()
        for a, b in zip(samples, samples[1:], strict=False)
    }
    assert gaps == {10.0}
    assert samples[0].timestamp == CLOCK.start
    assert samples[-1].timestamp.date().day == 30  # 00:05 next day
    assert all(s.timestamp.utcoffset() == CLOCK.start.utcoffset() for s in samples)


def test_zone_follows_patrol_and_leg_boundary_belongs_to_next_leg():
    config = _config()
    samples = {s.timestamp: s for s in simulate_telemetry(CLOCK, config)}
    assert samples[CLOCK.at(290)].zone is Zone.MAIN_GATE
    assert samples[CLOCK.at(300)].zone is Zone.RESTRICTED_ZONE
    assert config.zone_at(600) is Zone.RESTRICTED_ZONE
    with pytest.raises(ValueError):
        config.zone_at(601)


def test_positions_stay_near_synthetic_zone_centres_with_plausible_values():
    for s in simulate_telemetry(CLOCK, _config()):
        lat, lon = ZONE_CENTERS[s.zone]
        assert abs(s.latitude - lat) <= 0.00002
        assert abs(s.longitude - lon) <= 0.00002
        assert 28 <= s.altitude_m <= 32
        assert 0 <= s.battery_percent <= 100


def test_battery_never_increases_and_is_clamped_at_zero():
    batteries = [s.battery_percent for s in simulate_telemetry(CLOCK, _config())]
    assert batteries == sorted(batteries, reverse=True)
    drained = simulate_telemetry(
        CLOCK, _config(battery_start_percent=1, battery_drain_percent_per_min=1)
    )
    assert drained[-1].battery_percent == 0.0


def test_dropout_removes_samples_without_changing_the_others():
    full = simulate_telemetry(CLOCK, _config())
    with_gap = simulate_telemetry(
        CLOCK, _config(dropouts=[{"start_offset_s": 100, "end_offset_s": 200}])
    )
    kept = {s.timestamp: s for s in full if not 100 <= CLOCK.offset(s.timestamp) <= 200}
    assert len(full) - len(with_gap) == 11  # offsets 100,110,...,200
    assert {s.timestamp: s for s in with_gap} == kept


def test_invalid_config_rejected():
    with pytest.raises(ValidationError):
        _config(patrol=[])
    with pytest.raises(ValidationError):
        _config(patrol=[{"zone": "main_gate", "duration_s": 0}])
    with pytest.raises(ValidationError):
        _config(dropouts=[{"start_offset_s": 50, "end_offset_s": 10}])
