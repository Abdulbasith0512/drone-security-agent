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


# ---- anomalies, signal and geofence geometry (M9) ---------------------------------------------
def _by_offset(config: TelemetryConfig) -> dict[int, object]:
    return {int(CLOCK.offset(s.timestamp)): s for s in simulate_telemetry(CLOCK, config)}


def test_signal_is_absent_unless_the_scenario_asks_for_one():
    assert {s.signal_percent for s in simulate_telemetry(CLOCK, _config())} == {None}
    samples = simulate_telemetry(CLOCK, _config(signal_baseline_percent=90))
    assert all(88 <= s.signal_percent <= 92 for s in samples)


def test_enabling_signal_leaves_position_altitude_and_battery_unchanged_for_first_sample():
    plain = simulate_telemetry(CLOCK, _config())[0]
    signalled = simulate_telemetry(CLOCK, _config(signal_baseline_percent=90))[0]
    assert (plain.latitude, plain.longitude, plain.altitude_m) == (
        signalled.latitude,
        signalled.longitude,
        signalled.altitude_m,
    )


def test_battery_drop_steps_the_battery_down_and_it_keeps_draining():
    samples = _by_offset(
        _config(
            battery_start_percent=100,
            battery_drain_percent_per_min=0.6,
            anomalies=[{"kind": "battery_drop", "start_offset_s": 300, "value": 8}],
        )
    )
    assert samples[290].battery_percent > 90
    assert samples[300].battery_percent == 8.0
    assert samples[360].battery_percent == pytest.approx(7.4, abs=0.05)


def test_signal_fade_ramps_down_holds_then_recovers():
    samples = _by_offset(
        _config(
            signal_baseline_percent=95,
            anomalies=[
                {
                    "kind": "signal_fade",
                    "start_offset_s": 100,
                    "end_offset_s": 200,
                    "value": 10,
                    "ramp_s": 40,
                }
            ],
        )
    )
    assert samples[90].signal_percent > 90
    assert 10 < samples[120].signal_percent < 90  # mid-ramp
    assert samples[150].signal_percent == pytest.approx(10, abs=0.2)
    assert samples[210].signal_percent > 90


def test_altitude_excursion_only_inside_its_window():
    samples = _by_offset(
        _config(
            anomalies=[
                {
                    "kind": "altitude_excursion",
                    "start_offset_s": 100,
                    "end_offset_s": 150,
                    "value": 140,
                }
            ]
        )
    )
    assert samples[90].altitude_m < 33
    assert all(138 <= samples[o].altitude_m <= 142 for o in (100, 120, 150))
    assert samples[160].altitude_m < 33


def test_geofence_drift_moves_the_drone_east_then_returns():
    from drone_agent.telemetry import distance_m

    config = _config(
        anomalies=[
            {"kind": "geofence_drift", "start_offset_s": 100, "end_offset_s": 200, "value": 150}
        ]
    )
    samples = _by_offset(config)
    home = ZONE_CENTERS[Zone.MAIN_GATE]
    moved = distance_m(samples[200].latitude, samples[200].longitude, *home)
    assert 148 < moved < 152
    assert distance_m(samples[150].latitude, samples[150].longitude, *home) == pytest.approx(
        75, abs=3
    )
    assert distance_m(samples[210].latitude, samples[210].longitude, *home) < 3


def test_geometry_helpers_are_consistent():
    from drone_agent.telemetry import GEOFENCE_CENTER, GEOFENCE_RADIUS_M, distance_m

    assert distance_m(0, 0, 0, 0) == 0
    assert distance_m(0, 0, 0.001, 0) == pytest.approx(111.32, abs=0.01)
    # every patrol zone lies inside the default fence with margin
    for lat, lon in ZONE_CENTERS.values():
        assert distance_m(lat, lon, *GEOFENCE_CENTER) < GEOFENCE_RADIUS_M - 20


def test_invalid_anomalies_rejected():
    for bad in (
        {"kind": "battery_drop", "start_offset_s": 0, "value": 120},
        {"kind": "signal_fade", "start_offset_s": 10, "value": 5},  # needs an end
        {"kind": "signal_fade", "start_offset_s": 10, "end_offset_s": 5, "value": 5},
        {"kind": "geofence_drift", "start_offset_s": 0, "end_offset_s": 5, "value": 0},
        {"kind": "altitude_excursion", "start_offset_s": 0, "end_offset_s": 5, "value": 900},
        {"kind": "unknown", "start_offset_s": 0, "end_offset_s": 5, "value": 1},
    ):
        with pytest.raises(ValidationError):
            _config(anomalies=[bad])


def test_garage_is_a_patrol_zone():
    samples = simulate_telemetry(
        CLOCK, _config(patrol=[{"zone": "garage", "duration_s": 60}], sample_interval_s=30)
    )
    assert {s.zone for s in samples} == {Zone.GARAGE}
