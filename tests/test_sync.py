from datetime import datetime, timedelta

import pytest
from conftest import IST

from drone_agent.telemetry import TelemetrySynchronizer

T0 = datetime(2026, 9, 29, 23, 55, tzinfo=IST)


@pytest.fixture
def sync(make_telemetry):
    samples = [
        make_telemetry(T0 + timedelta(seconds=s), battery_percent=90 - s / 10) for s in (0, 10, 20)
    ]
    return TelemetrySynchronizer(samples, stale_after_s=30)


def test_exact_timestamp_selects_that_sample_with_zero_age(sync):
    assoc = sync.associate(T0 + timedelta(seconds=10))
    assert assoc.telemetry.timestamp == T0 + timedelta(seconds=10)
    assert assoc.age_seconds == 0
    assert not assoc.stale


def test_selects_most_recent_sample_at_or_before_never_a_later_one(sync):
    assoc = sync.associate(T0 + timedelta(seconds=19.9))
    assert assoc.telemetry.timestamp == T0 + timedelta(seconds=10)
    assert assoc.age_seconds == pytest.approx(9.9)


def test_timestamp_before_first_sample_has_no_association(sync):
    assert sync.associate(T0 - timedelta(seconds=1)) is None


def test_stale_boundary_age_equal_to_threshold_is_fresh(sync):
    last = T0 + timedelta(seconds=20)
    assert not sync.associate(last + timedelta(seconds=30)).stale
    assert sync.associate(last + timedelta(seconds=30.001)).stale


def test_unsorted_samples_are_handled(make_telemetry):
    samples = [make_telemetry(T0 + timedelta(seconds=s)) for s in (20, 0, 10)]
    sync = TelemetrySynchronizer(samples)
    assert sync.associate(T0 + timedelta(seconds=15)).telemetry.timestamp == T0 + timedelta(
        seconds=10
    )


def test_negative_threshold_rejected(make_telemetry):
    with pytest.raises(ValueError):
        TelemetrySynchronizer([make_telemetry(T0)], stale_after_s=-1)


def test_attach_fills_observation_telemetry_without_mutating_original(sync):
    from drone_agent.models import Observation, Zone

    obs = Observation(
        observation_id="o1",
        frame_id=1,
        timestamp=T0 + timedelta(seconds=12),
        source="scripted",
        zone=Zone.MAIN_GATE,
        description="x",
    )
    attached = sync.attach(obs)
    assert obs.telemetry is None
    assert attached.telemetry.age_seconds == pytest.approx(2)
