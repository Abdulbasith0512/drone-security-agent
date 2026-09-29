from datetime import datetime

import pytest
from conftest import IST

from drone_agent.clock import SimulationClock


def test_clock_advances_by_offsets_and_keeps_timezone():
    clock = SimulationClock(datetime(2026, 9, 29, 23, 55, tzinfo=IST))
    assert clock.at(0) == clock.start
    ts = clock.at(60)
    assert ts.isoformat() == "2026-09-29T23:56:00+05:30"
    assert clock.offset(ts) == 60


def test_clock_crosses_midnight_into_next_date():
    clock = SimulationClock(datetime(2026, 9, 29, 23, 55, tzinfo=IST))
    before, at, after = clock.at(299), clock.at(300), clock.at(360)
    assert (before.date().day, before.hour, before.minute) == (29, 23, 59)
    assert (at.date().day, at.hour, at.minute) == (30, 0, 0)
    assert (after.date().day, after.hour, after.minute) == (30, 0, 1)
    assert after.utcoffset() == clock.start.utcoffset()


def test_clock_rejects_naive_start():
    with pytest.raises(ValueError):
        SimulationClock(datetime(2026, 9, 29, 23, 55))
