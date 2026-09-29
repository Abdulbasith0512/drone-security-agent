from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from drone_agent.models import Telemetry, Zone

IST = timezone(timedelta(hours=5, minutes=30))
ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT / "scenarios"


@pytest.fixture
def make_telemetry():
    def _make(timestamp: datetime, zone: Zone = Zone.MAIN_GATE, **overrides) -> Telemetry:
        data = dict(
            timestamp=timestamp,
            latitude=0.0001,
            longitude=0.0001,
            altitude_m=30.0,
            zone=zone,
            battery_percent=90.0,
        )
        data.update(overrides)
        return Telemetry(**data)

    return _make


@pytest.fixture
def store(tmp_path):
    """A fresh, isolated on-disk database per test."""
    from drone_agent.store import Store

    with Store.open(tmp_path / "test.db") as s:
        yield s
