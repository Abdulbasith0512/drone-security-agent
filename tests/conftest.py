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


def write_video(path, frames, fps: float = 10.0):
    """Write BGR frames as an MJPG .avi (each frame is its own JPEG, so it is deterministic)."""
    import cv2

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory):
    """The synthetic 20 s, 10 fps demo video, generated once per test session (never committed)."""
    from drone_agent.video import VideoError
    from drone_agent.video.synthetic import write_demo_video

    try:
        return write_demo_video(tmp_path_factory.mktemp("video") / "demo.avi")
    except VideoError:
        pytest.skip("this OpenCV build cannot write MJPG video")


@pytest.fixture(scope="session")
def demo_cassette_path(demo_video, tmp_path_factory):
    """A cassette with canned responses for every 1 s frame of the demo video."""
    from drone_agent.video.synthetic import build_demo_cassette

    path = tmp_path_factory.mktemp("cassette") / "demo.cassette.json"
    build_demo_cassette(demo_video).save(path)
    return path
