"""Run registry, the offline demo bundle, frame provider and the dashboard entry point."""

import socket
from pathlib import Path

import cv2
import numpy as np
import pytest

from drone_agent.store import Store
from drone_agent.web import __main__ as web_main
from drone_agent.web.bundle import BundleError, build_demo_runs
from drone_agent.web.frames import MAX_WIDTH, MIN_WIDTH, FrameNotFound, FrameProvider
from drone_agent.web.registry import (
    DATA_KINDS,
    RegistryError,
    RunRegistry,
    make_run,
    parse_run_spec,
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("dash")
    try:
        runs = build_demo_runs(out)
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise
    return out, runs


@pytest.fixture
def empty_db(tmp_path):
    path = tmp_path / "x.db"
    Store.open(path).close()
    return path


# ---- registry: the data kind is declared, never inferred --------------------------------------
def test_the_three_kinds_have_distinct_banners_and_honest_details():
    assert [k.label for k in DATA_KINDS.values()] == ["SCRIPTED", "SYNTHETIC DEMO", "REAL VIDEO"]
    assert "no image exists" in DATA_KINDS["scripted"].detail
    assert "NOT the output of any vision model" in DATA_KINDS["synthetic-demo"].detail
    assert "no model is called now" in DATA_KINDS["real-video"].detail
    assert not DATA_KINDS["scripted"].has_pixels and DATA_KINDS["real-video"].has_pixels
    origins = {k.description_origin for k in DATA_KINDS.values()}
    assert len(origins) == 3 and any("not a vision model" in o for o in origins)


def test_a_run_spec_needs_an_explicit_kind(empty_db):
    with pytest.raises(RegistryError, match="declared, never inferred"):
        parse_run_spec(f"mine={empty_db}")
    run = parse_run_spec(f"mine={empty_db},kind=scripted")
    assert (run.name, run.kind.label, run.video, run.has_video) == (
        "mine", "SCRIPTED", None, False
    )  # fmt: skip


def test_the_declared_kind_wins_over_what_the_stored_source_looks_like(demo):
    """The scripted database has source 'scripted'; declaring it REAL VIDEO must be honoured
    (and shown), proving the label is not read from observation.source."""
    out, _ = demo
    video = out / "demo" / "demo.avi"
    run = make_run("odd", out / "security_events.db", "real-video", video)
    assert run.kind.label == "REAL VIDEO"


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("", "expected NAME=DB"),
        ("nodb", "expected NAME=DB"),
        ("bad name=x.db,kind=scripted", "must be 1-40 letters"),
        ("../x=x.db,kind=scripted", "must be 1-40 letters"),
        ("a=missing.db,kind=scripted", "database not found"),
    ],
)
def test_bad_run_specs_are_rejected_with_a_clear_message(spec, message):
    with pytest.raises(RegistryError, match=message):
        parse_run_spec(spec)


def test_unknown_kinds_options_and_duplicates_are_rejected(empty_db, tmp_path):
    with pytest.raises(RegistryError, match="kind must be one of"):
        parse_run_spec(f"a={empty_db},kind=live-camera")
    with pytest.raises(RegistryError, match="allowed: kind=, video="):
        parse_run_spec(f"a={empty_db},kind=scripted,color=red")
    with pytest.raises(RegistryError, match="given twice"):
        parse_run_spec(f"a={empty_db},kind=scripted,kind=scripted")
    with pytest.raises(RegistryError, match="has no video"):
        make_run("a", empty_db, "scripted", empty_db)
    with pytest.raises(RegistryError, match="video not found"):
        make_run("a", empty_db, "real-video", tmp_path / "nope.mp4")


def test_registry_rejects_duplicates_and_being_empty(empty_db):
    run = make_run("a", empty_db, "scripted")
    with pytest.raises(RegistryError, match="registered twice"):
        RunRegistry([run, run])
    with pytest.raises(RegistryError, match="no runs registered"):
        RunRegistry([])
    registry = RunRegistry([run])
    assert registry.get("a") is run and registry.get("b") is None and registry.all() == [run]


# ---- the demo bundle ----------------------------------------------------------------------------
def test_the_demo_bundle_builds_the_two_offline_runs(demo):
    _, runs = demo
    scripted, video = runs
    assert (scripted.name, scripted.kind.key, scripted.video) == (
        "security_events",
        "scripted",
        None,
    )
    assert (video.name, video.kind.key) == ("synthetic_video", "synthetic-demo")
    assert video.video is not None and video.video.is_file()
    with Store.open(scripted.db) as s:
        assert (s.count_observations(), s.count_events(), s.count_entities()) == (27, 7, 2)
        assert s.count_alerts() >= 6  # the six camera alerts; a newer backend adds safety alerts
        assert set(s.observation_counts_by("source")) == {"scripted"}
    with Store.open(video.db) as s:
        assert (s.count_observations(), s.count_events(), s.count_entities(), s.count_alerts()) == (
            10, 2, 1, 1
        )  # fmt: skip
        assert set(s.observation_counts_by("source")) == {"cassette", "cassette+carried"}


def test_the_bundle_reuses_existing_runs_and_rebuilds_on_request(demo):
    out, first = demo
    with Store.open(first[0].db) as s:  # opening for write may touch the file: read, THEN stamp
        first_alerts = s.count_alerts()
    stamp = first[0].db.stat().st_mtime_ns
    again = build_demo_runs(out)
    assert again[0].db.stat().st_mtime_ns == stamp  # reused, not rebuilt
    rebuilt = build_demo_runs(out, rebuild=True)
    assert rebuilt[0].db.stat().st_mtime_ns > stamp
    with Store.open(rebuilt[0].db) as s:
        assert s.count_alerts() == first_alerts  # a rebuild does not double the data


def test_building_the_bundle_never_reads_dotenv_or_touches_the_network(tmp_path, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the demo bundle must not read .env or open a socket")

    monkeypatch.setattr("dotenv.load_dotenv", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        build_demo_runs(tmp_path / "offline")
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise


def test_a_failing_build_step_reports_its_command_and_output(tmp_path):
    with pytest.raises(BundleError, match="ingest"):
        build_demo_runs(tmp_path / "bad", scenarios_dir=tmp_path / "no-scenarios")


# ---- frames: real pixels of the registered video only -------------------------------------------
def test_frames_are_decoded_from_the_registered_video_and_are_jpeg(demo):
    _, runs = demo
    provider = FrameProvider(runs[1].video)
    assert provider.frame_count == 200
    data = provider.jpeg(40, 320)
    assert data[:3] == b"\xff\xd8\xff"
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape[1] == 320  # width honoured (the source is 320 wide)
    assert provider.jpeg(40, 320) is provider.jpeg(40, 320)  # served from the cache
    assert cv2.imdecode(np.frombuffer(provider.jpeg(0, 64), np.uint8), 1).shape[1] == 64


def test_frame_width_is_clamped_and_out_of_range_frames_are_not_found(demo):
    _, runs = demo
    provider = FrameProvider(runs[1].video)
    tiny = cv2.imdecode(np.frombuffer(provider.jpeg(10, 1), np.uint8), 1)
    assert tiny.shape[1] == MIN_WIDTH
    big = cv2.imdecode(np.frombuffer(provider.jpeg(10, 99999), np.uint8), 1)
    assert big.shape[1] <= MAX_WIDTH
    for bad in (-1, 200, 10**9):
        with pytest.raises(FrameNotFound):
            provider.jpeg(bad)


def test_the_frame_cache_is_bounded(demo, monkeypatch):
    _, runs = demo
    monkeypatch.setattr("drone_agent.web.frames._CACHE_ITEMS", 5)
    provider = FrameProvider(runs[1].video)
    for i in range(20):
        provider.jpeg(i)
    assert len(provider._cache) == 5


def test_an_unreadable_video_is_refused_at_startup(tmp_path):
    from drone_agent.video import VideoError

    junk = tmp_path / "junk.avi"
    junk.write_bytes(b"not a video")
    with pytest.raises(VideoError):
        FrameProvider(junk)


# ---- the entry point ---------------------------------------------------------------------------
def test_the_entry_point_needs_something_to_serve(capsys):
    assert web_main.main([]) == 2


def test_a_run_without_a_kind_is_an_error_not_a_guess(empty_db, capsys):
    assert web_main.main(["--run", f"a={empty_db}"]) == 2


def test_the_server_can_only_be_bound_to_loopback():
    parser = web_main.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--demo", "--host", "0.0.0.0"])
    assert parser.parse_args(["--demo"]).host == "127.0.0.1"


def test_the_entry_point_serves_until_interrupted_and_prints_the_kinds(empty_db, monkeypatch):
    from rich.console import Console

    class FakeServer:
        server_port = 4321
        closed = False

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            FakeServer.closed = True

    monkeypatch.setattr(web_main, "make_server", lambda api, host, port: FakeServer())
    out = Console(record=True, width=120)
    code = web_main.main(["--run", f"a={empty_db},kind=scripted"], out)
    text = out.export_text()
    assert code == 0 and FakeServer.closed
    assert "http://127.0.0.1:4321/" in text and "SCRIPTED" in text and "read-only" in text


def test_an_outdated_database_is_reported_before_serving(tmp_path):
    import sqlite3

    from rich.console import Console

    old = tmp_path / "old.db"
    conn = sqlite3.connect(old)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    out = Console(record=True, width=120)
    assert web_main.main(["--run", f"old={old},kind=scripted"], out) == 2
    assert "schema version 1" in " ".join(out.export_text().split())


def test_the_dashboard_sources_stay_inside_the_repo_layout():
    package = Path(web_main.__file__).parent
    names = {p.name for p in package.glob("*.py")}
    assert {"api.py", "bundle.py", "frames.py", "guard.py", "registry.py", "server.py"} <= names
