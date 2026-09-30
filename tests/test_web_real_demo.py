"""The bundled real-video demo: real pixels, a RECORDED model replay, no model and no credential.

`--demo` opens on the real clip in demo_assets/real_clip/: the exact file, the cassette that holds
what a vision model answered for it when it was recorded, and a manifest that pins both by sha256.
Nothing is analysed live: every description is read back from the cassette, so the run is the same
every time. There is no ground truth for the clip, so nothing here scores a description; these
tests check integrity, reproducibility, honesty of the labels and that the dashboard serves the
real frames.
"""

import json
import shutil
import socket
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from rich.console import Console

from drone_agent.eval.real_clip import scan_for_secrets
from drone_agent.store import Store
from drone_agent.web import __main__ as web_main
from drone_agent.web.api import ApiError, DashboardApi, open_readonly
from drone_agent.web.bundle import REAL_ASSETS, BundleError, build_real_run, load_real_assets
from drone_agent.web.registry import RunRegistry

ROOT = Path(__file__).resolve().parents[1]
REAL = "real_clip"
CLIP_SHA256 = "d7e3ceeba55902bee5f6c9efbe68c8799266e699c7c5cb2d1a7180bcc7027ea2"
SAMPLED_FRAMES = [0, 75, 150, 225, 300, 375, 450, 525, 600]  # one every 3 s of a 25 fps clip


@pytest.fixture(scope="module")
def real_run(tmp_path_factory):
    """Built with no credential, no .env and every network connection refused."""
    mp = pytest.MonkeyPatch()
    mp.delenv("GEMINI_API_KEY", raising=False)
    mp.delenv("GEMINI_MODEL", raising=False)

    def no_dotenv(*args, **kwargs):
        raise AssertionError("the demo must not load .env")

    def no_network(self, *args, **kwargs):
        raise AssertionError("the demo must not open a network connection")

    mp.setattr("dotenv.load_dotenv", no_dotenv)
    mp.setattr(socket.socket, "connect", no_network)
    try:
        yield build_real_run(tmp_path_factory.mktemp("real"))
    finally:
        mp.undo()


@pytest.fixture(scope="module")
def api(real_run):
    return DashboardApi(RunRegistry([real_run]))


def rows(run):
    """The stored run as plain data (ids, times, descriptions, objects, events, entities)."""
    with open_readonly(run.db) as store:
        return {
            "observations": [o.model_dump(mode="json") for o in store.query_observations()],
            "events": [e.model_dump(mode="json") for e in store.query_events()],
            "entities": [n.model_dump(mode="json") for n in store.list_entities()],
            "alerts": [a.model_dump(mode="json") for a in store.query_alerts()],
        }


def copy_assets(tmp_path):
    dest = tmp_path / "assets"
    shutil.copytree(REAL_ASSETS, dest)
    return dest


# ---- the shipped files are exactly the ones the manifest pins --------------------------------
def test_the_bundled_files_match_the_manifest_byte_for_byte():
    manifest = load_real_assets()  # raises BundleError on any mismatch
    assert manifest["clip"]["sha256"] == CLIP_SHA256
    assert manifest["clip"]["size_bytes"] == 10_797_538
    assert manifest["cassette"]["provider"] == "gemini:gemini-3.6-flash"
    assert manifest["cassette"]["entries"] == 8
    assert manifest["replay"] == {
        "start": "2026-09-29T14:00:00+05:30",
        "zone": "main_gate",
        "sample_every_seconds": 3,
        "simulated": "start time, zone and telemetry are supplied here; a video carries none",
    }
    assert not any("licen" in k or "source" in k for k in manifest)  # nothing unknown is claimed


def test_the_clip_decodes_as_the_manifest_says(real_run):
    meta = load_real_assets()["clip"]
    capture = cv2.VideoCapture(str(real_run.video))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == meta["frame_count"] == 617
        assert capture.get(cv2.CAP_PROP_FPS) == meta["fps"] == 25.0
        assert (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), meta["width"]) == (1920, 1920)
        assert (int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), meta["height"]) == (1080, 1080)
    finally:
        capture.release()


@pytest.mark.parametrize("part", ["clip", "cassette"])
def test_a_modified_or_substituted_file_is_refused_with_a_clear_error(tmp_path, part):
    assets = copy_assets(tmp_path)
    target = assets / ("clip.mp4" if part == "clip" else "clip.cassette.json")
    data = bytearray(target.read_bytes())
    data[len(data) // 2] ^= 0x01  # same size, one flipped bit: only the hash can tell
    target.write_bytes(bytes(data))
    with pytest.raises(BundleError, match="does not match its manifest"):
        load_real_assets(assets)
    with pytest.raises(BundleError, match="does not match its manifest"):
        build_real_run(tmp_path / "out", assets_dir=assets)
    assert not (tmp_path / "out" / f"{REAL}.db").exists()  # nothing is built from a bad file


def test_a_truncated_clip_is_refused(tmp_path):
    assets = copy_assets(tmp_path)
    clip = assets / "clip.mp4"
    clip.write_bytes(clip.read_bytes()[:-1])
    with pytest.raises(BundleError, match="does not match its manifest"):
        load_real_assets(assets)


@pytest.mark.parametrize("name", ["clip.mp4", "clip.cassette.json", "manifest.json"])
def test_missing_assets_fail_loudly_not_silently(tmp_path, name):
    assets = copy_assets(tmp_path)
    (assets / name).unlink()
    with pytest.raises(BundleError, match="missing"):
        load_real_assets(assets)


def test_a_malformed_manifest_is_refused(tmp_path):
    assets = copy_assets(tmp_path)
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    (assets / "manifest.json").write_text(json.dumps({**manifest, "format": "x"}), encoding="utf-8")
    with pytest.raises(BundleError, match="unknown manifest format"):
        load_real_assets(assets)
    del manifest["replay"]
    (assets / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="not a valid real-clip manifest"):
        load_real_assets(assets)


# ---- the run: a recorded replay, reproducible, no model ---------------------------------------
def test_the_real_run_is_declared_real_video_and_carries_the_clip(real_run):
    assert (real_run.name, real_run.kind.key, real_run.kind.label) == (
        REAL,
        "real-video",
        "REAL VIDEO",
    )
    assert real_run.video == REAL_ASSETS / "clip.mp4" and real_run.has_video


def test_the_replay_is_the_recorded_v03_result_with_the_expected_shape(real_run):
    with Store.open(real_run.db) as s:
        assert (s.count_observations(), s.count_events(), s.count_entities()) == (9, 28, 12)
        assert s.count_alerts() == 0  # the deterministic rules raise nothing on this footage
        assert set(s.observation_counts_by("source")) == {"cassette", "cassette+carried"}
        obs = s.query_observations()
    assert [o.frame_id for o in obs] == SAMPLED_FRAMES
    assert sum(len(o.detected_objects) for o in obs) == 33
    assert {d.confidence for o in obs for d in o.detected_objects} == {
        0.5
    }  # the labelled placeholder
    assert [o.source for o in obs].count("cassette+carried") == 1  # the one gate-skipped frame
    assert {str(o.timestamp.utcoffset()) for o in obs} == {"5:30:00"}  # the simulated clock
    assert {o.zone.value for o in obs} == {"main_gate"}


def test_the_run_is_reproducible(real_run, tmp_path):
    again = build_real_run(tmp_path / "again")
    assert rows(again) == rows(real_run)
    rebuilt = build_real_run(tmp_path / "again", rebuild=True)
    assert rows(rebuilt) == rows(real_run)


def test_the_run_is_built_by_the_existing_cli_replaying_the_cassette(tmp_path, monkeypatch):
    from drone_agent.web import bundle

    calls = []
    real_cli = bundle._cli
    monkeypatch.setattr(bundle, "_cli", lambda *argv: (calls.append(argv), real_cli(*argv))[1])
    build_real_run(tmp_path / "cmds")
    assert [c[0] for c in calls] == ["ingest-video", "build-events", "analyze"]
    ingest = calls[0]
    option = {k: ingest[i + 1] for i, k in enumerate(ingest) if k.startswith("--")}
    assert option["--provider"] == "cassette"  # replay: never a live provider
    assert Path(option["--cassette"]).name == "clip.cassette.json"
    assert (option["--start"], option["--zone"]) == ("2026-09-29T14:00:00+05:30", "main_gate")
    assert option["--sample-every"] == "3" and "--cassette-mode" not in option


def test_every_description_comes_from_the_cassette(real_run):
    cassette = json.loads(load_real_assets()["cassette_path"].read_text(encoding="utf-8"))
    recorded = {e["frame_index"]: e["output"]["description"] for e in cassette["entries"]}
    with Store.open(real_run.db) as s:
        stored = {o.frame_id: o.description for o in s.query_observations()}
    for frame, description in recorded.items():
        assert stored[frame] == description
    assert stored[525] == stored[450]  # the carried frame repeats the previous answer


# ---- the dashboard serves the real frames and stays honest ------------------------------------
def test_the_dashboard_lists_the_run_with_the_existing_real_video_labels(api):
    listed = api.runs()["runs"][0]
    assert (listed["name"], listed["label"], listed["has_video"]) == (REAL, "REAL VIDEO", True)
    assert "no model is called now" in listed["detail"]
    assert "recorded vision-model output" in listed["description_origin"]
    assert "no perception ground truth" in listed["detail"]
    assert listed["video"] == {
        "name": "clip.mp4", "frame_count": 617, "fps": 25.0,
        "duration_seconds": 24.68, "width": 1920, "height": 1080,
    }  # fmt: skip


def test_frames_are_real_jpegs_of_the_video(api):
    for frame in SAMPLED_FRAMES:
        data = api.frame_jpeg(REAL, frame)
        assert data[:3] == b"\xff\xd8\xff"
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (270, 480)  # 1920x1080 scaled to the default width
        assert image.std() > 10  # a picture, not a blank placeholder
    full = cv2.imdecode(np.frombuffer(api.frame_jpeg(REAL, 75, 960), np.uint8), cv2.IMREAD_COLOR)
    assert full.shape[:2] == (540, 960)
    first, later = (api.frame_jpeg(REAL, f) for f in (0, 300))
    assert first != later  # different instants give different pictures


def test_every_observation_has_its_picture_and_frames_outside_the_clip_do_not(api):
    observations = api.observations(REAL)["observations"]
    assert [o["frame_id"] for o in observations] == SAMPLED_FRAMES
    assert all(o["image"] for o in observations)
    for bad in (617, -1):
        with pytest.raises(ApiError) as excinfo:
            api.frame_jpeg(REAL, bad)
        assert excinfo.value.status == 404


def test_events_entities_and_the_empty_alert_list_are_served(api):
    overview = api.overview(REAL)
    assert overview["run"]["label"] == "REAL VIDEO"
    assert overview["counts"]["events"] == 28 and overview["counts"]["alerts"] == 0
    assert any("simulated" in line for line in overview["limitations"])
    assert len(api.events(REAL)["events"]) == 28
    assert len(api.entities(REAL)["entities"]) == 12
    alerts = api.alerts(REAL)
    assert alerts["total"] == 0 and alerts["alerts"] == []  # stated, not invented


def test_the_analyst_answers_offline_from_stored_evidence(api):
    answer = api.ask(REAL, "What happened at the main gate?")
    assert answer["mode"] == "template" and not answer["refused"]
    kinds = {c["kind"] for c in answer["citations"]}
    assert {"event", "frame", "obs"} <= kinds
    assert "9 stored observation(s) and 28 event(s)" in answer["text"]
    none = api.ask(REAL, "List the alerts.")
    assert none["text"].startswith("Insufficient evidence") and none["citations"] == []


# ---- `--demo`: the real run first, the two existing runs kept --------------------------------
def test_demo_opens_on_the_real_run_and_keeps_the_scripted_and_synthetic_runs(
    tmp_path, monkeypatch
):
    seen = []

    class FakeServer:
        server_port = 8765

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def fake_make_server(api, *args, **kwargs):
        seen.append(api.runs()["runs"])
        return FakeServer()

    monkeypatch.setattr(web_main, "make_server", fake_make_server)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = Console(record=True, width=160)
    assert web_main.main(["--demo", "--out-dir", str(tmp_path / "out")], out) == 0
    listed = seen[0]
    assert [r["name"] for r in listed] == [REAL, "security_events", "synthetic_video"]
    assert [r["label"] for r in listed] == ["REAL VIDEO", "SCRIPTED", "SYNTHETIC DEMO"]
    assert [r["has_video"] for r in listed] == [True, False, True]  # the UI opens on the first
    text = out.export_text()
    assert REAL in text and "REAL VIDEO" in text


def test_the_provenance_note_matches_the_manifest_and_invents_nothing():
    manifest = load_real_assets()
    note = (REAL_ASSETS / "PROVENANCE.md").read_text(encoding="utf-8")
    assert manifest["clip"]["sha256"] in note
    assert f"{manifest['clip']['size_bytes']:,} bytes" in note
    assert manifest["cassette"]["provider"] in note
    assert "http" not in note  # no source or licence URL is claimed
    assert "Not recorded in this repository" in note  # what is unknown is said to be unknown
    assert "no ground truth" in note and "simulated" in note


# ---- no credential anywhere in what is shipped ------------------------------------------------
def test_the_shipped_assets_contain_no_credential_and_no_local_path():
    files = [p for p in REAL_ASSETS.iterdir() if p.is_file()]
    assert {p.name for p in files} >= {"clip.mp4", "clip.cassette.json", "manifest.json"}
    real_key = None
    env = ROOT / ".env"
    if env.exists():  # compare with the actual key if there is one; it is never printed
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                real_key = line.split("=", 1)[1].strip().strip("\"'") or None
    assert scan_for_secrets(files, [real_key] if real_key else []) == []
    text = "".join(p.read_text(encoding="utf-8") for p in files if p.suffix in (".json", ".md"))
    assert "GEMINI" not in text and "AIza" not in text
    for marker in ("C:" + chr(92), "/Users/", "/home/", "Admin"):
        assert marker not in text, marker


# ---- repository hygiene: exactly one video is whitelisted, everything else stays ignored ------
def ignored(path: str) -> bool:
    done = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=ROOT, capture_output=True, text=True
    )
    if done.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    return done.returncode == 0


def test_only_the_demo_clip_is_whitelisted_and_the_original_artifacts_stay_ignored():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    assert not ignored("demo_assets/real_clip/clip.mp4")
    assert not ignored("demo_assets/real_clip/clip.cassette.json")
    assert not ignored("demo_assets/real_clip/manifest.json")
    for path in (
        "demo_assets/real_clip/other.mp4",
        "demo_assets/clip.mp4",
        "clip.mp4",
        "output/real/clip.mp4",
        "output/real/v03/clip.v03.cassette.json",
        "output/real/v03/manifest.json",
        "output/dashboard/real_clip.db",
        "demo_assets/real_clip/run.db",
        ".env",
    ):
        assert ignored(path), path


def test_the_bundled_assets_are_pinned_against_line_ending_conversion():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "demo_assets/** -text" in attributes
