"""V03: reproducibility checks around the ONE recorded real-clip experiment.

The clip, its cassette and the manifest live under `output/real/` (gitignored: the clip's origin and
licence are not recorded, so it is never committed). The tests that need them skip when they are
absent. The tests of the evidence tooling itself (hashing, parsing, secret scanning) always run.

What these tests do and do not say: they show the recorded real-model answers replay into the same
structured observations, events and entities, that frame-to-time mapping is preserved, and that no
credential was persisted. They say NOTHING about whether the model's descriptions are correct: the
clip has no ground truth and nothing was scored.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import ROOT, write_video

from drone_agent.eval.real_clip import (
    clip_provenance,
    parse_record_log,
    replay_summary,
    scan_for_secrets,
)
from drone_agent.ingest import ingest_observations
from drone_agent.models import Zone
from drone_agent.pipeline import build_events
from drone_agent.store import Store
from drone_agent.video import (
    SamplingConfig,
    VideoIngestConfig,
    VideoSource,
    run_video_ingest,
    single_zone_timeline,
)
from drone_agent.vision import Cassette, CassetteDescriber, DescriberProvider
from drone_agent.vision.gemini import PLACEHOLDER_CONFIDENCE

IST = timezone(timedelta(hours=5, minutes=30))
START = datetime(2026, 9, 29, 14, 0, tzinfo=IST)
REAL = ROOT / "output" / "real"
CLIP = REAL / "clip.mp4"
CASSETTE = REAL / "v03" / "clip.v03.cassette.json"
MANIFEST = REAL / "v03" / "manifest.json"


@pytest.fixture(scope="module")
def evidence():
    if not (CLIP.exists() and CASSETTE.exists() and MANIFEST.exists()):
        pytest.skip("the local real-clip artifacts (output/real/) are not present")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture
def no_credentials(monkeypatch):
    """Replay must work with no key anywhere: remove it and block .env loading."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)


def replay():
    return replay_summary(CLIP, CASSETTE, start=START, sample_every=3.0)


def canonical(data) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False)


# ---- the recorded experiment (needs the local artifacts) ----------
def test_the_clip_is_the_one_the_manifest_describes_and_no_provenance_is_invented(evidence):
    clip = clip_provenance(CLIP)
    for key in (
        "size_bytes",
        "sha256",
        "width",
        "height",
        "fps",
        "frame_count",
        "duration_seconds",
    ):
        assert clip[key] == evidence["clip"][key], key
    assert clip["decoded_sha256_matches_file"] is True
    assert (clip["width"], clip["height"], clip["fps"], clip["frame_count"]) == (
        1920,
        1080,
        25.0,
        617,
    )
    assert clip["sha256"].startswith("d7e3ceeba55902be")  # the prefix the cassette keys use
    assert evidence["clip"]["source"] == "not recorded" == evidence["clip"]["licence"]
    assert evidence["ground_truth"].startswith("none")  # the evidence never claims accuracy


def test_the_cassette_replays_without_any_credential_or_provider_call(evidence, no_credentials):
    summary = replay()
    counts = summary["counts"]
    assert counts["failed"] == 0 and counts["provider_calls"] == 8 and counts["carried"] == 1
    assert (
        summary["cassette_replays"] == counts["provider_calls"]
    )  # every answer came from the tape
    assert summary["cassette_content_mismatches"] == 0  # the decoded pixels match the recording


def test_the_structured_observations_are_reproducible(evidence, no_credentials):
    first, second = replay(), replay()
    assert canonical(first) == canonical(second)  # deterministic, including events and entities
    assert canonical(first) == canonical(evidence["replay"])  # and equal to what was recorded
    assert evidence["checks"]["replay_structured_observations_equal_recorded"] is True
    assert evidence["checks"]["replay_made_no_provider_call"] is True


def test_frame_and_time_mapping_is_preserved(evidence, no_credentials):
    frames = replay()["frames"]
    fps = evidence["clip"]["fps"]
    assert [f["frame_index"] for f in frames] == list(range(0, 601, 75))  # every 3 s
    for f in frames:
        assert f["frame_index"] / fps == f["offset_s"]  # index <-> video time
        obs = f["observation"]
        assert obs is not None
        stamp = datetime.fromisoformat(f["timestamp"])
        assert stamp == START + timedelta(seconds=f["offset_s"])  # video time <-> simulated clock
    actions = {f["offset_s"]: f["action"] for f in frames}
    assert [t for t, a in actions.items() if a == "carried"] == [21.0]  # the one gate-skipped frame
    carried = next(f for f in frames if f["action"] == "carried")["observation"]
    previous = next(f for f in frames if f["offset_s"] == 18.0)["observation"]
    assert carried["source"].endswith("+carried") and not previous["source"].endswith("+carried")
    assert carried["description"] == previous["description"]  # propagation, not a new inference
    assert carried["objects"] == previous["objects"]


def test_downstream_processing_is_deterministic_and_matches_the_batch_path(
    evidence, no_credentials
):
    live = replay()
    source = VideoSource(CLIP)
    describer = CassetteDescriber(Cassette.load(CASSETTE))
    provider = DescriberProvider(describer, PLACEHOLDER_CONFIDENCE)
    timeline = single_zone_timeline(START, Zone.MAIN_GATE, source.metadata.duration_seconds)
    config = VideoIngestConfig(sampling=SamplingConfig(every_seconds=3.0))
    ingested = run_video_ingest(source, provider, timeline, config)
    with Store.open(":memory:") as store:
        ingest_observations(store, "clip.mp4", ingested.telemetry, ingested.observations)
        build_events(store)
        events, entities = store.query_events(), store.list_entities()
    assert [e.event_id for e in events] == [e["id"] for e in live["events"]]
    assert [e.frame_ids for e in events] == [e["frames"] for e in live["events"]]
    assert [(n.entity_id, n.event_ids) for n in entities] == [
        (n["id"], n["events"]) for n in live["entities"]
    ]


def test_the_stored_confidence_is_the_labelled_placeholder_never_a_model_claim(evidence):
    objects = [
        o
        for f in evidence["replay"]["frames"]
        if f["observation"]
        for o in f["observation"]["objects"]
    ]
    assert objects and {o["confidence"] for o in objects} == {PLACEHOLDER_CONFIDENCE}
    assert "uncalibrated" in evidence["model"]["confidence"]
    assert evidence["recording"]["failed_frames"] == 0  # and any failure would have been recorded


def test_no_credential_is_persisted_in_any_artifact(evidence):
    files = [p for p in (REAL / "v03").iterdir() if p.is_file()] + [
        REAL / "clip.gemini.cassette.json"
    ]
    assert len(files) >= 5
    real_key = None
    env = ROOT / ".env"
    if env.exists():  # compare with the actual key if there is one; it is never printed
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                real_key = line.split("=", 1)[1].strip() or None
    leaked = scan_for_secrets(files, [real_key] if real_key else [])
    assert leaked == [], f"credential-shaped content in: {leaked}"


# ---- the evidence tooling (always runs) ----------
def test_provenance_records_only_what_is_known(tmp_path):
    import numpy as np

    video = write_video(tmp_path / "tiny.avi", [np.full((90, 160, 3), 50, np.uint8)] * 6, fps=2.0)
    unknown = clip_provenance(video)
    assert unknown["source"] == "not recorded" and unknown["licence"] == "not recorded"
    assert len(unknown["sha256"]) == 64 and unknown["size_bytes"] == video.stat().st_size
    assert unknown["frame_count"] == 6 and unknown["fps"] == 2.0
    known = clip_provenance(video, source="a named dataset", licence="CC0 (example)")
    assert (known["source"], known["licence"]) == ("a named dataset", "CC0 (example)")


def test_the_recording_log_is_parsed_not_retyped():
    log = (
        "| Provider                     | gemini:gemini-3.6-flash                      |\n"
        "| Provider calls               | 8, cassette replays 0                        |\n"
        "| Gemini API requests          | 12 (retries 4, total latency 124.5s)         |\n"
    )
    parsed = parse_record_log(log, wall_seconds=130.8)
    assert parsed["api_requests"] == 12 and parsed["retries"] == 4
    assert parsed["total_api_latency_s"] == 124.5 and parsed["provider_calls"] == 8
    assert parsed["provider"] == "gemini:gemini-3.6-flash" and parsed["wall_clock_seconds"] == 130.8
    assert parsed["failed_frames"] == 0
    assert parse_record_log("PERCEPTION FAILURES | 2 frame(s)")["failed_frames"] == 1


def test_the_secret_scan_finds_key_shaped_content_and_named_secrets(tmp_path):
    clean = tmp_path / "clean.json"
    clean.write_text('{"description": "a street", "confidence": 0.5}', encoding="utf-8")
    shaped = tmp_path / "shaped.log"
    shaped.write_text("key=AIza" + "x" * 35, encoding="utf-8")  # Google-key shaped, made up
    named = tmp_path / "named.db"
    named.write_bytes(b"\x00\x01TOTALLY-FAKE-SECRET-VALUE\x02")
    assert scan_for_secrets([clean]) == []
    assert scan_for_secrets([clean, shaped, named], ["TOTALLY-FAKE-SECRET-VALUE"]) == [
        "shaped.log",
        "named.db",
    ]
    assert scan_for_secrets([clean, named], [""]) == []  # an empty secret never matches everything
