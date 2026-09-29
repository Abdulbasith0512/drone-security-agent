import json
from datetime import datetime

import pytest
from conftest import IST

from drone_agent.models import ObjectType, Observation, Zone
from drone_agent.video import SamplingConfig, VideoSource, sample_frames
from drone_agent.video.synthetic import DEMO_PROVIDER_LABEL, build_demo_cassette, demo_output
from drone_agent.vision import (
    PROVIDER_INSTRUCTIONS,
    Cassette,
    CassetteDescriber,
    CassetteFormatError,
    CassetteMismatch,
    CassetteMiss,
    CassetteMode,
    DescriberProvider,
    FrameInput,
    ProviderOutput,
    ProviderOutputError,
    VisionProvider,
    observation_from_output,
    parse_provider_output,
    provider_output_schema,
)
from drone_agent.vision.cassette import CassetteEntry, CassetteVideo

GOOD = {
    "description": "A person stands near the gate.",
    "objects": [
        {
            "type": "person",
            "activity": "standing",
            "attributes": {"clothing": "dark jacket"},
            "confidence": 0.9,
        }
    ],
}
T = datetime(2026, 9, 29, 23, 40, tzinfo=IST)


def frame(index=20, zone=Zone.MAIN_GATE, key="k:20", chash="abc") -> FrameInput:
    return FrameInput(
        frame_id=index,
        timestamp=T,
        zone=zone,
        offset_seconds=2.0,
        frame_key=key,
        content_hash=chash,
    )


# ---- provider output schema -----------------------------------------------------------------
def test_valid_output_parses_from_dict_json_and_a_fenced_block():
    from_dict = parse_provider_output(GOOD)
    assert from_dict.objects[0].type is ObjectType.PERSON
    assert parse_provider_output(json.dumps(GOOD)) == from_dict
    assert parse_provider_output(f"```json\n{json.dumps(GOOD)}\n```") == from_dict
    assert parse_provider_output({"description": "Nothing visible."}).objects == []


@pytest.mark.parametrize(
    "raw,fragment",
    [
        ("not json at all", "not valid JSON"),
        ('["a", "list"]', "must be a JSON object"),
        ({}, "description"),
        ({"description": "  "}, "blank"),
        ({"description": "x", "extra": 1}, "extra"),
        ({"description": "x", "objects": [{"type": "truck"}]}, "type"),
        ({"description": "x", "objects": [{"type": "person", "confidence": 1.5}]}, "confidence"),
        ({"description": "x", "objects": [{"type": "person", "confidence": "high"}]}, "confidence"),
        (
            {"description": "x", "objects": [{"type": "person", "attributes": {"Color": "red"}}]},
            "lower_snake",
        ),
        (
            {"description": "x", "objects": [{"type": "person", "attributes": {"age": 30}}]},
            "attributes",
        ),
        ({"description": "x", "objects": [{"type": "person", "activity": ""}]}, "activity"),
        ({"description": "x", "objects": [{"type": "person", "unknown": 1}]}, "unknown"),
        ({"description": "x", "objects": [{"type": "other"}] * 51}, "objects"),
    ],
)
def test_malformed_output_fails_clearly_and_nothing_is_invented(raw, fragment):
    with pytest.raises(ProviderOutputError, match=fragment):
        parse_provider_output(raw)


def test_schema_and_instructions_are_available_for_a_real_provider():
    schema = provider_output_schema()
    assert set(schema["properties"]) == {"description", "objects"}
    text = PROVIDER_INSTRUCTIONS.lower()
    for forbidden in ("identity", "intent", "criminality", "suspicious"):
        assert forbidden in text  # the prompt explicitly forbids these conclusions
    assert "do not" in text and "json" in text


# ---- ProviderOutput -> Observation ----------------------------------------------------------
def test_observation_is_built_from_output_without_inventing_fields():
    obs = observation_from_output(frame(), parse_provider_output(GOOD), source="cassette")
    assert isinstance(obs, Observation)
    assert (obs.observation_id, obs.frame_id, obs.zone, obs.source) == (
        "obs-20",
        20,
        Zone.MAIN_GATE,
        "cassette",
    )
    assert obs.timestamp == T and obs.telemetry is None
    (person,) = obs.detected_objects
    assert (person.object_type, person.activity, person.confidence) == (
        ObjectType.PERSON,
        "standing",
        0.9,
    )
    assert person.attributes == {"clothing": "dark jacket"} and person.object_id is None


def test_missing_confidence_needs_an_explicit_default():
    output = parse_provider_output({"description": "d", "objects": [{"type": "vehicle"}]})
    with pytest.raises(ProviderOutputError, match="no confidence"):
        observation_from_output(frame(), output, "cassette")
    obs = observation_from_output(frame(), output, "cassette", default_confidence=0.5)
    assert obs.detected_objects[0].confidence == 0.5
    with pytest.raises(ValueError, match="between 0 and 1"):
        observation_from_output(frame(), output, "cassette", default_confidence=2.0)


def test_a_frame_without_a_zone_cannot_become_an_observation():
    no_zone = FrameInput(frame_id=1, timestamp=T)
    with pytest.raises(ValueError, match="zone"):
        observation_from_output(no_zone, parse_provider_output(GOOD), "cassette")


def test_describer_provider_is_a_vision_provider_returning_ordinary_observations():
    class Describer:
        name = "fake"

        def describe(self, frame):
            return parse_provider_output(GOOD)

    provider = DescriberProvider(Describer())
    assert isinstance(provider, VisionProvider) and provider.name == "fake"
    assert isinstance(provider.analyze(frame()), Observation)


# ---- cassette file --------------------------------------------------------------------------
def test_demo_cassette_round_trips_and_saves_deterministically(demo_video, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    cassette = build_demo_cassette(demo_video)
    cassette.save(a)
    Cassette.load(a).save(b)
    assert a.read_bytes() == b.read_bytes()
    assert len(Cassette.load(a)) == 20 and Cassette.load(a).data.provider == DEMO_PROVIDER_LABEL
    indexes = [e["frame_index"] for e in json.loads(a.read_text())["entries"]]
    assert indexes == sorted(indexes) == list(range(0, 200, 10))


def test_cassette_holds_only_safe_metadata(demo_video, tmp_path):
    path = tmp_path / "c.json"
    build_demo_cassette(demo_video).save(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert set(data) == {"format", "provider", "contract_version", "video", "entries"}
    assert set(data["video"]) == {"name", "sha256", "fps", "frame_count", "width", "height"}
    assert set(data["entries"][0]) == {
        "key",
        "frame_index",
        "offset_seconds",
        "content_hash",
        "output",
    }
    assert data["video"]["name"] == "demo.avi"
    assert str(demo_video.parent) not in text and "\\" not in text  # no local paths
    assert not any(word in text.lower() for word in ("api_key", "secret", "token", "password"))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(format="something-else"),
        lambda d: d.pop("video"),
        lambda d: d["entries"].append(d["entries"][0]),  # duplicate key
        lambda d: d["entries"][0]["output"].update(objects=[{"type": "truck"}]),
        lambda d: d.update(unexpected=1),
    ],
)
def test_invalid_cassette_files_are_rejected(demo_video, tmp_path, mutate):
    path = tmp_path / "c.json"
    build_demo_cassette(demo_video).save(path)
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(CassetteFormatError):
        Cassette.load(path)


def test_unreadable_cassette_files_are_rejected(tmp_path):
    with pytest.raises(CassetteFormatError):
        Cassette.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(CassetteFormatError):
        Cassette.load(bad)


# ---- replay ---------------------------------------------------------------------------------
def demo_cassette(demo_cassette_path) -> Cassette:
    return Cassette.load(demo_cassette_path)


def demo_frames(demo_video, seconds=1.0):
    return list(sample_frames(VideoSource(demo_video), SamplingConfig(every_seconds=seconds)))


def as_input(f, zone=Zone.MAIN_GATE) -> FrameInput:
    return FrameInput(
        frame_id=f.frame_id,
        timestamp=T,
        zone=zone,
        frame_key=f.frame_key,
        content_hash=f.content_hash,
        offset_seconds=f.offset_seconds,
    )


def test_replay_returns_the_recorded_output_and_counts_replays(demo_video, demo_cassette_path):
    describer = CassetteDescriber(demo_cassette(demo_cassette_path))
    frames = demo_frames(demo_video)
    person_frame = next(f for f in frames if f.frame_index == 50)  # 5.0 s: person scene
    output = describer.describe(as_input(person_frame))
    assert output == demo_output("person") and describer.replayed == 1
    assert describer.content_mismatches == [] and describer.name == "cassette"


def test_replay_miss_is_an_error_naming_the_frame(demo_video, demo_cassette_path):
    describer = CassetteDescriber(demo_cassette(demo_cassette_path))
    half_second = demo_frames(demo_video, seconds=0.5)[1]  # index 5: not in a 1 s cassette
    with pytest.raises(CassetteMiss, match="frame index 5"):
        describer.describe(as_input(half_second))


def test_replay_never_calls_an_inner_describer_and_needs_frame_identity(demo_cassette_path):
    describer = CassetteDescriber(demo_cassette(demo_cassette_path))
    with pytest.raises(ValueError, match="frame_key"):
        describer.describe(FrameInput(frame_id=1, timestamp=T, zone=Zone.MAIN_GATE))
    with pytest.raises(ValueError, match="inner describer"):
        CassetteDescriber(demo_cassette(demo_cassette_path), CassetteMode.RECORD)


def test_pixel_hash_differences_are_recorded_not_fatal(demo_video, demo_cassette_path):
    describer = CassetteDescriber(demo_cassette(demo_cassette_path))
    f = demo_frames(demo_video)[0]
    tampered = as_input(f).model_copy(update={"content_hash": "different"})
    describer.describe(tampered)
    assert describer.content_mismatches == [f.frame_key]


def test_cassette_for_another_video_is_refused_up_front(demo_cassette_path, tmp_path):
    import numpy as np
    from conftest import write_video

    other = write_video(tmp_path / "other.avi", [np.full((36, 64, 3), 9, np.uint8)] * 5)
    describer = CassetteDescriber(demo_cassette(demo_cassette_path))
    with pytest.raises(CassetteMismatch, match="different video"):
        describer.check_video(VideoSource(other).metadata.sha256)


def test_frame_identity_is_stable_across_decodes_and_distinguishes_frames(demo_video):
    first = {f.frame_index: f for f in demo_frames(demo_video)}
    second = {f.frame_index: f for f in demo_frames(demo_video)}
    assert {i: (f.frame_key, f.content_hash) for i, f in first.items()} == {
        i: (f.frame_key, f.content_hash) for i, f in second.items()
    }
    assert first[0].content_hash == first[160].content_hash  # both are the empty yard...
    assert first[0].frame_key != first[160].frame_key  # ...but they are different frames
    assert first[0].content_hash != first[50].content_hash  # empty vs person


# ---- record / live plumbing (a fake inner describer stands in for a future real VLM) -------
class FakeDescriber:
    name = "fake-vlm"

    def __init__(self):
        self.calls = 0

    def describe(self, frame):
        self.calls += 1
        return ProviderOutput(description=f"fake description {frame.frame_id}")


def empty_cassette(demo_video) -> Cassette:
    meta = VideoSource(demo_video).metadata
    return Cassette.new(
        "fake-vlm",
        CassetteVideo(
            name=meta.source_name,
            sha256=meta.sha256,
            fps=meta.fps,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
        ),
    )


def test_record_mode_calls_the_inner_describer_once_per_frame_then_replays(demo_video, tmp_path):
    inner, path = FakeDescriber(), tmp_path / "rec.json"
    describer = CassetteDescriber(empty_cassette(demo_video), CassetteMode.RECORD, inner, path)
    f = as_input(demo_frames(demo_video, seconds=5)[1])
    first = describer.describe(f)
    again = describer.describe(f)
    assert first == again and inner.calls == 1
    assert (describer.recorded, describer.replayed, describer.name) == (1, 1, "fake-vlm")
    describer.flush()
    reloaded = Cassette.load(path)
    assert len(reloaded) == 1 and reloaded.lookup(f.frame_key).output == first
    replay = CassetteDescriber(reloaded)  # a later run needs no provider at all
    assert replay.describe(f) == first


def test_live_mode_always_calls_the_inner_describer_and_leaves_the_cassette_alone(demo_video):
    inner = FakeDescriber()
    cassette = empty_cassette(demo_video)
    describer = CassetteDescriber(cassette, CassetteMode.LIVE, inner)
    f = as_input(demo_frames(demo_video, seconds=5)[0])
    describer.describe(f)
    describer.describe(f)
    assert inner.calls == 2 and describer.live_calls == 2 and len(cassette) == 0


def test_a_misbehaving_inner_describer_is_rejected(demo_video):
    class Bad:
        name = "bad"

        def describe(self, frame):
            return {"description": "a dict, not a ProviderOutput"}

    describer = CassetteDescriber(empty_cassette(demo_video), CassetteMode.LIVE, Bad())
    with pytest.raises(ProviderOutputError):
        describer.describe(as_input(demo_frames(demo_video, seconds=5)[0]))


def test_cassette_entries_are_validated_on_add(demo_video):
    with pytest.raises(ValueError):
        CassetteEntry(
            key="k", frame_index=-1, offset_seconds=0, content_hash="h", output=demo_output("empty")
        )
