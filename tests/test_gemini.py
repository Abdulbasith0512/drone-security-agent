"""Gemini adapter tests. The SDK boundary is faked: no network, no real key."""

import json
from io import StringIO
from types import SimpleNamespace

import cv2
import httpx
import numpy as np
import pytest
from conftest import IST, write_video
from google.genai import errors as genai_errors
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.models import Zone
from drone_agent.vision import DescriberProvider, FrameInput, ProviderOutput
from drone_agent.vision import gemini as gem
from drone_agent.vision.gemini import (
    GeminiAuthError,
    GeminiConfig,
    GeminiConfigError,
    GeminiDescriber,
    GeminiModelError,
    GeminiOutputError,
    GeminiRateLimitError,
    GeminiRequestError,
    GeminiTransientError,
    prepare_image,
    response_schema,
)

SECRET = "AIzaSyFAKE-not-a-real-key-1234567890"
GOOD = json.dumps(
    {
        "description": "A blue pickup truck is parked by the gate.",
        "objects": [
            {
                "type": "vehicle",
                "activity": "parked",
                "attributes": {"color": "blue", "body_type": "pickup"},
            }
        ],
    }
)


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """The developer's real .env must never leak into tests."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


class FakeClient:
    def __init__(self, *script):
        self.models = FakeModels(script)


def api_error(code, message="boom"):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


def make_frame(image=None):
    if image is None:
        image = np.zeros((90, 160, 3), np.uint8)
    return FrameInput(frame_id=7, timestamp="2026-09-29T23:40:00+05:30", image=image)


def describer(*script, **overrides):
    config = GeminiConfig(model="gemini-test", api_key=SECRET, **overrides)
    sleeps = []
    d = GeminiDescriber(config, client=FakeClient(*script), sleep=sleeps.append)
    d.slept = sleeps
    return d


# ---- configuration and key handling ---------------------------------------------------------
def test_missing_key_fails_immediately_with_a_concise_error():
    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY is not set"):
        GeminiConfig.from_env(load_env_file=False)


def test_key_and_model_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_MODEL", "env-model")
    assert GeminiConfig.from_env(load_env_file=False).model == "env-model"
    assert GeminiConfig.from_env("cli-model", load_env_file=False).model == "cli-model"
    monkeypatch.delenv("GEMINI_MODEL")
    assert GeminiConfig.from_env(load_env_file=False).model == gem.DEFAULT_MODEL


def test_config_never_shows_the_key_and_validates_settings():
    config = GeminiConfig(model="m", api_key=SECRET)
    assert SECRET not in repr(config) and SECRET not in str(config)
    for bad in ({"max_dimension": 10}, {"jpeg_quality": 0}, {"timeout_s": 0}, {"model": " "}):
        with pytest.raises(GeminiConfigError):
            GeminiConfig(**{"model": "m", "api_key": SECRET, **bad})


def test_the_dotenv_file_is_loaded_without_overriding_the_process_environment(
    monkeypatch, tmp_path
):
    seen = {}

    def fake_load(path, override):
        seen["override"] = override
        monkeypatch.setenv("GEMINI_API_KEY", SECRET)

    monkeypatch.setattr("dotenv.load_dotenv", fake_load)
    assert GeminiConfig.from_env().api_key == SECRET
    assert seen["override"] is False


# ---- image preparation ----------------------------------------------------------------------
def test_image_is_encoded_as_jpeg_with_correct_colors_and_no_resize_when_small():
    bgr = np.zeros((90, 160, 3), np.uint8)
    bgr[:, :] = (255, 0, 0)  # pure BLUE in OpenCV's BGR order
    data, width, height = prepare_image(bgr, max_dimension=1024, jpeg_quality=90)
    assert (width, height) == (160, 90) and data[:2] == b"\xff\xd8"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # back to BGR
    b, g, r = decoded[45, 80]
    assert b > 200 and g < 40 and r < 40  # still blue: channels were not swapped


def test_large_image_is_downscaled_keeping_aspect_ratio_and_never_upscaled():
    big = np.zeros((1080, 1920, 3), np.uint8)
    _, width, height = prepare_image(big, 640, 80)
    assert (width, height) == (640, 360)
    _, width, height = prepare_image(np.zeros((100, 50, 3), np.uint8), 640, 80)
    assert (width, height) == (50, 100)


@pytest.mark.parametrize(
    "bad", [None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 4), np.uint8)]
)
def test_prepare_image_rejects_non_bgr_input(bad):
    with pytest.raises(ValueError, match="BGR"):
        prepare_image(bad, 1024, 85)


# ---- structured output ----------------------------------------------------------------------
def test_valid_reply_becomes_a_validated_provider_output_and_the_request_is_shaped_correctly():
    d = describer(GOOD)
    out = d.describe(make_frame())
    assert isinstance(out, ProviderOutput)
    assert out.objects[0].type.value == "vehicle" and out.objects[0].confidence is None
    call = d.client.models.calls[0]
    assert call.model == "gemini-test"  # the selected model is what is sent
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_json_schema == response_schema()
    assert call.config.temperature == 0.0
    assert gem.PROVIDER_INSTRUCTIONS in call.config.system_instruction
    part = call.contents[0]
    assert part.inline_data.mime_type == "image/jpeg" and part.inline_data.data[:2] == b"\xff\xd8"
    assert (d.api_calls, d.retries, d.last_image_size) == (1, 0, (160, 90))
    assert d.name == "gemini:gemini-test"


def test_schema_omits_confidence_but_keeps_the_rest_of_the_contract():
    props = response_schema()["$defs"]["ProviderObject"]["properties"]
    assert "confidence" not in props
    assert set(props) == {"type", "activity", "attributes"}
    # the shared contract schema is untouched
    from drone_agent.vision import provider_output_schema

    assert "confidence" in provider_output_schema()["$defs"]["ProviderObject"]["properties"]


def test_confidence_placeholder_is_applied_downstream_and_labelled_uncalibrated():
    d = describer(GOOD)
    obs = DescriberProvider(d, gem.PLACEHOLDER_CONFIDENCE).analyze(
        FrameInput(
            frame_id=7,
            timestamp="2026-09-29T23:40:00+05:30",
            zone=Zone.MAIN_GATE,
            image=np.zeros((90, 160, 3), np.uint8),
        )
    )
    assert obs.source == "gemini:gemini-test"
    assert obs.detected_objects[0].confidence == gem.PLACEHOLDER_CONFIDENCE == 0.5
    assert obs.timestamp.utcoffset() == IST.utcoffset(None)


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        "[]",
        json.dumps({"description": "x", "objects": [{"type": "alien"}]}),
        json.dumps({"description": "x", "objects": [], "verdict": "suspicious"}),
        json.dumps({"objects": []}),
        "",
        None,
    ],
)
def test_materially_malformed_output_is_rejected_after_exactly_one_retry(bad):
    d = describer(bad, bad)
    with pytest.raises(GeminiOutputError):
        d.describe(make_frame())
    assert d.api_calls == 2  # one retry, no more
    assert d.client.models.script == []


def test_malformed_then_valid_output_succeeds_on_the_retry():
    d = describer("garbage", GOOD)
    assert d.describe(make_frame()).description.startswith("A blue pickup")
    assert (d.api_calls, d.retries) == (2, 1)


def test_output_is_not_regex_repaired_beyond_the_one_fence_the_contract_allows():
    d = describer('Sure! Here is the JSON: {"description": "x", "objects": []}', GOOD)
    d.describe(make_frame())
    assert d.api_calls == 2  # prose around JSON is a malformed reply, not extracted


# ---- failures -------------------------------------------------------------------------------
@pytest.mark.parametrize("code", [401, 403])
def test_auth_failure_is_not_retried(code):
    d = describer(api_error(code))
    with pytest.raises(GeminiAuthError):
        d.describe(make_frame())
    assert d.api_calls == 1 and d.slept == []


def test_bad_key_reported_as_a_400_is_classified_as_auth():
    d = describer(api_error(400, "API key not valid. Please pass a valid API key."))
    with pytest.raises(GeminiAuthError):
        d.describe(make_frame())
    assert d.api_calls == 1


def test_model_not_found_fails_without_retry_or_silent_fallback():
    d = describer(api_error(404, "models/gemini-test is not found"))
    with pytest.raises(GeminiModelError, match="gemini-test.*no fallback"):
        d.describe(make_frame())
    assert d.api_calls == 1
    assert {c.model for c in d.client.models.calls} == {"gemini-test"}


def test_other_bad_requests_fail_immediately():
    d = describer(api_error(400, "Invalid JSON payload"))
    with pytest.raises(GeminiRequestError):
        d.describe(make_frame())
    assert d.api_calls == 1


def test_transient_errors_are_retried_a_bounded_number_of_times_then_succeed():
    d = describer(api_error(503), httpx.ConnectTimeout("t"), GOOD, retry_delays=(1.0, 3.0))
    assert d.describe(make_frame()).objects
    assert d.api_calls == 3 and d.slept == [1.0, 3.0] and d.retries == 2


def test_transient_errors_give_up_after_the_bounded_retries():
    d = describer(*[api_error(500)] * 3, retry_delays=(1.0, 3.0))
    with pytest.raises(GeminiTransientError):
        d.describe(make_frame())
    assert d.api_calls == 3 and d.slept == [1.0, 3.0]  # never an infinite loop


def test_rate_limit_is_retried_then_reported_as_rate_limited():
    d = describer(*[api_error(429)] * 3, retry_delays=(1.0, 3.0))
    with pytest.raises(GeminiRateLimitError):
        d.describe(make_frame())
    assert d.api_calls == 3


def test_timeouts_and_network_errors_count_as_transient():
    d = describer(httpx.ReadTimeout("slow"), TimeoutError(), ConnectionError(), retry_delays=(0, 0))
    with pytest.raises(GeminiTransientError):
        d.describe(make_frame())


def test_the_api_key_never_appears_in_error_text_even_if_the_sdk_echoes_it():
    leaks = [
        api_error(403, f"denied for key {SECRET}"),
        api_error(400, f"bad request with {SECRET}"),
        api_error(404, f"model missing ({SECRET})"),
        RuntimeError(f"sdk exploded: {SECRET}"),
    ]
    for leak in leaks:
        d = describer(leak)
        with pytest.raises(gem.GeminiError) as info:
            d.describe(make_frame())
        assert SECRET not in str(info.value) and SECRET not in repr(info.value)
        assert info.value.__cause__ is None  # the raw SDK exception is not chained
    d = describer("nope", f'{{"description": "{SECRET}", "objects": 5}}')
    with pytest.raises(GeminiOutputError) as info:
        d.describe(make_frame())
    assert SECRET not in repr(d.config)


# ---- CLI ------------------------------------------------------------------------------------
def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


@pytest.fixture
def fake_gemini(monkeypatch):
    """Route the CLI's GeminiDescriber to one with a scripted fake client."""
    box = SimpleNamespace(script=[], instances=[])

    def factory(config):
        d = GeminiDescriber(config, client=FakeClient(*box.script), sleep=lambda s: None)
        box.instances.append(d)
        return d

    monkeypatch.setattr("drone_agent.__main__.GeminiDescriber", factory)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    return box


@pytest.fixture
def video(tmp_path):
    frames = [np.full((72, 128, 3), 30, np.uint8)] * 20 + [
        np.full((72, 128, 3), 200, np.uint8)
    ] * 20
    return write_video(tmp_path / "v.avi", frames, fps=10.0)  # 4 s, two scenes


def test_describe_frame_cli_reports_the_result_and_stores_nothing(video, fake_gemini, tmp_path):
    fake_gemini.script = [GOOD]
    code, text = run_cli("describe-frame", video, "--time", 1.5, "--model", "gemini-cli-model")
    assert code == 0, text
    for expected in (
        "GEMINI_API_KEY configured: yes",
        "gemini-cli-model",
        "index 15",
        "128x72 JPEG",
        "API requests",
        "ProviderOutput valid",
        "blue pickup",
        "vehicle",
        "parked",
    ):
        assert expected in text
    assert SECRET not in text
    assert fake_gemini.instances[0].api_calls == 1
    assert not list(tmp_path.glob("*.db"))


def test_describe_frame_cli_errors_are_clear_and_secret_free(video, fake_gemini, monkeypatch):
    fake_gemini.script = [api_error(403, f"key {SECRET} denied")]
    code, text = run_cli("describe-frame", video, "--time", 1)
    assert code == 2 and "rejected the API key" in text and SECRET not in text
    assert run_cli("describe-frame", video, "--time", 99)[0] == 2  # beyond the end
    monkeypatch.delenv("GEMINI_API_KEY")
    code, text = run_cli("describe-frame", video, "--time", 1)
    assert code == 2 and "GEMINI_API_KEY is not set" in text


def test_ingest_video_record_calls_gemini_only_for_gate_accepted_frames_then_replays_free(
    video, fake_gemini, tmp_path
):
    fake_gemini.script = [GOOD, GOOD]
    cassette = tmp_path / "rec.cassette.json"
    args = ["ingest-video", video, "--sample-every", 1, "--cassette", cassette]
    code, text = run_cli(*args, "--db", tmp_path / "a.db", "--provider", "gemini",
                         "--cassette-mode", "record", "--model", "m1")  # fmt: skip
    assert code == 0, text
    # 4 samples: t=0 (accepted), t=1 (skipped), t=2 (scene change: accepted), t=3 (skipped)
    assert fake_gemini.instances[0].api_calls == 2
    for expected in (
        "gemini:m1", "Gemini API requests", "recorded 2", "carried 2",
        "UNCALIBRATED placeholder 0.5",
    ):  # fmt: skip
        assert expected in text
    data = json.loads(cassette.read_text())
    assert data["provider"] == "gemini:m1" and len(data["entries"]) == 2
    assert SECRET not in cassette.read_text() and "api_key" not in cassette.read_text().lower()
    for entry in data["entries"]:  # nothing invented in the recording
        assert [o["confidence"] for o in entry["output"]["objects"]] == [None]

    # replay: zero Gemini calls (the fake client would raise IndexError on any call)
    fake_gemini.instances.clear()
    code, replay_text = run_cli(*args, "--db", tmp_path / "b.db", "--provider", "cassette")
    assert code == 0, replay_text
    assert not fake_gemini.instances and "cassette replays 2" in replay_text
    from drone_agent.store import Store

    rows = []
    for name in ("a.db", "b.db"):
        with Store.open(tmp_path / name) as s:
            rows.append([s.get_observation_by_frame(i) for i in (0, 10, 20, 30)])
    for rec, rep in zip(*rows, strict=True):
        assert rec is not None and rep is not None
        assert (rec.description, rec.detected_objects, rec.zone, rec.timestamp) == (
            rep.description,
            rep.detected_objects,
            rep.zone,
            rep.timestamp,
        )
    assert [o.source for o in rows[0]] == [
        "gemini:m1", "gemini:m1+carried", "gemini:m1", "gemini:m1+carried"
    ]  # fmt: skip
    assert [o.source for o in rows[1]] == [
        "cassette", "cassette+carried", "cassette", "cassette+carried"
    ]  # fmt: skip


def test_record_mode_flushes_the_cassette_when_a_later_call_fails(video, fake_gemini, tmp_path):
    fake_gemini.script = [GOOD, api_error(403)]  # 2nd accepted frame fails
    cassette = tmp_path / "partial.json"
    code, text = run_cli("ingest-video", video, "--sample-every", 1, "--cassette", cassette,
                         "--db", tmp_path / "p.db", "--provider", "gemini",
                         "--cassette-mode", "record")  # fmt: skip
    assert code == 2 and "rejected the API key" in text and "saved with 1 recorded" in text
    assert len(json.loads(cassette.read_text())["entries"]) == 1
    assert not (tmp_path / "p.db").exists()  # nothing half-ingested


def test_record_mode_resumes_from_an_existing_cassette(video, fake_gemini, tmp_path):
    cassette = tmp_path / "c.json"
    common = ["ingest-video", video, "--sample-every", 1, "--cassette", cassette,
              "--provider", "gemini", "--cassette-mode", "record", "--model", "m"]  # fmt: skip
    fake_gemini.script = [GOOD, api_error(403)]
    assert run_cli(*common, "--db", tmp_path / "1.db")[0] == 2
    fake_gemini.script, fake_gemini.instances = [GOOD], []
    code, text = run_cli(*common, "--db", tmp_path / "2.db")
    assert code == 0, text
    assert fake_gemini.instances[0].api_calls == 1  # only the missing frame is paid for again
    assert len(json.loads(cassette.read_text())["entries"]) == 2


def test_cli_rejects_inconsistent_provider_and_mode_combinations(video, fake_gemini, tmp_path):
    db = ["--db", tmp_path / "x.db"]
    assert run_cli("ingest-video", video, *db, "--provider", "gemini")[0] == 2  # replay + gemini
    assert (
        run_cli("ingest-video", video, *db, "--provider", "gemini", "--cassette-mode", "record")[0]
        == 2
    )  # record needs a cassette path
    assert (
        run_cli("ingest-video", video, *db, "--provider", "cassette", "--cassette-mode", "record",
                "--cassette", tmp_path / "n.json")[0] == 2
    )  # fmt: skip
    other = tmp_path / "other.json"
    fake_gemini.script = [GOOD, GOOD]
    run_cli("ingest-video", video, *db, "--provider", "gemini", "--cassette-mode", "record",
            "--cassette", other, "--model", "first")  # fmt: skip
    code, text = run_cli(
        "ingest-video", video, "--db", tmp_path / "y.db", "--provider", "gemini",
        "--cassette-mode", "record", "--cassette", other, "--model", "second",
    )  # fmt: skip
    assert code == 2 and "recorded by 'gemini:first'" in text  # never mixes models in a cassette


def test_live_mode_uses_gemini_without_a_cassette(video, fake_gemini, tmp_path):
    fake_gemini.script = [GOOD, GOOD]
    code, text = run_cli("ingest-video", video, "--sample-every", 1, "--db", tmp_path / "l.db",
                         "--provider", "gemini", "--cassette-mode", "live")  # fmt: skip
    assert code == 0, text
    assert fake_gemini.instances[0].api_calls == 2 and "Cassette" not in text
