from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.models import ObjectType, Observation, Zone
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.vision import FrameInput, ScriptedProvider, UnknownFrameError, VisionProvider

SECURITY = SCENARIO_DIR / "security_events.yaml"


@pytest.fixture(scope="module")
def security():
    return load_scenario(SECURITY)


def test_scripted_provider_produces_validated_observation(security):
    provider = ScriptedProvider(security)
    assert isinstance(provider, VisionProvider)
    ts = security.clock.at(60)
    obs = provider.analyze(FrameInput(frame_id=1, timestamp=ts))
    assert isinstance(obs, Observation)
    assert (obs.observation_id, obs.frame_id, obs.source, obs.zone) == (
        "obs-1",
        1,
        "scripted",
        Zone.MAIN_GATE,
    )
    assert obs.timestamp == ts and obs.telemetry is None
    person = obs.detected_objects[0]
    assert person.object_type is ObjectType.PERSON
    assert person.activity == "standing"
    assert person.attributes == {"clothing": "dark jacket"}
    assert person.confidence == 0.94


def test_scripted_provider_rejects_unknown_frame_and_wrong_timestamp(security):
    provider = ScriptedProvider(security)
    with pytest.raises(UnknownFrameError):
        provider.analyze(FrameInput(frame_id=999, timestamp=security.clock.at(60)))
    with pytest.raises(ValueError, match="does not match"):
        provider.analyze(FrameInput(frame_id=1, timestamp=security.clock.at(61)))


def test_run_scenario_is_deterministic(security):
    first, second = run_scenario(security), run_scenario(load_scenario(SECURITY))
    assert first.telemetry == second.telemetry
    assert first.observations == second.observations
    assert [o.model_dump_json() for o in first.observations] == [
        o.model_dump_json() for o in second.observations
    ]


def test_every_observation_is_synchronised_and_only_the_dropout_frame_is_stale(security):
    result = run_scenario(security)
    assert len(result.observations) == 27
    stale = [o.frame_id for o in result.observations if o.telemetry.stale]
    assert stale == [18]
    frame18 = result.observations[17]
    assert frame18.telemetry.age_seconds == 60  # last sample delivered at offset 1560
    for obs in result.observations:
        assert obs.telemetry.telemetry.timestamp <= obs.timestamp
        assert obs.telemetry.telemetry.zone == obs.zone


def test_normal_day_has_no_stale_telemetry():
    result = run_scenario(load_scenario(SCENARIO_DIR / "normal_day.yaml"))
    assert len(result.observations) == 11
    assert not any(o.telemetry.stale for o in result.observations)


def test_cli_simulate_prints_timeline():
    out = StringIO()
    code = main(["simulate", str(SECURITY)], Console(file=out, width=200, force_terminal=False))
    text = out.getvalue()
    assert code == 0
    for expected in (
        "security_events",
        "23:41:00",
        "main_gate",
        "restricted_zone",
        "Blue pickup truck stopped at the main gate",
        "STALE",
        "dark jacket",
    ):
        assert expected in text


def test_cli_reports_bad_scenario_without_traceback(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [oops", encoding="utf-8")
    out = StringIO()
    assert main(["simulate", str(bad)], Console(file=out, width=200)) == 2
    assert "error" in out.getvalue()
