"""Perception agent: fallback chain, failure isolation, timeouts, guard, provenance."""

import time
from datetime import datetime

import pytest
from conftest import IST

from drone_agent.agents import GuardConfig, PerceptionAgent
from drone_agent.models import DetectedObject, ObjectType, Observation, Provenance, Zone
from drone_agent.vision import FrameInput, ProviderOutputError
from drone_agent.vision.contract import DescriberProvider, parse_provider_output

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)
FRAME = FrameInput(frame_id=1, timestamp=T0, zone=Zone.MAIN_GATE)


def person(**kw) -> DetectedObject:
    return DetectedObject(object_type=ObjectType.PERSON, confidence=kw.pop("confidence", 0.9), **kw)


def observation(objects=(), description="A person at the gate", **kw) -> Observation:
    return Observation(
        observation_id="obs-1",
        frame_id=1,
        timestamp=T0,
        source="fake",
        zone=Zone.MAIN_GATE,
        description=description,
        detected_objects=list(objects),
        **kw,
    )


class Fake:
    """A provider that returns a fixed observation, raises, or sleeps first."""

    def __init__(self, name, result=None, error=None, delay=0.0):
        self.name, self.result, self.error, self.delay, self.calls = name, result, error, delay, 0

    def analyze(self, frame):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.error:
            raise self.error
        return self.result


# ---- fallback chain -------------------------------------------------------------------------
def test_the_first_healthy_provider_answers_and_later_ones_are_not_called():
    primary, backup = Fake("live", observation([person()])), Fake("scripted", observation())
    result = PerceptionAgent([primary, backup]).perceive(FRAME)
    assert result.trace.status == "ok" and result.trace.provider == "live"
    assert primary.calls == 1 and backup.calls == 0
    assert [(a.provider, a.ok) for a in result.trace.attempts] == [("live", True)]


def test_a_failing_provider_falls_back_and_the_failure_is_recorded_not_hidden():
    primary = Fake("live", error=ProviderOutputError("provider output is not valid JSON: x"))
    backup = Fake("scripted", observation([person()]))
    agent = PerceptionAgent([primary, backup])
    result = agent.perceive(FRAME)
    assert result.trace.status == "fallback" and result.trace.provider == "scripted"
    first, second = result.trace.attempts
    assert not first.ok and "ProviderOutputError" in first.error and second.ok
    assert result.observation is not None
    assert agent.stats.fallbacks == 1 and agent.stats.ok == 0 and agent.stats.failed == 0


def test_when_every_provider_fails_the_frame_is_dropped_never_invented():
    agent = PerceptionAgent(
        [Fake("a", error=RuntimeError("boom")), Fake("b", error=TimeoutError("slow"))]
    )
    result = agent.perceive(FRAME)
    assert result.observation is None and result.trace.status == "failed"
    assert [a.ok for a in result.trace.attempts] == [False, False]
    assert agent.stats.failed == 1 and agent.stats.frames == 1


def test_one_bad_frame_does_not_stop_the_next_one():
    flaky = Fake("live", observation([person()]))
    agent = PerceptionAgent([flaky])
    flaky.error = RuntimeError("transient")
    assert agent.perceive(FRAME).observation is None
    flaky.error = None
    assert agent.perceive(FRAME).observation is not None
    assert (agent.stats.failed, agent.stats.ok) == (1, 1)


def test_error_text_in_the_trace_is_short_and_carries_no_newlines():
    long = RuntimeError("line one\n" + "x" * 500)
    result = PerceptionAgent([Fake("a", error=long)]).perceive(FRAME)
    error = result.trace.attempts[0].error
    assert "\n" not in error and len(error) < 200


def test_a_hung_provider_times_out_and_the_chain_moves_on():
    slow = Fake("slow", observation(), delay=0.4)
    quick = Fake("quick", observation([person()]))
    agent = PerceptionAgent([slow, quick], call_timeout_s=0.05)
    result = agent.perceive(FRAME)
    agent.close()
    assert result.trace.status == "fallback" and result.trace.provider == "quick"
    assert "TimeoutError" in result.trace.attempts[0].error
    assert agent.stats.timeouts == 1


def test_an_agent_needs_a_provider():
    with pytest.raises(ValueError):
        PerceptionAgent([])


def test_name_lists_the_chain():
    assert PerceptionAgent(
        [Fake("gemini", observation()), Fake("scripted", observation())]
    ).name == ("gemini+scripted")


# ---- guard ----------------------------------------------------------------------------------
def test_duplicate_object_ids_are_collapsed_and_flagged():
    obs = observation([person(object_id="p1"), person(object_id="p1"), person(object_id="p2")])
    result = PerceptionAgent([Fake("a", obs)]).perceive(FRAME)
    assert len(result.observation.detected_objects) == 2
    assert "dropped_duplicate_ids:1" in result.trace.flags


def test_objects_without_ids_are_never_collapsed():
    """Two people in dark jackets are two people; without positions we cannot say otherwise."""
    obs = observation(
        [person(attributes={"clothing": "dark"}), person(attributes={"clothing": "dark"})]
    )
    result = PerceptionAgent([Fake("a", obs)]).perceive(FRAME)
    assert len(result.observation.detected_objects) == 2 and result.trace.flags == []


def test_low_confidence_and_excess_objects_are_bounded():
    objs = [person(confidence=0.1), *[person(confidence=0.9) for _ in range(5)]]
    guard = GuardConfig(min_confidence=0.5, max_objects=3)
    result = PerceptionAgent([Fake("a", observation(objs))], guard=guard).perceive(FRAME)
    assert len(result.observation.detected_objects) == 3
    assert {"dropped_low_confidence:1", "capped_objects:2"} <= set(result.trace.flags)


def test_overlong_descriptions_are_truncated_and_short_ones_untouched():
    text = "x" * 900
    result = PerceptionAgent([Fake("a", observation(description=text))]).perceive(FRAME)
    assert len(result.observation.description) == 500 and "truncated_description" in (
        result.trace.flags
    )
    clean = PerceptionAgent([Fake("a", observation([person()]))]).perceive(FRAME)
    assert clean.trace.flags == []


def test_an_unknown_object_class_is_flagged_when_it_was_coerced():
    coerced = person(attributes={"reported_class": "truck"})
    result = PerceptionAgent([Fake("a", observation([coerced]))]).perceive(FRAME)
    assert "coerced_class:truck" in result.trace.flags


# ---- lenient class parsing (vision contract) ---------------------------------------------------
STRICT_BAD = '{"description": "a truck", "objects": [{"type": "truck", "confidence": 0.9}]}'


def test_unknown_class_is_rejected_by_default_and_coerced_only_on_request():
    with pytest.raises(ProviderOutputError):
        parse_provider_output(STRICT_BAD)
    out = parse_provider_output(STRICT_BAD, lenient_types=True)
    (obj,) = out.objects
    assert obj.type is ObjectType.OTHER and obj.attributes == {"reported_class": "truck"}


def test_lenient_mode_still_rejects_everything_else_that_is_malformed():
    for bad in (
        '{"description": "x", "objects": [{"type": "person", "confidence": 7}]}',
        '{"objects": []}',
        "not json",
    ):
        with pytest.raises(ProviderOutputError):
            parse_provider_output(bad, lenient_types=True)


def test_known_classes_are_untouched_in_lenient_mode():
    out = parse_provider_output(
        '{"description": "d", "objects": [{"type": "vehicle", "attributes": {"color": "blue"}}]}',
        lenient_types=True,
    )
    assert out.objects[0].type is ObjectType.VEHICLE and out.objects[0].attributes == {
        "color": "blue"
    }


# ---- provenance -----------------------------------------------------------------------------
def test_describer_provider_records_how_the_answer_was_obtained():
    from drone_agent.vision.contract import ProviderOutput

    class Replay:
        name = "cassette"
        last_provenance = Provenance.REPLAYED

        def describe(self, frame):
            return ProviderOutput(description="a car", objects=[])

    obs = DescriberProvider(Replay()).analyze(
        FrameInput(frame_id=1, timestamp=T0, zone=Zone.MAIN_GATE)
    )
    assert obs.provenance is Provenance.REPLAYED


def test_a_describer_that_says_nothing_leaves_provenance_unknown():
    from drone_agent.vision.contract import ProviderOutput

    class Plain:
        name = "plain"

        def describe(self, frame):
            return ProviderOutput(description="a car", objects=[])

    obs = DescriberProvider(Plain()).analyze(
        FrameInput(frame_id=1, timestamp=T0, zone=Zone.MAIN_GATE)
    )
    assert obs.provenance is None
