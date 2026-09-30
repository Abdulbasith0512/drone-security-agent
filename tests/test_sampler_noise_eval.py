"""The sampler and noise experiments: their machinery is correct and their claims are checkable."""

import random

import numpy as np
import pytest
from conftest import SCENARIO_DIR

from drone_agent.eval import noise_exp, sampler_exp
from drone_agent.eval.noise_exp import NoisyProvider
from drone_agent.models import ObjectType
from drone_agent.scenario import load_scenario
from drone_agent.video.gate import FrameGate, GateConfig
from drone_agent.vision import FrameInput, ScriptedProvider


def scenario(name):
    return load_scenario(SCENARIO_DIR / f"{name}.yaml")


# ---- sampler experiment ------------------------------------------------------------------------
def test_the_fast_pixel_signal_equals_the_real_frame_gate_score():
    stream = sampler_exp.generate(0, keep_images=True)
    real = FrameGate(GateConfig(threshold=0.0))
    fast = sampler_exp.PixelSignal(stream)
    for i in range(0, 120, 3):
        assert real.score(stream.images[i]) == fast.score(i)
        real.set_reference(stream.images[i])
        fast.set_reference(i)


def test_streams_are_deterministic_and_carry_labelled_changes():
    a, b = sampler_exp.generate(3), sampler_exp.generate(3)
    assert [c.frame for c in a.changes] == [c.frame for c in b.changes]
    assert all(np.array_equal(x, y) for x, y in zip(a.small, b.small, strict=True))
    meaningful = [c for c in a.changes if c.meaningful]
    assert meaningful and {c.kind for c in meaningful} == {"appear", "leave"}
    assert any(not c.meaningful for c in a.changes)  # illumination steps are the distractors


def test_scoring_counts_a_change_only_when_a_frame_is_sent_in_time():
    stream = sampler_exp.generate(0)
    first = next(c for c in stream.changes if c.meaningful).frame
    on_time = sampler_exp.score_policy(stream, [first + sampler_exp.DETECT_WINDOW])
    too_late = sampler_exp.score_policy(stream, [first + sampler_exp.DETECT_WINDOW + 1])
    assert on_time["detected"] >= too_late["detected"]
    assert sampler_exp.score_policy(stream, list(range(sampler_exp.FRAMES)))["recall"] == 1.0
    assert sampler_exp.score_policy(stream, [])["detected"] == 0


def test_the_experiment_reports_calls_together_with_misses_for_every_policy():
    result = sampler_exp.run(seeds=(0, 1, 2))
    policies = result.measured["policies"]
    assert policies["every frame (gate off)"]["recall"] == 1.0
    for row in policies.values():
        assert {"calls_mean", "avoided_pct_mean", "recall", "missed_small", "missed_large"} <= set(
            row
        )
    pixel = policies["pixel gate, threshold 6 (previous default)"]
    adaptive = policies["adaptive (pixel + telemetry + event watch + refresh)"]
    # The claim being tested: adaptive detects more than the pixel gate alone, and it pays for it.
    assert adaptive["recall"] > pixel["recall"] and adaptive["calls_mean"] > pixel["calls_mean"]
    assert pixel["missed_small"] > 0  # the documented blind spot, reproduced
    text = " ".join(result.interpretation)
    assert "honest baseline" in text and "Ablations" in text


def test_the_experiment_is_deterministic_and_states_its_assumptions():
    a, b = sampler_exp.run(seeds=(0, 1)), sampler_exp.run(seeds=(0, 1))
    assert a.measured == b.measured
    assert any("assumption" in line for line in a.limitations)


# ---- noise experiment ----------------------------------------------------------------------------
def observation(name="loiter_midnight", frame_id=1):
    sc = scenario(name)
    frame = next(f for f in sc.frames if f.frame_id == frame_id)
    return ScriptedProvider(sc), FrameInput(
        frame_id=frame_id, timestamp=sc.clock.at(frame.timestamp_offset)
    )


def test_zero_noise_changes_nothing_for_every_kind():
    provider, frame = observation()
    clean = provider.analyze(frame)
    for kind in noise_exp.KINDS:
        assert NoisyProvider(provider, kind, 0.0, random.Random(1)).analyze(frame) == clean


def test_miss_drops_objects_and_only_objects():
    provider, frame = observation()
    noisy = NoisyProvider(provider, "miss", 1.0, random.Random(1)).analyze(frame)
    assert noisy.detected_objects == [] and noisy.description == provider.analyze(frame).description


def test_confuse_changes_one_attribute_but_never_the_class():
    provider, frame = observation()
    clean = provider.analyze(frame)
    noisy = NoisyProvider(provider, "confuse", 1.0, random.Random(1)).analyze(frame)
    assert noisy.detected_objects[0].object_type is ObjectType.PERSON
    assert noisy.detected_objects[0].attributes != clean.detected_objects[0].attributes
    assert noisy.detected_objects[0].confidence == clean.detected_objects[0].confidence


def test_fail_raises_and_the_perception_agent_turns_it_into_a_dropped_frame():
    provider, frame = observation()
    with pytest.raises(RuntimeError):
        NoisyProvider(provider, "fail", 1.0, random.Random(1)).analyze(frame)
    agent = noise_exp.perturbed_perception(scenario("loiter_midnight"), "fail", 1.0, 0)
    assert agent.perceive(frame).observation is None


def test_noise_is_reproducible_per_scenario_kind_rate_and_seed():
    sc = scenario("loiter_midnight")
    frame = FrameInput(frame_id=1, timestamp=sc.clock.at(60))
    first = noise_exp.perturbed_perception(sc, "confuse", 0.5, 7).perceive(frame).observation
    again = noise_exp.perturbed_perception(sc, "confuse", 0.5, 7).perceive(frame).observation
    assert first == again


def test_a_small_sweep_shows_scores_falling_with_perception_noise():
    scenarios = [scenario("loiter_midnight"), scenario("flicker_noise"), scenario("repeat_pickup")]
    result = noise_exp.run(scenarios, seeds=2, rates=(0.0, 0.3), kinds=("miss", "confuse", "fail"))
    series = result.measured["series"]
    for kind, cells in series.items():
        assert cells[0]["f1"] == 1.0 and cells[0]["exact_rate"] == 1.0  # the ceiling
        assert cells[-1]["exact_rate"] < 1.0, kind  # noise breaks exact agreement
    confuse = series["confuse"][-1]
    assert confuse["f1"] < 1.0 and confuse["fp"] > 0  # inconsistent attributes raise false alerts
    assert series["fail"][-1]["dropped_frames"] > 0
    assert any("ceiling" in line for line in result.interpretation)
    assert any("SIMULATED" in line for line in result.limitations)
