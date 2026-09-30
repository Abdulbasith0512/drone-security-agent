"""Experiment 9 (RQ9): how do precision, recall and latency degrade when perception is imperfect?

The ground-truth scenarios use perfect scripted detections, so every score there is 1.0 by
construction. That says the pipeline logic matches its author's intent; it does not say how the
system behaves when the vision model is wrong. This experiment perturbs the perception boundary
(the observations the perception agent returns) at increasing rates and re-scores every scenario:

* `miss`     each detected object is dropped with probability p (the model does not see it);
* `confuse`  with probability p an object's key attribute (clothing, colour, body type) is
             replaced by a different plausible value (the model describes it inconsistently: the
             pattern seen in the real Gemini run, where 20 of 25 events were single-frame);
* `fail`     with probability p the whole frame's perception fails (timeout / malformed output) and
             the frame is dropped by the perception agent's failure isolation.

Each cell is the pooled result over all scenarios with ground truth and several seeds. Perturbation
is seeded per (scenario, kind, rate, seed), so the sweep is reproducible. No model is called.

What it shows and does not: the shape of the degradation of THIS rule set on THESE scenarios under
SIMULATED error. It is not a measurement of any real vision model's error rates.
"""

import random
import statistics
from dataclasses import dataclass

from drone_agent.agents.perception import PerceptionAgent
from drone_agent.eval.base import CONTROLLED_QUALIFICATION, ExperimentResult, fmt
from drone_agent.eval.ground_truth import (
    ScenarioScore,
    aggregate,
    run_scenario_scored,
    scenario_paths,
)
from drone_agent.models import DetectedObject, Observation
from drone_agent.scenario import Scenario, load_scenario
from drone_agent.vision import FrameInput, ScriptedProvider

KINDS = ("miss", "confuse", "fail")
RATES = (0.0, 0.05, 0.1, 0.2, 0.3)
DEFAULT_SEEDS = 3

_VOCABULARY = {
    "clothing": [
        "dark jacket", "dark coat", "black hoodie", "grey coat", "dark clothing",
        "blue polo", "hi-vis vest", "dark hoodie",
    ],
    "color": ["blue", "red", "white", "black", "grey", "green", "silver"],
    "body_type": ["pickup", "sedan", "van", "hatchback", "suv", "truck"],
}  # fmt: skip


class NoisyProvider:
    """A scripted provider whose answers are deliberately imperfect."""

    def __init__(self, inner: ScriptedProvider, kind: str, rate: float, rng: random.Random) -> None:
        self.inner, self.kind, self.rate, self.rng = inner, kind, rate, rng
        self.name = f"scripted+{kind}@{rate:g}"

    def analyze(self, frame: FrameInput) -> Observation:
        if self.kind == "fail" and self.rng.random() < self.rate:
            raise RuntimeError("simulated perception failure")
        obs = self.inner.analyze(frame)
        objects: list[DetectedObject] = []
        for obj in obs.detected_objects:
            if self.kind == "miss" and self.rng.random() < self.rate:
                continue
            if self.kind == "confuse" and self.rng.random() < self.rate:
                obj = self._confuse(obj)
            objects.append(obj)
        return obs.model_copy(update={"detected_objects": objects})

    def _confuse(self, obj: DetectedObject) -> DetectedObject:
        keys = [k for k in _VOCABULARY if k in obj.attributes]
        if not keys:
            return obj
        key = self.rng.choice(keys)
        choices = [v for v in _VOCABULARY[key] if v != obj.attributes[key]]
        return obj.model_copy(
            update={"attributes": {**obj.attributes, key: self.rng.choice(choices)}}
        )


def perturbed_perception(scenario: Scenario, kind: str, rate: float, seed: int) -> PerceptionAgent:
    rng = random.Random(f"{scenario.name}:{kind}:{rate}:{seed}")
    return PerceptionAgent([NoisyProvider(ScriptedProvider(scenario), kind, rate, rng)])


@dataclass
class Cell:
    kind: str
    rate: float
    scores: list[ScenarioScore]
    dropped_frames: int = 0

    def summary(self) -> dict:
        agg = aggregate(self.scores)
        n = len(self.scores)
        return {
            "kind": self.kind,
            "rate": self.rate,
            "runs": n,
            "precision": agg["precision"],
            "recall": agg["recall"],
            "f1": agg["f1"],
            "tp": agg["tp"],
            "fp": agg["fp"],
            "fn": agg["fn"],
            "exact_rate": round(agg["exact_scenarios"] / n, 4) if n else None,
            "severity_path_accuracy": agg["severity_path_accuracy"],
            "latency_mean_s": agg["detection_latency_s"]["mean"],
            "dropped_frames": self.dropped_frames,
        }


def run_cell(kind: str, rate: float, scenarios: list[Scenario], seeds: int) -> Cell:
    scores: list[ScenarioScore] = []
    dropped = 0
    for scenario in scenarios:
        for seed in range(1 if rate == 0 else seeds):  # no noise: every seed is identical
            perception = perturbed_perception(scenario, kind, rate, seed)
            score, summary = run_scenario_scored(scenario, perception=perception)
            scores.append(score)
            dropped += summary.dropped
    return Cell(kind, rate, scores, dropped)


def run(
    scenarios: list[Scenario] | None = None,
    seeds: int = DEFAULT_SEEDS,
    rates: tuple[float, ...] = RATES,
    kinds: tuple[str, ...] = KINDS,
) -> ExperimentResult:
    scenarios = scenarios or [load_scenario(p) for p in scenario_paths()]
    baseline = run_cell("none", 0.0, scenarios, 1).summary()
    series: dict[str, list[dict]] = {}
    for kind in kinds:
        series[kind] = [baseline | {"kind": kind}] + [
            run_cell(kind, rate, scenarios, seeds).summary() for rate in rates if rate > 0
        ]
    rows = [{"experiment": "noise", **cell} for cells in series.values() for cell in cells]
    worst = {
        kind: min(cells, key=lambda c: c["f1"] if c["f1"] is not None else 1.0)
        for kind, cells in series.items()
    }
    interpretation = [
        f"With perfect perception the {len(scenarios)} scenarios score precision "
        f"{fmt(baseline['precision'])}, recall {fmt(baseline['recall'])}, F1 {fmt(baseline['f1'])} "
        f"({baseline['tp']} TP, {baseline['fp']} FP, {baseline['fn']} FN); that is the ceiling, and "
        "it is 1.0 by construction.",
    ]
    for kind, cells in series.items():
        top = cells[-1]
        interpretation.append(
            f"{kind}: at rate {top['rate']:g} F1 is {fmt(top['f1'])} (precision "
            f"{fmt(top['precision'])}, recall {fmt(top['recall'])}, "
            f"{top['fp']} FP, {top['fn']} FN); {fmt(top['exact_rate'])} of runs still match their "
            f"ground truth exactly; lowest F1 in the sweep {fmt(worst[kind]['f1'])} at rate "
            f"{worst[kind]['rate']:g}."
        )
    interpretation.append(
        "Each point pools every scenario and "
        f"{seeds} seed(s); miss = objects dropped, confuse = attributes described inconsistently, "
        "fail = whole frames lost to a perception failure."
    )
    return ExperimentResult(
        key="noise",
        title="Perception-noise sweep: how precision, recall and F1 degrade",
        research_question="RQ9: when perception is imperfect, which failure mode hurts the "
        "deterministic pipeline most, and how gracefully?",
        definition=(
            f"{len(scenarios)} ground-truth scenarios x rates {list(rates)} x {seeds} seeds x "
            f"kinds {list(kinds)}; the scripted provider's answers are perturbed at the perception "
            "boundary (drop objects / swap attributes / fail frames) and every run is re-scored "
            "against the same author-written ground truth."
        ),
        measured={"series": series, "scenarios": [s.name for s in scenarios], "seeds": seeds},
        rows=rows,
        interpretation=interpretation,
        limitations=[
            CONTROLLED_QUALIFICATION,
            "The error model is SIMULATED and i.i.d. per object or frame; real vision errors are "
            "correlated (occlusion, lighting, small distant objects) and structured differently.",
            "Scenarios are short and few; a handful of alerts per cell means individual cells are "
            "noisy, which is why several seeds are pooled.",
            "These numbers describe this rule set's sensitivity to perception error, not any "
            "real model's accuracy.",
        ],
    )


__all__ = ["KINDS", "RATES", "NoisyProvider", "perturbed_perception", "run", "run_cell"]

_ = statistics  # kept available for callers extending the summaries
