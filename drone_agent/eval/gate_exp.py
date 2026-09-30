"""Experiment 1 (RQ1): frame-change gate ablation on deterministic synthetic frames.

The gate benchmark measures whether a sampled frame WOULD require a vision-model call; no model
is called. Ground truth is exact because the fixture is generated: every transition has a known
index, and each is labelled `meaningful` (a security-relevant scene change) or not (illumination
only). The gate is pixel-based, so the fixture also shows where pixel change and semantic
importance disagree.
"""

from dataclasses import dataclass

import numpy as np

from drone_agent.eval.base import ExperimentResult, fmt, pct
from drone_agent.video.gate import FrameGate, GateConfig

WIDTH, HEIGHT = 320, 180
CANDIDATES = 40  # sampled frames (e.g. one every 2 s)
NOISE_SIGMA = 1.5  # deterministic sensor noise (seeded per frame)
DEFAULT_THRESHOLD = GateConfig().threshold
THRESHOLDS = (0.0, 0.5, 1.0, 3.0, 4.0, DEFAULT_THRESHOLD, 12.0, 24.0)


@dataclass(frozen=True)
class Transition:
    index: int
    name: str
    kind: str  # "object" | "illumination"
    meaningful: bool
    # object: (x, y, w, h, appears) where appears False = the object leaves; illumination: delta
    box: tuple[int, int, int, int] | None = None
    appears: bool = True
    brightness_delta: int = 0
    object_key: str = ""


# Known ground truth. Sizes are fractions of the 320x180 frame: large 20.8%, medium 2.8%,
# small 0.44%. The object is dark (value 20) on a mid-grey background.
TRANSITIONS = (
    Transition(5, "large object appears", "object", True, (100, 40, 120, 100), True, 0, "large"),
    Transition(12, "illumination step +25", "illumination", False, None, True, 25),
    Transition(18, "small object appears", "object", True, (250, 20, 16, 16), True, 0, "small"),
    Transition(24, "medium object appears", "object", True, (40, 120, 40, 40), True, 0, "medium"),
    Transition(30, "illumination step +8", "illumination", False, None, True, 8),
    Transition(36, "large object leaves", "object", True, (100, 40, 120, 100), False, 0, "large"),
)


def _background() -> np.ndarray:
    x = np.linspace(90, 130, WIDTH, dtype=np.float32)[None, :].repeat(HEIGHT, axis=0)
    texture = ((np.arange(HEIGHT)[:, None] // 12 + np.arange(WIDTH)[None, :] // 12) % 2) * 6
    return x + texture.astype(np.float32)


_BASE = _background()


def render(index: int) -> np.ndarray:
    """The BGR uint8 frame for candidate `index` (deterministic)."""
    brightness = sum(
        t.brightness_delta for t in TRANSITIONS if t.kind == "illumination" and t.index <= index
    )
    gray = _BASE + brightness
    present: dict[str, tuple[int, int, int, int]] = {}
    for t in TRANSITIONS:
        if t.kind == "object" and t.index <= index and t.box is not None:
            if t.appears:
                present[t.object_key] = t.box
            else:
                present.pop(t.object_key, None)
    for x, y, w, h in present.values():
        gray[y : y + h, x : x + w] = 20
    noise = np.random.default_rng(1000 + index).normal(0, NOISE_SIGMA, gray.shape)
    gray = np.clip(gray + noise, 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def frame_scores() -> list[float | None]:
    """Score of each frame against the PREVIOUS frame (a threshold-0 gate accepts everything, so
    its reference is always the previous frame). The distribution used to choose thresholds."""
    gate = FrameGate(GateConfig(threshold=0.0))
    return [gate.check(render(i)).score for i in range(CANDIDATES)]


def run_threshold(threshold: float) -> dict:
    gate = FrameGate(GateConfig(threshold=threshold))
    accepted = [gate.check(render(i)).accepted for i in range(CANDIDATES)]
    n_accepted = sum(accepted)
    meaningful = [t for t in TRANSITIONS if t.meaningful]
    detected = [t for t in meaningful if accepted[t.index]]
    irrelevant = [t for t in TRANSITIONS if not t.meaningful]
    return {
        "threshold": threshold,
        "candidates": CANDIDATES,
        "accepted": n_accepted,
        "skipped": CANDIDATES - n_accepted,
        "call_reduction_pct": pct(CANDIDATES - n_accepted, CANDIDATES),
        "transitions_total": len(TRANSITIONS),
        "meaningful_transitions": len(meaningful),
        "meaningful_detected": len(detected),
        "meaningful_missed": len(meaningful) - len(detected),
        "meaningful_recall": None if not meaningful else round(len(detected) / len(meaningful), 4),
        "missed_names": [t.name for t in meaningful if not accepted[t.index]],
        "irrelevant_transitions_accepted": sum(1 for t in irrelevant if accepted[t.index]),
        "irrelevant_transitions": len(irrelevant),
        "accepted_indices": [i for i, a in enumerate(accepted) if a],
    }


def run() -> ExperimentResult:
    scores = frame_scores()
    transition_index = {t.index for t in TRANSITIONS}
    static = [s for i, s in enumerate(scores) if s is not None and i not in transition_index]
    per_transition = [
        {
            "index": t.index,
            "name": t.name,
            "meaningful": t.meaningful,
            "score_vs_previous_frame": scores[t.index],
        }
        for t in TRANSITIONS
    ]
    sweeps = [run_threshold(th) for th in THRESHOLDS]
    default = next(s for s in sweeps if s["threshold"] == DEFAULT_THRESHOLD)
    off = next(s for s in sweeps if s["threshold"] == 0.0)
    by_name = {p["name"]: p["score_vs_previous_frame"] for p in per_transition}
    small = by_name["small object appears"]
    step8 = by_name["illumination step +8"]
    step25 = by_name["illumination step +25"]
    measured = {
        "candidates": CANDIDATES,
        "default_threshold": DEFAULT_THRESHOLD,
        "static_frame_score": {
            "max": round(max(static), 4),
            "median": round(float(np.median(static)), 4),
        },
        "transition_scores": per_transition,
        "sweep": sweeps,
        "baseline_gate_off_calls": off["accepted"],
        "default_calls": default["accepted"],
        "pixel_vs_semantic": {
            "small_meaningful_change_score": small,
            "illumination_only_scores": [step8, step25],
            "small_change_scores_below_irrelevant_illumination": small < step8,
        },
    }
    rows = [
        {
            "experiment": "gate",
            "threshold": s["threshold"],
            "accepted": s["accepted"],
            "skipped": s["skipped"],
            "call_reduction_pct": s["call_reduction_pct"],
            "meaningful_detected": s["meaningful_detected"],
            "meaningful_missed": s["meaningful_missed"],
            "meaningful_recall": s["meaningful_recall"],
            "irrelevant_accepted": s["irrelevant_transitions_accepted"],
        }
        for s in sweeps
    ]
    interpretation = [
        f"Threshold 0 sends all {off['accepted']} of {CANDIDATES} candidate frames to the vision "
        f"model; the default {DEFAULT_THRESHOLD:g} sends {default['accepted']} "
        f"({fmt(default['call_reduction_pct'], 1)}% fewer calls) while detecting "
        f"{default['meaningful_detected']} of {default['meaningful_transitions']} meaningful "
        f"transitions (recall {fmt(default['meaningful_recall'])}).",
        "Missed at the default threshold: "
        + (", ".join(default["missed_names"]) or "none")
        + ". A missed change is absorbed silently: the reference frame is not updated, so the "
        "vision model is never asked about it.",
        f"Pixel change and semantic importance are not equivalent: the small meaningful object "
        f"scores {fmt(small)} while an irrelevant illumination step scores {fmt(step8)} "
        f"(+8) and {fmt(step25)} (+25); "
        f"{default['irrelevant_transitions_accepted']} of {default['irrelevant_transitions']} "
        "illumination-only changes are accepted (wasted calls) at the default threshold.",
        f"Static-frame noise peaks at score {fmt(measured['static_frame_score']['max'])}, so "
        "thresholds above that separate 'no change' from any real change.",
    ]
    limitations = [
        "One synthetic 40-frame sequence with a flat background and dark rectangles; real scenes "
        "have far richer pixel statistics (shadows, foliage, weather, camera shake).",
        "MAD on a 64x36 greyscale copy is pixel-based, not semantic; a small person far from "
        "the camera can fall below any threshold that also suppresses illumination flicker.",
        "Thresholds were chosen from the observed score distribution (see transition_scores); "
        "the default 6.0 was NOT tuned to this fixture.",
        "Transition 'detected' means the transition frame itself was accepted.",
    ]
    return ExperimentResult(
        key="gate",
        title="Frame-gate threshold ablation",
        research_question="RQ1: how many vision-model calls does the gate avoid, and what "
        "meaningful transitions does it miss?",
        definition=(
            f"{CANDIDATES} generated {WIDTH}x{HEIGHT} candidate frames with seeded sensor noise "
            f"(sigma {NOISE_SIGMA}); 6 known transitions (4 meaningful object changes of size 20.8%, "
            "0.44%, 2.8% and 20.8% of the frame, 2 illumination-only steps). The real FrameGate "
            f"runs at thresholds {list(THRESHOLDS)}; no model is called."
        ),
        measured=measured,
        rows=rows,
        interpretation=interpretation,
        limitations=limitations,
    )
