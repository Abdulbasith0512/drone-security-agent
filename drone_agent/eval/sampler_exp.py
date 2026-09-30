"""Experiment 8 (RQ8): adaptive VLM sampling versus a pixel gate, at what cost and with what misses.

A pixel gate saves model calls but is blind to small changes (the earlier gate experiment: 5 of 40
frames sent, 2 of 4 meaningful changes seen). The obvious cheap fix is "also refresh every N
frames". The adaptive sampler claims to do better by also using telemetry and event state. This
experiment tests that claim against that baseline rather than against a straw man.

Setup. Generated streams of candidate frames (one every 2 s) with a known schedule of changes:
objects of three sizes appear and leave (meaningful: 20.8%, 2.8% and 0.44% of the frame),
illumination drifts (irrelevant), and a fraction of the meaningful changes coincide with the
drone repositioning (a telemetry cue; the fraction is an ASSUMPTION stated below). The pixel signal
is the real FrameGate difference against the last frame sent to the model. The event-state signal
is what the last model call actually saw (the system cannot know about objects it never looked at).

Score. A meaningful change is DETECTED if some frame within K frames after it (and while it still
holds) was sent to the model. Reported together, never separately: model calls (and % avoided),
detection rate, mean lag, and calls per detected change.

Every configuration is fixed before looking at results; nothing is tuned to this fixture.
"""

import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np

from drone_agent.agents.sampling import AdaptiveSampler, SamplerConfig
from drone_agent.eval.base import CONTROLLED_QUALIFICATION, ExperimentResult, fmt
from drone_agent.models import Telemetry, Zone
from drone_agent.telemetry.geo import east_offset_degrees
from drone_agent.video.gate import FrameGate, GateConfig

IST = timezone(timedelta(hours=5, minutes=30))
T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)

WIDTH, HEIGHT = 160, 90
FRAMES = 600  # 20 minutes at one candidate frame every 2 s
FRAME_SECONDS = 2
DETECT_WINDOW = 5  # frames (10 s) within which a change must be seen to count as detected
NOISE_SIGMA = 1.5
P_CUE = 0.3  # ASSUMPTION: share of meaningful changes accompanied by a drone reposition
SEEDS = tuple(range(8))
SIZES = {"large": (60, 50), "medium": (20, 20), "small": (8, 8)}  # 20.8%, 2.8%, 0.44% of the frame


@dataclass
class Change:
    frame: int
    kind: str  # "appear" | "leave" | "illumination"
    meaningful: bool
    size: str = ""
    cue: bool = False  # the drone repositioned at this frame


@dataclass
class Stream:
    telemetry: list[Telemetry]
    changes: list[Change]
    present: list[int]  # objects actually in the scene at each frame (ground truth)
    # gate-sized greyscale copy of every frame; the full frames are dropped to save memory
    small: list[np.ndarray]
    images: list[np.ndarray] = field(default_factory=list)  # only kept when keep_images=True


def _background() -> np.ndarray:
    x = np.linspace(90, 130, WIDTH, dtype=np.float32)[None, :].repeat(HEIGHT, axis=0)
    texture = ((np.arange(HEIGHT)[:, None] // 6 + np.arange(WIDTH)[None, :] // 6) % 2) * 6
    return x + texture.astype(np.float32)


_BASE = _background()


@lru_cache(maxsize=16)
def generate(seed: int, keep_images: bool = False) -> Stream:
    """A deterministic stream. Only gate-sized copies are kept (full frames are ~3 MB each)."""
    rng = random.Random(seed)
    visits: list[tuple[int, int, str, tuple[int, int]]] = []
    frame = rng.randint(15, 40)
    while frame < FRAMES - 60:
        length = rng.randint(25, 70)
        size = rng.choice(["large", "medium", "small", "large", "medium", "small"])
        box = (rng.randint(5, WIDTH - 70), rng.randint(5, HEIGHT - 55))
        visits.append((frame, min(FRAMES - 1, frame + length), size, box))
        frame += length + rng.randint(30, 80)
    changes: list[Change] = []
    for start, end, size, _ in visits:
        changes.append(Change(start, "appear", True, size, rng.random() < P_CUE))
        changes.append(Change(end, "leave", True, size, rng.random() < P_CUE))
    for _ in range(len(visits)):
        changes.append(Change(rng.randint(5, FRAMES - 5), "illumination", False))
    changes.sort(key=lambda c: c.frame)
    brightness_steps = {
        c.frame: rng.choice([-25, -12, 8, 12, 25]) for c in changes if not c.meaningful
    }

    offset_by_frame: dict[int, int] = {}
    offset = 0
    for c in changes:
        if c.cue:
            offset += 22 if (len(offset_by_frame) % 2 == 0) else -22
            offset_by_frame[c.frame] = offset
    images, telemetry, present = [], [], []
    reference_gate = FrameGate(GateConfig())
    small: list[np.ndarray] = []
    brightness, east, live = 0, 0, {}
    for i in range(FRAMES):
        brightness += brightness_steps.get(i, 0)
        east = offset_by_frame.get(i, east)
        for start, end, size, box in visits:
            if i == start:
                live[(start, size)] = (box, SIZES[size])
            if i == end:
                live.pop((start, size), None)
        gray = _BASE + brightness
        for (x, y), (w, h) in live.values():
            gray[y : y + h, x : x + w] = 20
        gray = gray + np.random.default_rng(seed * 10_000 + i).normal(0, NOISE_SIGMA, gray.shape)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        frame_bgr = np.repeat(gray[:, :, None], 3, axis=2)
        small.append(reference_gate._small(frame_bgr))
        if keep_images:
            images.append(frame_bgr)
        telemetry.append(
            Telemetry(
                timestamp=T0 + timedelta(seconds=i * FRAME_SECONDS),
                latitude=0.0004,
                longitude=0.00045 + east_offset_degrees(float(east), 0.0004),
                altitude_m=30.0,
                zone=Zone.MAIN_GATE,
                battery_percent=90.0,
                signal_percent=95.0,
            )
        )
        present.append(len(live))
    return Stream(telemetry, changes, present, small, images)


class PixelSignal:
    """The FrameGate metric (mean absolute difference of the gate-sized greyscale copies) over a
    stream whose copies were computed once. Equal to FrameGate.score, checked in the tests."""

    def __init__(self, stream: Stream) -> None:
        self._small = stream.small
        self._reference: np.ndarray | None = None

    def score(self, index: int) -> float | None:
        if self._reference is None:
            return None
        return round(float(np.abs(self._small[index] - self._reference).mean()), 4)

    def set_reference(self, index: int) -> None:
        self._reference = self._small[index]


# ---- policies: decide(frame index) -> analyse?; each returns the sorted list of analysed frames ------
def _pixel_only(stream: Stream, threshold: float, refresh: int | None = None) -> list[int]:
    gate = PixelSignal(stream)
    sent: list[int] = []
    for i in range(FRAMES):
        score = gate.score(i)
        due = refresh is not None and sent and i - sent[-1] >= refresh
        if score is None or score >= threshold or due:
            gate.set_reference(i)
            sent.append(i)
    return sent


def _adaptive(stream: Stream, config: SamplerConfig, use_telemetry: bool = True) -> list[int]:
    gate = PixelSignal(stream)
    sampler = AdaptiveSampler(config)
    sent: list[int] = []
    observed_open = 0  # what the last model call saw: the system knows nothing else
    for i in range(FRAMES):
        decision = sampler.decide(
            now=stream.telemetry[i].timestamp,
            zone=Zone.MAIN_GATE,
            scene_score=gate.score(i),
            telemetry=stream.telemetry[i] if use_telemetry else None,
            open_events=observed_open,
        )
        if decision.analyze:
            gate.set_reference(i)
            sent.append(i)
            observed_open = stream.present[i]
    return sent


def score_policy(stream: Stream, sent: list[int]) -> dict:
    sent_set = set(sent)
    meaningful = [c for c in stream.changes if c.meaningful]
    lags, missed_sizes = [], []
    for c in meaningful:
        hit = next(
            (f for f in range(c.frame, min(FRAMES, c.frame + DETECT_WINDOW + 1)) if f in sent_set),
            None,
        )
        if hit is None:
            missed_sizes.append(c.size)
        else:
            lags.append(hit - c.frame)
    detected = len(meaningful) - len(missed_sizes)
    return {
        "calls": len(sent),
        "avoided_pct": round(100 * (1 - len(sent) / FRAMES), 1),
        "meaningful": len(meaningful),
        "detected": detected,
        "recall": round(detected / len(meaningful), 4) if meaningful else None,
        "mean_lag_frames": round(statistics.fmean(lags), 2) if lags else None,
        "calls_per_detected": round(len(sent) / detected, 2) if detected else None,
        "missed_small": sum(1 for s in missed_sizes if s == "small"),
        "missed_medium": sum(1 for s in missed_sizes if s == "medium"),
        "missed_large": sum(1 for s in missed_sizes if s == "large"),
    }


# Fixed before looking at results. Durations are simulated seconds (frames are 2 s apart).
ADAPTIVE = SamplerConfig(scene_threshold=6.0, watch_interval_s=20.0, max_skip_seconds=120.0)
POLICIES: dict[str, tuple[str, callable]] = {
    "every frame (gate off)": ("send everything", lambda s: list(range(FRAMES))),
    "pixel gate, threshold 6 (previous default)": ("pixel only", lambda s: _pixel_only(s, 6.0)),
    "pixel gate, threshold 3": ("pixel only, more sensitive", lambda s: _pixel_only(s, 3.0)),
    "pixel gate + refresh every 15 frames": ("pixel + periodic", lambda s: _pixel_only(s, 6.0, 15)),
    "pixel gate + refresh every 30 frames": ("pixel + periodic", lambda s: _pixel_only(s, 6.0, 30)),
    "adaptive (pixel + telemetry + event watch + refresh)": (
        "adaptive",
        lambda s: _adaptive(s, ADAPTIVE),
    ),
    "adaptive without telemetry signal": (
        "ablation",
        lambda s: _adaptive(s, ADAPTIVE, use_telemetry=False),
    ),
    "adaptive without event watch": (
        "ablation",
        lambda s: _adaptive(s, ADAPTIVE.model_copy(update={"watch_interval_s": 10_000.0})),
    ),
    "adaptive without forced refresh": (
        "ablation",
        lambda s: _adaptive(s, ADAPTIVE.model_copy(update={"max_skip_seconds": 10_000.0})),
    ),
}


@dataclass
class _Acc:
    rows: list[dict] = field(default_factory=list)

    def mean(self, key: str) -> float | None:
        values = [r[key] for r in self.rows if r[key] is not None]
        return round(statistics.fmean(values), 3) if values else None


def run(seeds: tuple[int, ...] = SEEDS) -> ExperimentResult:
    streams = [generate(seed) for seed in seeds]
    results: dict[str, dict] = {}
    for name, (family, policy) in POLICIES.items():
        acc = _Acc([score_policy(stream, policy(stream)) for stream in streams])
        total_meaningful = sum(r["meaningful"] for r in acc.rows)
        detected = sum(r["detected"] for r in acc.rows)
        calls = sum(r["calls"] for r in acc.rows)
        results[name] = {
            "family": family,
            "calls_mean": acc.mean("calls"),
            "avoided_pct_mean": acc.mean("avoided_pct"),
            "recall": round(detected / total_meaningful, 4),
            "mean_lag_frames": acc.mean("mean_lag_frames"),
            "calls_per_detected": round(calls / detected, 2) if detected else None,
            "missed_small": sum(r["missed_small"] for r in acc.rows),
            "missed_medium": sum(r["missed_medium"] for r in acc.rows),
            "missed_large": sum(r["missed_large"] for r in acc.rows),
            "meaningful_changes": total_meaningful,
        }
    adaptive = results["adaptive (pixel + telemetry + event watch + refresh)"]
    pixel = results["pixel gate, threshold 6 (previous default)"]
    periodic = results["pixel gate + refresh every 15 frames"]
    rows = [{"experiment": "sampler", "policy": name, **r} for name, r in results.items()]
    interpretation = [
        f"On {len(seeds)} generated {FRAMES}-frame streams ({adaptive['meaningful_changes']} "
        f"meaningful changes in total): the previous pixel-only default sent "
        f"{fmt(pixel['calls_mean'], 1)} frames per stream ({fmt(pixel['avoided_pct_mean'], 1)}% "
        f"avoided) and detected {fmt(pixel['recall'])} of the meaningful changes within "
        f"{DETECT_WINDOW} frames; it missed {pixel['missed_small']} small, "
        f"{pixel['missed_medium']} medium and {pixel['missed_large']} large ones.",
        f"The adaptive sampler sent {fmt(adaptive['calls_mean'], 1)} frames "
        f"({fmt(adaptive['avoided_pct_mean'], 1)}% avoided) and detected "
        f"{fmt(adaptive['recall'])}, mean lag {fmt(adaptive['mean_lag_frames'], 2)} frames, "
        f"{fmt(adaptive['calls_per_detected'], 2)} calls per detected change.",
        f"The honest baseline is pixel gate plus a fixed refresh: every 15 frames it sent "
        f"{fmt(periodic['calls_mean'], 1)} frames and detected {fmt(periodic['recall'])} "
        f"({fmt(periodic['calls_per_detected'], 2)} calls per detected change).",
        "Ablations remove one adaptive signal at a time (telemetry, event watch, forced refresh): "
        + "; ".join(
            f"{name.replace('adaptive ', '')} {fmt(r['recall'])} recall at "
            f"{fmt(r['calls_mean'], 1)} calls"
            for name, r in results.items()
            if r["family"] == "ablation"
        )
        + ".",
    ]
    return ExperimentResult(
        key="sampler",
        title="Adaptive sampling versus a pixel gate: calls saved AND changes missed",
        research_question="RQ8: does combining pixel change, telemetry change, event state and a "
        "staleness bound detect more meaningful changes per model call than a pixel gate, "
        "including a pixel gate with a fixed refresh?",
        definition=(
            f"{len(seeds)} seeded {FRAMES}-frame streams (frames 2 s apart, {WIDTH}x{HEIGHT}, "
            f"noise sigma {NOISE_SIGMA}) with objects of three sizes appearing and leaving, "
            f"illumination steps, and a {int(P_CUE * 100)}% telemetry-cue rate; the real FrameGate "
            f"pixel score, the real AdaptiveSampler; a change counts as detected if a frame within "
            f"{DETECT_WINDOW} frames was sent. Configurations fixed in advance: adaptive = "
            f"watch {ADAPTIVE.watch_interval_s:g} s, forced refresh {ADAPTIVE.max_skip_seconds:g} s."
        ),
        measured={
            "policies": results,
            "streams": len(seeds),
            "frames_per_stream": FRAMES,
            "detect_window_frames": DETECT_WINDOW,
            "p_cue": P_CUE,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            CONTROLLED_QUALIFICATION,
            f"The share of meaningful changes that come with a telemetry cue ({int(P_CUE * 100)}%) is "
            "an assumption; with none, the telemetry signal contributes nothing (see the ablation).",
            "Generated scenes are flat backgrounds with dark rectangles; real scenes have far richer "
            "pixel statistics, so absolute recall and call counts will differ.",
            "Detection means a model call was made in time, not that the model would have described "
            "the change correctly (that is perception accuracy, which is not measured here).",
        ],
    )
