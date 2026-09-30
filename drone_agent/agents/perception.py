"""Perception agent: turn a frame into a validated Observation, whatever goes wrong.

The agent owns the *calling policy* around one or more vision providers (scripted, cassette,
a real model); it decides nothing about security. Its job:

* try providers in order (a fallback chain), each with a wall-clock timeout, and record every
  attempt, so a dead API, a malformed answer or a hung call costs one frame, not the whole run;
* guard the result: an object class the model invented is coerced to `other` (opt-in, see
  the vision contract), duplicate object ids are collapsed, the object count and text lengths
  are bounded, and nothing is ever added that the provider did not report;
* return a per-frame trace (provider used, attempts, latency, guard actions) that the run
  records, so "what did the vision model see and why did we trust it" is answerable later.

It never invents an observation for a failed frame: the caller gets `observation=None` and a trace
that says why, and decides (the streaming engine drops the frame and counts it).
"""

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Literal

from pydantic import Field

from drone_agent.models import DomainModel, Observation
from drone_agent.vision.base import FrameInput, VisionProvider


class GuardConfig(DomainModel):
    max_objects: int = Field(default=25, ge=1)
    min_confidence: float = Field(default=0.0, ge=0, le=1)
    max_description_chars: int = Field(default=500, ge=20)
    drop_duplicate_object_ids: bool = True


@dataclass(frozen=True)
class PerceptionAttempt:
    provider: str
    ok: bool
    latency_s: float
    error: str | None = None  # class and short message, never a key or a prompt


@dataclass
class PerceptionTrace:
    frame_id: int
    status: Literal["ok", "fallback", "failed"]
    provider: str | None = None  # the provider whose answer was used
    attempts: list[PerceptionAttempt] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)  # guard actions applied to the answer
    latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "status": self.status,
            "provider": self.provider,
            "latency_s": round(self.latency_s, 4),
            "flags": list(self.flags),
            "attempts": [
                {
                    "provider": a.provider,
                    "ok": a.ok,
                    "latency_s": round(a.latency_s, 4),
                    **({"error": a.error} if a.error else {}),
                }
                for a in self.attempts
            ],
        }


@dataclass
class PerceptionResult:
    observation: Observation | None
    trace: PerceptionTrace


@dataclass
class PerceptionStats:
    frames: int = 0
    ok: int = 0
    fallbacks: int = 0  # answered by a provider other than the first
    failed: int = 0  # no provider produced an answer
    timeouts: int = 0
    guard_actions: int = 0
    total_latency_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "ok": self.ok,
            "fallbacks": self.fallbacks,
            "failed": self.failed,
            "timeouts": self.timeouts,
            "guard_actions": self.guard_actions,
            "total_latency_s": round(self.total_latency_s, 4),
        }


def _short(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ")
    return f"{type(exc).__name__}: {text[:160]}"


class PerceptionAgent:
    def __init__(
        self,
        providers: Sequence[VisionProvider],
        *,
        guard: GuardConfig | None = None,
        call_timeout_s: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not providers:
            raise ValueError("a perception agent needs at least one provider")
        self.providers = list(providers)
        self.guard = guard or GuardConfig()
        self.call_timeout_s = call_timeout_s
        self._clock = clock
        self.stats = PerceptionStats()
        self._pool: ThreadPoolExecutor | None = None

    @property
    def name(self) -> str:
        return "+".join(p.name for p in self.providers)

    # ---- one frame -------------------------------------------------------------------------
    def perceive(self, frame: FrameInput) -> PerceptionResult:
        started = self._clock()
        trace = PerceptionTrace(frame_id=frame.frame_id, status="failed")
        observation: Observation | None = None
        for index, provider in enumerate(self.providers):
            began = self._clock()
            try:
                observation = self._call(provider, frame)
            except Exception as exc:  # noqa: BLE001 - the boundary: record it, try the next one
                if isinstance(exc, FutureTimeout):
                    self.stats.timeouts += 1
                    exc = TimeoutError(f"no answer within {self.call_timeout_s}s")
                trace.attempts.append(
                    PerceptionAttempt(provider.name, False, self._clock() - began, _short(exc))
                )
                continue
            trace.attempts.append(PerceptionAttempt(provider.name, True, self._clock() - began))
            trace.provider = provider.name
            trace.status = "ok" if index == 0 else "fallback"
            break
        if observation is not None:
            observation = self._guard(observation, trace)
        trace.latency_s = self._clock() - started
        self._count(trace)
        return PerceptionResult(observation, trace)

    def _call(self, provider: VisionProvider, frame: FrameInput) -> Observation:
        if self.call_timeout_s is None:
            return provider.analyze(frame)
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perception")
        future = self._pool.submit(provider.analyze, frame)
        try:
            return future.result(timeout=self.call_timeout_s)
        except FutureTimeout:
            future.cancel()  # a call already running cannot be stopped; its answer is discarded
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # ---- guard -----------------------------------------------------------------------------
    def _guard(self, obs: Observation, trace: PerceptionTrace) -> Observation:
        cfg = self.guard
        objects = list(obs.detected_objects)
        for o in objects:
            reported = o.attributes.get("reported_class")
            if reported:
                trace.flags.append(f"coerced_class:{reported}")
        kept = [o for o in objects if o.confidence >= cfg.min_confidence]
        if len(kept) != len(objects):
            trace.flags.append(f"dropped_low_confidence:{len(objects) - len(kept)}")
        if cfg.drop_duplicate_object_ids:
            seen: set[str] = set()
            unique = []
            for o in kept:
                if o.object_id is not None:
                    if o.object_id in seen:
                        continue
                    seen.add(o.object_id)
                unique.append(o)
            if len(unique) != len(kept):
                trace.flags.append(f"dropped_duplicate_ids:{len(kept) - len(unique)}")
            kept = unique
        if len(kept) > cfg.max_objects:
            trace.flags.append(f"capped_objects:{len(kept) - cfg.max_objects}")
            kept = kept[: cfg.max_objects]
        description = obs.description
        if len(description) > cfg.max_description_chars:
            trace.flags.append("truncated_description")
            description = description[: cfg.max_description_chars - 3] + "..."
        if not trace.flags:
            return obs
        return obs.model_copy(update={"detected_objects": kept, "description": description})

    def _count(self, trace: PerceptionTrace) -> None:
        s = self.stats
        s.frames += 1
        s.total_latency_s += trace.latency_s
        s.guard_actions += len(trace.flags)
        if trace.status == "ok":
            s.ok += 1
        elif trace.status == "fallback":
            s.fallbacks += 1
        else:
            s.failed += 1
