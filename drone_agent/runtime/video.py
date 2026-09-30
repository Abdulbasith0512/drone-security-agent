"""Progressive video processing: a video file through the SAME downstream pipeline as a patrol.

    video frame -> sampler -> gate -> vision provider -> Observation
                -> persist -> events / entities -> rules -> alerts        (per frame, not per video)

`run_video_live` walks the lazy `VideoStream` and hands every step to the shared
`ObservationPipeline` (runtime/consumer.py), the very object the scripted patrol engine uses. There
is no video-specific event, entity or rule code: a video observation and a scripted one are
indistinguishable once they reach the consumer, except for their own `source` and `provenance`.

Because each step is committed before the next frame is decoded and described, observation N, the
event it extends and any alert it raises are in the store (and announced to the listener) while the
video is still being processed. The simulated telemetry of the timeline is fed in time order next to
the frames, so the drone-safety rules also fire as their data arrives.

The batch path (`run_video_ingest`, then `build-events` and `analyze`) is unchanged and reaches the
same final events, entities and alerts; the test-suite compares the two.

This is about WHEN things are computed, not about how good the descriptions are: a cassette or a
scripted provider exercises the mechanics; how accurate a real model's descriptions are is a
separate question that this module does not answer.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime

from drone_agent.agents.perception import PerceptionAgent
from drone_agent.models import Observation, Telemetry
from drone_agent.pipeline.config import EventConfig
from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.runtime.consumer import Listener, ObservationPipeline, emitter
from drone_agent.runtime.events import RunEventKind
from drone_agent.store import Store
from drone_agent.video.ingest import (
    FailurePolicy,
    PerceptionFailed,
    VideoIngestConfig,
    VideoIngestResult,
    VideoStep,
    VideoStream,
)
from drone_agent.video.source import VideoSource
from drone_agent.video.timeline import VideoTimeline
from drone_agent.vision.base import FrameInput, VisionProvider


@dataclass(frozen=True)
class LiveStep:
    """The state of the store right after ONE sampled frame was fully processed and committed."""

    frame_index: int
    offset_s: float
    action: str  # analyzed | carried | dropped | failed
    observations: int  # observations persisted so far
    events: int  # events known so far (open ones included)
    open_events: int
    alerts: int  # alerts persisted so far
    new_alerts: tuple[str, ...]  # alert ids first raised at this step
    error: str = ""  # for a failed frame: "<kind>: <class>: <short message>"


@dataclass
class LiveVideoResult:
    result: VideoIngestResult  # the same shape the batch path returns
    steps: list[LiveStep] = field(default_factory=list)
    pipeline: ObservationPipeline | None = None

    @property
    def failures(self) -> list[LiveStep]:
        """Frames whose perception failed: no observation was stored or invented for them."""
        return [s for s in self.steps if s.action == "failed"]

    @property
    def first_alert_step(self) -> LiveStep | None:
        """The earliest step at which any alert existed (None if the video raised none)."""
        return next((s for s in self.steps if s.alerts), None)


def run_video_live(
    source: VideoSource,
    provider: VisionProvider,
    timeline: VideoTimeline,
    store: Store,
    *,
    config: VideoIngestConfig | None = None,
    events: EventConfig | None = None,
    rules: SecurityRulesConfig | None = None,
    listener: Listener | None = None,
    on_step: Callable[[LiveStep], None] | None = None,
    on_failure: FailurePolicy = FailurePolicy.CONTINUE,
    call_timeout_s: float | None = None,
) -> LiveVideoResult:
    """Process `source` frame by frame, persisting and evaluating everything as it goes.

    Each sampled frame is one transaction: its telemetry due, its observation, the events and
    entities it changes and the alerts it raises are committed together before the next frame is
    read. `on_step` is called after each commit (the store can be queried from there).

    A provider failure does not stop the run by default: the frame becomes an explicit "failed"
    step (no observation is stored or invented for it, so it cannot raise or clear an alert), the
    telemetry keeps flowing through the safety rules, and the next frame is a fresh attempt. Pass
    `FailurePolicy.RAISE` to abort instead; what was committed before stays committed either way.

    `call_timeout_s` gives each provider call that long before the frame counts as a timeout
    failure (the perception agent's timeout: a call already running cannot be interrupted, its
    answer is discarded). None waits for the provider.
    """
    if call_timeout_s is not None and call_timeout_s <= 0:
        raise ValueError("call_timeout_s must be positive (or None to wait)")
    config = replace(config or VideoIngestConfig(), on_failure=on_failure)
    agent = analyzer = None
    if call_timeout_s is not None:
        agent = PerceptionAgent([provider], call_timeout_s=call_timeout_s)
        analyzer = _timeout_analyzer(agent)
    stream = VideoStream(source, provider, timeline, config, analyzer=analyzer)
    emit = emitter(listener)
    pipeline = ObservationPipeline(
        store,
        events=events,
        rules=rules,
        stale_after_s=timeline.telemetry.stale_after_s,
        emit=emit,
    )
    pending: deque[Telemetry] = deque(sorted(stream.telemetry, key=lambda t: t.timestamp))
    kept: list = []  # the stored observations (telemetry attached), for the result
    steps: list[LiveStep] = []
    last_at: datetime = timeline.start
    last_offset = 0.0

    def feed_telemetry(until: datetime | None) -> None:
        nonlocal last_at, last_offset
        while pending and (until is None or pending[0].timestamp <= until):
            sample = pending.popleft()
            offset = (sample.timestamp - timeline.start).total_seconds()
            pipeline.record_telemetry(sample)
            pipeline.assess(
                sample.timestamp, offset, camera=False
            )  # a sample cannot change a camera rule
            last_at, last_offset = max(last_at, sample.timestamp), max(last_offset, offset)

    def process(step: VideoStep) -> LiveStep:
        nonlocal last_at, last_offset
        known_alerts = {a.alert_id for a in pipeline.alerts}
        with store.transaction():
            feed_telemetry(step.timestamp)
            cand = step.candidate
            emit(
                RunEventKind.FRAME,
                step.timestamp,
                step.offset_seconds,
                frame_id=step.frame_id,
                zone=cand.zone,
                analyzed=cand.action == "analyzed",
                reason=cand.gate_reason,
                detail=step.error
                or ("" if cand.gate_score is None else f"gate score {cand.gate_score:.2f}"),
                dropped=cand.action in ("dropped", "failed"),
            )
            if step.observation is not None:
                kept.append(pipeline.consume(step.observation, step.timestamp, step.offset_seconds))
        last_at, last_offset = max(last_at, step.timestamp), max(last_offset, step.offset_seconds)
        return LiveStep(
            frame_index=step.candidate.frame_index,
            offset_s=step.offset_seconds,
            action=step.candidate.action,
            observations=pipeline.n_observations,
            events=len(pipeline.events),
            open_events=sum(1 for e in pipeline.events if e.is_open),
            alerts=len(pipeline.alerts),
            new_alerts=tuple(a.alert_id for a in pipeline.alerts if a.alert_id not in known_alerts),
            error=step.error,
        )

    try:
        for step in stream:
            live = process(step)
            steps.append(live)
            if on_step is not None:
                on_step(live)
    finally:
        if agent is not None:
            agent.close()
    with store.transaction():  # the video ended: the rest of the telemetry, then close everything
        feed_telemetry(None)
        pipeline.finish(last_at, last_offset)
    result = VideoIngestResult(
        stream.metadata, stream.report, stream.candidates, kept, stream.telemetry
    )
    return LiveVideoResult(result, steps, pipeline)


def _timeout_analyzer(agent: PerceptionAgent) -> Callable[[FrameInput], Observation]:
    """Analyse through the perception agent (its timeout), raising a classified failure if no
    observation comes back."""

    def analyze(frame: FrameInput) -> Observation:
        result = agent.perceive(frame)
        if result.observation is None:
            error = result.trace.attempts[-1].error if result.trace.attempts else None
            kind = "timeout" if (error or "").startswith("TimeoutError") else "provider_error"
            raise PerceptionFailed(kind, error or "no provider produced an answer")
        return result.observation

    return analyze
