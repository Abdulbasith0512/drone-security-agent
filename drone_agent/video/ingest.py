"""video -> sampling -> frame-change gate -> VisionProvider -> Observations (+ SIMULATED telemetry).

The stages are one lazy stream (`VideoStream`): each step is one sampled frame, turned into an
Observation (analysed by the provider, or carried over a gate-skipped frame) the moment it is asked
for. Two consumers use it:

* `run_video_ingest` drains it and returns plain lists of `Observation`s and `Telemetry`, exactly
  what the scripted path produces (the batch path: nothing is stored here);
* `runtime.video.run_video_live` feeds each step to the shared observation consumer as it
  arrives, so events and alerts exist before the video ends.

Either way the store, events, entities and rules cannot tell a video from a script.

Carrying is bounded. A frame the gate calls "similar" is represented by a copy of the last
observation (labelled `+carried`), but never forever: after `max_carry_seconds` (and optionally
`max_carried_frames`) the next frame is sent to the provider anyway (gate reason "forced_refresh"),
so a change too small for the pixel gate is noticed within a known time. A refreshed frame is a
fresh perception, not a carry.

A provider failure never produces an observation. With `FailurePolicy.CONTINUE` the frame gets an
explicit "failed" outcome (with a classified reason), the carry state is cleared so nothing stale
is carried over a frame that may have changed, and the next frame makes a fresh attempt.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from drone_agent.models import Observation, Telemetry
from drone_agent.telemetry import TelemetrySynchronizer, simulate_telemetry
from drone_agent.video.gate import FrameGate, GateConfig
from drone_agent.video.sampling import SamplingConfig, SamplingReport, sample_frames
from drone_agent.video.source import VideoMetadata, VideoSource
from drone_agent.video.timeline import VideoTimeline
from drone_agent.vision.base import FrameInput, VisionProvider

# The longest a scene may be carried before a fresh perception is forced. It is the adaptive
# sampler's `max_skip_seconds` (agents/sampling.py): one policy, one number for both paths. A test
# keeps the two equal.
DEFAULT_MAX_CARRY_SECONDS = 300.0


class SkippedPolicy(StrEnum):
    # Frames the gate skipped as "similar" are represented by a copy of the last accepted
    # observation (labelled source "<provider>+carried"), so a static scene stays one continuous
    # activity for the event aggregator. The vision model is NOT called for them.
    CARRY = "carry"
    # Skipped frames produce no observation. Cheaper, but a static scene then has long gaps that
    # the aggregator may split into separate events.
    DROP = "drop"


class FailurePolicy(StrEnum):
    # A provider failure aborts the run (the batch default: nothing is stored, the error is raised).
    RAISE = "raise"
    # The failed frame gets an explicit "failed" outcome and the run goes on. No observation is
    # invented for it, and what the next frame does is a fresh attempt.
    CONTINUE = "continue"


@dataclass(frozen=True)
class VideoIngestConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    skipped: SkippedPolicy = SkippedPolicy.CARRY
    # Bounded carry: force a fresh perception once a scene has been carried this long (video
    # seconds since the last fresh perception) or for this many frames. None switches a bound off.
    max_carry_seconds: float | None = DEFAULT_MAX_CARRY_SECONDS
    max_carried_frames: int | None = None
    on_failure: FailurePolicy = FailurePolicy.RAISE

    def __post_init__(self) -> None:
        if self.max_carry_seconds is not None and self.max_carry_seconds <= 0:
            raise ValueError("max_carry_seconds must be positive (or None for no bound)")
        if self.max_carried_frames is not None and self.max_carried_frames < 1:
            raise ValueError("max_carried_frames must be at least 1 (or None for no bound)")


class PerceptionFailed(Exception):
    """A provider call produced no observation (raised by an injected analyzer, for example one
    that enforces a timeout; its `kind` says why)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def classify_failure(exc: BaseException) -> str:
    """A short stable reason for a failed perception (never the key or the prompt)."""
    if isinstance(exc, PerceptionFailed):
        return exc.kind
    names = {c.__name__ for c in type(exc).__mro__}
    if names & {"ProviderOutputError"}:
        return "malformed_output"
    if names & {"CassetteMiss"}:
        return "replay_miss"
    if names & {"TimeoutError", "FutureTimeout", "ReadTimeout", "ConnectTimeout"}:
        return "timeout"
    return "provider_error"


def _failure_text(exc: BaseException) -> str:
    text = str(exc).replace("\n", " ")[:160]
    return f"{classify_failure(exc)}: {type(exc).__name__}: {text}"


@dataclass(frozen=True)
class CandidateRecord:
    """What happened to one sample candidate (for inspection)."""

    frame_index: int
    offset_seconds: float
    zone: str
    gate_score: float | None
    gate_reason: str  # first | changed | similar | gate_disabled | forced_refresh
    action: str  # "analyzed" | "carried" | "dropped" | "failed"
    error: str = ""  # for "failed": "<kind>: <class>: <short message>"


@dataclass(frozen=True)
class VideoIngestResult:
    metadata: VideoMetadata
    sampling: SamplingReport
    candidates: list[CandidateRecord]
    observations: list[Observation]
    telemetry: list[Telemetry]

    @property
    def forced_refreshes(self) -> int:
        """Frames the gate called similar that were sent to the provider anyway (bounded carry)."""
        return sum(1 for c in self.candidates if c.gate_reason == "forced_refresh")

    @property
    def failed(self) -> int:
        return sum(1 for c in self.candidates if c.action == "failed")

    @property
    def gate_accepted(self) -> int:
        """Frames the GATE sent to the provider (forced refreshes are counted separately)."""
        attempted = sum(1 for c in self.candidates if c.action in ("analyzed", "failed"))
        return attempted - self.forced_refreshes

    @property
    def gate_skipped(self) -> int:
        """Frames the gate called similar: carried, dropped, or sent anyway by a forced refresh."""
        return sum(1 for c in self.candidates if c.action in ("carried", "dropped")) + (
            self.forced_refreshes
        )

    @property
    def provider_calls(self) -> int:
        """Calls made to the provider, successful or not."""
        return sum(1 for c in self.candidates if c.action in ("analyzed", "failed"))

    @property
    def carried(self) -> int:
        return sum(1 for c in self.candidates if c.action == "carried")


@dataclass(frozen=True)
class VideoStep:
    """One sampled frame after the gate and the provider have dealt with it."""

    candidate: CandidateRecord
    frame_id: int
    timestamp: datetime  # simulated wall-clock time of the frame
    offset_seconds: float  # seconds into the video
    # analysed by the provider, or carried over a skipped frame (source "...+carried"); None when
    # the frame was dropped or its perception failed (nothing is ever invented for it). Not yet
    # joined to telemetry: the consumer does that.
    observation: Observation | None

    @property
    def error(self) -> str:
        return self.candidate.error


class VideoStream:
    """Runs the video stages lazily, one sampled frame per step, and keeps the running totals.

    Nothing is stored here. Iterate once; `candidates`, `report` and `telemetry` are complete when
    the iteration has finished (and meaningful so far while it runs).
    """

    def __init__(
        self,
        source: VideoSource,
        provider: VisionProvider,
        timeline: VideoTimeline,
        config: VideoIngestConfig | None = None,
        analyzer: Callable[[FrameInput], Observation] | None = None,
    ) -> None:
        self.config = config or VideoIngestConfig()
        # How a frame becomes an observation. By default the provider is called directly; the
        # runtime layer may inject a wrapper (a timeout, say) without this package knowing it.
        self._analyzer = analyzer or provider.analyze
        self.source, self.provider, self.timeline = source, provider, timeline
        self.metadata = source.metadata
        self.clock = timeline.clock
        timeline.check_covers((self.metadata.frame_count - 1) / self.metadata.fps)
        self.telemetry = simulate_telemetry(self.clock, timeline.telemetry)
        self.report = SamplingReport(
            self.metadata.frame_count, self.metadata.duration_seconds, self.metadata.fps
        )
        self.candidates: list[CandidateRecord] = []

    def __iter__(self) -> Iterator[VideoStep]:
        config, timeline, clock = self.config, self.timeline, self.clock
        gate = FrameGate(config.gate)
        last: Observation | None = None  # the last FRESH observation of the current scene
        last_fresh_at: datetime | None = None
        carried_since = 0
        for sampled in sample_frames(self.source, config.sampling, self.report):
            zone = timeline.telemetry.zone_at(sampled.offset_seconds)
            if last is not None and zone != last.zone:
                gate.reset()  # a new zone is a new scene: never carry objects across it
                last = last_fresh_at = None
            decision = gate.check(sampled.image)
            frame = FrameInput(
                frame_id=sampled.frame_id,
                timestamp=clock.at(sampled.offset_seconds),
                zone=zone,
                source_name=sampled.source,
                offset_seconds=sampled.offset_seconds,
                frame_key=sampled.frame_key,
                content_hash=sampled.content_hash,
                image=sampled.image,
            )
            accepted, reason, error = decision.accepted, decision.reason, ""
            if not accepted and last is not None and last_fresh_at is not None:
                age = (frame.timestamp - last_fresh_at).total_seconds()
                too_old = config.max_carry_seconds is not None and age >= config.max_carry_seconds
                too_many = (
                    config.max_carried_frames is not None
                    and carried_since >= config.max_carried_frames
                )
                if too_old or too_many:  # bounded carry: look again, whatever the pixels say
                    gate.set_reference(sampled.image)
                    accepted, reason = True, "forced_refresh"
            if accepted:
                try:
                    obs = self._analyzer(frame)
                except Exception as exc:  # noqa: BLE001 - the provider boundary
                    if config.on_failure is FailurePolicy.RAISE:
                        raise
                    # Nothing is invented for this frame. The last fresh observation describes a
                    # scene this frame may have changed, so it must not be carried any further,
                    # and the gate forgets its reference so the next frame is a fresh attempt.
                    obs, action, error = None, "failed", _failure_text(exc)
                    last = last_fresh_at = None
                    carried_since = 0
                    gate.reset()
                else:
                    last, last_fresh_at, carried_since = obs, frame.timestamp, 0
                    action = "analyzed"
            elif config.skipped is SkippedPolicy.CARRY and last is not None:
                obs = Observation(
                    observation_id=f"obs-{frame.frame_id}",
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                    source=f"{last.source}+carried",
                    zone=zone,
                    description=last.description,
                    detected_objects=last.detected_objects,
                    provenance=last.provenance,
                )
                action = "carried"
                carried_since += 1
            else:
                obs = None
                action = "dropped"
                carried_since += 1 if last is not None else 0
            sampled.release()  # the pixels are no longer needed
            candidate = CandidateRecord(
                sampled.frame_index,
                sampled.offset_seconds,
                zone.value,
                decision.score,
                reason,
                action,
                error,
            )
            self.candidates.append(candidate)
            yield VideoStep(candidate, frame.frame_id, frame.timestamp, sampled.offset_seconds, obs)


def run_video_ingest(
    source: VideoSource,
    provider: VisionProvider,
    timeline: VideoTimeline,
    config: VideoIngestConfig | None = None,
) -> VideoIngestResult:
    """Run the video pipeline to the end. Nothing is stored here; an error leaves no partial
    result. This is the batch path: the whole video is processed before anything is returned."""
    stream = VideoStream(source, provider, timeline, config)
    sync = TelemetrySynchronizer(stream.telemetry, timeline.telemetry.stale_after_s)
    observations = [
        sync.attach(step.observation) for step in stream if step.observation is not None
    ]
    return VideoIngestResult(
        stream.metadata, stream.report, stream.candidates, observations, stream.telemetry
    )
