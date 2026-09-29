"""video -> sampling -> frame-change gate -> VisionProvider -> Observations (+ SIMULATED telemetry).

The result is a plain list of `Observation`s and `Telemetry`, exactly what the scripted path
produces, so the store, events, entities and rules cannot tell the difference.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from drone_agent.models import Observation, Telemetry
from drone_agent.telemetry import TelemetrySynchronizer, simulate_telemetry
from drone_agent.video.gate import FrameGate, GateConfig
from drone_agent.video.sampling import SamplingConfig, SamplingReport, sample_frames
from drone_agent.video.source import VideoMetadata, VideoSource
from drone_agent.video.timeline import VideoTimeline
from drone_agent.vision.base import FrameInput, VisionProvider


class SkippedPolicy(StrEnum):
    # Frames the gate skipped as "similar" are represented by a copy of the last accepted
    # observation (labelled source "<provider>+carried"), so a static scene stays one continuous
    # activity for the event aggregator. The vision model is NOT called for them.
    CARRY = "carry"
    # Skipped frames produce no observation. Cheaper, but a static scene then has long gaps that
    # the aggregator may split into separate events.
    DROP = "drop"


@dataclass(frozen=True)
class VideoIngestConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    skipped: SkippedPolicy = SkippedPolicy.CARRY
    default_confidence: float | None = None


@dataclass(frozen=True)
class CandidateRecord:
    """What happened to one sample candidate (for inspection)."""

    frame_index: int
    offset_seconds: float
    zone: str
    gate_score: float | None
    gate_reason: str
    action: str  # "analyzed" | "carried" | "dropped"


@dataclass(frozen=True)
class VideoIngestResult:
    metadata: VideoMetadata
    sampling: SamplingReport
    candidates: list[CandidateRecord]
    observations: list[Observation]
    telemetry: list[Telemetry]

    @property
    def gate_accepted(self) -> int:
        return sum(1 for c in self.candidates if c.action == "analyzed")

    @property
    def gate_skipped(self) -> int:
        return sum(1 for c in self.candidates if c.action != "analyzed")

    @property
    def provider_calls(self) -> int:
        return self.gate_accepted

    @property
    def carried(self) -> int:
        return sum(1 for c in self.candidates if c.action == "carried")


def run_video_ingest(
    source: VideoSource,
    provider: VisionProvider,
    timeline: VideoTimeline,
    config: VideoIngestConfig | None = None,
) -> VideoIngestResult:
    """Run the video pipeline. Nothing is stored here; an error leaves no partial result."""
    config = config or VideoIngestConfig()
    meta = source.metadata
    clock = timeline.clock
    timeline.check_covers((meta.frame_count - 1) / meta.fps)
    telemetry = simulate_telemetry(clock, timeline.telemetry)
    sync = TelemetrySynchronizer(telemetry, timeline.telemetry.stale_after_s)

    report = SamplingReport(meta.frame_count, meta.duration_seconds, meta.fps)
    gate = FrameGate(config.gate)
    candidates: list[CandidateRecord] = []
    observations: list[Observation] = []
    last: Observation | None = None

    for sampled in sample_frames(source, config.sampling, report):
        zone = timeline.telemetry.zone_at(sampled.offset_seconds)
        if last is not None and zone != last.zone:
            gate.reset()  # a new zone is a new scene: never carry objects across it
            last = None
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
        if decision.accepted:
            obs = provider.analyze(frame)
            last = obs
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
            )
            action = "carried"
        else:
            obs = None
            action = "dropped"
        sampled.release()  # the pixels are no longer needed
        candidates.append(
            CandidateRecord(
                sampled.frame_index,
                sampled.offset_seconds,
                zone.value,
                decision.score,
                decision.reason,
                action,
            )
        )
        if obs is not None:
            observations.append(sync.attach(obs))

    return VideoIngestResult(meta, report, candidates, observations, telemetry)
