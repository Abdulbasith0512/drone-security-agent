"""Real video ingestion: OpenCV source, sampling, frame-change gate, SIMULATED timeline."""

from drone_agent.video.gate import FrameGate, GateConfig, GateDecision
from drone_agent.video.ingest import (
    CandidateRecord,
    SkippedPolicy,
    VideoIngestConfig,
    VideoIngestResult,
    run_video_ingest,
)
from drone_agent.video.sampling import (
    SampledFrame,
    SamplingConfig,
    SamplingReport,
    plan_samples,
    read_frame_at,
    sample_frames,
)
from drone_agent.video.source import VideoError, VideoMetadata, VideoSource
from drone_agent.video.timeline import (
    TimelineError,
    VideoTimeline,
    load_timeline,
    single_zone_timeline,
)

__all__ = [
    "CandidateRecord",
    "FrameGate",
    "GateConfig",
    "GateDecision",
    "SampledFrame",
    "SamplingConfig",
    "SamplingReport",
    "SkippedPolicy",
    "TimelineError",
    "VideoError",
    "VideoIngestConfig",
    "VideoIngestResult",
    "VideoMetadata",
    "VideoSource",
    "VideoTimeline",
    "load_timeline",
    "plan_samples",
    "read_frame_at",
    "run_video_ingest",
    "sample_frames",
    "single_zone_timeline",
]
