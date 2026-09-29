"""A tiny deterministic SYNTHETIC video for tests and demos (never a real camera).

Four obvious scenes, drawn with plain shapes and no noise, so sampling and the frame-change gate
are easy to reason about: empty yard -> a person (static for several seconds) -> a blue vehicle
-> empty yard. Written as MJPG .avi (every frame is its own JPEG, so identical input frames give
identical decoded frames). The demo cassette pairs these scenes with canned descriptions written
by THIS module, not produced by any vision model; its provider label says so.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drone_agent.models import ObjectType
from drone_agent.video.sampling import SamplingConfig, sample_frames
from drone_agent.video.source import VideoError, VideoMetadata, VideoSource
from drone_agent.vision.cassette import Cassette, CassetteEntry, CassetteVideo
from drone_agent.vision.contract import ProviderObject, ProviderOutput

DEMO_PROVIDER_LABEL = "synthetic-demo (canned descriptions, not a vision model)"


@dataclass(frozen=True)
class Scene:
    start_seconds: float
    end_seconds: float
    kind: str  # "empty" | "person" | "vehicle"


DEMO_SCENES = (
    Scene(0, 4, "empty"),
    Scene(4, 12, "person"),  # static for 8 s: several identical frames for the gate
    Scene(12, 16, "vehicle"),
    Scene(16, 20, "empty"),
)


def render_frame(kind: str, size: tuple[int, int] = (320, 180)) -> np.ndarray:
    width, height = size
    frame = np.full((height, width, 3), (70, 75, 70), np.uint8)  # grey-green yard
    cv2.rectangle(frame, (0, int(height * 0.7)), (width, height), (95, 100, 105), -1)  # road
    # Shapes are large and high-contrast on purpose: scene changes must be obvious to the gate.
    if kind == "person":
        x = width // 2
        cv2.rectangle(
            frame, (x - 40, int(height * 0.25)), (x + 40, int(height * 0.95)), (15, 15, 20), -1
        )
        cv2.circle(frame, (x, int(height * 0.16)), 28, (200, 200, 235), -1)
    elif kind == "vehicle":
        cv2.rectangle(
            frame, (20, int(height * 0.4)), (width - 20, int(height * 0.9)), (230, 90, 10), -1
        )
        cv2.rectangle(
            frame,
            (width // 2 - 60, int(height * 0.18)),
            (width // 2 + 90, int(height * 0.4)),
            (250, 160, 60),
            -1,
        )
    elif kind != "empty":
        raise ValueError(f"unknown scene kind: {kind}")
    return frame


def scene_at(offset_seconds: float, scenes: tuple[Scene, ...] = DEMO_SCENES) -> Scene:
    for scene in scenes:
        if scene.start_seconds <= offset_seconds < scene.end_seconds:
            return scene
    return scenes[-1]


def write_demo_video(
    path: str | Path,
    fps: float = 10.0,
    size: tuple[int, int] = (320, 180),
    scenes: tuple[Scene, ...] = DEMO_SCENES,
) -> Path:
    """Write the synthetic video. Raises VideoError if this OpenCV build cannot write MJPG."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        raise VideoError("this OpenCV build cannot write MJPG video")
    try:
        total = int(round(scenes[-1].end_seconds * fps))
        for i in range(total):
            writer.write(render_frame(scene_at(i / fps, scenes).kind, size))
    finally:
        writer.release()
    return target


_SCENE_OUTPUTS = {
    "empty": ProviderOutput(description="An empty yard with a road; nothing notable is visible."),
    "person": ProviderOutput(
        description="A person stands in the yard.",
        objects=[
            ProviderObject(
                type=ObjectType.PERSON,
                activity="standing",
                attributes={"clothing": "dark jacket"},
                confidence=0.9,
            )
        ],
    ),
    "vehicle": ProviderOutput(
        description="A blue pickup truck is parked in the yard.",
        objects=[
            ProviderObject(
                type=ObjectType.VEHICLE,
                activity="parked",
                attributes={"color": "blue", "body_type": "pickup"},
                confidence=0.92,
            )
        ],
    ),
}


def demo_output(kind: str) -> ProviderOutput:
    return _SCENE_OUTPUTS[kind]


def build_demo_cassette(
    video_path: str | Path,
    every_seconds: float = 1.0,
    scenes: tuple[Scene, ...] = DEMO_SCENES,
) -> Cassette:
    """A cassette with a canned response for every `every_seconds`-th frame of the demo video.

    Frames are matched by video hash and index, so it replays only against this exact file.
    """
    source = VideoSource(video_path)
    meta: VideoMetadata = source.metadata
    cassette = Cassette.new(
        DEMO_PROVIDER_LABEL,
        CassetteVideo(
            name=meta.source_name,
            sha256=meta.sha256,
            fps=meta.fps,
            frame_count=meta.frame_count,
            width=meta.width,
            height=meta.height,
        ),
    )
    for frame in sample_frames(source, SamplingConfig(every_seconds=every_seconds)):
        cassette.add(
            CassetteEntry(
                key=frame.frame_key,
                frame_index=frame.frame_index,
                offset_seconds=frame.offset_seconds,
                content_hash=frame.content_hash,
                output=demo_output(scene_at(frame.offset_seconds, scenes).kind),
            )
        )
        frame.release()
    return cassette
