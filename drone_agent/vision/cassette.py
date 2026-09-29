"""Cassettes: recorded provider responses replayed for reproducible, key-free runs.

A cassette file holds ONLY the provider's validated responses plus the metadata needed to match
them to frames (video file name and hash, frame index, offset, a content hash). It never holds
images, API keys, prompts with secrets, or absolute paths.

Modes (for a future real provider; only REPLAY is exercised with real data in M5A):
  replay  answer from the cassette; a missing frame is an error; the inner provider is never called
  record  answer from the cassette when possible, otherwise call the inner provider and store it
  live    always call the inner provider; the cassette is not read or written
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from drone_agent.models import DomainModel
from drone_agent.vision.base import FrameInput
from drone_agent.vision.contract import FrameDescriber, ProviderOutput, ProviderOutputError

CASSETTE_FORMAT = "drone-agent-cassette/1"


class CassetteError(Exception):
    """Base class for cassette problems."""


class CassetteFormatError(CassetteError):
    """The cassette file is unreadable or malformed."""


class CassetteMiss(CassetteError):
    """Replay was asked for a frame that has no recorded response."""


class CassetteMismatch(CassetteError):
    """The cassette was recorded for a different video file."""


class CassetteVideo(DomainModel):
    name: str  # file name only, never a path
    sha256: str = Field(min_length=64, max_length=64)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class CassetteEntry(DomainModel):
    key: str = Field(min_length=1)  # "<video sha256[:16]>:<frame index>"
    frame_index: int = Field(ge=0)
    offset_seconds: float = Field(ge=0)
    content_hash: str = Field(min_length=1)
    output: ProviderOutput


class CassetteFile(DomainModel):
    format: Literal["drone-agent-cassette/1"] = CASSETTE_FORMAT
    provider: str = Field(min_length=1)  # who produced the responses, e.g. "synthetic-demo"
    contract_version: str = "1"
    video: CassetteVideo
    entries: list[CassetteEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_keys(self) -> "CassetteFile":
        keys = [e.key for e in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("cassette entry keys must be unique")
        return self


class Cassette:
    def __init__(self, data: CassetteFile) -> None:
        self.data = data
        self._by_key = {e.key: e for e in data.entries}

    @classmethod
    def new(cls, provider: str, video: CassetteVideo) -> "Cassette":
        return cls(CassetteFile(provider=provider, video=video))

    @classmethod
    def load(cls, path: str | Path) -> "Cassette":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            return cls(CassetteFile.model_validate(raw))
        except (OSError, json.JSONDecodeError) as exc:
            raise CassetteFormatError(f"cannot read cassette {path}: {exc}") from exc
        except ValueError as exc:  # pydantic ValidationError is a ValueError
            raise CassetteFormatError(f"invalid cassette {path}: {exc}") from exc

    def save(self, path: str | Path) -> None:
        """Deterministic output: entries sorted by frame index, sorted keys, stable indentation."""
        entries = sorted(self._by_key.values(), key=lambda e: (e.frame_index, e.key))
        data = self.data.model_copy(update={"entries": entries})
        text = json.dumps(data.model_dump(mode="json"), indent=2, sort_keys=True)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")

    def lookup(self, key: str) -> CassetteEntry | None:
        return self._by_key.get(key)

    def add(self, entry: CassetteEntry) -> None:
        self._by_key[entry.key] = entry
        self.data = self.data.model_copy(update={"entries": list(self._by_key.values())})

    def __len__(self) -> int:
        return len(self._by_key)


class CassetteMode(StrEnum):
    REPLAY = "replay"
    RECORD = "record"
    LIVE = "live"


class CassetteDescriber:
    """A FrameDescriber backed by a cassette (and, when recording/live, an inner describer)."""

    def __init__(
        self,
        cassette: Cassette,
        mode: CassetteMode = CassetteMode.REPLAY,
        inner: FrameDescriber | None = None,
        path: str | Path | None = None,
    ) -> None:
        if mode is not CassetteMode.REPLAY and inner is None:
            raise ValueError(f"mode '{mode.value}' needs an inner describer")
        self.cassette = cassette
        self.mode = CassetteMode(mode)
        self.inner = inner
        self.path = Path(path) if path else None
        self.replayed = 0
        self.recorded = 0
        self.live_calls = 0
        self.content_mismatches: list[str] = []

    @property
    def name(self) -> str:
        if self.mode is CassetteMode.REPLAY or self.inner is None:
            return "cassette"
        return self.inner.name

    def check_video(self, video_sha256: str) -> None:
        """Fail early if this cassette belongs to a different video file."""
        recorded = self.cassette.data.video
        if self.mode is not CassetteMode.LIVE and recorded.sha256 != video_sha256:
            raise CassetteMismatch(
                f"cassette was recorded for a different video "
                f"({recorded.name}, sha256 {recorded.sha256[:12]}...), "
                f"not this one (sha256 {video_sha256[:12]}...)"
            )

    def describe(self, frame: FrameInput) -> ProviderOutput:
        if self.mode is CassetteMode.LIVE:
            return self._call_inner(frame)
        if frame.frame_key is None or frame.content_hash is None:
            raise ValueError("cassette lookup needs frame.frame_key and frame.content_hash")
        entry = self.cassette.lookup(frame.frame_key)
        if entry is not None:
            self.replayed += 1
            if entry.content_hash != frame.content_hash:
                self.content_mismatches.append(frame.frame_key)  # advisory: decoder differences
            return entry.output
        if self.mode is CassetteMode.REPLAY:
            raise CassetteMiss(
                f"no recorded response for frame index {frame.frame_id} (key {frame.frame_key}); "
                "re-record the cassette or sample the frames it contains"
            )
        output = self._call_inner(frame, count_live=False)
        self.cassette.add(
            CassetteEntry(
                key=frame.frame_key,
                frame_index=frame.frame_id,
                offset_seconds=frame.offset_seconds or 0.0,
                content_hash=frame.content_hash,
                output=output,
            )
        )
        self.recorded += 1
        return output

    def _call_inner(self, frame: FrameInput, count_live: bool = True) -> ProviderOutput:
        assert self.inner is not None
        output = self.inner.describe(frame)
        if not isinstance(output, ProviderOutput):
            raise ProviderOutputError("inner describer must return a ProviderOutput")
        if count_live:
            self.live_calls += 1
        return output

    def flush(self) -> None:
        """Write the cassette to `path` (record mode only)."""
        if self.mode is CassetteMode.RECORD and self.path is not None:
            self.cassette.save(self.path)
