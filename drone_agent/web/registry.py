"""Run registry: which databases (and videos) the dashboard may serve, and what each one is.

The data kind is DECLARED by whoever registers a run. It is never inferred from
`observation.source`: a replayed recording and canned demo text are both stored as "cassette", so
the stored label cannot tell them apart. The dashboard shows the declared kind next to every run
and the stored source next to every observation.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class RegistryError(ValueError):
    """A run could not be registered (bad spec, missing file, unknown kind)."""


@dataclass(frozen=True)
class DataKind:
    key: str
    label: str  # the banner text
    detail: str  # one honest sentence about what the data is
    description_origin: str  # how a NEW observation's description was obtained
    has_pixels: bool


DATA_KINDS: dict[str, DataKind] = {
    "scripted": DataKind(
        "scripted",
        "SCRIPTED",
        "Frames are text descriptions written by the author in a scenario file; no image exists. "
        "Zone, time and telemetry are simulated. This is deterministic test data, not perception.",
        "authored text from the scenario file",
        has_pixels=False,
    ),
    "synthetic-demo": DataKind(
        "synthetic-demo",
        "SYNTHETIC DEMO",
        "A tiny synthetic video made of shapes, not a camera. The pixels are the real frames of "
        "that file; the descriptions are canned text replayed from a cassette, NOT the output of "
        "any vision model. Zone, time and telemetry are simulated.",
        "canned text from a cassette (not a vision model)",
        has_pixels=True,
    ),
    "real-video": DataKind(
        "real-video",
        "REAL VIDEO",
        "One recorded real-video experiment: real footage whose frame descriptions came from a "
        "vision model earlier and are stored here (no model is called now). Zone, time and "
        "telemetry are simulated, and there is no perception ground truth.",
        "recorded vision-model output, replayed from storage (no live call)",
        has_pixels=True,
    ),
}

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")


@dataclass(frozen=True)
class Run:
    name: str
    db: Path
    kind: DataKind
    video: Path | None = None

    @property
    def has_video(self) -> bool:
        return self.video is not None


def make_run(name: str, db: str | Path, kind: str, video: str | Path | None = None) -> Run:
    """Validate and build a Run. Files must already exist; a database is never created here."""
    if not _NAME.match(name):
        raise RegistryError(f"run name '{name}' must be 1-40 letters, digits, '_' or '-'")
    if kind not in DATA_KINDS:
        allowed = ", ".join(DATA_KINDS)
        raise RegistryError(f"run '{name}': kind must be one of: {allowed} (got '{kind}')")
    db_path = Path(db)
    if not db_path.is_file():
        raise RegistryError(f"run '{name}': database not found: {db_path}")
    video_path = None
    if video is not None:
        video_path = Path(video)
        if not DATA_KINDS[kind].has_pixels:
            raise RegistryError(f"run '{name}': a {DATA_KINDS[kind].label} run has no video")
        if not video_path.is_file():
            raise RegistryError(f"run '{name}': video not found: {video_path}")
    return Run(name, db_path, DATA_KINDS[kind], video_path)


def parse_run_spec(spec: str) -> Run:
    """`NAME=DB.db,kind=KIND[,video=PATH]` (kind is required; it is never guessed)."""
    name, sep, rest = spec.partition("=")
    if not sep or not rest:
        raise RegistryError(f"bad --run '{spec}': expected NAME=DB,kind=KIND[,video=PATH]")
    parts = rest.split(",")
    db, options = parts[0], {}
    for part in parts[1:]:
        key, eq, value = part.partition("=")
        if not eq or key not in ("kind", "video") or not value:
            raise RegistryError(f"bad option '{part}' in --run '{spec}' (allowed: kind=, video=)")
        if key in options:
            raise RegistryError(f"option '{key}' given twice in --run '{spec}'")
        options[key] = value
    if "kind" not in options:
        raise RegistryError(
            f"--run '{name}' needs kind=scripted|synthetic-demo|real-video "
            "(the data kind is declared, never inferred)"
        )
    return make_run(name.strip(), db, options["kind"], options.get("video"))


class RunRegistry:
    def __init__(self, runs: Iterable[Run]) -> None:
        self._runs: dict[str, Run] = {}
        for run in runs:
            if run.name in self._runs:
                raise RegistryError(f"run name '{run.name}' is registered twice")
            self._runs[run.name] = run
        if not self._runs:
            raise RegistryError("no runs registered (use --demo and/or --run)")

    def get(self, name: str) -> Run | None:
        return self._runs.get(name)

    def all(self) -> list[Run]:
        return list(self._runs.values())
