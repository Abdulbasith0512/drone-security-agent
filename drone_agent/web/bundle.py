"""`--demo`: build the offline demo runs by calling the EXISTING CLI commands in-process.

Nothing here re-implements the pipeline. The scripted run is `ingest` -> `build-events` ->
`analyze` on scenarios/security_events.yaml; the synthetic-video run is `make-demo-video` ->
`ingest-video --provider cassette` (replay of the canned cassette) -> `build-events` -> `analyze`.
No model, no network and no API key are involved; the Gemini configuration is never read.
"""

import io
from pathlib import Path

from rich.console import Console

from drone_agent.__main__ import main as cli_main
from drone_agent.web.registry import Run, make_run

SCRIPTED_NAME = "security_events"
VIDEO_NAME = "synthetic_video"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class BundleError(RuntimeError):
    """A demo build step failed; the message holds that step's console output."""


def _cli(*argv: str) -> None:
    out = io.StringIO()
    code = cli_main(list(argv), Console(file=out, width=140, force_terminal=False))
    if code != 0:
        raise BundleError(f"`drone_agent {' '.join(argv)}` exited with {code}:\n{out.getvalue()}")


def _remove(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


def build_demo_runs(
    out_dir: str | Path, rebuild: bool = False, scenarios_dir: str | Path | None = None
) -> list[Run]:
    """Build (or reuse) the two demo runs under `out_dir` and return them registered."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = Path(scenarios_dir) if scenarios_dir else _REPO_ROOT / "scenarios"

    scripted_db = out / f"{SCRIPTED_NAME}.db"
    if rebuild or not scripted_db.is_file():
        _remove(scripted_db)
        scenario = scenarios / "security_events.yaml"
        _cli("ingest", str(scenario), "--db", str(scripted_db))
        _cli("build-events", "--db", str(scripted_db))
        _cli("analyze", "--db", str(scripted_db))

    video_db = out / f"{VIDEO_NAME}.db"
    video = out / "demo" / "demo.avi"
    if rebuild or not video_db.is_file() or not video.is_file():
        _remove(video_db)
        _cli("make-demo-video", "--out-dir", str(out / "demo"))
        _cli(
            "ingest-video",
            str(video),
            "--db",
            str(video_db),
            "--provider",
            "cassette",
            "--cassette",
            str(out / "demo" / "demo.cassette.json"),
            "--sample-every",
            "2",
        )
        _cli("build-events", "--db", str(video_db))
        _cli("analyze", "--db", str(video_db))

    return [
        make_run(SCRIPTED_NAME, scripted_db, "scripted"),
        make_run(VIDEO_NAME, video_db, "synthetic-demo", video),
    ]
