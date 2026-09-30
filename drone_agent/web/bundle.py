"""`--demo`: build the offline demo runs by calling the EXISTING CLI commands in-process.

Nothing here re-implements the pipeline. The scripted run is `ingest` -> `build-events` ->
`analyze` on scenarios/security_events.yaml; the synthetic-video run is `make-demo-video` ->
`ingest-video --provider cassette` (replay of the canned cassette) -> `build-events` -> `analyze`.
The real-video run is the same `ingest-video --provider cassette` -> `build-events` -> `analyze`
over the bundled real clip (demo_assets/real_clip/) and the cassette that holds what a vision model
answered for it when it was recorded. No model, no network and no API key are involved; the Gemini
configuration is never read.
"""

import hashlib
import io
import json
from pathlib import Path

from rich.console import Console

from drone_agent.__main__ import main as cli_main
from drone_agent.web.registry import Run, make_run

SCRIPTED_NAME = "security_events"
VIDEO_NAME = "synthetic_video"
REAL_NAME = "real_clip"
_REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_ASSETS = _REPO_ROOT / "demo_assets" / "real_clip"
_REAL_MANIFEST_FORMAT = "drone-agent-demo-asset/1"


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_real_assets(assets_dir: str | Path | None = None) -> dict:
    """Read the real-clip manifest and check the clip and cassette against it, byte for byte.

    The cassette is keyed to the clip's bytes, so a trimmed, re-encoded or substituted file would
    not replay; this says so up front, in one sentence, instead of failing deep inside ingestion.
    Returns the manifest with `clip_path` and `cassette_path` added."""
    assets = Path(assets_dir) if assets_dir else REAL_ASSETS
    manifest_path = assets / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"the real-clip demo assets are missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["format"] != _REAL_MANIFEST_FORMAT:
            raise BundleError(f"{manifest_path}: unknown manifest format {manifest['format']!r}")
        for part in ("clip", "cassette"):
            entry = manifest[part]
            path = assets / entry["file"]
            if not path.is_file():
                raise BundleError(f"the real-clip demo {part} is missing: {path}")
            if path.stat().st_size != entry["size_bytes"] or _sha256(path) != entry["sha256"]:
                raise BundleError(
                    f"the real-clip demo {part} {path.name} does not match its manifest "
                    "(size or sha256): it must be the exact bundled file, unmodified"
                )
            manifest[f"{part}_path"] = path
        if not {"start", "zone", "sample_every_seconds"} <= manifest["replay"].keys():
            raise BundleError(f"{manifest_path}: the replay settings are incomplete")
    except (KeyError, TypeError, ValueError) as exc:
        raise BundleError(f"{manifest_path} is not a valid real-clip manifest: {exc!r}") from exc
    return manifest


def build_real_run(
    out_dir: str | Path, rebuild: bool = False, assets_dir: str | Path | None = None
) -> Run:
    """Build (or reuse) the real-video run: the bundled clip's frames, a RECORDED model replay."""
    manifest = load_real_assets(assets_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / f"{REAL_NAME}.db"
    clip, cassette = manifest["clip_path"], manifest["cassette_path"]
    if rebuild or not db.is_file():
        _remove(db)
        replay = manifest["replay"]
        _cli(
            "ingest-video",
            str(clip),
            "--db",
            str(db),
            "--provider",
            "cassette",
            "--cassette",
            str(cassette),
            "--sample-every",
            f"{replay['sample_every_seconds']:g}",
            "--start",
            replay["start"],
            "--zone",
            replay["zone"],
        )
        _cli("build-events", "--db", str(db))
        _cli("analyze", "--db", str(db))
    return make_run(REAL_NAME, db, "real-video", clip)


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
