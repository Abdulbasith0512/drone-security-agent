"""`python -m drone_agent.web`: start the evaluator dashboard on localhost.

    python -m drone_agent.web --demo
    python -m drone_agent.web --run mine=output/security.db,kind=scripted
    python -m drone_agent.web --run clip=output/real/clip.db,kind=real-video,video=clip.mp4

Offline and read-only: no model, no network, no API key. `--demo` rebuilds the two demo runs with
the existing CLI commands.

Public hosting (opt-in), for example as a single Render web service:

    python -m drone_agent.web --demo --host 0.0.0.0 --port $PORT
        --public-host $RENDER_EXTERNAL_HOSTNAME        (one command; wrapped for reading)

`--public-host` names the ONE hostname the service answers to (and its https origin); it is what
allows binding 0.0.0.0. Without it nothing changes: loopback only, loopback Host/Origin only. There
is still no authentication and the data is read-only demo data; no credential is read or needed.
"""

import argparse
import sys
import webbrowser
from collections.abc import Sequence

from rich.console import Console

from drone_agent.store import StoreError
from drone_agent.video import VideoError
from drone_agent.web.api import DashboardApi, open_readonly
from drone_agent.web.bundle import BundleError, build_demo_runs
from drone_agent.web.registry import RegistryError, Run, RunRegistry, parse_run_spec
from drone_agent.web.server import (
    LOOPBACK_HOSTS,
    PUBLIC_BIND_HOST,
    STATIC_DIR,
    STATIC_FILES,
    make_server,
    normalise_public_host,
)


class _Parser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        parsed = super().parse_args(args, namespace)
        if parsed.host == PUBLIC_BIND_HOST and not parsed.public_host:
            self.error(
                f"--host {PUBLIC_BIND_HOST} needs --public-host HOSTNAME: the dashboard is "
                "loopback-only unless it is told which hostname it is served under"
            )
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m drone_agent.web", description="Evaluator dashboard (offline, read-only)."
    )
    parser.add_argument(
        "--demo", action="store_true", help="build/reuse the offline demo runs and serve them"
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=DB,kind=KIND[,video=PATH]",
        help="register an existing run; kind is required: scripted | synthetic-demo | real-video",
    )
    parser.add_argument(
        "--out-dir", default="output/dashboard", help="where --demo builds its runs"
    )
    parser.add_argument(
        "--rebuild", action="store_true", help="rebuild the --demo runs from scratch"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        choices=(*LOOPBACK_HOSTS, PUBLIC_BIND_HOST),
        help=f"default 127.0.0.1; {PUBLIC_BIND_HOST} is accepted only together with --public-host",
    )
    parser.add_argument("--port", type=int, default=8765, help="default 8765 (hosts give $PORT)")
    parser.add_argument(
        "--public-host",
        default=None,
        metavar="HOSTNAME",
        help="opt in to public hosting: bind 0.0.0.0 and accept exactly this hostname (and its "
        "https origin), for example app.onrender.com",
    )
    parser.add_argument("--open", action="store_true", help="open the page in the default browser")
    return parser


def main(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    console = console or Console()
    args = build_parser().parse_args(argv)
    if not args.demo and not args.run:
        console.print("[red]error:[/red] give --demo and/or at least one --run")
        return 2
    public_host = None
    if args.public_host is not None:
        try:
            public_host = normalise_public_host(args.public_host)
        except ValueError as exc:
            console.print(f"[red]error:[/red] --public-host: {exc}")
            return 2
    host = PUBLIC_BIND_HOST if public_host else args.host  # public mode always binds 0.0.0.0
    missing = [name for name, _ in STATIC_FILES.values() if not (STATIC_DIR / name).is_file()]
    if missing:
        console.print(
            f"[red]error:[/red] the dashboard UI files are missing from {STATIC_DIR}: "
            f"{', '.join(missing)}. Run from a full clone of the repository "
            "(the UI lives in the top-level frontend/ folder)."
        )
        return 2
    try:
        runs: list[Run] = []
        if args.demo:
            console.print("Building the offline demo runs (existing CLI commands, no model) ...")
            runs += build_demo_runs(args.out_dir, rebuild=args.rebuild)
        runs += [parse_run_spec(spec) for spec in args.run]
        registry = RunRegistry(runs)
        for run in registry.all():  # fail early on an unreadable or outdated database
            open_readonly(run.db).close()
        api = DashboardApi(registry)
        if public_host:
            server = make_server(api, host, args.port, public_host=public_host)
        else:  # exactly as before
            server = make_server(api, host, args.port)
    except (RegistryError, BundleError, StoreError, VideoError, OSError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        return 2

    if public_host:
        url = f"https://{public_host}/"
        console.print(
            f"\nDashboard (public mode): [bold]{url}[/bold]   bound {host}:{server.server_port}; "
            "read-only demo data, no authentication"
        )
    else:
        url = f"http://{host}:{server.server_port}/"
        console.print(f"\nDashboard: [bold]{url}[/bold]   (read-only, offline; Ctrl+C to stop)")
    for run in registry.all():
        video = f", video {run.video}" if run.video else ""
        console.print(f"  - {run.name}: [bold]{run.kind.label}[/bold] ({run.db}{video})")
    if args.open and not public_host:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
