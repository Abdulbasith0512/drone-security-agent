"""CLI entry point: `python -m drone_agent simulate <scenario.yaml>`."""

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console
from rich.table import Table

from drone_agent.models import DetectedObject, Observation
from drone_agent.scenario import ScenarioError, load_scenario
from drone_agent.simulate import run_scenario


def _format_object(obj: DetectedObject) -> str:
    extra = ", ".join(f"{k}={v}" for k, v in obj.attributes.items())
    detail = "; ".join(part for part in (obj.activity, extra) if part)
    return f"{obj.object_type.value} ({detail}) {obj.confidence:.2f}"


def _format_telemetry(obs: Observation) -> str:
    assoc = obs.telemetry
    if assoc is None:
        return "none"
    t = assoc.telemetry
    status = "STALE" if assoc.stale else "ok"
    return (
        f"{status} age={assoc.age_seconds:.0f}s "
        f"alt={t.altitude_m:.1f}m bat={t.battery_percent:.1f}%"
    )


def simulate_command(path: str, console: Console) -> int:
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        console.print(f"[red]error:[/red] {exc}", markup=True, highlight=False)
        return 2
    result = run_scenario(scenario)

    console.print(f"[bold]{scenario.name}[/bold]: {scenario.description}", highlight=False)
    console.print(
        f"start {scenario.start.isoformat()}  |  {len(result.observations)} frames  |  "
        f"{len(result.telemetry)} telemetry samples (synthetic coordinates)"
    )
    table = Table(show_lines=False)
    for col in ("Time", "Frame", "Zone", "Description", "Objects", "Telemetry"):
        table.add_column(col, overflow="fold")
    for obs in result.observations:
        table.add_row(
            obs.timestamp.strftime("%m-%d %H:%M:%S"),
            str(obs.frame_id),
            obs.zone.value,
            obs.description,
            "\n".join(_format_object(o) for o in obs.detected_objects) or "-",
            _format_telemetry(obs),
        )
    console.print(table)
    return 0


def main(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drone_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="run a scenario and show observations + telemetry sync")
    sim.add_argument("scenario", help="path to a scenario YAML file")
    args = parser.parse_args(argv)
    console = console or Console()
    if args.command == "simulate":
        return simulate_command(args.scenario, console)
    return 1


if __name__ == "__main__":
    sys.exit(main())
