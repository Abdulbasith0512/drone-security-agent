"""CLI entry point: `python -m drone_agent <simulate|ingest|query> ...`."""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from drone_agent.ingest import ingest_result
from drone_agent.models import DetectedObject, ObjectType, Observation, Zone
from drone_agent.scenario import ScenarioError, load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import DuplicateRecordError, Fts5UnavailableError, Store, StoreError


def _format_object(obj: DetectedObject) -> str:
    extra = ", ".join(f"{k}={v}" for k, v in obj.attributes.items())
    detail = "; ".join(part for part in (obj.activity, extra) if part)
    return f"{obj.object_type.value} ({detail}) {obj.confidence:.2f}"


def _format_objects(obs: Observation) -> str:
    return "\n".join(_format_object(o) for o in obs.detected_objects) or "-"


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


def _error(console: Console, message: str) -> None:
    console.print(f"[red]error:[/red] {escape(message)}", highlight=False)


def simulate_command(path: str, console: Console) -> int:
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        _error(console, str(exc))
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
            _format_objects(obs),
            _format_telemetry(obs),
        )
    console.print(table)
    return 0


def ingest_command(path: str, db: str, console: Console) -> int:
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        _error(console, str(exc))
        return 2
    result = run_scenario(scenario)
    try:
        with Store.open(db) as store:
            summary = ingest_result(store, result)
    except DuplicateRecordError as exc:
        _error(
            console,
            f"{exc}\nThe database already holds these records. Ingestion is atomic, so nothing "
            "was changed. Use a new --db path (or delete the file) to ingest again.",
        )
        return 1
    except Fts5UnavailableError as exc:
        _error(console, str(exc))
        return 3
    except StoreError as exc:
        _error(console, str(exc))
        return 1

    table = Table(title="Ingest summary", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Scenario", summary.scenario)
    table.add_row("Frames ingested", str(summary.frames))
    table.add_row("Observations ingested", str(summary.observations))
    table.add_row("Objects ingested", str(summary.objects))
    table.add_row("Telemetry samples", str(summary.telemetry_samples))
    table.add_row("Stale telemetry associations", str(summary.stale_associations))
    table.add_row("Database", db)
    console.print(table)
    return 0


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"'{value}' needs a UTC offset, e.g. 2026-09-29T23:45:00+05:30"
        )
    return parsed


def query_command(args: argparse.Namespace, console: Console) -> int:
    filters = {
        "start": args.start,
        "end": args.end,
        "zone": Zone(args.zone) if args.zone else None,
        "object_type": ObjectType(args.object) if args.object else None,
        "text": args.search,
        "limit": args.limit,
    }
    try:
        with Store.open(args.db) as store:
            if args.explain:
                for line in store.explain_query(**filters):
                    console.print(line, markup=False, highlight=False)
                return 0
            rows = store.query_observations(**filters)
    except (StoreError, ValueError) as exc:
        _error(console, str(exc))
        return 1

    table = Table(title=f"{len(rows)} observation(s)")
    for col in ("Time", "Frame", "Observation", "Zone", "Description", "Objects"):
        table.add_column(col, overflow="fold")
    for obs in rows:
        table.add_row(
            obs.timestamp.isoformat(timespec="seconds"),
            str(obs.frame_id),
            obs.observation_id,
            obs.zone.value,
            obs.description,
            _format_objects(obs),
        )
    console.print(table)
    return 0


def main(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    parser = argparse.ArgumentParser(prog="drone_agent")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="run a scenario and show observations + telemetry sync")
    sim.add_argument("scenario", help="path to a scenario YAML file")

    ing = sub.add_parser("ingest", help="simulate a scenario and store it in a SQLite database")
    ing.add_argument("scenario", help="path to a scenario YAML file")
    ing.add_argument("--db", required=True, help="SQLite database path (created if missing)")

    qry = sub.add_parser("query", help="query stored observations (filters are ANDed)")
    qry.add_argument("--db", required=True, help="SQLite database path")
    qry.add_argument("--zone", choices=[z.value for z in Zone])
    qry.add_argument("--object", choices=[t.value for t in ObjectType], help="object type")
    qry.add_argument("--search", help='full-text search over descriptions, e.g. "blue pickup"')
    qry.add_argument("--start", type=_parse_time, help="inclusive, ISO-8601 with UTC offset")
    qry.add_argument("--end", type=_parse_time, help="inclusive, ISO-8601 with UTC offset")
    qry.add_argument("--limit", type=int)
    qry.add_argument("--explain", action="store_true", help="print the SQLite query plan only")

    args = parser.parse_args(argv)
    console = console or Console()
    if args.command == "simulate":
        return simulate_command(args.scenario, console)
    if args.command == "ingest":
        return ingest_command(args.scenario, args.db, console)
    return query_command(args, console)


if __name__ == "__main__":
    sys.exit(main())
