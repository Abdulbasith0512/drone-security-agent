"""CLI: run a scenario as a live simulated patrol, list scenarios, work with alert lifecycle.

    python -m drone_agent run --scenario loiter_midnight --speed 5
    python -m drone_agent scenarios
    python -m drone_agent alert-action --db output/runs/loiter_midnight.db ALR-... acknowledge

`run` processes frames and telemetry progressively on a simulated clock, persists everything to
SQLite as it happens and shows a live dashboard. Real versus simulated is stated on screen.
"""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from drone_agent.agents.sampling import SamplerConfig
from drone_agent.pipeline import ConfigError, load_config
from drone_agent.rules import AlertManager, LifecycleError, load_rules_config
from drone_agent.runtime import EngineOptions, Pacer, PatrolEngine, RunSummary
from drone_agent.runtime.view import LiveView, LogView
from drone_agent.scenario import Scenario, ScenarioError, load_scenario
from drone_agent.store import Store, StoreError

SCENARIO_DIR = Path(__file__).resolve().parents[1] / "scenarios"
COMMANDS = ("run", "scenarios", "alert-action")

_SEVERITY_STYLE = {
    "low": "cyan",
    "medium": "yellow",
    "high": "bold red",
    "critical": "bold white on red",
}


def _cell(value: object, style: str = "") -> Text:
    return Text(str(value), style=style)


def _error(console: Console, message: str) -> None:
    console.print(f"[red]error:[/red] {escape(message)}", highlight=False)


# ---- scenario discovery -----------------------------------------------------------------------
def available_scenarios() -> list[Path]:
    return sorted(SCENARIO_DIR.glob("*.yaml")) if SCENARIO_DIR.is_dir() else []


def resolve_scenario(name_or_path: str) -> Path:
    """A path to a scenario file, or the name of one shipped in scenarios/."""
    direct = Path(name_or_path)
    if direct.is_file():
        return direct
    named = SCENARIO_DIR / f"{name_or_path}.yaml"
    if named.is_file():
        return named
    known = ", ".join(p.stem for p in available_scenarios()) or "none found"
    raise ScenarioError(f"unknown scenario '{name_or_path}' (available: {known})")


def scenarios_command(console: Console) -> int:
    table = Table(title="Scenarios (scripted frames + simulated telemetry)")
    for col in ("Name", "Start", "Duration", "Frames", "Ground truth", "What it exercises"):
        table.add_column(col, overflow="fold")
    for path in available_scenarios():
        try:
            sc = load_scenario(path)
        except ScenarioError as exc:
            table.add_row(_cell(path.stem), "-", "-", "-", "-", _cell(f"INVALID: {exc}"))
            continue
        truth = (
            f"{len(sc.expect.alerts)} alerts, {len(sc.expect.forbidden)} forbidden"
            if sc.expect
            else "none (prose only)"
        )
        table.add_row(
            _cell(sc.name),
            _cell(sc.start.strftime("%m-%d %H:%M")),
            _cell(f"{sc.telemetry.duration_s // 60} min"),
            _cell(len(sc.frames)),
            _cell(truth),
            _cell(sc.description),
        )
    console.print(table)
    console.print("Run one with: python -m drone_agent run --scenario NAME --speed 5")
    return 0


# ---- run --------------------------------------------------------------------------------------
def _spec_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(args: argparse.Namespace, console: Console) -> int:
    try:
        path = resolve_scenario(args.scenario)
        scenario = load_scenario(path)
        rules = load_rules_config(args.rules)
        events = load_config(args.config)
    except (ScenarioError, ConfigError) as exc:
        _error(console, str(exc))
        return 2

    db = Path(args.db) if args.db else Path("output/runs") / f"{scenario.name}.db"
    if db.exists():
        if not args.overwrite:
            _error(
                console,
                f"{db} already exists; add --overwrite to replace it or pass another --db",
            )
            return 1
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    options = EngineOptions(
        rules=rules,
        events=events,
        sampler=SamplerConfig(enabled=not args.no_sampler),
    )
    live = console.is_terminal and not args.no_live and not args.quiet
    view = LiveView(console, args.speed, len(scenario.expect.alerts) if scenario.expect else 0)
    log = LogView(console)
    started = time.monotonic()
    last_draw = [0.0]

    try:
        with Store.open(db) as store:
            if live:
                with Live(
                    view.render(), console=console, refresh_per_second=10, screen=False
                ) as ui:

                    def listener(event, ui=ui):
                        view(event)
                        now = time.monotonic()
                        if now - last_draw[0] > 0.05 or event.kind.value in ("alert", "finished"):
                            last_draw[0] = now
                            ui.update(view.render())

                    engine = PatrolEngine(
                        scenario,
                        store,
                        options=options,
                        listener=listener,
                        pacer=Pacer(args.speed, args.max_wait),
                        spec_sha256=_spec_sha256(path),
                    )
                    summary = engine.run()
                    ui.update(view.render())
            else:
                engine = PatrolEngine(
                    scenario,
                    store,
                    options=options,
                    listener=None if args.quiet else log,
                    pacer=Pacer(args.speed, args.max_wait),
                    spec_sha256=_spec_sha256(path),
                )
                summary = engine.run()
            print_result(console, scenario, summary, store, db)
    except (StoreError, ValueError) as exc:
        _error(console, str(exc))
        return 1
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(
            json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        console.print(_cell(f"wrote {args.json}"))
    console.print(_cell(f"(wall time {time.monotonic() - started:.1f}s)", "dim"))
    return 0


def print_result(
    console: Console, scenario: Scenario, summary: RunSummary, store: Store, db: Path
) -> None:
    table = Table(title=f"Patrol summary: {summary.scenario}", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row(
        "Simulated span", _cell(f"{summary.sim_start:%m-%d %H:%M} -> {summary.sim_end:%H:%M}")
    )
    table.add_row(
        "Telemetry (SIMULATED)",
        _cell(
            f"{summary.telemetry_delivered} delivered of {summary.telemetry_scheduled} scheduled"
        ),
    )
    table.add_row(
        "Frames / model calls",
        _cell(
            f"{summary.frames} frames, {summary.model_calls} perception calls, "
            f"{summary.skipped} carried forward ({summary.avoided_pct:g}% of calls avoided)"
        ),
    )
    reasons = ", ".join(f"{k} {v}" for k, v in summary.sampler["by_reason"].items())
    table.add_row("Sampler decisions", _cell(reasons or "-"))
    table.add_row("Provenance", _cell(", ".join(f"{k} {v}" for k, v in summary.provenance.items())))
    table.add_row(
        "Observations / events / entities",
        _cell(f"{summary.observations} / {summary.events} / {summary.entities}"),
    )
    status = ", ".join(f"{k} {v}" for k, v in summary.alerts_by_status.items()) or "-"
    table.add_row("Alerts", _cell(f"{summary.alerts} ({status})"))
    table.add_row("Database", _cell(db))
    console.print(table)
    print_alerts(console, store)
    _print_ground_truth(console, scenario, store)
    console.print(
        _cell(
            f"Next: python -m drone_agent alerts --db {db} --show <ALERT_ID>   |   "
            f'python -m drone_agent ask --db {db} "What happened?"',
            "dim",
        )
    )


def print_alerts(console: Console, store: Store) -> None:
    alerts = store.query_alerts()
    if not alerts:
        console.print("no alerts were raised")
        return
    table = Table(title=f"{len(alerts)} alert(s) with lifecycle", show_lines=False)
    for col in ("Alert", "Sev", "Status", "Raised", "Latency", "Ladder"):
        table.add_column(col, overflow="fold")
    for a in alerts:
        path = " > ".join(s.severity.value for s in a.ladder) or a.severity.value
        latency = a.detection_latency_seconds
        table.add_row(
            _cell(a.alert_id),
            _cell(a.severity.value, _SEVERITY_STYLE.get(a.severity.value, "")),
            _cell(a.status.value),
            _cell(a.triggered_at.strftime("%H:%M:%S")),
            _cell("-" if latency is None else f"{latency:.0f}s"),
            _cell(path),
        )
    console.print(table)
    console.print(Text("Recommended actions (fixed table by rule and severity, not a model):"))
    for a in alerts:
        confidence = "-" if a.confidence is None else f"{a.confidence:.2f}"
        verdict = a.verdict.value if a.verdict else "-"
        console.print(Text(f"  {a.alert_id} [{verdict}, confidence {confidence}]", style="bold"))
        console.print(Text(f"    {a.recommended_action or '-'}"))


def _print_ground_truth(console: Console, scenario: Scenario, store: Store) -> None:
    if scenario.expect is None:
        return
    try:
        from drone_agent.eval.ground_truth import render_scores, score_store
    except ImportError:  # the evaluation harness is optional at this point of the build
        return
    scores = score_store(scenario, store)
    render_scores(console, scores)


# ---- alert lifecycle actions -------------------------------------------------------------------
def alert_action_command(args: argparse.Namespace, console: Console) -> int:
    """Acknowledge or resolve an alert as an operator, recording who did what and when."""
    at = args.at or datetime.now(UTC)
    try:
        with Store.open(args.db) as store:
            manager = AlertManager(store.query_alerts())
            act = manager.acknowledge if args.action == "acknowledge" else manager.resolve
            change = act(args.alert_id, at, args.reason) if args.reason else act(args.alert_id, at)
            store.save_alert(change.alert)
            alert = change.alert
    except (StoreError, LifecycleError) as exc:
        _error(console, str(exc))
        return 1
    console.print(
        _cell(
            f"{alert.alert_id}: {alert.history[-1].kind.value} by operator "
            f"-> status {alert.status.value}"
        )
    )
    return 0


# ---- registration ------------------------------------------------------------------------------
def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            f"'{value}' needs a UTC offset, e.g. 2026-09-30T00:10:00+05:30"
        )
    return parsed


def add_parsers(sub) -> None:
    run = sub.add_parser(
        "run",
        help="run a scenario as a live simulated patrol (frames + telemetry arrive progressively)",
    )
    run.add_argument("--scenario", required=True, help="a scenario name (see `scenarios`) or path")
    run.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="simulated seconds per wall second (5 = 5x real time; 0 = as fast as possible)",
    )
    run.add_argument(
        "--max-wait",
        type=float,
        default=0.6,
        help="longest wall-clock pause between two ticks in seconds (keeps quiet spells watchable)",
    )
    run.add_argument("--db", help="SQLite path (default output/runs/<scenario>.db)")
    run.add_argument("--overwrite", action="store_true", help="replace an existing database")
    run.add_argument(
        "--no-live", action="store_true", help="print a plain log instead of the dashboard"
    )
    run.add_argument("--quiet", action="store_true", help="print only the final summary")
    run.add_argument("--no-sampler", action="store_true", help="analyse every frame (sampler off)")
    run.add_argument(
        "--rules", help="rules YAML (default: built-in, see config/security_rules.yaml)"
    )
    run.add_argument("--config", help="event/entity thresholds YAML (see config/events.yaml)")
    run.add_argument("--json", help="also write the run summary as JSON to this path")

    sub.add_parser("scenarios", help="list the scenarios that ship with the project")

    act = sub.add_parser(
        "alert-action", help="acknowledge or resolve an alert as an operator (recorded in history)"
    )
    act.add_argument("--db", required=True, help="SQLite database path")
    act.add_argument("alert_id")
    act.add_argument("action", choices=["acknowledge", "resolve"])
    act.add_argument("--reason", help="why (stored in the alert's history)")
    act.add_argument("--at", type=_parse_time, help="when, ISO-8601 with UTC offset (default: now)")


def dispatch(args: argparse.Namespace, console: Console) -> int:
    if args.command == "run":
        return run_command(args, console)
    if args.command == "scenarios":
        return scenarios_command(console)
    return alert_action_command(args, console)
