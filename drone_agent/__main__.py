"""CLI entry point: `python -m drone_agent <command> ...` (run with -h for the command list)."""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from drone_agent.ingest import ingest_result
from drone_agent.models import (
    DetectedObject,
    Entity,
    Event,
    MatchLevel,
    ObjectType,
    Observation,
    Severity,
    Zone,
)
from drone_agent.pipeline import ConfigError, build_events, load_config
from drone_agent.rules import analyze, load_rules_config
from drone_agent.scenario import ScenarioError, load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import DuplicateRecordError, Fts5UnavailableError, Store, StoreError


# ---- formatting helpers ---------------------------------------------------------------------
def _cell(value: object) -> Text:
    """Table cell shown literally (never parsed as Rich markup)."""
    return Text(str(value))


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


def _frame_ranges(frames: Sequence[int]) -> str:
    """[1, 2, 3, 4, 6, 7] -> '1-4,6-7'."""
    ordered = sorted(set(frames))
    parts: list[str] = []
    start = prev = None
    for f in ordered:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = f
    if start is not None:
        parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def _clock(ts: datetime) -> str:
    return ts.strftime("%m-%d %H:%M:%S")


def _duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def _attrs(attributes: dict[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in attributes.items()) or "-"


def _error(console: Console, message: str) -> None:
    console.print(f"[red]error:[/red] {escape(message)}", highlight=False)


def _note_migration(console: Console, store: Store) -> None:
    if store.migrated_from is not None:
        console.print(f"note: upgraded database schema from v{store.migrated_from} (data kept)")


# ---- M1/M2 commands -------------------------------------------------------------------------
def simulate_command(path: str, console: Console) -> int:
    try:
        scenario = load_scenario(path)
    except ScenarioError as exc:
        _error(console, str(exc))
        return 2
    result = run_scenario(scenario)

    console.print(Text.assemble((scenario.name, "bold"), f": {scenario.description}"))
    console.print(
        f"start {scenario.start.isoformat()}  |  {len(result.observations)} frames  |  "
        f"{len(result.telemetry)} telemetry samples (synthetic coordinates)"
    )
    table = Table(show_lines=False)
    for col in ("Time", "Frame", "Zone", "Description", "Objects", "Telemetry"):
        table.add_column(col, overflow="fold")
    for obs in result.observations:
        table.add_row(
            _cell(obs.timestamp.strftime("%m-%d %H:%M:%S")),
            _cell(obs.frame_id),
            _cell(obs.zone.value),
            _cell(obs.description),
            _cell(_format_objects(obs)),
            _cell(_format_telemetry(obs)),
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
            _note_migration(console, store)
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
    table.add_row("Scenario", _cell(summary.scenario))
    table.add_row("Frames ingested", _cell(summary.frames))
    table.add_row("Observations ingested", _cell(summary.observations))
    table.add_row("Objects ingested", _cell(summary.objects))
    table.add_row("Telemetry samples", _cell(summary.telemetry_samples))
    table.add_row("Stale telemetry associations", _cell(summary.stale_associations))
    table.add_row("Database", _cell(db))
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
            _note_migration(console, store)
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
            _cell(obs.timestamp.isoformat(timespec="seconds")),
            _cell(obs.frame_id),
            _cell(obs.observation_id),
            _cell(obs.zone.value),
            _cell(obs.description),
            _cell(_format_objects(obs)),
        )
    console.print(table)
    return 0


# ---- M3 commands ----------------------------------------------------------------------------
def build_events_command(db: str, config_path: str | None, console: Console) -> int:
    try:
        config = load_config(config_path)
        with Store.open(db) as store:
            _note_migration(console, store)
            summary = build_events(store, config)
    except (ConfigError, StoreError) as exc:
        _error(console, str(exc))
        return 1

    ratio = summary.compression_ratio
    table = Table(title="Build events summary", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Observations analyzed", _cell(summary.observations))
    table.add_row("Object detections", _cell(summary.detections))
    table.add_row(
        "Events generated",
        _cell(
            f"{summary.events} (person {summary.person_events}, "
            f"vehicle {summary.vehicle_events}, other {summary.other_events})"
        ),
    )
    table.add_row(
        "Compression ratio",
        _cell(
            f"{summary.detections} detections -> {summary.events} events"
            + (f" = {ratio:.2f} detections/event" if ratio else "")
        ),
    )
    table.add_row("Vehicle entities (similar-vehicle groups)", _cell(summary.entities))
    table.add_row("Multi-appearance entities", _cell(summary.multi_appearance_entities))
    table.add_row(
        "Replaced previous",
        _cell(f"{summary.replaced_events} events, {summary.replaced_entities} entities"),
    )
    if summary.discarded_alerts:
        table.add_row(
            "Alerts discarded (they cited old events)",
            _cell(f"{summary.discarded_alerts}; run `analyze` again"),
        )
    console.print(table)
    return 0


def _event_table(events: list[Event], title: str) -> Table:
    table = Table(title=title)
    for col in (
        "Event",
        "Type",
        "Zone",
        "First seen",
        "Last seen",
        "Duration",
        "Frames",
        "Attributes",
    ):
        table.add_column(col, overflow="fold")
    table.add_column("Entity", overflow="fold")
    for ev in events:
        table.add_row(
            _cell(ev.event_id),
            _cell(ev.object_type.value),
            _cell(ev.zone.value),
            _cell(_clock(ev.first_seen)),
            _cell(_clock(ev.last_seen)),
            _cell(_duration(ev.duration_seconds)),
            _cell(f"{ev.evidence_count} obs: {_frame_ranges(ev.frame_ids)}"),
            _cell(_attrs(ev.attributes)),
            _cell(ev.entity_id or "-"),
        )
    return table


def _entity_summary(entity: Entity) -> str:
    kind = (
        "confirmed by explicit plate"
        if entity.match_level is MatchLevel.CONFIRMED
        else ("COARSE match (similar vehicle, identity not confirmed)")
    )
    return f"{entity.entity_id}: {entity.label}, {entity.appearance_count} appearance(s), {kind}"


def _show_event_evidence(store: Store, event_id: str, console: Console) -> int:
    event = store.get_event(event_id)
    if event is None:
        _error(console, f"no event {event_id}")
        return 1
    console.print(_event_table([event], f"Event {event_id}"))
    console.print(_cell(f"description: {event.description}"))
    if event.activities:
        console.print(_cell(f"activities: {', '.join(event.activities)}"))
    console.print(_cell(f"closed by: {event.close_reason.value}"))
    if event.entity_id:
        entity = store.get_entity(event.entity_id)
        if entity:
            console.print(_cell(f"entity: {_entity_summary(entity)}"))

    evidence = {o.observation_id: o for o in store.get_event_observations(event_id)}
    span = store.query_observations(zone=event.zone, start=event.first_seen, end=event.last_seen)
    table = Table(title="Evidence: event -> observations -> frames")
    for col in ("Frame", "Time", "Observation", "Role", "Description"):
        table.add_column(col, overflow="fold")
    for obs in span:
        used = obs.observation_id in evidence
        table.add_row(
            _cell(obs.frame_id),
            _cell(_clock(obs.timestamp)),
            _cell(obs.observation_id),
            _cell("EVIDENCE" if used else "no detection (gap tolerated)"),
            _cell(obs.description),
        )
    console.print(table)
    return 0


def events_command(args: argparse.Namespace, console: Console) -> int:
    try:
        with Store.open(args.db) as store:
            _note_migration(console, store)
            if args.evidence:
                return _show_event_evidence(store, args.evidence, console)
            events = store.query_events(
                zone=Zone(args.zone) if args.zone else None,
                object_type=ObjectType(args.object) if args.object else None,
                entity_id=args.entity,
            )
    except StoreError as exc:
        _error(console, str(exc))
        return 1
    if not events:
        console.print("no events found (run `build-events` first?)")
        return 0
    console.print(_event_table(events, f"{len(events)} event(s)"))
    return 0


def entities_command(args: argparse.Namespace, console: Console) -> int:
    try:
        with Store.open(args.db) as store:
            _note_migration(console, store)
            if args.show:
                entity = store.get_entity(args.show)
                if entity is None:
                    _error(console, f"no entity {args.show}")
                    return 1
                console.print(_cell(_entity_summary(entity)))
                events = store.query_events(entity_id=entity.entity_id)
                console.print(_event_table(events, f"Appearances of {entity.entity_id}"))
                console.print(
                    _cell(
                        "Note: linked on normalised coarse attributes "
                        f"({_attrs(entity.attributes)}); this does not prove it is the same "
                        "physical vehicle."
                    )
                )
                return 0
            entities = store.list_entities()
    except StoreError as exc:
        _error(console, str(exc))
        return 1
    table = Table(title=f"{len(entities)} entity(ies)")
    for col in ("Entity", "Label", "Match", "Appearances", "First seen", "Last seen", "Events"):
        table.add_column(col, overflow="fold")
    for ent in entities:
        table.add_row(
            _cell(ent.entity_id),
            _cell(ent.label),
            _cell(ent.match_level.value),
            _cell(ent.appearance_count),
            _cell(_clock(ent.first_seen)),
            _cell(_clock(ent.last_seen)),
            _cell(", ".join(ent.event_ids)),
        )
    console.print(table)
    return 0


# ---- M4 commands ----------------------------------------------------------------------------
def analyze_command(db: str, rules_path: str | None, console: Console) -> int:
    try:
        config = load_rules_config(rules_path)
        with Store.open(db) as store:
            _note_migration(console, store)
            summary = analyze(store, config)
    except (ConfigError, StoreError) as exc:
        _error(console, str(exc))
        return 1

    table = Table(
        title="Security analysis summary (deterministic rules, no LLM)", show_header=False
    )
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Events analyzed", _cell(summary.events_analyzed))
    table.add_row("Entities analyzed", _cell(summary.entities_analyzed))
    table.add_row("Alerts generated", _cell(summary.alerts))
    by_rule = ", ".join(f"{k}: {v}" for k, v in summary.by_rule.items()) or "-"
    table.add_row("Alerts by rule", _cell(by_rule))
    order = {s.value: s.rank for s in Severity}
    by_sev = sorted(summary.by_severity.items(), key=lambda kv: -order[kv[0]])
    table.add_row("Alerts by severity", _cell(", ".join(f"{k}: {v}" for k, v in by_sev) or "-"))
    table.add_row("Replaced previous alerts", _cell(summary.replaced_alerts))
    console.print(table)
    if summary.events_analyzed == 0:
        console.print("no events to analyze (run `build-events` first?)")
    return 0


def _show_alert(store: Store, alert_id: str, console: Console) -> int:
    alert = store.get_alert(alert_id)
    if alert is None:
        _error(console, f"no alert {alert_id}")
        return 1
    console.print(Text.assemble((alert.alert_id, "bold"), f"  {alert.title}"))
    for label, value in (
        ("rule", alert.rule_id),
        ("severity", alert.severity.value),
        ("triggered at", alert.triggered_at.isoformat()),
        ("evidence span", f"{alert.first_seen.isoformat()} -> {alert.last_seen.isoformat()}"),
        ("zone", alert.zone.value if alert.zone else "-"),
        ("reason", alert.message),
    ):
        console.print(_cell(f"{label + ':':14} {value}"))

    facts = Table(title="Rule facts (why it fired)", show_header=False)
    facts.add_column(style="bold")
    facts.add_column()
    for key, value in alert.facts.items():
        facts.add_row(_cell(key), _cell(value))
    console.print(facts)

    if alert.entity_id:
        entity = store.get_entity(alert.entity_id)
        if entity:
            console.print(_cell(f"entity: {_entity_summary(entity)}"))
    events = [e for e in (store.get_event(eid) for eid in alert.event_ids) if e]
    console.print(_event_table(events, "Events that triggered the alert"))

    evidence = Table(title="Evidence: alert -> events -> observations -> frames")
    for col in ("Frame", "Time", "Observation", "Event", "Description"):
        evidence.add_column(col, overflow="fold")
    for ev in events:
        for obs in store.get_event_observations(ev.event_id):
            evidence.add_row(
                _cell(obs.frame_id),
                _cell(_clock(obs.timestamp)),
                _cell(obs.observation_id),
                _cell(ev.event_id),
                _cell(obs.description),
            )
    console.print(evidence)
    console.print(
        _cell(
            f"trace: {alert.alert_id} -> events {', '.join(alert.event_ids)} -> "
            f"frames {_frame_ranges(alert.frame_ids)}"
        )
    )
    return 0


def alerts_command(args: argparse.Namespace, console: Console) -> int:
    try:
        with Store.open(args.db) as store:
            _note_migration(console, store)
            if args.show:
                return _show_alert(store, args.show, console)
            alerts = store.query_alerts(
                rule_id=args.rule,
                severity=Severity(args.severity) if args.severity else None,
                zone=Zone(args.zone) if args.zone else None,
            )
    except StoreError as exc:
        _error(console, str(exc))
        return 1
    if not alerts:
        console.print("no alerts found (none fired, or `analyze` has not been run yet)")
        return 0
    table = Table(title=f"{len(alerts)} alert(s)")
    for col in ("Alert", "Rule", "Severity", "Triggered", "Zone", "Summary", "Events"):
        table.add_column(col, overflow="fold")
    for a in alerts:
        table.add_row(
            _cell(a.alert_id),
            _cell(a.rule_id),
            _cell(a.severity.value),
            _cell(_clock(a.triggered_at)),
            _cell(a.zone.value if a.zone else "-"),
            _cell(a.title),
            _cell(", ".join(a.event_ids)),
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

    bld = sub.add_parser(
        "build-events", help="derive events and vehicle entities from stored observations"
    )
    bld.add_argument("--db", required=True, help="SQLite database path")
    bld.add_argument("--config", help="YAML thresholds (default: built-in, see config/events.yaml)")

    evt = sub.add_parser("events", help="list events, or show one event's evidence")
    evt.add_argument("--db", required=True, help="SQLite database path")
    evt.add_argument("--zone", choices=[z.value for z in Zone])
    evt.add_argument("--object", choices=[t.value for t in ObjectType], help="object type")
    evt.add_argument("--entity", help="only events linked to this entity id")
    evt.add_argument("--evidence", metavar="EVENT_ID", help="trace an event to its frames")

    ent = sub.add_parser("entities", help="list similar-vehicle entities, or show one")
    ent.add_argument("--db", required=True, help="SQLite database path")
    ent.add_argument("--show", metavar="ENTITY_ID", help="show an entity's appearances")

    ana = sub.add_parser("analyze", help="run the deterministic security rules over stored events")
    ana.add_argument("--db", required=True, help="SQLite database path")
    ana.add_argument(
        "--rules", help="rules YAML (default: built-in, see config/security_rules.yaml)"
    )

    alr = sub.add_parser("alerts", help="list alerts, or trace one alert to its evidence")
    alr.add_argument("--db", required=True, help="SQLite database path")
    alr.add_argument("--rule", help="only this rule id, e.g. loitering")
    alr.add_argument("--severity", choices=[s.value for s in Severity])
    alr.add_argument("--zone", choices=[z.value for z in Zone])
    alr.add_argument("--show", metavar="ALERT_ID", help="show reason, facts and evidence")

    args = parser.parse_args(argv)
    console = console or Console()
    if args.command == "simulate":
        return simulate_command(args.scenario, console)
    if args.command == "ingest":
        return ingest_command(args.scenario, args.db, console)
    if args.command == "query":
        return query_command(args, console)
    if args.command == "build-events":
        return build_events_command(args.db, args.config, console)
    if args.command == "events":
        return events_command(args, console)
    if args.command == "entities":
        return entities_command(args, console)
    if args.command == "analyze":
        return analyze_command(args.db, args.rules, console)
    return alerts_command(args, console)


if __name__ == "__main__":
    sys.exit(main())
