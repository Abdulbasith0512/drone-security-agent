"""CLI entry point: `python -m drone_agent <command> ...` (run with -h for the command list)."""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from drone_agent import cli_run
from drone_agent.agent.build import (
    build_gemini_analyst,
    build_planner_analyst,
    build_replay_analyst,
)
from drone_agent.agent.config import AnalystConfig, AnalystConfigError
from drone_agent.agent.session import AnalystSession, Answer
from drone_agent.ingest import ingest_observations, ingest_result, summarize_ingest
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
from drone_agent.runtime.video import LiveVideoResult, run_video_live
from drone_agent.runtime.view import LogView
from drone_agent.scenario import ScenarioError, load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import DuplicateRecordError, Fts5UnavailableError, Store, StoreError
from drone_agent.video import (
    DEFAULT_MAX_CARRY_SECONDS,
    FailurePolicy,
    GateConfig,
    SamplingConfig,
    SkippedPolicy,
    TimelineError,
    VideoError,
    VideoIngestConfig,
    VideoSource,
    load_timeline,
    read_frame_at,
    run_video_ingest,
    single_zone_timeline,
)
from drone_agent.vision import (
    Cassette,
    CassetteDescriber,
    CassetteError,
    CassetteMode,
    DescriberProvider,
    FrameInput,
    ProviderOutputError,
)
from drone_agent.vision.cassette import CassetteVideo
from drone_agent.vision.gemini import (
    PLACEHOLDER_CONFIDENCE,
    GeminiConfig,
    GeminiDescriber,
    GeminiError,
)


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
    if event.departure_observed_at is not None:
        console.print(_cell(f"departure observed at: {event.departure_observed_at.isoformat()}"))
    elif not event.is_open:
        console.print(_cell("departure: not observed (coverage ended; this is not proof it left)"))
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


# ---- M5A commands: real video -----------------------------------------------------------------
def _parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"'{value}' must look like 64x36") from exc


DEFAULT_VIDEO_START = "2026-09-29T23:40:00+05:30"


def _gemini_config(args: argparse.Namespace, console: Console) -> GeminiConfig:
    """Build the Gemini config (key from the environment/.env only) and report only whether it
    exists, never any part of it."""
    config = GeminiConfig.from_env(
        args.model, max_dimension=args.max_dimension, jpeg_quality=args.jpeg_quality
    )
    console.print("GEMINI_API_KEY configured: yes")
    return config


def _build_video_provider(
    args: argparse.Namespace, meta, console: Console
) -> tuple[DescriberProvider, CassetteDescriber | None, GeminiDescriber | None]:
    """Returns (provider, cassette describer or None, gemini describer or None)."""
    mode = CassetteMode(args.cassette_mode)
    default_confidence = args.default_confidence
    if args.provider == "cassette":
        if mode is not CassetteMode.REPLAY:
            raise CassetteError("--provider cassette only replays; use --provider gemini to record")
        if not args.cassette:
            raise CassetteError("--provider cassette needs --cassette FILE")
        cassette = Cassette.load(args.cassette)
        describer = CassetteDescriber(cassette, CassetteMode.REPLAY)
        describer.check_video(meta.sha256)
        if default_confidence is None and cassette.data.provider.startswith("gemini"):
            default_confidence = PLACEHOLDER_CONFIDENCE
        return DescriberProvider(describer, default_confidence), describer, None

    if mode is CassetteMode.REPLAY:
        raise CassetteError("--provider gemini needs --cassette-mode record or live")
    gemini = GeminiDescriber(_gemini_config(args, console))
    if default_confidence is None:
        default_confidence = PLACEHOLDER_CONFIDENCE
    if mode is CassetteMode.LIVE:
        return DescriberProvider(gemini, default_confidence), None, gemini
    if not args.cassette:
        raise CassetteError("--cassette-mode record needs --cassette FILE to write")
    if Path(args.cassette).exists():
        cassette = Cassette.load(args.cassette)
        if cassette.data.provider != gemini.name:
            raise CassetteError(
                f"cassette was recorded by '{cassette.data.provider}', not '{gemini.name}'; "
                "use a new cassette path or the same --model"
            )
    else:
        cassette = Cassette.new(
            gemini.name,
            CassetteVideo(
                name=meta.source_name,
                sha256=meta.sha256,
                fps=meta.fps,
                frame_count=meta.frame_count,
                width=meta.width,
                height=meta.height,
            ),
        )
    describer = CassetteDescriber(cassette, CassetteMode.RECORD, inner=gemini, path=args.cassette)
    describer.check_video(meta.sha256)
    return DescriberProvider(describer, default_confidence), describer, gemini


def ingest_video_command(args: argparse.Namespace, console: Console) -> int:
    if (args.call_timeout is not None or args.fail_fast) and not args.live:
        _error(console, "--call-timeout and --fail-fast apply to --live runs only")
        return 2
    describer: CassetteDescriber | None = None
    gemini: GeminiDescriber | None = None
    try:
        source = VideoSource(args.video)
        meta = source.metadata
        if args.timeline:
            timeline = load_timeline(args.timeline)
        else:
            timeline = single_zone_timeline(args.start, Zone(args.zone), meta.duration_seconds)
        provider, describer, gemini = _build_video_provider(args, meta, console)
        width, height = args.gate_resize
        config = VideoIngestConfig(
            sampling=SamplingConfig(every_seconds=args.sample_every),
            gate=GateConfig(
                enabled=not args.no_gate,
                resize_width=width,
                resize_height=height,
                threshold=args.gate_threshold,
            ),
            skipped=SkippedPolicy(args.skipped),
            max_carry_seconds=args.max_carry_seconds or None,  # 0 switches the bound off
        )
        live: LiveVideoResult | None = None
        try:
            if args.live:
                # progressive: each frame's observation, events and alerts are committed (and
                # printed) as soon as the frame is processed, not after the whole video
                try:
                    with Store.open(args.db) as store:
                        _note_migration(console, store)
                        live = run_video_live(
                            source, provider, timeline, store, config=config,
                            listener=LogView(console),
                            on_failure=FailurePolicy.RAISE if args.fail_fast
                            else FailurePolicy.CONTINUE,
                            call_timeout_s=args.call_timeout,
                        )  # fmt: skip
                except DuplicateRecordError as exc:
                    _error(
                        console,
                        f"{exc}\nThe database already holds these records. Nothing was changed. "
                        "Use a new --db path (or delete the file) to ingest again.",
                    )
                    return 1
                except StoreError as exc:
                    _error(console, str(exc))
                    return 1
                result = live.result
            else:
                result = run_video_ingest(source, provider, timeline, config)
        finally:
            # Keep every response already paid for, even if a later frame fails.
            if describer is not None and describer.recorded:
                describer.flush()
    except (
        VideoError,
        TimelineError,
        CassetteError,
        ProviderOutputError,
        GeminiError,
        ValueError,
    ) as exc:
        _error(console, str(exc))
        if describer is not None and describer.recorded:
            console.print(
                _cell(
                    f"cassette saved with {describer.recorded} recorded frame(s): {args.cassette}"
                )
            )
        return 2

    try:
        if live is not None:  # already stored, frame by frame
            summary = summarize_ingest(meta.source_name, result.telemetry, result.observations)
        else:
            with Store.open(args.db) as store:
                _note_migration(console, store)
                summary = ingest_observations(
                    store, meta.source_name, result.telemetry, result.observations
                )
    except DuplicateRecordError as exc:
        _error(
            console,
            f"{exc}\nThe database already holds these records. Ingestion is atomic, so nothing "
            "was changed. Use a new --db path (or delete the file) to ingest again.",
        )
        return 1
    except StoreError as exc:
        _error(console, str(exc))
        return 1

    rep = result.sampling
    table = Table(title="Video ingest summary", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row(
        "Video", _cell(f"{meta.source_name} ({meta.width}x{meta.height}, {meta.fps:g} fps)")
    )
    table.add_row("Video duration", _cell(f"{meta.duration_seconds:.2f}s"))
    table.add_row("Original frames", _cell(rep.original_frames))
    table.add_row(
        "Sample candidates",
        _cell(
            f"{rep.planned} every {args.sample_every:g}s (decoded {rep.decoded}, "
            f"failed {rep.failed}, beyond end {rep.beyond_end})"
        ),
    )
    gate_note = "gate disabled" if args.no_gate else f"threshold {args.gate_threshold:g}"
    table.add_row("Frame gate accepted", _cell(f"{result.gate_accepted} ({gate_note})"))
    table.add_row(
        "Frame gate skipped (similar)",
        _cell(f"{result.gate_skipped} (policy: {args.skipped}; carried {result.carried})"),
    )
    if result.forced_refreshes:
        table.add_row(
            "Forced refreshes",
            _cell(
                f"{result.forced_refreshes} (a scene was carried for "
                f"{args.max_carry_seconds:g}s, so the provider was asked again)"
            ),
        )
    replays = f", cassette replays {describer.replayed}" if describer else ""
    table.add_row("Provider", _cell(provider.name))
    table.add_row("Provider calls", _cell(f"{result.provider_calls}{replays}"))
    if result.failed:
        table.add_row(
            "PERCEPTION FAILURES",
            _cell(f"{result.failed} frame(s): no observation stored for them (see below)"),
        )
    if gemini is not None:
        table.add_row(
            "Gemini API requests",
            _cell(
                f"{gemini.api_calls} (retries {gemini.retries}, "
                f"total latency {gemini.total_latency_s:.1f}s)"
            ),
        )
        if describer is not None:
            table.add_row(
                "Cassette",
                _cell(
                    f"recorded {describer.recorded}, replayed {describer.replayed}: {args.cassette}"
                ),
            )
        table.add_row(
            "Confidence",
            _cell(f"UNCALIBRATED placeholder {provider.default_confidence} (Gemini reports none)"),
        )
    if describer and describer.content_mismatches:
        table.add_row(
            "Cassette content mismatches",
            _cell(f"{len(describer.content_mismatches)} (decoded pixels differ from recording)"),
        )
    table.add_row("Observations written", _cell(summary.observations))
    table.add_row("Objects written", _cell(summary.objects))
    table.add_row(
        "SIMULATED telemetry samples",
        _cell(
            f"{summary.telemetry_samples} (not from the video; start {timeline.start.isoformat()})"
        ),
    )
    table.add_row("Stale telemetry associations", _cell(summary.stale_associations))
    table.add_row("Database", _cell(args.db))
    if live is not None and live.pipeline is not None:
        pipe = live.pipeline
        first = live.first_alert_step
        table.add_row(
            "Processing",
            _cell("LIVE: each frame committed before the next one was read (not batch)"),
        )
        table.add_row(
            "Events / entities / alerts",
            _cell(f"{len(pipe.events)} / {len(pipe.entities)} / {len(pipe.alerts)}"),
        )
        table.add_row(
            "First alert",
            _cell(
                "none raised"
                if first is None
                else f"after frame {first.frame_index} ({first.offset_s:g}s into the video), "
                f"with {len(live.steps) - live.steps.index(first) - 1} sampled frame(s) to come"
            ),
        )
    console.print(table)

    if args.verbose:
        detail = Table(title="Sample candidates")
        for col in ("Frame", "Offset", "Zone", "Gate score", "Gate", "Action"):
            detail.add_column(col)
        for c in result.candidates:
            score = "-" if c.gate_score is None else f"{c.gate_score:.2f}"
            detail.add_row(
                _cell(c.frame_index),
                _cell(f"{c.offset_seconds:.2f}s"),
                _cell(c.zone),
                _cell(score),
                _cell(c.gate_reason),
                _cell(c.action),
            )
        console.print(detail)
    failed = [c for c in result.candidates if c.action == "failed"]
    for c in failed:
        _error(console, f"frame {c.frame_index} ({c.offset_seconds:.2f}s): {c.error}")
    return 3 if failed else 0  # 3: finished, but some frames have no observation


def describe_frame_command(args: argparse.Namespace, console: Console) -> int:
    """Smoke test: one frame -> Gemini -> validated ProviderOutput. Persists nothing."""
    try:
        source = VideoSource(args.video)
        sampled = read_frame_at(source, args.time)
        gemini = GeminiDescriber(_gemini_config(args, console))
        frame = FrameInput(
            frame_id=sampled.frame_id,
            timestamp=datetime.fromisoformat(DEFAULT_VIDEO_START),
            source_name=sampled.source,
            offset_seconds=sampled.offset_seconds,
            image=sampled.image,
        )
        output = gemini.describe(frame)
    except (VideoError, GeminiError, ProviderOutputError, ValueError) as exc:
        _error(console, str(exc))
        return 2
    width, height = gemini.last_image_size or (0, 0)
    table = Table(title="Single-frame Gemini smoke test (nothing stored)", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Model", _cell(gemini.config.model))
    table.add_row(
        "Frame",
        _cell(f"index {sampled.frame_index} at {sampled.offset_seconds:.2f}s of {args.video}"),
    )
    table.add_row("Original size", _cell(f"{sampled.width}x{sampled.height}"))
    table.add_row("Encoded image", _cell(f"{width}x{height} JPEG, {gemini.last_jpeg_bytes} bytes"))
    table.add_row("API requests", _cell(f"{gemini.api_calls} (retries {gemini.retries})"))
    table.add_row("Latency", _cell(f"{gemini.total_latency_s:.2f}s"))
    table.add_row("Validation", _cell("ProviderOutput valid (strict Pydantic schema)"))
    table.add_row("Description", _cell(output.description))
    console.print(table)
    objects = Table(title="Objects (confidence: none reported by the model)")
    for col in ("Type", "Activity", "Attributes"):
        objects.add_column(col)
    for obj in output.objects:
        attrs = ", ".join(f"{k}={v}" for k, v in obj.attributes.items())
        objects.add_row(_cell(obj.type.value), _cell(obj.activity or "-"), _cell(attrs or "-"))
    console.print(objects)
    return 0


def _build_analyst_llm(args: argparse.Namespace, console: Console):
    """None for the offline template analyst; else a tool-calling analyst whose model is the live
    Gemini model, a replay of a recorded one, or the scripted planner (no LLM at all)."""
    if args.offline or args.provider == "offline":
        return None
    if args.provider == "planner":
        console.print("Analyst: scripted planner in the real tool-calling loop (NO LLM, no key)")
        return build_planner_analyst()
    if args.provider == "replay":
        if not args.analyst_cassette:
            raise AnalystConfigError("--provider replay needs --analyst-cassette FILE")
        llm = build_replay_analyst(args.analyst_cassette)
        console.print(f"Analyst: REPLAYED model turns ({llm.name}); no live call")
        return llm
    config = AnalystConfig.from_env(args.model)
    llm = build_gemini_analyst(config, args.record_cassette)
    console.print("GEMINI_API_KEY configured: yes")
    console.print(f"Analyst model: {config.model}")
    if args.record_cassette:
        console.print(f"Recording model turns to {args.record_cassette}")
    return llm


def _print_answer(console: Console, question: str, answer: Answer) -> None:
    # Text objects, not markup: citations look like [event:EVT-V-001].
    console.print(Text(f"\nQ: {question}", style="bold"))
    console.print(Text(f"[{answer.label}]", style="dim"))
    console.print(Text(answer.text))
    if answer.citations:
        console.print(Text(f"Verified citations: {' '.join(answer.citations)}", style="dim"))
    for note in answer.notes:
        console.print(Text(f"note: {note}", style="yellow"))
    if answer.metrics:  # LLM-backed answers only; template mode makes no model requests
        console.print(Text(f"Analyst metrics: {answer.metrics.line()}", style="dim"))
    if TRACE and answer.tool_trace:
        for step in answer.tool_trace:
            args = ", ".join(f"{k}={v}" for k, v in step["args"].items())
            outcome = f"{step['refs']} ref(s)" if step["ok"] else f"error: {step.get('error', '')}"
            console.print(Text(f"  tool {step['tool']}({args}) -> {outcome}", style="dim"))


TRACE = False  # set by --trace: show every tool call behind an answer


def _analyst_session(args: argparse.Namespace, console: Console, store: Store) -> AnalystSession:
    global TRACE
    TRACE = bool(getattr(args, "trace", False))
    return AnalystSession(store, _build_analyst_llm(args, console), record=True)


def _flush_recording(session: AnalystSession, console: Console) -> None:
    recorder = getattr(session.llm, "recorder", None)
    if recorder is not None:
        recorder.flush()
        console.print(Text(f"analyst cassette written: {recorder.path}", style="dim"))


def _follow_ups(args, console, session: AnalystSession) -> None:
    for question in args.follow_up or []:
        _print_answer(console, question, session.ask(question))
    if getattr(args, "interactive", False):
        while True:
            try:
                question = console.input("\nFollow-up (blank to quit)> ").strip()
            except EOFError:
                break
            if not question:
                break
            _print_answer(console, question, session.ask(question))


def _open_existing_db(path: str) -> Store:
    if not Path(path).is_file():
        raise StoreError(f"database not found: {path}")
    return Store.open(path)


def ask_command(args: argparse.Namespace, console: Console) -> int:
    if not args.question and not args.interactive:
        _error(console, "give a QUESTION, or use --interactive")
        return 2
    try:
        with _open_existing_db(args.db) as store:
            session = _analyst_session(args, console, store)
            if args.question:
                _print_answer(console, args.question, session.ask(args.question))
            _follow_ups(args, console, session)
            _flush_recording(session, console)
    except (StoreError, AnalystConfigError, ValueError) as exc:
        _error(console, str(exc))
        return 2
    return 0


def summarize_command(args: argparse.Namespace, console: Console) -> int:
    try:
        with _open_existing_db(args.db) as store:
            session = _analyst_session(args, console, store)
            _print_answer(console, "Summarize the run.", session.summarize())
            _follow_ups(args, console, session)
            _flush_recording(session, console)
    except (StoreError, AnalystConfigError, ValueError) as exc:
        _error(console, str(exc))
        return 2
    return 0


def evaluate_command(args: argparse.Namespace, console: Console) -> int:
    """Run the deterministic evaluation harness (no network, no API key) and write results."""
    from drone_agent.eval.runner import run_all, write_results

    try:
        sizes = tuple(int(x) for x in args.scale_sizes.split(",") if x.strip())
        if not sizes or any(n < 10 for n in sizes):
            raise ValueError("--scale-sizes must be comma-separated integers >= 10")
        results = run_all(
            args.scale, sizes, args.repeats, noise_enabled=args.noise, noise_seeds=args.noise_seeds
        )
        paths = write_results(results, args.output)
    except (ValueError, OSError) as exc:
        _error(console, str(exc))
        return 2
    table = Table(title="Ablation table (every value computed by the harness)")
    for col in ("Component", "Metric", "Baseline", "Ablated", "Effect"):
        table.add_column(col)
    for row in results["ablation_table"]:
        table.add_row(
            _cell(row["component"]),
            _cell(row["metric"]),
            _cell(f"{row['baseline']}  [{row['baseline_setting']}]"),
            _cell(f"{row['ablated']}  [{row['ablated_setting']}]"),
            _cell(row["effect"]),
        )
    console.print(table)
    for exp in results["experiments"].values():
        console.print(Text("\n" + exp["title"], style="bold"))
        for line in exp["interpretation"]:
            console.print(Text(f"  {line}"))
    console.print(Text("\n" + " ".join(results["meta"]["qualifications"]), style="dim"))
    for path in paths:
        console.print(_cell(f"wrote {path}"))
    return 0


def make_demo_video_command(out_dir: str, console: Console) -> int:
    from drone_agent.video.synthetic import (
        DEMO_PROVIDER_LABEL,
        build_demo_cassette,
        write_demo_video,
    )

    try:
        video = write_demo_video(Path(out_dir) / "demo.avi")
        cassette = build_demo_cassette(video)
    except VideoError as exc:
        _error(console, str(exc))
        return 2
    cassette_path = Path(out_dir) / "demo.cassette.json"
    cassette.save(cassette_path)
    console.print(_cell(f"synthetic video: {video}"))
    console.print(_cell(f"demo cassette:   {cassette_path} ({len(cassette)} entries)"))
    console.print(_cell(f"note: descriptions come from {DEMO_PROVIDER_LABEL}"))
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

    vid = sub.add_parser(
        "ingest-video", help="sample a video, describe frames via a provider, store observations"
    )
    vid.add_argument("video", help="path to a local video file")
    vid.add_argument("--db", required=True, help="SQLite database path (created if missing)")
    vid.add_argument("--provider", choices=["cassette", "gemini"], default="cassette")
    vid.add_argument("--cassette", help="cassette JSON: replayed, or written with record mode")
    vid.add_argument(
        "--cassette-mode",
        choices=[m.value for m in CassetteMode],
        default=CassetteMode.REPLAY.value,
        help="replay (default, no API), record (gemini; saves responses) or live (no cassette)",
    )
    vid.add_argument("--model", help="Gemini model (default: GEMINI_MODEL from env/.env)")
    vid.add_argument("--max-dimension", type=int, default=1024, help="longest side sent to Gemini")
    vid.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality sent to Gemini")
    vid.add_argument(
        "--default-confidence",
        type=float,
        default=None,
        help="placeholder confidence for objects the provider gives none (Gemini: 0.5, "
        "uncalibrated)",
    )
    vid.add_argument("--sample-every", type=float, default=2.0, metavar="SECONDS")
    vid.add_argument("--no-gate", action="store_true", help="send every sampled frame")
    vid.add_argument("--gate-threshold", type=float, default=6.0, help="mean abs pixel diff, 0-255")
    vid.add_argument("--gate-resize", type=_parse_size, default=(64, 36), metavar="WxH")
    vid.add_argument(
        "--skipped",
        choices=["carry", "drop"],
        default="carry",
        help="gate-skipped frames: carry the last observation forward, or drop them",
    )
    vid.add_argument("--timeline", help="YAML with a SIMULATED start time and patrol (zones)")
    vid.add_argument(
        "--start",
        type=_parse_time,
        default=DEFAULT_VIDEO_START,
        help="SIMULATED wall-clock time of video start (used when no --timeline)",
    )
    vid.add_argument(
        "--zone",
        choices=[z.value for z in Zone],
        default=Zone.MAIN_GATE.value,
        help="SIMULATED zone for the whole video (used when no --timeline)",
    )
    vid.add_argument(
        "--live",
        action="store_true",
        help="process progressively: commit and print each frame's observation, events and alerts "
        "as it is processed, instead of ingesting everything first (no build-events or analyze "
        "step needed)",
    )
    vid.add_argument(
        "--max-carry-seconds",
        type=float,
        default=DEFAULT_MAX_CARRY_SECONDS,
        metavar="SECONDS",
        help="never carry one observation over similar frames longer than this: ask the provider "
        "again (forced refresh). 0 = no bound. Default: the adaptive sampler's max_skip_seconds",
    )
    vid.add_argument(
        "--call-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="with --live: treat a provider call that takes longer than this as a failed frame",
    )
    vid.add_argument(
        "--fail-fast",
        action="store_true",
        help="with --live: stop at the first provider failure instead of recording it and "
        "continuing (what was already committed stays)",
    )
    vid.add_argument("--verbose", action="store_true", help="list every sample candidate")

    one = sub.add_parser(
        "describe-frame",
        help="API smoke test: describe ONE video frame with Gemini (stores nothing)",
    )
    one.add_argument("video", help="path to a local video file")
    one.add_argument("--time", type=float, required=True, metavar="SECONDS")
    one.add_argument("--provider", choices=["gemini"], default="gemini")
    one.add_argument("--model", help="Gemini model (default: GEMINI_MODEL from env/.env)")
    one.add_argument("--max-dimension", type=int, default=1024)
    one.add_argument("--jpeg-quality", type=int, default=85)

    for name, help_text in (
        ("ask", "ask the analyst a question about the stored evidence (cites evidence ids)"),
        ("summarize", "summarize the stored run, then optionally ask follow-ups"),
    ):
        an = sub.add_parser(name, help=help_text)
        an.add_argument("--db", required=True, help="SQLite database path (must exist)")
        if name == "ask":
            an.add_argument("question", nargs="?", help="the question, in quotes")
            an.add_argument(
                "--interactive", "-i", action="store_true", help="keep asking follow-ups"
            )
        else:
            an.add_argument("--interactive", "-i", action="store_true", help="ask follow-ups")
        an.add_argument(
            "--follow-up", action="append", metavar="QUESTION", help="a follow-up (repeatable)"
        )
        an.add_argument(
            "--provider",
            choices=["offline", "gemini", "planner", "replay"],
            default="offline",
            help="offline = deterministic template analyst (default); planner = the real "
            "tool-calling loop with a scripted model (no LLM, no key); replay = recorded model "
            "turns (--analyst-cassette); gemini = live LLM (needs a key)",
        )
        an.add_argument(
            "--analyst-cassette", help="recorded model turns to replay (--provider replay)"
        )
        an.add_argument(
            "--record-cassette", help="with --provider gemini: also record the model's turns here"
        )
        an.add_argument(
            "--trace", action="store_true", help="show every tool call behind an answer"
        )
        an.add_argument("--offline", action="store_true", help="same as --provider offline")
        an.add_argument("--model", help="analyst model (default: ANALYST_MODEL from env/.env)")

    ev = sub.add_parser(
        "evaluate", help="run the deterministic evaluation/ablation harness (no API, no network)"
    )
    ev.add_argument("--output", default="output/eval", help="directory for results.json/csv")
    ev.add_argument("--scale", action="store_true", help="also run the timing benchmark")
    ev.add_argument("--scale-sizes", default="100,1000,10000", help="observation counts")
    ev.add_argument("--repeats", type=int, default=5, help="timing repeats (median reported)")
    ev.add_argument(
        "--noise", action="store_true", help="also run the perception-noise sweep (slower)"
    )
    ev.add_argument("--noise-seeds", type=int, default=3, help="seeds per noise cell")

    demo = sub.add_parser(
        "make-demo-video", help="write a tiny synthetic test video and a matching demo cassette"
    )
    demo.add_argument("--out-dir", default="output/demo")

    cli_run.add_parsers(sub)

    args = parser.parse_args(argv)
    console = console or Console()
    if args.command in cli_run.COMMANDS:
        return cli_run.dispatch(args, console)
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
    if args.command == "alerts":
        return alerts_command(args, console)
    if args.command == "ingest-video":
        return ingest_video_command(args, console)
    if args.command == "ask":
        return ask_command(args, console)
    if args.command == "summarize":
        return summarize_command(args, console)
    if args.command == "evaluate":
        return evaluate_command(args, console)
    if args.command == "describe-frame":
        return describe_frame_command(args, console)
    return make_demo_video_command(args.out_dir, console)


if __name__ == "__main__":
    sys.exit(main())
