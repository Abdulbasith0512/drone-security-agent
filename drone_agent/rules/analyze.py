"""stored events + entities + telemetry -> RuleEngine -> alert lifecycle -> persistence."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.rules.engine import RuleEngine
from drone_agent.rules.lifecycle import AlertManager
from drone_agent.store import Store


@dataclass(frozen=True)
class AnalysisSummary:
    events_analyzed: int
    entities_analyzed: int
    alerts: int
    by_rule: dict[str, int]
    by_severity: dict[str, int]
    replaced_alerts: int
    telemetry_samples: int = 0
    by_status: dict[str, int] | None = None


def data_end(store: Store) -> datetime | None:
    """The latest moment the stored data covers (observations, events or telemetry)."""
    latest: list[datetime] = []
    span = store.observation_span()
    if span:
        latest.append(span[1])
    telemetry = store.list_telemetry_rows()
    if telemetry:
        latest.append(telemetry[-1][1].timestamp)
    events = store.query_events()
    if events:
        latest.append(max(e.last_seen for e in events))
    return max(latest) if latest else None


def analyze(store: Store, config: SecurityRulesConfig | None = None) -> AnalysisSummary:
    """Run the deterministic rules over the stored data and persist the alerts.

    Idempotent by replacement: alerts are a pure function of (events, entities, telemetry, rules
    config), so the alerts table is rewritten in ONE transaction. Unchanged inputs give identical
    alerts (deterministic ids); a changed rules config simply yields the alerts of the new
    config. An alert that survives keeps its lifecycle (operator acknowledgements included).
    No LLM is called.
    """
    events = store.query_events()
    entities = store.list_entities()
    telemetry = store.list_telemetry()
    now = data_end(store)
    snapshots = RuleEngine(config).run(events, entities, telemetry, now=now)
    kept = {a.alert_id for a in snapshots}
    manager = AlertManager(a for a in store.query_alerts() if a.alert_id in kept)
    if now is not None:
        manager.reconcile(snapshots, now, live=False)
    alerts = manager.alerts
    replaced = store.replace_alerts(alerts)
    return AnalysisSummary(
        events_analyzed=len(events),
        entities_analyzed=len(entities),
        alerts=len(alerts),
        by_rule=dict(sorted(Counter(a.rule_id for a in alerts).items())),
        by_severity=dict(
            sorted(Counter(a.severity.value for a in alerts).items(), key=lambda kv: kv[0])
        ),
        replaced_alerts=replaced,
        telemetry_samples=len(telemetry),
        by_status=dict(sorted(Counter(a.status.value for a in alerts).items())),
    )
