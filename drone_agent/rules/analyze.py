"""stored events + entities -> RuleEngine -> alerts -> persistence."""

from collections import Counter
from dataclasses import dataclass

from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.rules.engine import RuleEngine
from drone_agent.store import Store


@dataclass(frozen=True)
class AnalysisSummary:
    events_analyzed: int
    entities_analyzed: int
    alerts: int
    by_rule: dict[str, int]
    by_severity: dict[str, int]
    replaced_alerts: int


def analyze(store: Store, config: SecurityRulesConfig | None = None) -> AnalysisSummary:
    """Run the deterministic rules over the stored events/entities and persist the alerts.

    Idempotent by replacement: alerts are a pure function of (events, entities, rules config), so
    the alerts table is rewritten in ONE transaction. Unchanged inputs give identical alerts
    (deterministic ids); a changed rules config simply yields the alerts of the new config.
    No LLM is called.
    """
    events = store.query_events()
    entities = store.list_entities()
    alerts = RuleEngine(config).run(events, entities)
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
    )
