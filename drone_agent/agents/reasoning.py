"""Security reasoning agent: temporal context + deterministic rules + alert lifecycle.

This is the decision-maker, and it is deliberately not a language model. "Reasoning" here means
combining what the drone SAW over time (events, entities) with what the drone FELT (telemetry) and
applying explicit rules with thresholds a human can read and change. Every conclusion carries its
evidence chain (alert -> rule facts -> events -> observations -> frames -> telemetry) and a
recommended action drawn from a fixed table. Nothing here can be talked into a different answer.

A language model may later *explain* an alert (the analyst does, with citations), but never
creates, suppresses, escalates or resolves one.
"""

from collections.abc import Sequence
from datetime import datetime

from drone_agent.models import Alert, Entity, Event, Telemetry
from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.rules.engine import RuleEngine
from drone_agent.rules.lifecycle import AlertChange, AlertManager


class SecurityReasoner:
    def __init__(
        self, rules: SecurityRulesConfig | None = None, manager: AlertManager | None = None
    ) -> None:
        self.engine = RuleEngine(rules)
        self.manager = manager or AlertManager()

    def assess(
        self,
        events: Sequence[Event],
        entities: Sequence[Entity],
        telemetry: Sequence[Telemetry],
        now: datetime,
        *,
        camera: bool = True,
        safety: bool = True,
    ) -> list[AlertChange]:
        """Evaluate the rules on the data so far and apply the result to the alert lifecycle.

        `camera` / `safety` choose which rule family to evaluate: a telemetry tick cannot change a
        camera rule and a frame cannot change a safety rule, so a streaming run evaluates only the
        family a tick could affect. Returns what changed at this step (a new alert, an
        escalation, a resolution, or new evidence) in chronological order; unchanged data gives [].
        """
        snapshots: list[Alert] = []
        if camera:
            snapshots += self.engine.run_events(events, entities)
        if safety:
            snapshots += self.engine.run_safety(telemetry, now)
        return self.manager.reconcile(snapshots, now, live=True)

    @property
    def alerts(self) -> list[Alert]:
        return self.manager.alerts
