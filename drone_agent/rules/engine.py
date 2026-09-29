"""RuleEngine: run the four rules over events/entities and return one deterministic alert list.

Overlap policy (simple on purpose): rules are independent. Different rules MAY alert on the same
event because they describe different risks (e.g. a night-time loiterer yields both a loitering
and an after-hours alert; group by `event_ids`). Alert identity is `ALR-<RULE>-<event or entity>`,
so the same rule never alerts twice on the same subject and re-running on unchanged facts gives
the same alerts. No rule suppresses another, and nothing here calls an LLM.
"""

from collections.abc import Sequence

from drone_agent.models import Alert, Entity, Event
from drone_agent.rules import after_hours, loitering, repeat_vehicle, restricted_zone
from drone_agent.rules.config import SecurityRulesConfig

_RULE_ORDER = {
    loitering.RULE_ID: 0,
    after_hours.RULE_ID: 1,
    repeat_vehicle.RULE_ID: 2,
    restricted_zone.RULE_ID: 3,
}


class RuleEngine:
    def __init__(self, config: SecurityRulesConfig | None = None) -> None:
        self.config = config or SecurityRulesConfig()

    def run(self, events: Sequence[Event], entities: Sequence[Entity]) -> list[Alert]:
        """Alerts for these events/entities, ordered by (triggered_at, rule order, alert_id)."""
        cfg = self.config
        produced = [
            *loitering.evaluate(events, cfg.loitering),
            *after_hours.evaluate(events, cfg.after_hours_presence),
            *repeat_vehicle.evaluate(events, entities, cfg.repeat_vehicle),
            *restricted_zone.evaluate(events, cfg.restricted_zone_entry),
        ]
        unique = {a.alert_id: a for a in produced}  # identical (rule, subject) collapses to one
        return sorted(
            unique.values(),
            key=lambda a: (a.triggered_at, _RULE_ORDER[a.rule_id], a.alert_id),
        )
