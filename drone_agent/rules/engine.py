"""RuleEngine: run the deterministic rules over events, entities and telemetry.

Overlap policy (simple on purpose): rules are independent. Different rules MAY alert on the same
event because they describe different risks (e.g. a night-time loiterer yields both a loitering
and an after-hours alert; group by `event_ids`). Alert identity is `ALR-<RULE>-<event or entity>`
(or the episode start for telemetry rules), so the same rule never alerts twice on the same
subject and re-running on unchanged facts gives the same alerts. No rule suppresses another, and
nothing here calls an LLM.

The output is a list of *snapshots*: what every alert should look like given the data so far.
`rules.lifecycle.AlertManager` turns snapshots into a lifecycle (raise, escalate, resolve).
"""

from collections.abc import Sequence
from datetime import datetime

from drone_agent.models import Alert, Entity, Event, Telemetry
from drone_agent.rules import (
    after_hours,
    loitering,
    repeat_vehicle,
    restricted_zone,
    safety,
)
from drone_agent.rules.config import SecurityRulesConfig

_RULE_ORDER = {
    loitering.RULE_ID: 0,
    after_hours.RULE_ID: 1,
    repeat_vehicle.RULE_ID: 2,
    restricted_zone.RULE_ID: 3,
    safety.RULE_LOW_BATTERY: 4,
    safety.RULE_SIGNAL_LOSS: 5,
    safety.RULE_ALTITUDE: 6,
    safety.RULE_GEOFENCE: 7,
}


class RuleEngine:
    def __init__(self, config: SecurityRulesConfig | None = None) -> None:
        self.config = config or SecurityRulesConfig()

    def run(
        self,
        events: Sequence[Event],
        entities: Sequence[Entity],
        telemetry: Sequence[Telemetry] = (),
        *,
        now: datetime | None = None,
    ) -> list[Alert]:
        """Alerts for these events/entities/telemetry, ordered by (triggered_at, rule, id).

        `events` may include OPEN events (a live snapshot). `now` is the end of the data; it lets
        the signal rule report a telemetry outage that is still going on.
        """
        return self._ordered([*self.run_events(events, entities), *self.run_safety(telemetry, now)])

    def run_events(self, events: Sequence[Event], entities: Sequence[Entity]) -> list[Alert]:
        """Only the camera rules (they depend on events and entities, never on telemetry)."""
        cfg = self.config
        return self._ordered(
            [
                *loitering.evaluate(events, cfg.loitering),
                *after_hours.evaluate(events, cfg.after_hours_presence),
                *repeat_vehicle.evaluate(events, entities, cfg.repeat_vehicle),
                *restricted_zone.evaluate(events, cfg.restricted_zone_entry),
            ]
        )

    def run_safety(
        self, telemetry: Sequence[Telemetry], now: datetime | None = None
    ) -> list[Alert]:
        """Only the drone-safety rules (they depend on telemetry and the clock, never on events)."""
        cfg = self.config
        return self._ordered(
            [
                *safety.low_battery(telemetry, cfg.low_battery),
                *safety.signal_loss(telemetry, cfg.signal_loss, now),
                *safety.altitude_anomaly(telemetry, cfg.altitude_anomaly),
                *safety.geofence_violation(telemetry, cfg.geofence_violation),
            ]
        )

    @staticmethod
    def _ordered(alerts: list[Alert]) -> list[Alert]:
        unique = {a.alert_id: a for a in alerts}  # identical (rule, subject) collapses to one
        return sorted(
            unique.values(),
            key=lambda a: (a.triggered_at, _RULE_ORDER[a.rule_id], a.alert_id),
        )
