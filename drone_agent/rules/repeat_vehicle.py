"""Repeat vehicle: a similar vehicle (coarse Entity) appears in enough separate events.

Entities are coarse attribute matches, so the wording says "a similar <vehicle> was observed in N
separate appearances", never "the same vehicle returned".
"""

from collections.abc import Iterable

from drone_agent.models import Alert, Entity, Event, MatchLevel
from drone_agent.rules.common import alert_id, frame_union, triage_fields
from drone_agent.rules.config import RepeatVehicleRule

RULE_ID = "repeat_vehicle"
CODE = "REPEAT"


def evaluate(
    events: Iterable[Event], entities: Iterable[Entity], cfg: RepeatVehicleRule
) -> list[Alert]:
    """One alert per entity with at least `minimum_appearances` events (not one per event)."""
    if not cfg.enabled:
        return []
    by_id = {ev.event_id: ev for ev in events}
    alerts: list[Alert] = []
    for entity in entities:
        count = entity.appearance_count
        if count < cfg.minimum_appearances:
            continue
        appearances = sorted(
            (by_id[eid] for eid in entity.event_ids), key=lambda e: (e.first_seen, e.event_id)
        )
        # The moment the count reached the minimum: start of the Nth appearance.
        reached = appearances[cfg.minimum_appearances - 1]
        coarse = entity.match_level is MatchLevel.COARSE
        caveat = (
            " This is a coarse attribute match; it does not confirm the same vehicle."
            if coarse
            else ""
        )
        alerts.append(
            Alert(
                alert_id=alert_id(CODE, entity.entity_id),
                rule_id=RULE_ID,
                severity=cfg.severity,
                title=f"Repeat vehicle: {entity.label}",
                message=(
                    f"A {entity.label} was observed in {count} separate appearances "
                    f"(events {', '.join(e.event_id for e in appearances)}).{caveat}"
                ),
                triggered_at=reached.first_seen,
                first_seen=min(e.first_seen for e in appearances),
                last_seen=max(e.last_seen for e in appearances),
                zone=reached.zone,
                event_ids=[e.event_id for e in appearances],
                entity_id=entity.entity_id,
                frame_ids=frame_union(appearances),
                facts={
                    "appearance_count": count,
                    "minimum_appearances": cfg.minimum_appearances,
                    "match_level": entity.match_level.value,
                    "entity_label": entity.label,
                    "zones": sorted({e.zone.value for e in appearances}),
                    "comparison": "appearance_count >= minimum_appearances",
                    **(
                        {"match_score": entity.match_score}
                        if entity.match_score is not None
                        else {}
                    ),
                },
                **triage_fields(
                    RULE_ID, cfg.severity, reached.first_seen, appearances, cfg.recommended_action
                ),
            )
        )
    return alerts
