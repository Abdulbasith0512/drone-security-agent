"""Recurrence HINTS for people: "was someone dressed like this seen here before?"

People are never identity-linked (the entity memory is vehicles only, by design). But an operator
does ask "was the person at the gate detected earlier?", and the honest answer is a hint, not a
match: earlier person events in the same zone whose described clothing is equal after
normalisation, ranked by how recent they are.

The hint is deliberately weak and labelled as such:
* it needs the SAME zone and equal normalised clothing (no clothing recorded means no hint);
* the score is capped at MAX_SCORE (0.6) so it can never read as a confident match;
* wording downstream must say "similar clothing", never "the same person".
"""

import math
from dataclasses import dataclass

from drone_agent.models import Event, ObjectType
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.normalize import normalize_attribute

MAX_SCORE = 0.6
_TAU_SECONDS = 6 * 3600.0


@dataclass(frozen=True)
class RecurrenceHint:
    event: Event
    score: float  # 0-MAX_SCORE, a heuristic, not a probability
    reason: str


def similar_earlier_person_events(
    target: Event, events: list[Event], config: EventConfig | None = None
) -> list[RecurrenceHint]:
    """Earlier person events in the same zone with equal normalised clothing, best first."""
    if target.object_type is not ObjectType.PERSON:
        return []
    cfg = config or EventConfig()
    clothing = target.attributes.get("clothing")
    if not clothing:
        return []
    wanted = normalize_attribute("clothing", clothing, cfg.normalization)
    hints = []
    for other in events:
        if other.event_id == target.event_id or other.object_type is not ObjectType.PERSON:
            continue
        if other.zone is not target.zone or other.last_seen > target.first_seen:
            continue
        theirs = other.attributes.get("clothing")
        if not theirs or normalize_attribute("clothing", theirs, cfg.normalization) != wanted:
            continue
        gap = (target.first_seen - other.last_seen).total_seconds()
        score = round(MAX_SCORE * (0.7 + 0.3 * math.exp(-gap / _TAU_SECONDS)), 3)
        hints.append(
            RecurrenceHint(
                other,
                score,
                f"same zone ({target.zone.value}) and similar clothing ('{theirs}'), "
                f"{gap / 60:.0f} min earlier",
            )
        )
    return sorted(hints, key=lambda h: (-h.score, h.event.event_id))
