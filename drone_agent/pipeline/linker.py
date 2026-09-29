"""Coarse vehicle entity memory: group separate vehicle events that look alike.

This is deliberately NOT re-identification. Two events are linked only when their normalised coarse
attributes (by default colour and body type) are equal, so the honest reading of an Entity is
"a similar vehicle appeared N times", not "the same vehicle came back". Only an explicit plate,
if the input supplied one on every linked event, upgrades an entity to MatchLevel.CONFIRMED.
People are never linked: separate person events stay separate.

An event may join an existing entity only if ALL of these hold:
1. It starts after the entity's latest event ended (overlapping vehicles are different vehicles).
2. Explicit plates do not conflict (if both sides have plates, at least one must be shared).
3. Either a shared explicit plate, or equal normalised coarse attributes on every required key.
   A missing required attribute means "unknown", which never links (nothing is invented).
Among several candidates the entity seen most recently wins.
"""

from dataclasses import dataclass, field

from drone_agent.models import Entity, Event, MatchLevel, ObjectType
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.normalize import normalize_attribute, normalize_plate


@dataclass
class _Group:
    entity_id: str
    coarse: dict[str, str]  # empty if the founding event lacked required attributes
    events: list[Event] = field(default_factory=list)
    plates: set[str] = field(default_factory=set)

    @property
    def last_seen(self):
        return max(e.last_seen for e in self.events)


class EntityLinker:
    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()

    def _coarse(self, event: Event) -> dict[str, str]:
        """Normalised required attributes, or {} if any is missing."""
        cfg = self.config
        out: dict[str, str] = {}
        for key in cfg.vehicle_matching.required_attributes:
            value = event.attributes.get(key)
            if not value:
                return {}
            out[key] = normalize_attribute(key, value, cfg.normalization)
        return out

    def _plate(self, event: Event) -> str | None:
        raw = event.attributes.get(self.config.vehicle_matching.plate_attribute)
        return normalize_plate(raw) if raw else None

    def link(self, events: list[Event]) -> tuple[list[Event], list[Entity]]:
        """Returns (events with entity_id set for vehicles, entities). Input order is irrelevant."""
        groups: list[_Group] = []
        assignment: dict[str, str] = {}
        for ev in sorted(events, key=lambda e: (e.first_seen, e.event_id)):
            if ev.object_type is not ObjectType.VEHICLE:
                continue  # people and other objects are never identity-linked
            coarse, plate = self._coarse(ev), self._plate(ev)
            candidates = [g for g in groups if self._matches(g, ev, coarse, plate)]
            if candidates:
                group = max(candidates, key=lambda g: g.last_seen)
            else:
                group = _Group(entity_id=f"ENT-V-{len(groups) + 1:03d}", coarse=coarse)
                groups.append(group)
            group.events.append(ev)
            if plate:
                group.plates.add(plate)
            assignment[ev.event_id] = group.entity_id

        linked = [ev.model_copy(update={"entity_id": assignment.get(ev.event_id)}) for ev in events]
        return linked, [self._to_entity(g) for g in groups]

    @staticmethod
    def _matches(group: _Group, ev: Event, coarse: dict[str, str], plate: str | None) -> bool:
        if ev.first_seen < group.last_seen:
            return False
        if plate and group.plates:
            return plate in group.plates  # shared plate links, conflicting plates never do
        return bool(coarse) and coarse == group.coarse

    def _to_entity(self, group: _Group) -> Entity:
        plates = [self._plate(e) for e in group.events]
        confirmed = all(plates) and len(set(plates)) == 1
        attributes = dict(group.coarse)
        if confirmed:
            attributes[self.config.vehicle_matching.plate_attribute] = plates[0] or ""
        return Entity(
            entity_id=group.entity_id,
            entity_type=ObjectType.VEHICLE,
            attributes=attributes,
            match_level=MatchLevel.CONFIRMED if confirmed else MatchLevel.COARSE,
            first_seen=min(e.first_seen for e in group.events),
            last_seen=group.last_seen,
            event_ids=[e.event_id for e in group.events],
        )
