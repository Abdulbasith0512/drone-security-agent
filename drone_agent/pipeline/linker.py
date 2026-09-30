"""Coarse vehicle entity memory: group separate vehicle events that look alike.

This is deliberately NOT re-identification. Two events are linked only when their normalised coarse
attributes (by default colour and body type) are equal, so the honest reading of an Entity is
"a similar vehicle appeared N times", not "the same vehicle came back". Only an explicit plate,
if the input supplied one on every linked event, upgrades an entity to MatchLevel.CONFIRMED.
People are never linked: separate person events stay separate.

An event may join an existing entity only if ALL of these hold (the hard gates):
1. It starts after the entity's latest event ended (overlapping vehicles are different vehicles).
2. Explicit plates do not conflict (if both sides have plates, at least one must be shared).
3. Either a shared explicit plate, or equal normalised coarse attributes on every required key.
   A missing required attribute means "unknown", which never links (nothing is invented).
Among several candidates the entity seen most recently wins.

Every link also gets a SCORE in [0, 1] with its components, so the answer to "is this the same
vehicle?" is "likely, 0.82" with the reasons, never "confirmed":

    score = attribute_agreement x temporal_factor x zone_factor        (shared plate: fixed score)

* attribute agreement is the mean over the required attributes, 1.0 for an exact match and
  `normalized_match_credit` when they are only equal after normalisation ("dark blue" ~ "blue");
* the temporal factor decays from 1 toward `temporal_floor` as the gap between sightings grows;
* the zone factor is 1.0 in the same zone and `zone_change_factor` otherwise.

The score is a transparent heuristic, not a calibrated probability. Candidates are found through
indexes on the coarse signature and on plates, so linking is linear in the number of events
(the earlier all-pairs scan was quadratic).
"""

import math
from dataclasses import dataclass, field

from drone_agent.models import Entity, Event, MatchLevel, ObjectType
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.normalize import normalize_attribute, normalize_plate


@dataclass
class _Group:
    entity_id: str
    coarse: dict[str, str]  # empty if the founding event lacked required attributes
    order: int  # creation order: ties between candidates go to the earliest-created group
    events: list[Event] = field(default_factory=list)
    plates: set[str] = field(default_factory=set)
    last_seen: object = None  # cached max(last_seen) of the events
    scores: dict[str, float] = field(default_factory=dict)
    components: dict[str, dict[str, float]] = field(default_factory=dict)

    def add(self, event: Event) -> None:
        self.events.append(event)
        if self.last_seen is None or event.last_seen > self.last_seen:  # type: ignore[operator]
            self.last_seen = event.last_seen


class EntityLinker:
    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()

    # ---- signatures ------------------------------------------------------------------------
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

    def _key(self, coarse: dict[str, str]) -> tuple[str, ...] | None:
        """The bucket key of a coarse signature (None when it is incomplete)."""
        if not coarse:
            return None
        return tuple(coarse[k] for k in self.config.vehicle_matching.required_attributes)

    # ---- linking ---------------------------------------------------------------------------
    def link(self, events: list[Event]) -> tuple[list[Event], list[Entity]]:
        """Returns (events with entity_id set for vehicles, entities). Input order is irrelevant."""
        groups: list[_Group] = []
        by_coarse: dict[tuple[str, ...], list[_Group]] = {}
        by_plate: dict[str, list[_Group]] = {}
        assignment: dict[str, str] = {}

        for ev in sorted(events, key=lambda e: (e.first_seen, e.event_id)):
            if ev.object_type is not ObjectType.VEHICLE:
                continue  # people and other objects are never identity-linked
            coarse, plate = self._coarse(ev), self._plate(ev)
            key = self._key(coarse)
            best: tuple[_Group, float, dict[str, float]] | None = None
            for group in self._candidates(key, plate, by_coarse, by_plate):
                if not self._matches(group, ev, coarse, plate):
                    continue
                score, parts = self._score(group, ev, coarse, plate)
                if score < self.config.vehicle_matching.min_link_score:
                    continue
                if best is None or group.last_seen > best[0].last_seen:  # type: ignore[operator]
                    best = (group, score, parts)  # ties keep the earliest-created group
            if best is not None:
                group, score, parts = best
                group.scores[ev.event_id] = score
                group.components[ev.event_id] = parts
            else:
                group = _Group(
                    entity_id=f"ENT-V-{len(groups) + 1:03d}", coarse=coarse, order=len(groups)
                )
                groups.append(group)
                if key is not None:
                    by_coarse.setdefault(key, []).append(group)
            group.add(ev)
            if plate and plate not in group.plates:
                group.plates.add(plate)
                by_plate.setdefault(plate, []).append(group)
            assignment[ev.event_id] = group.entity_id

        linked = [ev.model_copy(update={"entity_id": assignment.get(ev.event_id)}) for ev in events]
        return linked, [self._to_entity(g) for g in groups]

    @staticmethod
    def _candidates(
        key: tuple[str, ...] | None,
        plate: str | None,
        by_coarse: dict[tuple[str, ...], list[_Group]],
        by_plate: dict[str, list[_Group]],
    ) -> list[_Group]:
        """Groups that could possibly match, in creation order (index lookups, no full scan)."""
        found: dict[int, _Group] = {}
        if plate is not None:
            for group in by_plate.get(plate, ()):  # groups that already hold this plate
                found[group.order] = group
        if key is not None:
            for group in by_coarse.get(key, ()):
                # a plated event never joins a group that has a (different) plate
                if plate is None or not group.plates:
                    found[group.order] = group
        return [found[order] for order in sorted(found)]

    @staticmethod
    def _matches(group: _Group, ev: Event, coarse: dict[str, str], plate: str | None) -> bool:
        if ev.first_seen < group.last_seen:  # type: ignore[operator]
            return False
        if plate and group.plates:
            return plate in group.plates  # shared plate links, conflicting plates never do
        return bool(coarse) and coarse == group.coarse

    # ---- scoring ---------------------------------------------------------------------------
    def _score(
        self, group: _Group, ev: Event, coarse: dict[str, str], plate: str | None
    ) -> tuple[float, dict[str, float]]:
        cfg = self.config.vehicle_matching
        previous = max(group.events, key=lambda e: (e.last_seen, e.event_id))
        gap = max(0.0, (ev.first_seen - previous.last_seen).total_seconds())
        temporal = cfg.temporal_floor + (1 - cfg.temporal_floor) * math.exp(
            -gap / cfg.temporal_tau_seconds
        )
        zone = 1.0 if ev.zone == previous.zone else cfg.zone_change_factor
        parts: dict[str, float] = {"temporal": round(temporal, 4), "zone": round(zone, 4)}
        if plate and plate in group.plates:
            parts["plate"] = 1.0
            return cfg.plate_confirmed_score, parts
        credits = []
        for name in cfg.required_attributes:
            raw_new, raw_old = ev.attributes.get(name, ""), previous.attributes.get(name, "")
            exact = raw_new.strip().lower() == raw_old.strip().lower() and bool(raw_new)
            credit = 1.0 if exact else cfg.normalized_match_credit
            parts[name] = credit
            credits.append(credit)
        agreement = sum(credits) / len(credits) if credits else cfg.normalized_match_credit
        return round(agreement * temporal * zone, 4), parts

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
            last_seen=group.last_seen,  # type: ignore[arg-type]
            event_ids=[e.event_id for e in group.events],
            match_score=min(group.scores.values()) if group.scores else None,
            link_scores=dict(group.scores),
            link_components={k: dict(v) for k, v in group.components.items()},
        )
