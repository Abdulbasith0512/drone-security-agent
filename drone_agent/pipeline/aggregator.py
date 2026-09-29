"""Deterministic temporal association of detections into Events.

This is NOT computer-vision tracking. It never looks at pixels or identity; it groups detections
that agree on object type, zone and coarse stable attributes and follow each other closely in time.

Algorithm (per observation, in chronological order):
1. Close open events that this observation rules out:
   - the drone is now observing a different zone           -> ZONE_CHANGE
   - the last matching detection is older than max_gap      -> GAP
2. Match each detection to an open event of the same type/zone whose stable attributes are
   compatible (the most recently seen one wins); otherwise open a new event. Incompatible
   attributes never close an existing event; they start a separate one, so two different people
   standing together stay two events.
3. Every open event in this observation's zone that was NOT matched has missed a frame; more than
   max_missed_frames consecutive misses close it                -> MISSED_FRAMES
Events still open when the input ends are closed by `flush()`     -> SCENARIO_END
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime

from drone_agent.models import CloseReason, DetectedObject, Event, ObjectType, Observation, Zone
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.normalize import normalize_attribute

_PREFIX = {ObjectType.PERSON: "P", ObjectType.VEHICLE: "V", ObjectType.OTHER: "O"}


def _modal(counts: dict[str, int]) -> str:
    """Most common key; ties go to the one seen first (dict preserves insertion order)."""
    return max(counts.items(), key=lambda kv: kv[1])[0]


@dataclass
class _OpenEvent:
    event_id: str
    object_type: ObjectType
    zone: Zone
    first_seen: datetime
    last_seen: datetime
    description: str
    observation_ids: list[str] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    raw: dict[str, dict[str, int]] = field(default_factory=dict)  # key -> raw value -> count
    norm: dict[str, dict[str, int]] = field(default_factory=dict)  # key -> normalised -> count
    missed: int = 0

    def add(self, obs: Observation, det: DetectedObject, cfg: EventConfig) -> None:
        self.last_seen = obs.timestamp
        self.observation_ids.append(obs.observation_id)
        self.frame_ids.append(obs.frame_id)
        if det.activity and det.activity not in self.activities:
            self.activities.append(det.activity)
        for key, value in det.attributes.items():
            self.raw.setdefault(key, {})
            self.raw[key][value] = self.raw[key].get(value, 0) + 1
            n = normalize_attribute(key, value, cfg.normalization)
            self.norm.setdefault(key, {})
            self.norm[key][n] = self.norm[key].get(n, 0) + 1
        self.missed = 0

    def compatible(self, det: DetectedObject, cfg: EventConfig) -> bool:
        for key in cfg.aggregation.stable_attributes.get(self.object_type, []):
            if key in det.attributes and key in self.norm:
                value = normalize_attribute(key, det.attributes[key], cfg.normalization)
                if value != _modal(self.norm[key]):
                    return False
        return True

    def to_event(self, reason: CloseReason) -> Event:
        return Event(
            event_id=self.event_id,
            object_type=self.object_type,
            zone=self.zone,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            attributes={k: _modal(v) for k, v in self.raw.items()},
            activities=self.activities,
            description=self.description,
            observation_ids=self.observation_ids,
            frame_ids=self.frame_ids,
            close_reason=reason,
        )


class EventAggregator:
    """Streaming aggregator: feed observations with `process`, finish with `flush`.

    Event ids (`EVT-P-001`, `EVT-V-001`, ...) are assigned when an event opens, so they follow
    first_seen order per object type and do not depend on when events close.
    """

    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()
        self._open: list[_OpenEvent] = []
        self._counters: dict[ObjectType, int] = {}
        self._last_ts: datetime | None = None

    def process(self, obs: Observation) -> list[Event]:
        """Consume one observation; returns the events it closed."""
        if self._last_ts is not None and obs.timestamp < self._last_ts:
            raise ValueError(
                f"observations must be chronological: frame {obs.frame_id} is earlier than "
                f"the previous one"
            )
        self._last_ts = obs.timestamp
        cfg = self.config.aggregation
        closed: list[Event] = []

        still_open: list[_OpenEvent] = []
        for ev in self._open:
            if ev.zone != obs.zone:
                closed.append(ev.to_event(CloseReason.ZONE_CHANGE))
            elif (obs.timestamp - ev.last_seen).total_seconds() > cfg.max_gap_seconds:
                closed.append(ev.to_event(CloseReason.GAP))
            else:
                still_open.append(ev)
        self._open = still_open

        matched: set[str] = set()
        for det in obs.detected_objects:
            candidates = [
                ev
                for ev in self._open
                if ev.object_type is det.object_type
                and ev.event_id not in matched
                and ev.compatible(det, self.config)
            ]
            if candidates:
                # most recently seen wins; max() keeps the earliest-opened among equals
                target = max(candidates, key=lambda e: e.last_seen)
            else:
                target = self._open_event(obs, det)
            target.add(obs, det, self.config)
            matched.add(target.event_id)

        still_open = []
        for ev in self._open:
            if ev.event_id not in matched:
                ev.missed += 1
                if ev.missed > cfg.max_missed_frames:
                    closed.append(ev.to_event(CloseReason.MISSED_FRAMES))
                    continue
            still_open.append(ev)
        self._open = still_open
        return closed

    def flush(self) -> list[Event]:
        """Close every open event (end of input)."""
        closed = [ev.to_event(CloseReason.SCENARIO_END) for ev in self._open]
        self._open = []
        return closed

    def _open_event(self, obs: Observation, det: DetectedObject) -> _OpenEvent:
        n = self._counters.get(det.object_type, 0) + 1
        self._counters[det.object_type] = n
        ev = _OpenEvent(
            event_id=f"EVT-{_PREFIX[det.object_type]}-{n:03d}",
            object_type=det.object_type,
            zone=obs.zone,
            first_seen=obs.timestamp,
            last_seen=obs.timestamp,
            description=obs.description,
        )
        self._open.append(ev)
        return ev


def aggregate_observations(
    observations: Iterable[Observation], config: EventConfig | None = None
) -> list[Event]:
    """Aggregate a whole chronological sequence; events are returned in id (first_seen) order."""
    agg = EventAggregator(config)
    events: list[Event] = []
    for obs in observations:
        events.extend(agg.process(obs))
    events.extend(agg.flush())
    return sorted(events, key=lambda e: (e.first_seen, e.event_id))
