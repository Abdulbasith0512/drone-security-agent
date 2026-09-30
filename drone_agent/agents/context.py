"""Temporal context engine: observations -> events -> entities, incrementally.

The batch pipeline (`build_events`) aggregates a finished run. A live run needs the same answer
*while it happens*: an event that is still being extended (a person standing at the gate) must be
visible, and countable by the rules, before it closes. This engine wraps the very same
`EventAggregator` and `EntityLinker`, and after each observation exposes

* every closed event,
* the OPEN events as snapshots (same ids, growing evidence),
* the vehicle entities linked over all of them,
* which events changed at this step (so the store rewrites only those).

Memory is therefore layered exactly as the design says:
    FRAME -> OBSERVATION -> EVENT -> ENTITY -> (rules decide) -> ALERT.

Determinism: feeding the same observations gives the same events and entities as the batch path
(the final `flush` produces the identical result), which the test-suite checks directly.
"""

from dataclasses import dataclass, field

from drone_agent.models import Entity, Event, Observation
from drone_agent.pipeline.aggregator import EventAggregator
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.linker import EntityLinker


@dataclass
class ContextUpdate:
    closed: list[Event] = field(default_factory=list)  # closed by THIS step
    events: list[Event] = field(default_factory=list)  # every event now, open ones as snapshots
    entities: list[Entity] = field(default_factory=list)
    changed: list[Event] = field(default_factory=list)  # new or updated at this step

    @property
    def open_events(self) -> list[Event]:
        return [e for e in self.events if e.is_open]


def _fingerprint(ev: Event) -> tuple:
    return (ev.last_seen, ev.close_reason, len(ev.observation_ids), ev.entity_id, ev.confidence)


class ContextEngine:
    def __init__(self, config: EventConfig | None = None) -> None:
        self.config = config or EventConfig()
        self._aggregator = EventAggregator(self.config)
        self._linker = EntityLinker(self.config)
        self._closed: list[Event] = []
        self._seen: dict[str, tuple] = {}
        self._entities: list[Entity] = []

    # ---- feeding ---------------------------------------------------------------------------
    def process(self, obs: Observation) -> ContextUpdate:
        """Consume one observation (chronological order required)."""
        closed = self._aggregator.process(obs)
        self._closed.extend(closed)
        return self._snapshot(closed)

    def flush(self) -> ContextUpdate:
        """The input ended: close every open event (SCENARIO_END)."""
        closed = self._aggregator.flush()
        self._closed.extend(closed)
        return self._snapshot(closed)

    # ---- views -----------------------------------------------------------------------------
    @property
    def entities(self) -> list[Entity]:
        return list(self._entities)

    def _snapshot(self, closed: list[Event]) -> ContextUpdate:
        events = sorted(
            [*self._closed, *self._aggregator.snapshot_open()],
            key=lambda e: (e.first_seen, e.event_id),
        )
        linked, entities = self._linker.link(events)
        self._entities = entities
        changed = []
        for ev in linked:
            fingerprint = _fingerprint(ev)
            if self._seen.get(ev.event_id) != fingerprint:
                self._seen[ev.event_id] = fingerprint
                changed.append(ev)
        return ContextUpdate(closed=closed, events=linked, entities=entities, changed=changed)
