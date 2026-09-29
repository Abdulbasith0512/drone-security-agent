"""stored observations -> events -> entity memory -> persistence."""

from dataclasses import dataclass

from drone_agent.models import ObjectType
from drone_agent.pipeline.aggregator import aggregate_observations
from drone_agent.pipeline.config import EventConfig
from drone_agent.pipeline.linker import EntityLinker
from drone_agent.store import Store


@dataclass(frozen=True)
class BuildSummary:
    observations: int
    detections: int
    events: int
    person_events: int
    vehicle_events: int
    other_events: int
    entities: int
    multi_appearance_entities: int
    replaced_events: int
    replaced_entities: int
    discarded_alerts: int

    @property
    def compression_ratio(self) -> float | None:
        """Object detections per event (None when there are no events)."""
        return self.detections / self.events if self.events else None


def build_events(store: Store, config: EventConfig | None = None) -> BuildSummary:
    """Derive events and entities from the stored observations and persist them.

    Idempotent by replacement: the derived tables are cleared and rewritten in ONE transaction,
    so running it again (with the same or a changed config) never duplicates evidence.
    """
    config = config or EventConfig()
    observations = store.query_observations()  # chronological
    events = aggregate_observations(observations, config)
    events, entities = EntityLinker(config).link(events)
    replaced_events, replaced_entities, discarded_alerts = store.replace_derived(events, entities)

    def count(kind: ObjectType) -> int:
        return sum(1 for e in events if e.object_type is kind)

    return BuildSummary(
        observations=len(observations),
        detections=sum(len(o.detected_objects) for o in observations),
        events=len(events),
        person_events=count(ObjectType.PERSON),
        vehicle_events=count(ObjectType.VEHICLE),
        other_events=count(ObjectType.OTHER),
        entities=len(entities),
        multi_appearance_entities=sum(1 for e in entities if e.appearance_count > 1),
        replaced_events=replaced_events,
        replaced_entities=replaced_entities,
        discarded_alerts=discarded_alerts,
    )
