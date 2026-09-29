"""Persist a simulated scenario run into the store."""

from dataclasses import dataclass

from drone_agent.simulate import SimulationResult
from drone_agent.store import Store


@dataclass(frozen=True)
class IngestSummary:
    scenario: str
    frames: int
    observations: int
    objects: int
    telemetry_samples: int
    stale_associations: int


def ingest_result(store: Store, result: SimulationResult) -> IngestSummary:
    """Store telemetry then observations in ONE transaction: all rows are written or none.

    Re-ingesting data that is already stored raises DuplicateRecordError and changes nothing.
    """
    with store.transaction():
        for sample in result.telemetry:
            store.insert_telemetry(sample)
        for obs in result.observations:
            store.insert_observation(obs)
    return IngestSummary(
        scenario=result.scenario.name,
        frames=len(result.observations),
        observations=len(result.observations),
        objects=sum(len(o.detected_objects) for o in result.observations),
        telemetry_samples=len(result.telemetry),
        stale_associations=sum(1 for o in result.observations if o.telemetry and o.telemetry.stale),
    )
