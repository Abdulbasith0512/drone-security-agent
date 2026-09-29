"""Persist observations + telemetry into the store (scripted scenarios and real video alike)."""

from collections.abc import Sequence
from dataclasses import dataclass

from drone_agent.models import Observation, Telemetry
from drone_agent.simulate import SimulationResult
from drone_agent.store import Store


@dataclass(frozen=True)
class IngestSummary:
    scenario: str  # scenario name, or the video file name
    frames: int
    observations: int
    objects: int
    telemetry_samples: int
    stale_associations: int


def ingest_observations(
    store: Store, name: str, telemetry: Sequence[Telemetry], observations: Sequence[Observation]
) -> IngestSummary:
    """Store telemetry then observations in ONE transaction: all rows are written or none.

    Source-agnostic: it only sees `Observation` and `Telemetry`. Re-ingesting data that is already
    stored raises DuplicateRecordError and changes nothing.
    """
    with store.transaction():
        for sample in telemetry:
            store.insert_telemetry(sample)
        for obs in observations:
            store.insert_observation(obs)
    return IngestSummary(
        scenario=name,
        frames=len(observations),
        observations=len(observations),
        objects=sum(len(o.detected_objects) for o in observations),
        telemetry_samples=len(telemetry),
        stale_associations=sum(1 for o in observations if o.telemetry and o.telemetry.stale),
    )


def ingest_result(store: Store, result: SimulationResult) -> IngestSummary:
    """Store a scripted-scenario run (see `ingest_observations`)."""
    return ingest_observations(store, result.scenario.name, result.telemetry, result.observations)
