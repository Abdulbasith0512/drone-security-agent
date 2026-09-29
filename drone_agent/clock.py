"""Deterministic simulation clock: scenario offsets -> timezone-aware timestamps.

Nothing here reads the wall clock, so time-of-day and midnight-boundary behaviour is reproducible.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SimulationClock:
    start: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("SimulationClock start must be timezone-aware")

    def at(self, offset_seconds: float) -> datetime:
        """Timestamp `offset_seconds` after the scenario start (same UTC offset as start)."""
        return self.start + timedelta(seconds=offset_seconds)

    def offset(self, timestamp: datetime) -> float:
        """Seconds between the scenario start and `timestamp`."""
        return (timestamp - self.start).total_seconds()
