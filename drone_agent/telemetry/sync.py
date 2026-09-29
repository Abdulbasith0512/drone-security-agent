"""Nearest-prior telemetry synchronisation."""

from bisect import bisect_right
from collections.abc import Sequence
from datetime import datetime

from drone_agent.models import Observation, Telemetry, TelemetryAssociation


class TelemetrySynchronizer:
    """Associates a timestamp with the most recent telemetry sample at or before it.

    An association is `stale` when that sample is strictly older than `stale_after_s`
    (an age equal to the threshold is still fresh). A timestamp before the first sample
    has no association (None).
    """

    def __init__(self, samples: Sequence[Telemetry], stale_after_s: float = 30.0) -> None:
        if stale_after_s < 0:
            raise ValueError("stale_after_s must be >= 0")
        self._samples = sorted(samples, key=lambda s: s.timestamp)
        self._times = [s.timestamp for s in self._samples]
        self.stale_after_s = stale_after_s

    def associate(self, timestamp: datetime) -> TelemetryAssociation | None:
        idx = bisect_right(self._times, timestamp)  # number of samples with ts <= timestamp
        if idx == 0:
            return None
        sample = self._samples[idx - 1]
        age = (timestamp - sample.timestamp).total_seconds()
        return TelemetryAssociation(
            telemetry=sample, age_seconds=age, stale=age > self.stale_after_s
        )

    def attach(self, observation: Observation) -> Observation:
        """Copy of `observation` with its telemetry association filled in."""
        return observation.model_copy(update={"telemetry": self.associate(observation.timestamp)})
