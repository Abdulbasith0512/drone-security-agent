"""What a running patrol announces: one small record per thing that happened."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class RunEventKind(StrEnum):
    STARTED = "started"
    TELEMETRY = "telemetry"  # a scheduled delivery: data["delivered"] says whether one arrived
    FRAME = "frame"  # a camera frame and what the sampler decided about it
    OBSERVATION = "observation"  # a stored observation (analysed or carried)
    EVENT = "event"  # an event opened, grew or closed
    ALERT = "alert"  # an alert was raised, escalated, acknowledged or resolved
    FINISHED = "finished"


@dataclass(frozen=True)
class RunEvent:
    kind: RunEventKind
    at: datetime  # simulated time
    offset_s: float
    data: dict[str, Any] = field(default_factory=dict)
    provenance: str = ""  # SIMULATED / SCRIPTED / REPLAYED / LIVE, so a view can label honestly

    def to_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "at": self.at.isoformat(),
            "offset_s": self.offset_s,
            "provenance": self.provenance,
            **self.data,
        }
