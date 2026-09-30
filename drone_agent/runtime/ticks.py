"""The simulated patrol as an ordered stream of ticks (telemetry deliveries and camera frames).

A live drone does not hand over a finished file: telemetry arrives on a schedule and frames arrive
between it. This module lays a scenario out on the simulated clock so the engine can process it
progressively, one tick at a time, exactly as a live feed would arrive.

Telemetry ticks are the *scheduled* deliveries (every `sample_interval_s`). A dropout removes the
sample but not the schedule, so the engine still gets a tick at that moment, a "watchdog" tick with
nothing delivered, and can notice the silence promptly instead of only when data resumes.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from drone_agent.models import Telemetry
from drone_agent.scenario import Scenario, ScenarioFrame
from drone_agent.telemetry import simulate_telemetry


@dataclass(frozen=True)
class Tick:
    kind: Literal["telemetry", "frame"]
    offset_s: float  # seconds after the scenario start (simulated)
    at: datetime
    sample: Telemetry | None = None  # telemetry tick: the delivered sample; None = nothing arrived
    frame: ScenarioFrame | None = None  # frame tick


def build_ticks(scenario: Scenario) -> list[Tick]:
    """Every telemetry delivery slot and every frame, in simulated-time order.

    At the same instant telemetry comes first, so a frame sees the sample stamped at its own
    time (the same nearest-prior rule the batch path uses).
    """
    clock = scenario.clock
    delivered = {s.timestamp: s for s in simulate_telemetry(clock, scenario.telemetry)}
    interval = scenario.telemetry.sample_interval_s
    ticks = [
        Tick("telemetry", float(offset), clock.at(offset), sample=delivered.get(clock.at(offset)))
        for offset in range(0, scenario.telemetry.duration_s + 1, interval)
    ]
    ticks += [
        Tick("frame", frame.timestamp_offset, clock.at(frame.timestamp_offset), frame=frame)
        for frame in scenario.frames
    ]
    return sorted(ticks, key=lambda t: (t.offset_s, 0 if t.kind == "telemetry" else 1))
