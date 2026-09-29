"""Run a scenario through telemetry simulation, a VisionProvider and telemetry sync."""

from dataclasses import dataclass

from drone_agent.models import Observation, Telemetry
from drone_agent.scenario import Scenario
from drone_agent.telemetry import TelemetrySynchronizer, simulate_telemetry
from drone_agent.vision import FrameInput, ScriptedProvider, VisionProvider


@dataclass(frozen=True)
class SimulationResult:
    scenario: Scenario
    telemetry: list[Telemetry]
    observations: list[Observation]


def run_scenario(scenario: Scenario, provider: VisionProvider | None = None) -> SimulationResult:
    """Deterministically produce telemetry and telemetry-synchronised observations."""
    provider = provider or ScriptedProvider(scenario)
    clock = scenario.clock
    telemetry = simulate_telemetry(clock, scenario.telemetry)
    sync = TelemetrySynchronizer(telemetry, scenario.telemetry.stale_after_s)
    observations = [
        sync.attach(
            provider.analyze(
                FrameInput(frame_id=f.frame_id, timestamp=clock.at(f.timestamp_offset))
            )
        )
        for f in scenario.frames
    ]
    return SimulationResult(scenario, telemetry, observations)
