"""ScriptedProvider: scripted scenario frames -> validated Observations. No network, no API key."""

from drone_agent.models import Observation
from drone_agent.scenario import Scenario, ScenarioFrame
from drone_agent.vision.base import FrameInput


class UnknownFrameError(LookupError):
    """The scenario has no scripted frame with the requested frame_id."""


class ScriptedProvider:
    name = "scripted"

    def __init__(self, scenario: Scenario) -> None:
        self._clock = scenario.clock
        self._frames: dict[int, ScenarioFrame] = {f.frame_id: f for f in scenario.frames}

    def analyze(self, frame: FrameInput) -> Observation:
        scripted = self._frames.get(frame.frame_id)
        if scripted is None:
            raise UnknownFrameError(f"no scripted frame with frame_id={frame.frame_id}")
        expected = self._clock.at(scripted.timestamp_offset)
        if frame.timestamp != expected:
            raise ValueError(
                f"frame {frame.frame_id} timestamp {frame.timestamp.isoformat()} does not match "
                f"scripted time {expected.isoformat()}"
            )
        return Observation(
            observation_id=f"obs-{scripted.frame_id}",
            frame_id=scripted.frame_id,
            timestamp=frame.timestamp,
            source=self.name,
            zone=scripted.zone,
            description=scripted.description,
            detected_objects=scripted.objects,
        )
