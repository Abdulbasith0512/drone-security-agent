"""Wall-clock pacing of a simulated run.

`--speed 5` means one wall second passes for every five simulated seconds. Long quiet stretches
are capped (`max_wait_s`) so a 50-minute patrol at speed 5 does not sit idle for ten minutes
between two frames: what matters for a demo is the ORDER and spacing of events, not literal
idleness. `speed = 0` (or None) disables pacing entirely; tests and evaluations run that way.
"""

import time
from collections.abc import Callable


class Pacer:
    def __init__(
        self,
        speed: float | None,
        max_wait_s: float = 0.6,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if speed is not None and speed < 0:
            raise ValueError("speed must be >= 0 (0 = as fast as possible)")
        if max_wait_s < 0:
            raise ValueError("max_wait_s must be >= 0")
        self.speed = speed or 0.0
        self.max_wait_s = max_wait_s
        self._sleep = sleep
        self.waited_s = 0.0

    @property
    def paced(self) -> bool:
        return self.speed > 0

    def wait(self, simulated_seconds: float) -> None:
        """Sleep for the wall time that `simulated_seconds` of patrol represents."""
        if not self.paced or simulated_seconds <= 0:
            return
        seconds = min(simulated_seconds / self.speed, self.max_wait_s)
        self.waited_s += seconds
        self._sleep(seconds)
