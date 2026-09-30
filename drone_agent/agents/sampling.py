"""Adaptive VLM sampling: decide, per frame, whether a vision-model call is worth making.

A fixed pixel gate (video/gate.py) asks one question: "did the picture change?". That misses
things a drone operator cares about and wastes calls on things they do not. This sampler combines
four cheap signals, all available BEFORE the model is called:

* scene change   - pixel difference (real video) or, for scripted text frames, a scene-signature
                   distance (a stand-in that is labelled as such in every trace);
* telemetry      - the drone changed zone, moved, climbed, lost battery or its link changed state,
                   so the view has changed even if the pixels look alike;
* event state    - something of interest (an open event) is being tracked, so the model is asked
                   again at least every `watch_interval_s` instead of trusting a stale carry-over;
* staleness      - no model call for `max_skip_seconds` forces a refresh, so a slow change that
                   never crosses the pixel threshold cannot be skipped forever.

Every decision has a reason code. A skipped frame is not lost: the caller carries the last
observation forward and labels it as carried (propagation, not a new inference).

Savings are meaningless without misses, so `SamplerStats` counts what was skipped and by which
rule, and the evaluation reports the call reduction together with the recall of meaningful
changes (eval/sampler_exp.py). Never quote one without the other.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from drone_agent.models import DomainModel, Telemetry, Zone
from drone_agent.telemetry.geo import distance_m


class SamplingReason(StrEnum):
    DISABLED = "sampler_disabled"  # every frame is analysed
    FIRST = "first"  # nothing to compare with yet
    ZONE_CHANGE = "zone_change"  # a new scene: never carry objects across zones
    SCENE_CHANGE = "scene_change"  # pixels / scene signature changed enough
    TELEMETRY_CHANGE = "telemetry_change"  # drone state changed materially
    EVENT_WATCH = "event_watch"  # an open event has gone too long without a fresh look
    FORCED_REFRESH = "forced_refresh"  # no model call for max_skip_seconds
    SIMILAR = "similar"  # skipped: nothing worth a call changed


class SamplerConfig(DomainModel):
    enabled: bool = True
    # Scene score at or above this triggers a call (0-255 mean pixel difference; scripted scenes
    # score 0 or 100 so any change passes).
    scene_threshold: float = Field(default=6.0, ge=0, le=255)
    battery_step_percent: float = Field(default=2.0, gt=0)
    altitude_step_m: float = Field(default=5.0, gt=0)
    position_step_m: float = Field(default=15.0, gt=0)
    signal_degraded_below_percent: float = Field(default=40.0, gt=0, le=100)
    # While an open event exists, look again at least this often (seconds, simulated time).
    watch_interval_s: float = Field(default=120.0, gt=0)
    # Never go longer than this without a model call.
    max_skip_seconds: float = Field(default=300.0, gt=0)


@dataclass(frozen=True)
class SamplingDecision:
    analyze: bool
    reason: SamplingReason
    detail: str = ""  # e.g. which telemetry field changed


@dataclass
class SamplerStats:
    frames: int = 0
    analyzed: int = 0
    by_reason: dict[str, int] = field(default_factory=dict)

    @property
    def skipped(self) -> int:
        return self.frames - self.analyzed

    @property
    def avoided_pct(self) -> float:
        """Share of frames that did NOT need a model call (0-100)."""
        return round(100.0 * self.skipped / self.frames, 1) if self.frames else 0.0

    def to_dict(self) -> dict:
        return {
            "frames": self.frames,
            "model_calls": self.analyzed,
            "skipped": self.skipped,
            "avoided_pct": self.avoided_pct,
            "by_reason": dict(sorted(self.by_reason.items())),
        }


class AdaptiveSampler:
    """Stateful decision function; feed it one frame at a time, in time order."""

    def __init__(self, config: SamplerConfig | None = None) -> None:
        self.config = config or SamplerConfig()
        self.stats = SamplerStats()
        self._last_at: datetime | None = None
        self._last_zone: Zone | None = None
        self._last_telemetry: Telemetry | None = None

    def decide(
        self,
        *,
        now: datetime,
        zone: Zone,
        scene_score: float | None = None,
        telemetry: Telemetry | None = None,
        open_events: int = 0,
    ) -> SamplingDecision:
        """Decide for one frame and update the sampler's memory when a call is made."""
        decision = self._decide(now, zone, scene_score, telemetry, open_events)
        self.stats.frames += 1
        self.stats.by_reason[decision.reason.value] = (
            self.stats.by_reason.get(decision.reason.value, 0) + 1
        )
        if decision.analyze:
            self.stats.analyzed += 1
            self._last_at, self._last_zone = now, zone
            self._last_telemetry = telemetry or self._last_telemetry
        return decision

    def _decide(
        self,
        now: datetime,
        zone: Zone,
        scene_score: float | None,
        telemetry: Telemetry | None,
        open_events: int,
    ) -> SamplingDecision:
        cfg = self.config
        if not cfg.enabled:
            return SamplingDecision(True, SamplingReason.DISABLED)
        if self._last_at is None:
            return SamplingDecision(True, SamplingReason.FIRST)
        if zone != self._last_zone:
            return SamplingDecision(
                True,
                SamplingReason.ZONE_CHANGE,
                f"{self._last_zone.value} -> {zone.value}",  # type: ignore[union-attr]
            )
        if scene_score is not None and scene_score >= cfg.scene_threshold:
            return SamplingDecision(True, SamplingReason.SCENE_CHANGE, f"score {scene_score:.1f}")
        changed = self._telemetry_change(telemetry)
        if changed:
            return SamplingDecision(True, SamplingReason.TELEMETRY_CHANGE, changed)
        idle = (now - self._last_at).total_seconds()
        if open_events > 0 and idle >= cfg.watch_interval_s:
            return SamplingDecision(
                True, SamplingReason.EVENT_WATCH, f"{open_events} open event(s), {idle:.0f}s"
            )
        if idle >= cfg.max_skip_seconds:
            return SamplingDecision(True, SamplingReason.FORCED_REFRESH, f"{idle:.0f}s")
        return SamplingDecision(False, SamplingReason.SIMILAR)

    def _telemetry_change(self, current: Telemetry | None) -> str:
        """Which telemetry field moved enough since the last call (empty = none)."""
        cfg, last = self.config, self._last_telemetry
        if current is None or last is None:
            return ""
        if abs(last.battery_percent - current.battery_percent) >= cfg.battery_step_percent:
            return f"battery {last.battery_percent:.0f}% -> {current.battery_percent:.0f}%"
        if abs(last.altitude_m - current.altitude_m) >= cfg.altitude_step_m:
            return f"altitude {last.altitude_m:.0f} m -> {current.altitude_m:.0f} m"
        moved = distance_m(last.latitude, last.longitude, current.latitude, current.longitude)
        if moved >= cfg.position_step_m:
            return f"moved {moved:.0f} m"
        if last.signal_percent is not None and current.signal_percent is not None:
            was = last.signal_percent < cfg.signal_degraded_below_percent
            now = current.signal_percent < cfg.signal_degraded_below_percent
            if was != now:
                return f"signal {'degraded' if now else 'recovered'}"
        return ""
