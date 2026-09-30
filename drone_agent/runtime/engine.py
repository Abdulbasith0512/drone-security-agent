"""The streaming patrol engine: frame -> perception -> context -> rules -> alert, tick by tick.

    telemetry tick ->  store  -> rules (safety)            -> alert lifecycle
    frame tick     ->  sampler -> perception agent | carry -> store -> temporal context
                                                            -> rules -> alert lifecycle

Everything is persisted as it happens (SQLite), so the analyst, the CLI and the tests read the
same rows a live view would have shown, mid-run or afterwards. The engine never calls a language
model itself; the perception agent it is given decides which provider (scripted, cassette,
live) answers a frame.

Determinism: with pacing off (`speed 0`) two runs of the same scenario produce identical rows.
Wall-clock pacing only sleeps between ticks; it never changes what is computed.
"""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from drone_agent.agents.perception import PerceptionAgent, PerceptionTrace
from drone_agent.agents.sampling import AdaptiveSampler, SamplerConfig, SamplingReason
from drone_agent.models import Observation
from drone_agent.pipeline.config import EventConfig
from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.runtime.consumer import ObservationPipeline
from drone_agent.runtime.events import RunEvent, RunEventKind
from drone_agent.runtime.pacing import Pacer
from drone_agent.runtime.scene import SIGNAL_NAME, Signature, scene_score, signature
from drone_agent.runtime.ticks import Tick, build_ticks
from drone_agent.scenario import Scenario
from drone_agent.store import Store
from drone_agent.vision import FrameInput, ScriptedProvider

Listener = Callable[[RunEvent], None]
MAX_TRACE_ENTRIES = 400


@dataclass
class EngineOptions:
    rules: SecurityRulesConfig = field(default_factory=SecurityRulesConfig)
    events: EventConfig = field(default_factory=EventConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    perception_timeout_s: float | None = None
    carry_skipped: bool = True  # a skipped frame repeats the last observation, labelled carried


@dataclass(frozen=True)
class RunSummary:
    scenario: str
    mode: str
    sim_start: datetime
    sim_end: datetime
    ticks: int
    telemetry_scheduled: int
    telemetry_delivered: int
    frames: int
    model_calls: int  # frames the perception agent was asked about
    skipped: int  # frames answered by carrying the last observation forward
    dropped: int  # frames with no observation at all (perception failed, nothing to carry)
    avoided_pct: float  # of frames, how many needed no model call
    observations: int
    events: int
    entities: int
    alerts: int
    alerts_by_status: dict[str, int]
    alerts_by_severity: dict[str, int]
    detection_latency_s: dict[str, float]  # alert_id -> seconds between condition and detection
    sampler: dict
    perception: dict
    provenance: dict[str, int]
    wall_seconds: float
    run_id: int | None = None

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "mode": self.mode,
            "sim_start": self.sim_start.isoformat(),
            "sim_end": self.sim_end.isoformat(),
            "ticks": self.ticks,
            "telemetry_scheduled": self.telemetry_scheduled,
            "telemetry_delivered": self.telemetry_delivered,
            "frames": self.frames,
            "model_calls": self.model_calls,
            "skipped": self.skipped,
            "dropped": self.dropped,
            "avoided_pct": self.avoided_pct,
            "observations": self.observations,
            "events": self.events,
            "entities": self.entities,
            "alerts": self.alerts,
            "alerts_by_status": self.alerts_by_status,
            "alerts_by_severity": self.alerts_by_severity,
            "detection_latency_s": self.detection_latency_s,
            "sampler": self.sampler,
            "perception": self.perception,
            "provenance": self.provenance,
            "wall_seconds": round(self.wall_seconds, 3),
        }


class PatrolEngine:
    def __init__(
        self,
        scenario: Scenario,
        store: Store,
        *,
        perception: PerceptionAgent | None = None,
        options: EngineOptions | None = None,
        listener: Listener | None = None,
        pacer: Pacer | None = None,
        spec_sha256: str = "",
    ) -> None:
        self.scenario = scenario
        self.store = store
        self.options = options or EngineOptions()
        self.perception = perception or PerceptionAgent(
            [ScriptedProvider(scenario)], call_timeout_s=self.options.perception_timeout_s
        )
        self.listener = listener
        self.pacer = pacer or Pacer(None)
        self.spec_sha256 = (
            spec_sha256 or hashlib.sha256(scenario.model_dump_json().encode()).hexdigest()
        )
        self.sampler = AdaptiveSampler(self.options.sampler)
        # Everything downstream of perception is the shared consumer (runtime/consumer.py): the
        # same object the progressive video driver feeds. The engine only decides WHICH frames are
        # analysed and what an observation for a skipped frame is.
        self.pipeline = ObservationPipeline(
            store,
            events=self.options.events,
            rules=self.options.rules,
            stale_after_s=scenario.telemetry.stale_after_s,
            emit=self._emit,
        )
        self.context = self.pipeline.context
        self.reasoner = self.pipeline.reasoner

        self._last_obs: Observation | None = None
        self._reference: Signature | None = None
        self._last_offset = 0.0
        self._trace: list[dict] = []
        # counters
        self.n_ticks = 0
        self.n_scheduled = 0
        self.n_frames = 0
        self.n_model_calls = 0
        self.n_carried = 0
        self.n_dropped = 0

    @property
    def n_observations(self) -> int:
        return self.pipeline.n_observations

    # ---- public ----------------------------------------------------------------------------
    def run(self) -> RunSummary:
        """Process the whole patrol; returns a summary (also stored in `agent_runs`)."""
        started = time.perf_counter()
        scenario, store = self.scenario, self.store
        ticks = build_ticks(scenario)
        expect = scenario.expect.model_dump(mode="json") if scenario.expect else {}
        store.upsert_scenario(
            scenario.name,
            scenario.description,
            scenario.start.isoformat(),
            self.spec_sha256,
            expect,
        )
        mode = f"scripted frames, simulated telemetry, {self.perception.name}"
        run_id = store.add_agent_run("patrol", scenario.name, mode, "running")
        self._emit(
            RunEventKind.STARTED,
            scenario.start,
            0.0,
            scenario=scenario.name,
            description=scenario.description,
            ticks=len(ticks),
            duration_s=scenario.telemetry.duration_s,
            frames=len(scenario.frames),
            expected_alerts=len(scenario.expect.alerts) if scenario.expect else 0,
        )
        last_at = scenario.start
        for tick in ticks:
            self.pacer.wait(tick.offset_s - self._last_offset)
            self._last_offset = tick.offset_s
            last_at = tick.at
            self.n_ticks += 1
            with store.transaction():
                if tick.kind == "telemetry":
                    self._on_telemetry(tick)
                else:
                    self._on_frame(tick)
        with store.transaction():
            self.pipeline.finish(last_at, self._last_offset)
        summary = self._summarise(last_at, time.perf_counter() - started, run_id)
        store.update_agent_run(
            run_id,
            status="finished",
            metrics=summary.to_dict(),
            trace=self._trace[:MAX_TRACE_ENTRIES],
        )
        self._emit(RunEventKind.FINISHED, last_at, self._last_offset, **summary.to_dict())
        self.perception.close()
        return summary

    # ---- ticks -----------------------------------------------------------------------------
    def _on_telemetry(self, tick: Tick) -> None:
        self.n_scheduled += 1
        sample = tick.sample
        if sample is not None:
            self.pipeline.record_telemetry(sample)
        self._emit(
            RunEventKind.TELEMETRY,
            tick.at,
            tick.offset_s,
            provenance="SIMULATED",
            delivered=sample is not None,
            **(
                {
                    "zone": sample.zone.value,
                    "battery_percent": sample.battery_percent,
                    "signal_percent": sample.signal_percent,
                    "altitude_m": sample.altitude_m,
                    "latitude": sample.latitude,
                    "longitude": sample.longitude,
                }
                if sample
                else {}
            ),
        )
        self.pipeline.assess(tick.at, tick.offset_s, camera=False)

    def _on_frame(self, tick: Tick) -> None:
        frame = tick.frame
        assert frame is not None
        self.n_frames += 1
        sync = self.pipeline.synchronizer()
        association = sync.associate(tick.at)
        current = signature(frame)
        score = scene_score(self._reference, current)
        decision = self.sampler.decide(
            now=tick.at,
            zone=frame.zone,
            scene_score=score,
            telemetry=association.telemetry if association else None,
            open_events=sum(1 for e in self.pipeline.events if e.is_open),
        )
        analyze = decision.analyze
        reason, detail = decision.reason.value, decision.detail
        if not analyze and (self._last_obs is None or self._last_obs.zone != frame.zone):
            analyze, reason, detail = True, "no_reference", "nothing to carry forward"

        obs: Observation | None = None
        trace: PerceptionTrace | None = None
        if analyze:
            self.n_model_calls += 1
            result = self.perception.perceive(
                FrameInput(frame_id=frame.frame_id, timestamp=tick.at)
            )
            obs, trace = result.observation, result.trace
            if obs is not None:
                self._last_obs = obs
                self._reference = current
        elif self.options.carry_skipped and self._last_obs is not None:
            last = self._last_obs
            obs = Observation(
                observation_id=f"obs-{frame.frame_id}",
                frame_id=frame.frame_id,
                timestamp=tick.at,
                source=f"{last.source}+carried",
                zone=frame.zone,
                description=last.description,
                detected_objects=last.detected_objects,
                provenance=last.provenance,
            )
            self.n_carried += 1
        if obs is None:
            self.n_dropped += 1

        self._emit(
            RunEventKind.FRAME,
            tick.at,
            tick.offset_s,
            frame_id=frame.frame_id,
            zone=frame.zone.value,
            analyzed=analyze,
            reason=reason,
            detail=detail,
            scene_score=score,
            scene_signal=SIGNAL_NAME,
            perception=trace.to_dict() if trace else None,
            dropped=obs is None,
        )
        if len(self._trace) < MAX_TRACE_ENTRIES:
            self._trace.append(
                {
                    "frame_id": frame.frame_id,
                    "offset_s": tick.offset_s,
                    "analyzed": analyze,
                    "reason": reason,
                    "detail": detail,
                    "scene_score": score,
                    **({"perception": trace.to_dict()} if trace else {}),
                }
            )
        if obs is None:
            return  # nothing new was seen, so no camera rule can change

        self.pipeline.consume(obs, tick.at, tick.offset_s)

    # ---- summary ---------------------------------------------------------------------------
    def _summarise(self, sim_end: datetime, wall: float, run_id: int) -> RunSummary:
        alerts = self.pipeline.alerts
        by_status: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for a in alerts:
            by_status[a.status.value] = by_status.get(a.status.value, 0) + 1
            by_severity[a.severity.value] = by_severity.get(a.severity.value, 0) + 1
        frames = self.n_frames
        avoided = round(100.0 * (frames - self.n_model_calls) / frames, 1) if frames else 0.0
        return RunSummary(
            scenario=self.scenario.name,
            mode=f"scripted frames, simulated telemetry, {self.perception.name}",
            sim_start=self.scenario.start,
            sim_end=sim_end,
            ticks=self.n_ticks,
            telemetry_scheduled=self.n_scheduled,
            telemetry_delivered=len(self.pipeline.samples),
            frames=frames,
            model_calls=self.n_model_calls,
            skipped=self.n_carried,
            dropped=self.n_dropped,
            avoided_pct=avoided,
            observations=self.n_observations,
            events=len(self.pipeline.events),
            entities=len(self.pipeline.entities),
            alerts=len(alerts),
            alerts_by_status=dict(sorted(by_status.items())),
            alerts_by_severity=dict(sorted(by_severity.items())),
            detection_latency_s={
                a.alert_id: a.detection_latency_seconds
                for a in alerts
                if a.detection_latency_seconds is not None
            },
            sampler=self.sampler.stats.to_dict(),
            perception=self.perception.stats.to_dict(),
            provenance=dict(sorted(self.pipeline.provenance.items())),
            wall_seconds=wall,
            run_id=run_id,
        )

    def _emit(self, kind: RunEventKind, at: datetime, offset_s: float, **data) -> None:
        if self.listener is None:
            return
        provenance = data.pop("provenance", "")
        self.listener(RunEvent(kind, at, offset_s, data, provenance))


__all__ = ["EngineOptions", "PatrolEngine", "RunSummary", "SamplingReason"]
