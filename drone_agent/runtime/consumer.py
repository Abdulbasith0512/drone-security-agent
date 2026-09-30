"""The shared observation consumer: everything downstream of perception, one observation at a time.

    Observation -> persist -> event aggregation / entity linking -> deterministic rules -> alerts

Both producers of observations feed this same object:

* the streaming patrol engine (`PatrolEngine`: scripted frames, sampled and perceived tick by tick);
* the progressive video driver (`run_video_live`: a video file, sampled, gated and described).

There is exactly one implementation of "what happens to an observation once it exists": it is
written to the store, the temporal context (`ContextEngine`: the same `EventAggregator` and
`EntityLinker` the batch pipeline uses) is advanced, the rules (`SecurityReasoner`: the same
`RuleEngine` and alert lifecycle) are evaluated on what is known so far, and every change is
persisted and announced immediately, not after the input ends. Telemetry joins the same way (a
sample is stored, then the safety rules run), so the drone-safety rules also see it as it arrives.

The consumer never sees pixels and never calls a model. What produced an observation (a script, a
cassette, a live model) stays recorded in the observation's own `source` and `provenance`; a
carried observation arrives labelled `+carried` and is stored and announced as such.

Ending the input (`finish`) closes the open events as SCENARIO_END and takes a last look with every
rule. That is the end of the OBSERVATION COVERAGE, not evidence that anything left (see
`CloseReason`); departure is recorded only when the same zone was seen without the object.
"""

from collections.abc import Callable
from datetime import datetime

from drone_agent.agents.context import ContextEngine, ContextUpdate
from drone_agent.agents.reasoning import SecurityReasoner
from drone_agent.models import Alert, Entity, Event, Observation, Telemetry
from drone_agent.pipeline.config import EventConfig
from drone_agent.rules.config import SecurityRulesConfig
from drone_agent.rules.lifecycle import AlertChange
from drone_agent.runtime.events import RunEvent, RunEventKind
from drone_agent.store import Store
from drone_agent.telemetry import TelemetrySynchronizer

Listener = Callable[[RunEvent], None]
Emit = Callable[..., None]


def emitter(listener: Listener | None) -> Emit:
    """Turn a RunEvent listener into the `emit(kind, at, offset_s, **data)` the consumer calls."""

    def emit(kind: RunEventKind, at: datetime, offset_s: float, **data) -> None:
        if listener is None:
            return
        provenance = data.pop("provenance", "")
        listener(RunEvent(kind, at, offset_s, data, provenance))

    return emit


class ObservationPipeline:
    """Persist observations and telemetry, keep the events/entities/alerts up to date, announce."""

    def __init__(
        self,
        store: Store,
        *,
        events: EventConfig | None = None,
        rules: SecurityRulesConfig | None = None,
        stale_after_s: float = 30.0,
        emit: Emit | None = None,
    ) -> None:
        self.store = store
        self.context = ContextEngine(events)
        self.reasoner = SecurityReasoner(rules)
        self.stale_after_s = stale_after_s
        self._emit = emit or emitter(None)
        self.samples: list[Telemetry] = []
        self.events: list[Event] = []
        self.entities: list[Entity] = []
        self.provenance: dict[str, int] = {}
        self.n_observations = 0
        self._known_events: dict[str, str] = {}

    # ---- input -----------------------------------------------------------------------------
    def synchronizer(self) -> TelemetrySynchronizer:
        """Joins a frame time to the telemetry received so far (nearest prior sample)."""
        return TelemetrySynchronizer(self.samples, self.stale_after_s)

    def record_telemetry(self, sample: Telemetry) -> None:
        self.store.insert_telemetry(sample)
        self.samples.append(sample)

    def consume(self, obs: Observation, at: datetime, offset_s: float) -> Observation:
        """One observation: attach telemetry, persist, advance the context, evaluate the rules."""
        obs = self.synchronizer().attach(obs)
        self.store.insert_observation(obs)
        self.n_observations += 1
        carried = obs.source.endswith("+carried")
        label = (obs.provenance.value.upper() if obs.provenance else "UNKNOWN") + (
            "+CARRIED" if carried else ""
        )
        self.provenance[label] = self.provenance.get(label, 0) + 1
        self._emit(
            RunEventKind.OBSERVATION,
            at,
            offset_s,
            provenance=label,
            frame_id=obs.frame_id,
            description=obs.description,
            objects=[
                {
                    "type": o.object_type.value,
                    "activity": o.activity,
                    "attributes": o.attributes,
                    "confidence": o.confidence,
                }
                for o in obs.detected_objects
            ],
            carried=carried,
        )
        self._apply_context(self.context.process(obs), at, offset_s)
        self.assess(at, offset_s, safety=False)  # a frame cannot change a telemetry rule
        return obs

    def assess(
        self, now: datetime, offset_s: float, *, camera: bool = True, safety: bool = True
    ) -> None:
        """Run the rule families a new input could have changed; persist and announce changes."""
        changes = self.reasoner.assess(
            self.events, self.entities, self.samples, now, camera=camera, safety=safety
        )
        for change in changes:
            self.store.save_alert(change.alert)
            self._announce(change, now, offset_s)

    def finish(self, last_at: datetime, last_offset_s: float) -> None:
        """The input ended: close open events and take a final look with every rule."""
        self._apply_context(self.context.flush(), last_at, last_offset_s)
        self.assess(last_at, last_offset_s)

    @property
    def alerts(self) -> list[Alert]:
        return self.reasoner.alerts

    # ---- persistence and announcements -----------------------------------------------------
    def _apply_context(self, update: ContextUpdate, at: datetime, offset_s: float) -> None:
        self.events, self.entities = update.events, update.entities
        if update.changed:
            self.store.save_events(update.changed)
        self.store.save_entities(update.entities)
        for ev in update.changed:
            status = "open" if ev.is_open else "closed"
            previous = self._known_events.get(ev.event_id)
            self._known_events[ev.event_id] = status
            if previous == "closed":
                continue
            change = "opened" if previous is None else ("closed" if status == "closed" else "grew")
            self._emit(
                RunEventKind.EVENT,
                at,
                offset_s,
                event_id=ev.event_id,
                change=change,
                object_type=ev.object_type.value,
                zone=ev.zone.value,
                first_seen=ev.first_seen.isoformat(),
                last_seen=ev.last_seen.isoformat(),
                duration_s=ev.duration_seconds,
                observations=ev.evidence_count,
                close_reason=ev.close_reason.value,
                entity_id=ev.entity_id,
                attributes=ev.attributes,
            )

    def _announce(self, change: AlertChange, now: datetime, offset_s: float) -> None:
        alert: Alert = change.alert
        for step in change.transitions or [None]:
            self._emit(
                RunEventKind.ALERT,
                now,
                offset_s,
                alert_id=alert.alert_id,
                rule_id=alert.rule_id,
                title=alert.title,
                severity=alert.severity.value,
                status=alert.status.value,
                verdict=alert.verdict.value if alert.verdict else None,
                zone=alert.zone.value if alert.zone else None,
                transition=step.kind.value if step else "evidence",
                reason=step.reason if step else "new evidence",
                recommended_action=alert.recommended_action,
                confidence=alert.confidence,
                detection_latency_s=alert.detection_latency_seconds,
                event_ids=alert.event_ids,
                is_new=change.is_new,
            )
