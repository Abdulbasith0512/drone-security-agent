"""Alert: the output of a deterministic security rule, with the facts and evidence behind it."""

from enum import StrEnum

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, Severity, Zone

# Machine-readable rule facts must stay simple and JSON-serialisable.
FactValue = str | int | float | bool | list[str]


class AlertStatus(StrEnum):
    """Lifecycle state. Set by the alert manager or an operator, never by a language model."""

    OPEN = "open"  # raised, nobody has looked at it yet
    ACKNOWLEDGED = "acknowledged"  # an operator has seen it
    ESCALATED = "escalated"  # severity went up after it was raised (and it is still active)
    RESOLVED = "resolved"  # the condition ended (auto-resolving rules) or an operator closed it


class Verdict(StrEnum):
    """Coarse triage class derived from severity by a fixed table (see rules/common.py)."""

    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    CRITICAL = "critical"


class TransitionKind(StrEnum):
    RAISED = "raised"
    ESCALATED = "escalated"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class LadderStep(DomainModel):
    """One rung of an escalation ladder that the rule's condition has reached."""

    at: AwareDatetime  # when the condition first reached this rung (deterministic, from the data)
    severity: Severity
    label: str = Field(min_length=1)  # e.g. "warning", "critical"


class AlertTransition(DomainModel):
    """One recorded lifecycle change. `at` is when it was recorded, `effective_at` when the
    rule's condition actually began; they are equal in batch analysis and differ by the detection
    latency in a streaming run (simulated time in both cases)."""

    seq: int = Field(ge=1)
    kind: TransitionKind
    at: AwareDatetime
    effective_at: AwareDatetime
    from_status: AlertStatus | None = None
    to_status: AlertStatus
    from_severity: Severity | None = None
    to_severity: Severity
    reason: str = Field(min_length=1)
    actor: str = "system"  # "system" | "operator"


class Alert(DomainModel):
    """One alert, created only by the rule engine (never by an LLM).

    Traceable chain: Alert -> event_ids (and entity_id) -> observations -> frames, or, for safety
    alerts, Alert -> telemetry samples (`telemetry_at`). Only the links are stored; `frame_ids` is
    the sorted union of those events' frames, filled in when the alert is created or read back.
    """

    alert_id: str = Field(min_length=1)  # deterministic: ALR-<RULE>-<subject id>
    rule_id: str = Field(min_length=1)
    severity: Severity
    title: str = Field(min_length=1)  # short: what happened
    message: str = Field(min_length=1)  # why the rule fired, in words
    # When the rule's condition first held (see docs/architecture.md, "triggered_at").
    triggered_at: AwareDatetime
    first_seen: AwareDatetime  # time span of the triggering evidence
    last_seen: AwareDatetime
    zone: Zone | None = None
    event_ids: list[str] = Field(default_factory=list)
    entity_id: str | None = None
    frame_ids: list[int] = Field(default_factory=list)
    facts: dict[str, FactValue] = Field(default_factory=dict)  # numbers the rule compared

    # ---- lifecycle and triage (filled by the rule engine / alert manager) ----------------------
    status: AlertStatus = AlertStatus.OPEN
    # Perception confidence behind the alert: mean detection confidence for event alerts (a
    # placeholder when the provider reports none), 1.0 for alerts on measured telemetry.
    confidence: float | None = Field(default=None, ge=0, le=1)
    verdict: Verdict | None = None
    recommended_action: str | None = None
    ladder: list[LadderStep] = Field(default_factory=list)
    detected_at: AwareDatetime | None = None  # when the system first raised it (simulated time)
    updated_at: AwareDatetime | None = None
    resolved_at: AwareDatetime | None = None
    telemetry_at: list[AwareDatetime] = Field(default_factory=list)  # evidence samples (safety)
    history: list[AlertTransition] = Field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return self.status is not AlertStatus.RESOLVED

    @property
    def detection_latency_seconds(self) -> float | None:
        """Seconds between the rule's condition first holding and the system raising the alert."""
        if self.detected_at is None:
            return None
        return max(0.0, (self.detected_at - self.triggered_at).total_seconds())

    @model_validator(mode="after")
    def _consistent(self) -> "Alert":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        if not self.first_seen <= self.triggered_at <= self.last_seen:
            raise ValueError("triggered_at must lie within first_seen..last_seen")
        if not self.event_ids and not self.telemetry_at:
            raise ValueError("an alert needs evidence: event_ids or telemetry_at")
        if self.event_ids and not self.frame_ids:
            raise ValueError("an alert on events needs the frame_ids of those events")
        if self.status is AlertStatus.RESOLVED and self.resolved_at is None:
            raise ValueError("a resolved alert needs resolved_at")
        steps = self.ladder
        for lower, higher in zip(steps, steps[1:], strict=False):
            if higher.at < lower.at or higher.severity.rank <= lower.severity.rank:
                raise ValueError("ladder steps must be chronological with strictly rising severity")
        if steps and steps[-1].severity is not self.severity:
            raise ValueError("severity must equal the last ladder step's severity")
        return self
