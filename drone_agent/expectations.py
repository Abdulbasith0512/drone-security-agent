"""Machine-readable ground truth for a scenario (the `expect:` block of a scenario file).

A scenario author writes down what SHOULD happen, independently of how the system is
implemented: which events must exist, which alerts must be raised, with what severity ladder and
when the underlying condition first holds, and which alerts must NOT appear. The evaluation
(eval/ground_truth.py) then scores the system against it: precision, recall, F1, severity
accuracy and alert-detection latency.

Times are seconds after the scenario start ("condition times"): when the world first satisfies the
condition, not when the system noticed. Detection latency is measured against them.

These labels are hand-written by the scenario author, who also wrote the rules. They check that
the implementation matches the author's intent; they are not independent evidence about
real-world security performance (docs/evaluation.md says so wherever numbers are quoted).
"""

from pydantic import Field, model_validator

from drone_agent.models import AlertStatus, DomainModel, ObjectType, Severity, Zone


class ExpectedEvent(DomainModel):
    """An event that must exist: an object of this type in this zone lasting a plausible time."""

    object_type: ObjectType
    zone: Zone
    min_duration_s: float = Field(default=0.0, ge=0)
    max_duration_s: float | None = Field(default=None, ge=0)
    # Attribute values (case-insensitive) the event must carry, e.g. {"color": "red"}; used to tell
    # apart two events of the same type in the same zone.
    attributes: dict[str, str] = Field(default_factory=dict)
    note: str = ""

    @model_validator(mode="after")
    def _ordered(self) -> "ExpectedEvent":
        if self.max_duration_s is not None and self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must not be below min_duration_s")
        return self


class ExpectedStep(DomainModel):
    """One rung the alert must reach: the severity and when the condition first holds."""

    severity: Severity
    at_s: float = Field(ge=0)


class ExpectedAlert(DomainModel):
    rule_id: str = Field(min_length=1)
    zone: Zone | None = None  # None = any zone
    steps: list[ExpectedStep] = Field(min_length=1)  # the ladder, lowest rung first
    # The system must raise the alert within this many simulated seconds of the first rung's
    # condition time; a later detection still counts as found but is flagged as late.
    detect_within_s: float = Field(default=120.0, gt=0)
    final_status: AlertStatus | None = None  # e.g. resolved once the person leaves; None = any
    note: str = ""

    @model_validator(mode="after")
    def _ladder(self) -> "ExpectedAlert":
        for lower, higher in zip(self.steps, self.steps[1:], strict=False):
            if higher.severity.rank <= lower.severity.rank or higher.at_s < lower.at_s:
                raise ValueError("expected steps must rise in severity and time")
        return self

    @property
    def final_severity(self) -> Severity:
        return self.steps[-1].severity

    @property
    def condition_at_s(self) -> float:
        return self.steps[0].at_s


class ForbiddenAlert(DomainModel):
    """An alert that must not be raised (a benign situation the rules must leave alone)."""

    rule_id: str = Field(min_length=1)
    zone: Zone | None = None
    note: str = ""


class ExpectedEntity(DomainModel):
    """A "similar vehicle" the memory must have formed: its coarse signature and how many
    separate appearances it must link."""

    attributes: dict[str, str]  # normalised coarse attributes, e.g. {"color": "blue", ...}
    appearances: int = Field(ge=1)
    note: str = ""


class Expectations(DomainModel):
    events: list[ExpectedEvent] = Field(default_factory=list)
    entities: list[ExpectedEntity] = Field(default_factory=list)
    alerts: list[ExpectedAlert] = Field(default_factory=list)
    forbidden: list[ForbiddenAlert] = Field(default_factory=list)
    # A rung reached within this many seconds of its expected time counts as timely.
    step_tolerance_s: float = Field(default=5.0, ge=0)
    # Alerts the scenario does not label at all are treated as false positives unless a rule id
    # is listed here (e.g. a side effect the author deliberately does not score).
    unscored_rules: list[str] = Field(default_factory=list)
    # Alerts are always scored exhaustively (an unlabelled alert is a false positive). Events and
    # entities are SUBSET labels by default: "these must exist", and extra ones are reported but not
    # an error. Set this to declare the event and entity labels COMPLETE, so that any unlabelled
    # event or entity makes the scenario fail.
    exhaustive: bool = False
