"""Alert: the output of a deterministic security rule, with the facts and evidence behind it."""

from pydantic import AwareDatetime, Field, model_validator

from drone_agent.models.base import DomainModel, Severity, Zone

# Machine-readable rule facts must stay simple and JSON-serialisable.
FactValue = str | int | float | bool | list[str]


class Alert(DomainModel):
    """One alert, created only by the rule engine (never by an LLM).

    Traceable chain: Alert -> event_ids (and entity_id) -> observations -> frames. Only the links
    are stored; `frame_ids` is the sorted union of those events' frames, filled in when the
    alert is created or read back.
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
    event_ids: list[str] = Field(min_length=1)
    entity_id: str | None = None
    frame_ids: list[int] = Field(min_length=1)
    facts: dict[str, FactValue] = Field(default_factory=dict)  # numbers the rule compared

    @model_validator(mode="after")
    def _consistent(self) -> "Alert":
        if self.last_seen < self.first_seen:
            raise ValueError("last_seen must not be before first_seen")
        if not self.first_seen <= self.triggered_at <= self.last_seen:
            raise ValueError("triggered_at must lie within first_seen..last_seen")
        return self
