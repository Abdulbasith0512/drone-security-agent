"""Prototype: structured claim-evidence consistency checker (EVALUATION LAYER ONLY).

It verifies a deliberately narrow set of structured claims against the stored evidence by exact,
deterministic comparison. It does NOT read free text and does NOT establish general semantic
entailment: extracting claims from prose is the unsolved part and is out of scope. It is not used
by the production analyst.

Supported claim classes (everything else is reported as unsupported by this checker):
  event_zone, event_type, event_activity, event_duration_s, event_start_hhmm,
  alert_severity, alert_rule, entity_appearance_count
"""

from dataclasses import dataclass
from typing import Any

from drone_agent.store import Store

SUPPORTED_CLASSES = (
    "event_zone",
    "event_type",
    "event_activity",
    "event_duration_s",
    "event_start_hhmm",
    "alert_severity",
    "alert_rule",
    "entity_appearance_count",
)

DURATION_TOLERANCE_S = 1.0


@dataclass(frozen=True)
class Claim:
    kind: str  # one of SUPPORTED_CLASSES
    subject: str  # the cited id the claim is attached to (event / alert / entity id)
    value: Any


@dataclass(frozen=True)
class Verdict:
    claim: Claim
    status: str  # "supported" | "contradicted" | "unknown_subject" | "unsupported_class"
    actual: Any = None


def check_claim(store: Store, claim: Claim) -> Verdict:
    kind = claim.kind
    if kind not in SUPPORTED_CLASSES:
        return Verdict(claim, "unsupported_class")
    if kind.startswith("event_"):
        event = store.get_event(claim.subject)
        if event is None:
            return Verdict(claim, "unknown_subject")
        actual: Any = {
            "event_zone": event.zone.value,
            "event_type": event.object_type.value,
            "event_activity": event.activities,
            "event_duration_s": event.duration_seconds,
            "event_start_hhmm": event.first_seen.strftime("%H:%M"),
        }[kind]
        if kind == "event_activity":
            ok = str(claim.value) in actual
        elif kind == "event_duration_s":
            ok = abs(float(claim.value) - actual) <= DURATION_TOLERANCE_S
        else:
            ok = str(claim.value) == actual
    elif kind.startswith("alert_"):
        alert = store.get_alert(claim.subject)
        if alert is None:
            return Verdict(claim, "unknown_subject")
        actual = alert.severity.value if kind == "alert_severity" else alert.rule_id
        ok = str(claim.value) == actual
    else:  # entity_appearance_count
        entity = store.get_entity(claim.subject)
        if entity is None:
            return Verdict(claim, "unknown_subject")
        actual = entity.appearance_count
        ok = int(claim.value) == actual
    return Verdict(claim, "supported" if ok else "contradicted", actual)


def check_claims(store: Store, claims: list[Claim]) -> list[Verdict]:
    return [check_claim(store, c) for c in claims]


def all_supported(verdicts: list[Verdict]) -> bool:
    """True only if every claim was checkable and matched (no claims = nothing was checked)."""
    return all(v.status == "supported" for v in verdicts)
