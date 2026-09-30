"""Structured claim-evidence consistency checking, as used by the evaluation harness.

The comparison logic now lives in the production analyst (`drone_agent/agent/factcheck.py`), which
also extracts claims from the prose of model-written answers. The evaluation keeps this module
because the semantic-faithfulness probe feeds it HAND-WRITTEN structured claims: that measures the
comparison on its own, independently of how well claims are read out of text.

Supported claim classes (everything else is reported as unsupported by this checker):
  event_zone, event_type, event_activity, event_duration_s, event_start_hhmm,
  alert_severity, alert_rule, entity_appearance_count
"""

from drone_agent.agent.factcheck import (
    DURATION_TOLERANCE_S,
    SUPPORTED_CLASSES,
    all_supported,
    check_claim,
    check_claims,
)
from drone_agent.agent.factcheck import StructuredClaim as Claim
from drone_agent.agent.factcheck import StructuredVerdict as Verdict

__all__ = [
    "DURATION_TOLERANCE_S",
    "SUPPORTED_CLASSES",
    "Claim",
    "Verdict",
    "all_supported",
    "check_claim",
    "check_claims",
]
