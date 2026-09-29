"""Helpers shared by the rules."""

from collections.abc import Iterable

from drone_agent.models import Event


def alert_id(code: str, subject_id: str) -> str:
    """Deterministic alert identity: same rule + same event/entity -> same id."""
    return f"ALR-{code}-{subject_id}"


def frame_union(events: Iterable[Event]) -> list[int]:
    """Sorted, de-duplicated frame ids across events (the alert's evidence frames)."""
    return sorted({f for ev in events for f in ev.frame_ids})


def type_label(event: Event) -> str:
    """'Person', 'Vehicle' (used only in the alert wording)."""
    return event.object_type.value.capitalize()
