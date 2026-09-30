"""Shared types and metric helpers for the evaluation harness.

Nothing here touches production behaviour. Every number in the results comes from executable
experiment code; interpretation text is built from measured values, never typed in by hand.
"""

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from drone_agent.models import DetectedObject, ObjectType, Observation, Zone

IST = timezone(timedelta(hours=5, minutes=30))

CONTROLLED_QUALIFICATION = (
    "Controlled synthetic fixtures only: these results do not estimate real-world security "
    "precision, recall or perception accuracy."
)


@dataclass
class ExperimentResult:
    key: str
    title: str
    research_question: str
    definition: str  # what exactly was run
    measured: dict  # machine-readable results
    rows: list[dict] = field(default_factory=list)  # flat rows for CSV
    interpretation: list[str] = field(default_factory=list)  # derived from `measured`
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "title": self.title,
            "research_question": self.research_question,
            "definition": self.definition,
            "measured": self.measured,
            "rows": self.rows,
            "interpretation": self.interpretation,
            "limitations": self.limitations,
        }


# ---- metric helpers -------------------------------------------------------------------------
def ratio(numerator: float, denominator: float) -> float | None:
    """numerator / denominator, or None (reported as N/A) when the denominator is zero."""
    return None if denominator == 0 else numerator / denominator


def pct(numerator: float, denominator: float) -> float | None:
    value = ratio(numerator, denominator)
    return None if value is None else round(100 * value, 2)


def fmt(value: float | int | None, digits: int = 3) -> str:
    """Human text for a metric; None becomes 'N/A' (never an invented number)."""
    if value is None:
        return "N/A"
    return str(value) if isinstance(value, int) else f"{value:.{digits}f}"


def f1_score(tp: int, fp: int, fn: int) -> float | None:
    """F1 = 2TP / (2TP + FP + FN): the harmonic mean of precision and recall.

    It is 0.0 when there are errors but no true positive (going through precision and recall
    would report N/A there), and None only when nothing was at stake (no TP, FP or FN at all).
    """
    return ratio(2 * tp, 2 * tp + fp + fn)


@dataclass(frozen=True)
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float | None:
        return ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float | None:
        return f1_score(self.tp, self.fp, self.fn)

    @property
    def accuracy(self) -> float | None:
        return ratio(self.tp + self.tn, self.tp + self.fp + self.tn + self.fn)

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.tn + other.tn, self.fn + other.fn
        )

    def to_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": None if self.precision is None else round(self.precision, 4),
            "recall": None if self.recall is None else round(self.recall, 4),
            "f1": None if self.f1 is None else round(self.f1, 4),
        }


def count_confusion(expected: int, actual: int) -> Confusion:
    """One (scenario, rule) cell, count based: matched alerts are TP, extra are FP, missing are
    FN, and an empty-vs-empty cell is one TN."""
    if expected == 0 and actual == 0:
        return Confusion(tn=1)
    matched = min(expected, actual)
    return Confusion(tp=matched, fp=max(0, actual - expected), fn=max(0, expected - actual))


def median(values: list[float]) -> float:
    return statistics.median(values)


# ---- observation builders for controlled fixtures --------------------------------------------
def det(
    object_type: ObjectType | str, activity: str | None = None, **attributes: str
) -> DetectedObject:
    return DetectedObject(
        object_type=ObjectType(object_type),
        activity=activity,
        attributes=attributes,
        confidence=0.9,
    )


class Timeline:
    """Builds a chronological list of scripted observations (frame ids 1, 2, 3 ...)."""

    def __init__(self, start: datetime) -> None:
        self.start = start
        self.observations: list[Observation] = []

    def add(self, offset_s: float, zone: Zone | str, *objects: DetectedObject) -> "Timeline":
        frame_id = len(self.observations) + 1
        self.observations.append(
            Observation(
                observation_id=f"obs-{frame_id}",
                frame_id=frame_id,
                timestamp=self.start + timedelta(seconds=offset_s),
                source="scripted",
                zone=Zone(zone),
                description=f"controlled fixture frame {frame_id}",
                detected_objects=list(objects),
            )
        )
        return self

    def run(self, zone: Zone | str, offsets: list[float], *objects: DetectedObject) -> "Timeline":
        for offset in offsets:
            self.add(offset, zone, *objects)
        return self
