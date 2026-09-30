"""Deterministic factual-claim validation for model-written analyst answers.

The citation gate (citations.py) proves that the evidence an answer cites exists and was
retrieved. That does not make the sentence around the citation true: "The pickup was at the main
gate [event:EVT-V-001]" cites a real event that happened in the parking area. This module closes
that gap for the claims a program can check exactly, by comparing them with the stored evidence.

Two layers, deliberately separated:

* `Claim` / `check_claim`: one structured claim about one cited subject, compared exactly with the
  store (zone, type, activity, duration, start time, alert severity and rule, appearance count).
  The evaluation harness uses this layer directly with hand-written claims.
* `check_facts`: extracts claims from the sentences of an answer with fixed, conservative
  patterns, anchors each to the evidence cited in the same sentence, and classifies it as
  SUPPORTED, CONTRADICTED (the evidence has that fact and it differs) or UNSUPPORTED (the cited
  evidence carries no such fact). Only the last two fail validation.

What is checked: alert severity, status and rule; zones; person/vehicle types; activities; clock
times; durations; appearance counts; totals of alerts, events, observations and entities (with the
filters the sentence itself states); object counts in a cited frame. Recommendations are not
facts, but the premises inside them are checked (see `_RECOMMENDATION`).

What is NOT checked, by design: anything not stated in one of those forms. This is not natural
language understanding or entailment. A claim phrased in a way the patterns do not recognise is
simply not seen (it neither passes nor fails here), and the identity/intent and citation checks in
citations.py still apply to it. No language model takes part.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from drone_agent.agent.citations import CITATION_RE, Citation, _negated, claim_units
from drone_agent.models import (
    Alert,
    AlertStatus,
    Event,
    ObjectType,
    Observation,
    TransitionKind,
    Zone,
)
from drone_agent.store import Store

# ---- layer 1: one structured claim, compared exactly -------------------------------------------
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
class StructuredClaim:
    kind: str  # one of SUPPORTED_CLASSES
    subject: str  # the cited id the claim is attached to (event / alert / entity id)
    value: Any


@dataclass(frozen=True)
class StructuredVerdict:
    claim: StructuredClaim
    status: str  # "supported" | "contradicted" | "unknown_subject" | "unsupported_class"
    actual: Any = None


def check_claim(store: Store, claim: StructuredClaim) -> StructuredVerdict:
    kind = claim.kind
    if kind not in SUPPORTED_CLASSES:
        return StructuredVerdict(claim, "unsupported_class")
    if kind.startswith("event_"):
        event = store.get_event(claim.subject)
        if event is None:
            return StructuredVerdict(claim, "unknown_subject")
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
            return StructuredVerdict(claim, "unknown_subject")
        actual = alert.severity.value if kind == "alert_severity" else alert.rule_id
        ok = str(claim.value) == actual
    else:  # entity_appearance_count
        entity = store.get_entity(claim.subject)
        if entity is None:
            return StructuredVerdict(claim, "unknown_subject")
        actual = entity.appearance_count
        ok = int(claim.value) == actual
    return StructuredVerdict(claim, "supported" if ok else "contradicted", actual)


def check_claims(store: Store, claims: list[StructuredClaim]) -> list[StructuredVerdict]:
    return [check_claim(store, c) for c in claims]


def all_supported(verdicts: list[StructuredVerdict]) -> bool:
    """True only if every claim was checkable and matched (no claims = nothing was checked)."""
    return all(v.status == "supported" for v in verdicts)


# ---- layer 2: claims extracted from an answer ---------------------------------------------------
class Verdict(StrEnum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"  # the cited evidence has this fact and it differs
    UNSUPPORTED = "unsupported"  # the cited evidence carries no such fact


@dataclass(frozen=True)
class Claim:
    kind: str  # severity | status | rule | zone | type | activity | time | duration | ...
    values: tuple
    sentence: str
    text: str  # the words of the sentence the claim was read from
    present: bool = False  # status only: stated in the present tense ("is resolved")
    recommendation: bool = False
    header: bool = False  # total only: the sentence introduces a list (it ends with a colon)


@dataclass(frozen=True)
class Finding:
    claim: Claim
    verdict: Verdict
    expected: str  # what the stored evidence says, in words

    def describe(self) -> str:
        said = f'"{self.claim.text}"'
        if self.verdict is Verdict.CONTRADICTED:
            return f"{said} ({self.claim.kind}) contradicts the cited evidence: {self.expected}"
        return f"{said} ({self.claim.kind}) is not supported by the cited evidence: {self.expected}"


@dataclass
class FactReport:
    findings: list[Finding] = field(default_factory=list)  # every claim that was assessed

    @property
    def failed(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is not Verdict.SUPPORTED]

    @property
    def contradicted(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is Verdict.CONTRADICTED]

    @property
    def unsupported(self) -> list[Finding]:
        return [f for f in self.findings if f.verdict is Verdict.UNSUPPORTED]

    @property
    def ok(self) -> bool:
        return not self.failed


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}  # fmt: skip
_NUM = (
    r"\d+(?:\.\d+)?|"
    r"(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
    r"(?:one|two|three|four|five|six|seven|eight|nine)|"
    + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))
)
_ID = re.compile(r"\b(?:EVT|ALR|ENT)-[A-Za-z0-9\-]+\b|\bobs-\d+\b")


def _number(word: str) -> float | None:
    word = word.lower().strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", word):
        return float(word)
    if word in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[word])
    m = re.fullmatch(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ](\w+)", word)
    if m and m.group(2) in _NUMBER_WORDS and _NUMBER_WORDS[m.group(2)] < 10:
        return float(_NUMBER_WORDS[m.group(1)] + _NUMBER_WORDS[m.group(2)])
    return None


# ---- vocabularies -------------------------------------------------------------------------------
_SEV = "low|medium|high|critical"
_STATUS = "open|escalated|acknowledged|resolved"
_SEVERITY_PATTERNS = (
    re.compile(rf"\b({_SEV})[- ]?(?:severity|priority)\b"),
    re.compile(rf"\bseverity(?: level)?(?: is| was| of| as| at| to| :|:)?\s+({_SEV})\b"),
    re.compile(
        rf"\b(?:raised|escalated|upgraded|downgraded|rated|elevated|promoted|set|reached)\s+"
        rf"(?:it |the alert |this alert )?(?:to|as|at)\s+({_SEV})\b"
    ),
    re.compile(rf"\b({_SEV})\s+alerts?\b"),
    re.compile(rf"\b({_SEV})/(?:{_STATUS})\b"),
)
_PRESENT_STATUS = re.compile(
    rf"\b(?:status(?: is| was| of| :|:)?|currently|now|still|remains?|is)\s+"
    rf"(?:still\s+)?({_STATUS})\b"
)
_PAST_STATUS = re.compile(
    rf"\b(?:was|were|been|got)\s+(?:then\s+|later\s+|subsequently\s+|finally\s+)?({_STATUS})\b"
)
_SLASH_STATUS = re.compile(rf"\b(?:{_SEV})/({_STATUS})\b")
_NOT_RESOLVED = re.compile(r"\bunresolved\b|\bnot\s+(?:yet\s+|been\s+)?resolved\b")

_RULE_PHRASES = {
    "loitering": r"loiter(?:ing)?",
    "after_hours_presence": r"after[- ]hours",
    "repeat_vehicle": r"repeat[- ]vehicle",
    "restricted_zone_entry": r"restricted[- ]zone",
    "low_battery": r"low[- ]battery",
    "signal_loss": r"signal[- ]loss|lost[- ]link|link[- ]lost",
    "altitude_anomaly": r"altitude",
    "geofence_violation": r"geofence",
}
_RULE_RE = {
    rule: re.compile(
        rf"\b(?:{phrase})(?:\s+(?:presence|entry|anomaly|violation))?[\s-]+(?:alerts?|rules?)\b"
    )
    for rule, phrase in _RULE_PHRASES.items()
}
_RULE_PHRASE_ONLY = {
    rule: re.compile(rf"\b(?:{phrase})\b") for rule, phrase in _RULE_PHRASES.items()
}

_ZONE_RE = {
    Zone.MAIN_GATE: re.compile(r"\bmain[ _-]?gate\b|\bgate\b"),
    Zone.DRIVEWAY: re.compile(r"\bdriveway\b"),
    Zone.PARKING_AREA: re.compile(r"\bparking(?:[ _-]?area|[ _-]lot)?\b"),
    Zone.WAREHOUSE: re.compile(r"\bwarehouse\b"),
    Zone.RESTRICTED_ZONE: re.compile(r"\brestricted[ _-]?zone\b"),
    Zone.GARAGE: re.compile(r"\bgarage\b"),
}
_TYPE_RE = {
    ObjectType.PERSON: re.compile(
        r"\b(?:person|people|persons|man|woman|individual|pedestrian|visitor|intruder)\b"
    ),
    ObjectType.VEHICLE: re.compile(r"\b(?:vehicles?|pickups?|trucks?|cars?|vans?|sedans?|suvs?)\b"),
}
_ACTIVITY_CANON = {
    "standing": "standing", "stood": "standing",
    "walking": "walking", "walked": "walking",
    "running": "running", "ran": "running",
    "pacing": "pacing", "paced": "pacing",
    "sitting": "sitting",
    "crouching": "crouching", "crouched": "crouching",
    "climbing": "climbing", "climbed": "climbing",
    "driving": "driving", "drove": "driving", "drives": "driving", "driven": "driving",
    "parked": "parked", "parking": "parked",
    "stopped": "stopped", "waiting": "waiting", "waited": "waiting", "idling": "idling",
    "approaching": "approaching", "leaving": "leaving",
}  # fmt: skip
# "parking" is also a zone word ("parking area"), so a bare "parking" is only an activity when it is
# not followed by area/lot (handled where activities are extracted).
_WORD = re.compile(r"[a-z]+")

# Recommendations are the analyst's own words, not evidence. Their factual premises are still
# checked; a place or time a recommendation merely points at ("review the driveway footage") is not
# a premise, so zone/time/type/activity are only checked there when the sentence also states
# something as a fact.
_RECOMMENDATION = re.compile(
    r"\b(?:recommend\w*|should|suggest\w*|advis\w*|consider|next step|action:|i would|"
    r"you (?:may|might|could))\b",
    re.IGNORECASE,
)
_IMPERATIVE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:please\s+)?(?:check|review|verify|confirm|dispatch|notify|contact|inspect|"
    r"monitor|send|keep|investigate|escalate|log|compare|continue|hold|treat|assume|do not|don't)\b"
)
_TIMELINE = re.compile(
    r"\balert_(?:raised|escalated|resolved|acknowledged)\b|^\s*(?:[-*]\s*)?\d{4}-\d{2}-\d{2}"
)
_LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+[.)])\s")
_CONNECTOR = re.compile(r"(\b(?:because|since|given that)\b|[;:,]|\band\b|\bbut\b)")
_PAST_FACT = re.compile(r"\b(?:was|were|has been|have been|had)\b")


def _premise_text(body: str) -> str:
    """The clauses of a recommendation that state something as fact (past tense, or after
    "because" / "since"); the place a recommendation points at is not one of them."""
    kept, after = [], False
    for part in _CONNECTOR.split(body):
        if re.fullmatch(r"\b(?:because|since|given that)\b", part):
            after = True
        elif part == ";" or part == ":":
            after = False
        elif after or _PAST_FACT.search(part):
            kept.append(part)
    return " ".join(kept)


_HHMM = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?(?:\s*([ap])\.?m\.?)?\b")
_ISO = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
)
_UTC_OFFSET = re.compile(r"(?:(?<=\d)|(?<=[\s(])|(?<=UTC)|(?<=GMT))[+-]\d{2}:\d{2}\b")
_DURATION = re.compile(
    rf"(?<![:.\d/])\b({_NUM})(?:\s*-\s*|\s*)(seconds?|secs?|s|minutes?|mins?|hours?|hrs?)\b(?!\w)"
)
_APPEARANCES = re.compile(
    rf"(?<![:.\d/])\b(?P<n>{_NUM})\s+(?:(?:separate|distinct|different)\s+)?"
    r"(?:times|appearances?|visits?|sightings?|occasions)\b"
)
_APPEARED_ONCE_TWICE = re.compile(
    r"\b(?:seen|appeared|appears|observed|spotted|visited|sighted)\s+(?:only\s+|just\s+)?"
    r"(?P<n>once|twice|thrice)\b"
)
_RETURN_FORM = re.compile(r"\b(?:return(?:ed|s)?|came back|comes back|re-?entered|revisited)\b")
_OF_TOTAL = re.compile(
    rf"(?<![:.\d/])\b(?:{_NUM})\s+of\s+(?P<m>\d+)\s+(?P<noun>alerts?|events?|observations?|entities|entity)\b"
)
_TOTAL = re.compile(
    rf"(?<![:.\d/])\b(?P<n>{_NUM})\s+(?P<mods>(?:[\w\-/]+\s+){{0,3}}?)"
    r"(?P<noun>alerts?|events?|observations?(?:\(s\))?|entities|entity)\b"
    r"(?P<tail>(?:\s+(?:that\s+|which\s+)?(?:were|was|are|is|remain(?:s|ed)?)?\s*"
    rf"(?:{_STATUS}|unresolved))?"
    r"(?:\s+(?:in|at)\s+the\s+[\w ]+?(?=[,.;:)]|$|\s+(?:and|but|while)\b))?)"
)
_OBJECT_COUNT = re.compile(
    rf"(?<![:.\d/])\b(?P<n>{_NUM})\s+(?:distinct\s+)?"
    r"(?P<what>people|persons?|vehicles?|cars?|trucks?|pickups?|sedans?|vans?)\b"
)
_STOP_MODS = frozenset(
    {
        "stored", "total", "separate", "distinct", "similar", "different", "new", "more",
        "additional", "recorded", "raised", "fired", "generated", "detected", "observed", "the",
        "a", "an", "and", "of", "were", "was", "are", "is", "there",
    }
)  # fmt: skip
_GLOBAL_SCOPE = re.compile(
    r"\b(?:stored|total|in total|overall|altogether|in all|all told|the run|this run|"
    r"there (?:were|are|was|is)|fired|raised|generated|recorded|logged)\b"
)
_TRAILING_QUALIFIER = re.compile(
    r"\s*(?:for|about|involving|concerning|near|during|between|after|before|since|from|with|by|"
    r"related|linked|connected)\b"
)
_NOUN_KIND = {"alert": "alert", "event": "event", "observation": "observation", "entit": "entity"}


# ---- the evidence a sentence cites -------------------------------------------------------------
@dataclass
class _Facts:
    """Everything the stored evidence cited in ONE sentence says, pooled across its citations."""

    subjects: set[str] = field(default_factory=set)  # event | alert | entity | obs
    zones: set[str] = field(default_factory=set)
    types: set[str] = field(default_factory=set)
    activities: set[str] = field(default_factory=set)
    severities: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)  # current
    statuses_ever: set[str] = field(default_factory=set)  # history, incl. the current one
    rules: set[str] = field(default_factory=set)
    times: list[datetime] = field(default_factory=list)
    clock_minutes: set[int] = field(default_factory=set)  # configured clock times (closed hours)
    obs_ids: set[str] = field(default_factory=set)
    durations: list[float] = field(default_factory=list)  # how long things actually lasted
    # rule thresholds and ladder offsets: a duration only when the sentence is about a threshold
    thresholds: list[float] = field(default_factory=list)
    appearances: set[int] = field(default_factory=set)
    cited: Counter = field(default_factory=Counter)  # kind -> distinct cited ids
    object_counts: list[Counter] = field(default_factory=list)  # one per cited observation
    # counts that belong to a cited subject: observations behind each cited event, events behind
    # each cited entity or alert ("4 observation(s)" on an event line is NOT a stored total)
    evidence_counts: set[int] = field(default_factory=set)
    member_counts: set[int] = field(default_factory=set)


def _canon_activities(words: list[str]) -> set[str]:
    return {_ACTIVITY_CANON[w] for w in words if w in _ACTIVITY_CANON}


def _add_event(f: _Facts, ev: Event, cited: bool = False) -> None:
    if cited:
        f.evidence_counts.add(ev.evidence_count)
    f.zones.add(ev.zone.value)
    f.types.add(ev.object_type.value)
    for activity in [*ev.activities, ev.description]:
        f.activities |= _canon_activities(_WORD.findall(activity.lower()))
    f.times += [ev.first_seen, ev.last_seen]
    if ev.departure_observed_at:
        f.times.append(ev.departure_observed_at)
    f.durations.append(ev.duration_seconds)


def _add_alert_state(f: _Facts, al: Alert) -> None:
    """The severity, status and rule of an alert (also used for alerts linked to a cited event)."""
    f.rules.add(al.rule_id)
    f.severities.add(al.severity.value)
    f.severities |= {step.severity.value for step in al.ladder}
    f.statuses.add(al.status.value)
    f.statuses_ever.add(al.status.value)
    for h in al.history:
        f.severities.add(h.to_severity.value)
        if h.from_severity:
            f.severities.add(h.from_severity.value)
        f.statuses_ever.add(h.to_status.value)
        if h.kind is TransitionKind.RAISED:
            f.statuses_ever.add(AlertStatus.OPEN.value)


def _collect(store: Store, citations: list[Citation]) -> _Facts:
    f = _Facts()
    for c in citations:
        if c.kind == "event":
            ev = store.get_event(c.ref)
            if ev is None:
                continue
            f.subjects.add("event")
            f.cited["event"] += 1
            _add_event(f, ev, cited=True)
            for al in store.alerts_for_events([ev.event_id]):
                _add_alert_state(f, al)
        elif c.kind == "alert":
            al = store.get_alert(c.ref)
            if al is None:
                continue
            f.subjects.add("alert")
            f.cited["alert"] += 1
            _add_alert_state(f, al)
            if al.zone:
                f.zones.add(al.zone.value)
            f.times += [al.triggered_at, al.first_seen, al.last_seen]
            f.times += [t for t in (al.resolved_at, al.detected_at) if t]
            f.times += [step.at for step in al.ladder]
            f.times += [h.at for h in al.history] + [h.effective_at for h in al.history]
            if isinstance(al.facts.get("duration_seconds"), (int, float)):
                f.durations.append(float(al.facts["duration_seconds"]))
            for key in ("threshold_seconds", "critical_after_seconds"):
                if isinstance(al.facts.get(key), (int, float)):
                    f.thresholds.append(float(al.facts[key]))
            for value in al.facts.values():  # configured clock times, e.g. closed 22:00-06:00
                if isinstance(value, str) and re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value):
                    f.clock_minutes.add(int(value[:2]) * 60 + int(value[3:5]))
            f.durations.append((al.last_seen - al.first_seen).total_seconds())
            f.thresholds += [(step.at - al.first_seen).total_seconds() for step in al.ladder]
            if isinstance(al.facts.get("appearance_count"), int):
                f.appearances.add(al.facts["appearance_count"])
            f.member_counts.add(len(al.event_ids))
            for event_id in al.event_ids:
                ev = store.get_event(event_id)
                if ev is not None:
                    _add_event(f, ev)
        elif c.kind == "entity":
            ent = store.get_entity(c.ref)
            if ent is None:
                continue
            f.subjects.add("entity")
            f.cited["entity"] += 1
            f.types.add(ent.entity_type.value)
            f.appearances.add(ent.appearance_count)
            f.member_counts.add(ent.appearance_count)
            for event_id in ent.event_ids:
                ev = store.get_event(event_id)
                if ev is not None:
                    _add_event(f, ev)
        elif c.kind in ("obs", "frame"):
            obs: Observation | None = (
                store.get_observation(c.ref)
                if c.kind == "obs"
                else store.get_observation_by_frame(int(c.ref))
                if c.ref.isdigit()
                else None
            )
            if obs is None:
                continue
            f.subjects.add("obs")
            f.obs_ids.add(obs.observation_id)
            f.cited["observation"] = len(f.obs_ids)  # [obs:x] and [frame:n] are one observation
            f.zones.add(obs.zone.value)
            f.times.append(obs.timestamp)
            f.activities |= _canon_activities(_WORD.findall(obs.description.lower()))
            counts: Counter = Counter()
            for o in obs.detected_objects:
                f.types.add(o.object_type.value)
                counts[o.object_type.value] += 1
                if o.activity:
                    f.activities |= _canon_activities(_WORD.findall(o.activity.lower()))
            f.object_counts.append(counts)
        # telemetry citations carry no checkable claim here
    for counts in (f.evidence_counts, f.member_counts):
        if len(counts) > 1:
            counts.add(sum(counts))  # "9 observations" across three cited events
    return f


# ---- extraction ---------------------------------------------------------------------------------
def _clean(sentence: str) -> str:
    """Lower-case text with citations, ids, ISO timestamps reduced so they cannot be misread."""
    text = CITATION_RE.sub(" ", sentence)
    text = _ISO.sub(lambda m: f" {m.group(1)} ", text)
    text = _UTC_OFFSET.sub("", text)
    return _ID.sub(" ", text).lower()


def _spans_outside(text: str, spans: list[tuple[int, int]]) -> str:
    chars = list(text)
    for a, b in spans:
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def _mentions(text: str, regexes: dict) -> tuple[set, list[tuple[int, int]]]:
    found, spans = set(), []
    for key, rx in regexes.items():
        for m in rx.finditer(text):
            if not _negated(text, m.start()):
                found.add(key)
                spans.append(m.span())
    return found, spans


def extract_claims(sentence: str, *, has_obs_subject: bool = False) -> list[Claim]:
    """The claims one sentence makes, in the forms the checker understands (and no others)."""
    text = _clean(sentence)
    recommendation = bool(_RECOMMENDATION.search(text) or _IMPERATIVE.match(text))
    claims: list[Claim] = []

    def add(kind: str, values, words: str, **kw) -> None:
        claims.append(Claim(kind, tuple(values), sentence, words.strip(), **kw))

    # severity
    severities = set()
    for rx in _SEVERITY_PATTERNS:
        for m in rx.finditer(text):
            if not _negated(text, m.start()):
                severities.add(m.group(1))
    if severities:
        add("severity", sorted(severities), ", ".join(sorted(severities)))
    # status (present tense must match the current status; past tense may match the history)
    for m in _PRESENT_STATUS.finditer(text):
        if not _negated(text, m.start()):
            add("status", [m.group(1)], m.group(0), present=True)
    for m in _PAST_STATUS.finditer(text):
        if not _negated(text, m.start()):
            add("status", [m.group(1)], m.group(0), present=False)
    # "critical/escalated" is the alert's state now, except in a timeline entry, where it is the
    # state at that moment ("2026-09-30 00:06 alert_raised: Link lost: high/open")
    timeline = bool(_TIMELINE.search(sentence))
    for m in _SLASH_STATUS.finditer(text):
        add("status", [m.group(1)], m.group(0), present=not timeline)
    if _NOT_RESOLVED.search(text):
        add("status", ["not_resolved"], _NOT_RESOLVED.search(text).group(0), present=True)
    # rule ("the loitering alert", "restricted-zone rule")
    rules, rule_spans = _mentions(text, _RULE_RE)
    if rules:
        add("rule", sorted(rules), ", ".join(sorted(rules)))
    body = _spans_outside(text, rule_spans)  # a rule name is not a place
    if recommendation:
        body = _premise_text(body)
    header = sentence.rstrip().endswith(":")
    # zone / type / activity / time
    if body.strip() and not header:
        zones, _ = _mentions(body, _ZONE_RE)
        if zones:
            add("zone", sorted(z.value for z in zones), ", ".join(sorted(z.value for z in zones)))
        types, _ = _mentions(body, _TYPE_RE)
        if types:
            add("type", sorted(t.value for t in types), ", ".join(sorted(t.value for t in types)))
        # a bare "parking" is the zone; "parked" / "parking" followed by nothing zone-like is not
        activities = {
            _ACTIVITY_CANON[w]
            for w in _WORD.findall(_ZONE_RE[Zone.PARKING_AREA].sub(" ", body))
            if w in _ACTIVITY_CANON
        }
        activities |= {"parked"} if re.search(r"\bparked\b", body) else set()
        negated = {
            _ACTIVITY_CANON[m.group(0)]
            for m in re.finditer(r"[a-z]+", body)
            if m.group(0) in _ACTIVITY_CANON and _negated(body, m.start())
        }
        activities -= negated
        if activities:
            add("activity", sorted(activities), ", ".join(sorted(activities)))
        # clock times ("23:41", "11:41 pm"); hours like "12:30" elsewhere are read as times too
        times = []
        for m in _HHMM.finditer(body):
            hour, minute = int(m.group(1)), int(m.group(2))
            if m.group(3):
                hour = hour % 12 + (12 if m.group(3) == "p" else 0)
            times.append((hour, minute))
        if times:
            add("time", sorted(set(times)), ", ".join(f"{h:02d}:{m:02d}" for h, m in sorted(times)))
    # durations ("480 s", "8 minutes", "five minutes"): (seconds, stated in whole minutes/hours?)
    durations = []
    for m in _DURATION.finditer(_HHMM.sub(" ", text)):
        value = _number(m.group(1))
        if value is None:
            continue
        unit = m.group(2)
        scale = 3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1
        durations.append((value * scale, scale > 1))
    if durations:
        add("duration", durations, ", ".join(f"{s:g}s" for s, _ in durations))
    # appearances of a vehicle ("seen 3 times", "five separate appearances", "seen only twice")
    if not _RETURN_FORM.search(text):
        counts = [int(n) for m in _APPEARANCES.finditer(text) if (n := _number(m.group("n")))]
        counts += [
            {"once": 1, "twice": 2, "thrice": 3}[m.group("n")]
            for m in _APPEARED_ONCE_TWICE.finditer(text)
        ]
        if counts:
            add("appearances", counts, ", ".join(str(c) for c in counts))
    # totals of alerts / events / observations / entities, with the filters the sentence states
    scoped = bool(_GLOBAL_SCOPE.search(text))
    sentence_zones = sorted(
        z.value for z in _mentions(_spans_outside(text, rule_spans), _ZONE_RE)[0]
    )
    for m in _OF_TOTAL.finditer(text):  # "showing 20 of 25 events": only the total is a claim
        strict = scoped and not sentence_zones and not header
        add("total", [(_noun(m.group("noun")), int(m.group("m")), (), strict)], m.group(0))
    hidden = _spans_outside(text, [m.span() for m in _OF_TOTAL.finditer(text)])
    for m in _TOTAL.finditer(hidden):
        n = _number(m.group("n"))
        if n is None or n != int(n):
            continue
        filters = _count_filters(m.group("mods"), m.group("tail"))
        trailing = bool(_TRAILING_QUALIFIER.match(hidden[m.end() :]))
        if not filters and len(sentence_zones) == 1 and _noun(m.group("noun")) != "entity":
            filters = (("zone~", sentence_zones[0]),)  # "In main_gate: 5 stored observations"
        exact = bool(filters) and ("?", "") not in filters  # the phrase states its own scope
        global_total = not filters and scoped and not trailing and not sentence_zones and not header
        add(
            "total",
            [(_noun(m.group("noun")), int(n), filters, exact or global_total)],
            m.group(0),
        )
    # object counts in a cited frame ("two people")
    if has_obs_subject:
        for m in _OBJECT_COUNT.finditer(text):
            n = _number(m.group("n"))
            if n is not None and not _negated(text, m.start()):
                what = "person" if m.group("what").startswith(("people", "person")) else "vehicle"
                add("object_count", [(what, int(n))], m.group(0))
    return [
        Claim(
            c.kind,
            c.values,
            c.sentence,
            c.text,
            c.present,
            recommendation,
            header and c.kind == "total",
        )
        for c in claims
    ]


def _noun(word: str) -> str:
    return next(v for k, v in _NOUN_KIND.items() if word.startswith(k))


def _count_filters(mods: str, tail: str) -> tuple:
    """The filters a count phrase states itself, as sorted (family, value) pairs. A modifier that
    is not understood yields (("?", ""),): the count is then only checked against an upper bound."""
    filters: set[tuple[str, str]] = set()
    text = mods.lower()
    for rule, rx in _RULE_PHRASE_ONLY.items():
        if rx.search(text):
            filters.add(("rule", rule))
            text = rx.sub(" ", text)
    unknown = False
    for word in re.findall(r"[a-z]+", text):
        if word in _STOP_MODS:
            continue
        if word in _SEV.split("|"):
            filters.add(("severity", word))
        elif word in _STATUS.split("|"):
            filters.add(("status", word))
        elif word == "unresolved":
            filters.add(("status", "not_resolved"))
        elif _TYPE_RE[ObjectType.PERSON].fullmatch(word):
            filters.add(("type", "person"))
        elif _TYPE_RE[ObjectType.VEHICLE].fullmatch(word):
            filters.add(("type", "vehicle"))
        else:
            unknown = True
    tail = tail.lower()
    status = re.search(rf"\b({_STATUS}|unresolved)\b", tail)
    if status:
        filters.add(
            ("status", "not_resolved" if status.group(1) == "unresolved" else status.group(1))
        )
    for zone, rx in _ZONE_RE.items():
        if rx.search(tail):
            filters.add(("zone", zone.value))
    if unknown:
        filters.add(("?", ""))
    return tuple(sorted(filters))


# ---- assessment ---------------------------------------------------------------------------------
_TIME_TOLERANCE_MIN = 1
_DURATION_REL = 0.02
_ROUNDED_ABS_S = 30.0
_THRESHOLD_CONTEXT = re.compile(
    r"\b(?:threshold|escalat\w*|warn\w*|critical|ladder|rung|reach\w*|triggered|within)\b"
)


def _minutes(t: datetime) -> int:
    return t.hour * 60 + t.minute


def _says(subset, allowed: set, noun: str) -> tuple[bool, str]:
    return set(subset) <= allowed, f"{noun} {sorted(allowed)}"


def _stored_total(store: Store, noun: str, filters: tuple) -> tuple[int, str] | None:
    """How many stored items match the filters the sentence states (None: not computable)."""
    f: dict[str, set] = {}
    for family, value in filters:
        f.setdefault(family.rstrip("~"), set()).add(value)
    if noun == "alert":
        if set(f) - {"severity", "status", "rule", "zone"}:
            return None
        alerts = store.query_alerts()
        kept = [
            a
            for a in alerts
            if (not f.get("severity") or a.severity.value in f["severity"])
            and (not f.get("rule") or a.rule_id in f["rule"])
            and (not f.get("zone") or (a.zone and a.zone.value in f["zone"]))
            and (
                not f.get("status")
                or a.status.value in f["status"]
                or ("not_resolved" in f["status"] and a.status is not AlertStatus.RESOLVED)
            )
        ]
        return len(kept), "alerts"
    if noun == "event":
        if set(f) - {"type", "zone", "status"}:
            return None
        events = store.query_events()
        kept = [
            e
            for e in events
            if (not f.get("type") or e.object_type.value in f["type"])
            and (not f.get("zone") or e.zone.value in f["zone"])
            and (not f.get("status") or (e.is_open and "open" in f["status"]) or not e.is_open)
        ]
        return len(kept), "events"
    if noun == "observation":
        if set(f) - {"type", "zone"}:
            return None
        if not f:
            return store.count_observations(), "observations"
        zones = [Zone(z) for z in f.get("zone", [])] or [None]
        types = [ObjectType(t) for t in f.get("type", [])] or [None]
        seen = {
            o.observation_id
            for z in zones
            for t in types
            for o in store.query_observations(zone=z, object_type=t)
        }
        return len(seen), "observations"
    return (store.count_entities(), "entities") if not f else None


def _assess(store: Store, claim: Claim, f: _Facts) -> Finding | None:
    """Compare one claim with the evidence cited next to it. None = nothing to compare it with."""
    kind, values = claim.kind, claim.values
    none = Verdict.UNSUPPORTED

    def verdict(ok: bool, expected: str) -> Finding:
        return Finding(claim, Verdict.SUPPORTED if ok else Verdict.CONTRADICTED, expected)

    def missing(what: str) -> Finding:
        return Finding(claim, none, f"the cited evidence records no {what}")

    if kind == "total":
        noun, n, filters, strict = values[0]
        own = (
            f.evidence_counts
            if noun == "observation"
            else f.member_counts
            if noun == "event"
            else set()
        )
        if own and not claim.header and all(fam == "zone~" for fam, _ in filters):
            # the count belongs to the cited event/entity/alert ("... 4 observation(s)")
            return verdict(n in own, f"the cited evidence has {sorted(own)} {noun}(s)")
        stored = _stored_total(store, noun, tuple(filters))
        if stored is None:
            return None
        count, label = stored
        if claim.header and f.cited[noun]:
            # "6 alerts raised by the rules:" introduces a list; it must be as long as the list
            listed = f.cited[noun]
            return verdict(n == listed, f"{listed} {label} are cited under this heading")
        if strict:
            return verdict(
                n == count,
                f"{count} {label} are stored" + (f" with {_fmt(filters)}" if filters else ""),
            )
        return verdict(n <= count, f"at most {count} {label} are stored")
    if not f.subjects:
        return None  # only telemetry (or nothing) was cited: no stored fact to compare with
    if kind == "severity":
        if not f.severities:
            return missing("alert severity")
        return verdict(*_says(values, f.severities, "the cited alert(s) have severity"))
    if kind == "status":
        if not f.statuses:
            return missing("alert status")
        value = values[0]
        if value == "not_resolved":
            ok = any(s != "resolved" for s in f.statuses)
        elif claim.present:
            ok = (
                any(s != "resolved" for s in f.statuses) if value == "open" else value in f.statuses
            )
        else:
            ok = value in f.statuses_ever
        return verdict(ok, f"the cited alert(s) have status {sorted(f.statuses)}")
    if kind == "rule":
        if not f.rules:
            return missing("alert rule")
        return verdict(*_says(values, f.rules, "the cited alert(s) come from rule"))
    if kind == "zone":
        if not f.zones:
            return missing("zone")
        return verdict(*_says(values, f.zones, "the cited evidence is in zone"))
    if kind == "type":
        if not f.types:
            return missing("object type")
        return verdict(*_says(values, f.types, "the cited evidence concerns"))
    if kind == "activity":
        if not f.activities:
            return missing("activity")
        return verdict(*_says(values, f.activities, "the cited evidence records activity"))
    if kind == "time":
        if not f.times and not f.clock_minutes:
            return missing("time")
        known = [_minutes(t) for t in f.times] + sorted(f.clock_minutes)
        ok = all(
            any(
                min(abs(h * 60 + m - k), 1440 - abs(h * 60 + m - k)) <= _TIME_TOLERANCE_MIN
                for k in known
            )
            for h, m in values
        )
        stamps = sorted({t.strftime("%H:%M") for t in f.times})
        stamps += [f"{k // 60:02d}:{k % 60:02d}" for k in sorted(f.clock_minutes)]
        return verdict(ok, "the cited evidence is timed " + ", ".join(stamps))
    if kind == "duration":
        if not (f.subjects & {"event", "alert", "entity"}):
            return None  # a duration next to a frame citation is not about a stored duration
        # "stayed 300 seconds" must match how long it stayed; "escalated after 300 seconds" may
        # match the rule's threshold or ladder
        about_threshold = bool(_THRESHOLD_CONTEXT.search(claim.sentence.lower()))
        known_durations = f.durations + (f.thresholds if about_threshold else [])
        if not known_durations:
            return missing("duration")

        def matches(seconds: float, rounded: bool) -> bool:
            return any(
                abs(seconds - d) <= (_ROUNDED_ABS_S if rounded else max(1.0, _DURATION_REL * d))
                or abs(seconds - d) <= _DURATION_REL * d
                for d in known_durations
            )

        ok = all(matches(s, r) for s, r in values)
        known = ", ".join(f"{d:g}s" for d in sorted({round(d, 1) for d in known_durations}))
        return verdict(ok, f"the cited evidence has duration(s) {known}")
    if kind == "appearances":
        known_counts = set(f.appearances) or ({f.cited["event"]} if f.cited["event"] else set())
        if not known_counts:
            return None
        return verdict(set(values) <= known_counts, f"appearance count {sorted(known_counts)}")
    if kind == "object_count":
        if not f.object_counts:
            return None
        what, n = values[0]
        seen = sorted({c[what] for c in f.object_counts})
        return verdict(n in seen, f"the cited frame(s) show {seen} {what} object(s)")
    return None


def _fmt(filters) -> str:
    return ", ".join(v for _, v in filters)


def check_facts(store: Store, text: str) -> FactReport:
    """Extract the checkable factual claims of an answer and compare each with the store."""
    units = claim_units(text)
    cites = [[Citation(k, r) for k, r in dict.fromkeys(CITATION_RE.findall(u))] for u in units]
    report = FactReport()
    for i, unit in enumerate(units):
        unit_cites = list(cites[i])
        if not unit_cites and unit.rstrip().endswith(":"):  # a list header: its items carry cites
            for j in range(i + 1, len(units)):
                if not cites[j] or not _LIST_ITEM.match(units[j]):
                    break
                unit_cites += cites[j]
        facts = _collect(store, unit_cites)
        for claim in extract_claims(unit, has_obs_subject="obs" in facts.subjects):
            finding = _assess(store, claim, facts)
            if finding is not None:
                report.findings.append(finding)
    return report
