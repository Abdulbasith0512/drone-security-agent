"""Internal evidence citations and their validation.

Answers cite stored evidence with tokens like [event:EVT-V-001]. Only these kinds exist:

    [frame:10]  [obs:obs-10]  [event:EVT-V-001]  [entity:ENT-V-001]  [alert:ALR-REPEAT-ENT-V-001]
    [tel:123]   (a stored telemetry sample)

A citation is valid only if that id exists in the database (and, when a set of retrieved ids is
given, was actually returned by a tool in this session). These are references into our own store,
never external sources. This module has no LangChain dependency.

The verifier has three levels, from cheapest to strictest:

1. every cited id must exist and must have been retrieved this session (no invented evidence);
2. the answer must cite something, and must not use identity/intent overclaim wording;
3. (`sentence_level`, used for language-model answers) every FACTUAL sentence must carry its own
   citation, so one valid citation cannot launder a paragraph of uncited claims.

`repair_citations` fixes only formatting slips (spacing, capitalisation, zero-padding of an id that
was retrieved) before an answer is rejected; it never substitutes a different id.
"""

import re
from dataclasses import dataclass, field

from drone_agent.models import Zone
from drone_agent.store import Store

CITATION_RE = re.compile(r"\[(frame|obs|event|entity|alert|tel):([A-Za-z0-9_.\-]+)\]")
KNOWN_KINDS = ("frame", "obs", "event", "entity", "alert", "tel")
# Loose detector for things that look like citations but are malformed (e.g. "[event EVT-1]").
_MALFORMED_RE = re.compile(r"\[(?:frame|obs|event|entity|alert|tel)\b[^\]]*\]", re.IGNORECASE)
# A citation with harmless spacing or capitalisation slips, which repair can normalise.
_SLOPPY_RE = re.compile(
    r"\[\s*(frame|obs|event|entity|alert|tel)\s*[:\s]\s*([A-Za-z0-9_.\-]+)\s*\]", re.IGNORECASE
)
# Citation-shaped tokens of an UNRECOGNISED type: "[evt:EVT-1]", "[source:wikipedia]",
# "[Event:EVT-1]" (kinds are lower-case). They are never ignored: an answer containing one fails.
# Ordinary brackets without a "word:" prefix ("[1]", "[12:30]") are not affected.
_UNKNOWN_KIND_RE = re.compile(r"\[\s*([A-Za-z][A-Za-z_\-]*)\s*:[^\[\]]*\]")

# Wording an analyst answer must not contain (identity / intent overclaims). A match preceded by a
# negation ("not the same person") is allowed.
_OVERCLAIMS = re.compile(
    r"\b(the same (?:person|individual|man|woman|intruder)|"
    r"confirmed (?:to be )?the same (?:vehicle|truck|car|pickup)|"
    r"criminal(?:ly)?|burglar\w*|thie(?:f|ves)|committing a crime|intent to (?:steal|rob|harm))\b",
    re.IGNORECASE,
)
# Negation is recognised narrowly and syntactically, never semantically: a negation cue must sit in
# the same clause (no . ; : ! ? , or "but"/"however" in between) at most this many words before the
# phrase, and the words in between must not include "doubt" (so "no doubt this is the same person"
# is still rejected). Double negatives such as "I do not doubt ..." are not understood: a limit.
_NEGATION_WINDOW_WORDS = 8
_CLAUSE_BREAK = re.compile(r"[.;:!?,\n]|\b(?:but|however)\b", re.IGNORECASE)
_NEGATION_WORDS = frozenset({"not", "never", "no", "cannot", "without"})

# ---- sentence-level checking -----------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# "e.g." and friends end in a dot but not a sentence: hide that dot from the splitter.
_ABBREVIATION_PATTERN = r"(?<![A-Za-z])(?:e\.g|i\.e|etc|vs|approx|min|max)\."
_ABBREVIATION = re.compile(_ABBREVIATION_PATTERN, re.IGNORECASE)
_DOT = "․"
_META = re.compile(
    r"^\s*(?:[-*]\s*)?(?:insufficient evidence|i (?:could|can)\s?not|i cannot|i do not have|"
    r"there (?:is|are) no|no (?:\w+ ){0,4}(?:is|are|was|were)?\s*(?:stored|found|available)|"
    r"note:|limitations?:|verified citations|\(showing|\(the search|these are|this is a)",
    re.IGNORECASE,
)
_ZONE_WORDS = tuple(z.value.replace("_", " ") for z in Zone) + tuple(z.value for z in Zone)
_CLAIM_MARKERS = re.compile(
    r"\d|\b(?:evt|alr|ent)-|\bobs-|"
    r"\b(?:alert|alerts|event|events|person|people|vehicle|pickup|truck|car|van|sedan|drone|"
    r"battery|signal|altitude|frame|frames|observation|observations|detected|raised|escalated|"
    r"resolved|acknowledged|critical|severity|loiter\w*|rule)\b",
    re.IGNORECASE,
)
MIN_CLAIM_WORDS = 4


@dataclass(frozen=True)
class Citation:
    kind: str
    ref: str

    @property
    def token(self) -> str:
        return f"[{self.kind}:{self.ref}]"


def extract_citations(text: str) -> list[Citation]:
    """All well-formed citation tokens, in order, without duplicates."""
    seen: dict[tuple[str, str], Citation] = {}
    for kind, ref in CITATION_RE.findall(text):
        seen.setdefault((kind, ref), Citation(kind, ref))
    return list(seen.values())


def citation_exists(store: Store, citation: Citation) -> bool:
    kind, ref = citation.kind, citation.ref
    if kind == "frame":
        return ref.isdigit() and store.get_observation_by_frame(int(ref)) is not None
    if kind == "obs":
        return store.get_observation(ref) is not None
    if kind == "event":
        return store.get_event(ref) is not None
    if kind == "entity":
        return store.get_entity(ref) is not None
    if kind == "tel":
        return ref.isdigit() and store.get_telemetry(int(ref)) is not None
    return store.get_alert(ref) is not None


def _negated(text: str, start: int) -> bool:
    """Is the phrase starting at `start` preceded, in its own clause, by a negation cue?"""
    clause = _CLAUSE_BREAK.split(text[:start])[-1]
    words = clause.split()
    for i in range(len(words) - 1, -1, -1):
        word = re.sub(r"[^\w']", "", words[i].lower())
        if word in _NEGATION_WORDS or word.endswith("n't"):
            between = words[i + 1 :]
            return len(between) <= _NEGATION_WINDOW_WORDS and not any(
                "doubt" in w.lower() for w in between
            )
    return False


def find_unknown_kinds(text: str) -> list[str]:
    """Citation-shaped tokens whose type is not one of KNOWN_KINDS, in order of appearance."""
    return [m.group(0) for m in _UNKNOWN_KIND_RE.finditer(text) if m.group(1) not in KNOWN_KINDS]


def find_overclaims(text: str) -> list[str]:
    return [m.group(0) for m in _OVERCLAIMS.finditer(text) if not _negated(text, m.start())]


def claim_units(text: str) -> list[str]:
    """The answer split into sentences and list items (the unit a citation must accompany)."""
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        protected = _ABBREVIATION.sub(lambda m: m.group(0)[:-1] + _DOT, line)
        units.extend(
            u.strip().replace(_DOT, ".") for u in _SENTENCE_SPLIT.split(protected) if u.strip()
        )
    return units


def _is_factual(unit: str) -> bool:
    """A sentence that asserts something about the stored evidence (and so needs a citation)."""
    if len(unit.split()) < MIN_CLAIM_WORDS or _META.match(unit):
        return False
    return bool(_CLAIM_MARKERS.search(unit) or any(z in unit.lower() for z in _ZONE_WORDS))


def find_uncited_claims(text: str) -> list[str]:
    """Factual sentences that carry no citation of their own.

    A line that ends with a colon introduces a list and is covered by the cited items after it.
    """
    units = claim_units(text)
    missing: list[str] = []
    for i, unit in enumerate(units):
        if not _is_factual(unit) or CITATION_RE.search(unit):
            continue
        if unit.rstrip().endswith(":") and i + 1 < len(units) and CITATION_RE.search(units[i + 1]):
            continue
        missing.append(unit)
    return missing


def is_pure_refusal(text: str) -> bool:
    """True when the answer only says the evidence is insufficient (no factual claims at all)."""
    units = claim_units(text)
    return (
        bool(units)
        and units[0].lower().startswith("insufficient evidence")
        and not any(
            _is_factual(u) and not u.lower().startswith("insufficient evidence") for u in units
        )
    )


# ---- repair --------------------------------------------------------------------------------------
def _canonical(ident: str) -> str:
    """An id with case and zero-padding removed ("EVT-V-01" and "evt-v-001" agree)."""
    if ident.isdigit():
        return str(int(ident))
    return re.sub(r"(?<=\D)0+(?=\d)", "", ident.upper())


def repair_citations(text: str, retrieved: set[str]) -> tuple[str, list[tuple[str, str]]]:
    """Fix formatting slips in citations, using only ids that were actually retrieved.

    Repaired: spacing and capitalisation of the token ("[Event: EVT-V-001]"), and case or
    zero-padding of the id ("[event:evt-v-1]" for the retrieved EVT-V-001). Never repaired: an id
    that differs in ANY digit or letter beyond that, because a neighbouring id is a different
    piece of evidence (EVT-P-002 is not EVT-P-003) and swapping it would change the claim. Such
    ids stay invalid so validation rejects the answer. Returns the new text and the (before,
    after) pairs that were applied.
    """
    by_kind: dict[str, dict[str, list[str]]] = {}
    for item in retrieved:
        kind, _, ident = item.partition(":")
        by_kind.setdefault(kind, {}).setdefault(_canonical(ident), []).append(ident)
    applied: list[tuple[str, str]] = []

    def fix(match: re.Match[str]) -> str:
        original = match.group(0)
        kind, ident = match.group(1).lower(), match.group(2)
        candidates = by_kind.get(kind, {}).get(_canonical(ident), [])
        if len(candidates) != 1:  # not retrieved, or ambiguous: leave it alone
            return original
        fixed = f"[{kind}:{candidates[0]}]"
        if fixed != original:
            applied.append((original, fixed))
        return fixed

    return _SLOPPY_RE.sub(fix, text), applied


@dataclass
class CitationReport:
    cited: list[Citation] = field(default_factory=list)
    invalid: list[Citation] = field(default_factory=list)  # not in the database
    not_retrieved: list[Citation] = field(default_factory=list)  # exists, but no tool returned it
    malformed: list[str] = field(default_factory=list)
    unknown_kinds: list[str] = field(default_factory=list)  # e.g. "[evt:EVT-1]", "[source:x]"
    overclaims: list[str] = field(default_factory=list)
    missing: bool = False  # evidence was available but the answer cites nothing
    uncited_claims: list[str] = field(default_factory=list)  # factual sentences with no citation

    @property
    def ok(self) -> bool:
        return not (
            self.invalid
            or self.not_retrieved
            or self.malformed
            or self.unknown_kinds
            or self.overclaims
            or self.missing
            or self.uncited_claims
        )

    def problems(self) -> list[str]:
        out = []
        if self.invalid:
            out.append("cited ids that do not exist: " + ", ".join(c.token for c in self.invalid))
        if self.not_retrieved:
            out.append(
                "cited ids that no tool returned in this conversation: "
                + ", ".join(c.token for c in self.not_retrieved)
            )
        if self.malformed:
            out.append("malformed citation tokens: " + ", ".join(self.malformed))
        if self.unknown_kinds:
            out.append(
                "unknown citation type (allowed: "
                + ", ".join(KNOWN_KINDS)
                + "): "
                + ", ".join(self.unknown_kinds)
            )
        if self.overclaims:
            out.append("identity/intent wording is not allowed: " + ", ".join(self.overclaims))
        if self.missing:
            out.append("the answer states facts but cites no evidence")
        if self.uncited_claims:
            shown = "; ".join(f'"{c[:70]}"' for c in self.uncited_claims[:3])
            more = (
                f" (+{len(self.uncited_claims) - 3} more)" if len(self.uncited_claims) > 3 else ""
            )
            out.append(f"factual sentences without their own citation: {shown}{more}")
        return out


def validate_answer(
    store: Store,
    text: str,
    *,
    retrieved: set[str] | None = None,
    require_citation: bool = True,
    sentence_level: bool = False,
) -> CitationReport:
    """Check an analyst answer.

    `retrieved` holds "kind:id" strings returned by tools so far; when given, every citation must
    be in it. `require_citation` is False when no tool returned any evidence (the honest answer
    is then "insufficient evidence", which needs nothing to cite). `sentence_level` additionally
    requires each factual sentence to carry its own citation (used for model-written answers).
    """
    citations = extract_citations(text)
    report = CitationReport(cited=citations)
    for c in citations:
        if not citation_exists(store, c):
            report.invalid.append(c)
        elif retrieved is not None and f"{c.kind}:{c.ref}" not in retrieved:
            report.not_retrieved.append(c)
    well_formed = {m.group(0) for m in CITATION_RE.finditer(text)}
    report.malformed = [m for m in _MALFORMED_RE.findall(text) if m not in well_formed]
    report.unknown_kinds = find_unknown_kinds(text)
    report.overclaims = find_overclaims(text)
    report.missing = require_citation and not citations
    if sentence_level:
        report.uncited_claims = find_uncited_claims(text)
    return report
