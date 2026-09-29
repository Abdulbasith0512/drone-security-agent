"""Internal evidence citations and their validation.

Answers cite stored evidence with tokens like [event:EVT-V-001]. Only these kinds exist:

    [frame:10]  [obs:obs-10]  [event:EVT-V-001]  [entity:ENT-V-001]  [alert:ALR-REPEAT-ENT-V-001]

A citation is valid only if that id exists in the database (and, when a set of retrieved ids is
given, was actually returned by a tool in this session). These are references into our own store,
never external sources. This module has no LangChain dependency.
"""

import re
from dataclasses import dataclass, field

from drone_agent.store import Store

CITATION_RE = re.compile(r"\[(frame|obs|event|entity|alert):([A-Za-z0-9_.\-]+)\]")
# Loose detector for things that look like citations but are malformed (e.g. "[event EVT-1]").
_MALFORMED_RE = re.compile(r"\[(?:frame|obs|event|entity|alert)\b[^\]]*\]", re.IGNORECASE)

# Wording an analyst answer must not contain (identity / intent overclaims). A match preceded by a
# negation ("not the same person") is allowed.
_OVERCLAIMS = re.compile(
    r"\b(the same (?:person|individual|man|woman|intruder)|"
    r"confirmed (?:to be )?the same (?:vehicle|truck|car|pickup)|"
    r"criminal(?:ly)?|burglar\w*|thie(?:f|ves)|committing a crime|intent to (?:steal|rob|harm))\b",
    re.IGNORECASE,
)
_NEGATION_BEFORE = re.compile(
    r"(not|never|no|n't|cannot|without)\s+(?:\w+\s+){0,2}$", re.IGNORECASE
)


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
    return store.get_alert(ref) is not None


def find_overclaims(text: str) -> list[str]:
    found = []
    for m in _OVERCLAIMS.finditer(text):
        if not _NEGATION_BEFORE.search(text[max(0, m.start() - 30) : m.start()]):
            found.append(m.group(0))
    return found


@dataclass
class CitationReport:
    cited: list[Citation] = field(default_factory=list)
    invalid: list[Citation] = field(default_factory=list)  # not in the database
    not_retrieved: list[Citation] = field(default_factory=list)  # exists, but no tool returned it
    malformed: list[str] = field(default_factory=list)
    overclaims: list[str] = field(default_factory=list)
    missing: bool = False  # evidence was available but the answer cites nothing

    @property
    def ok(self) -> bool:
        return not (
            self.invalid or self.not_retrieved or self.malformed or self.overclaims or self.missing
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
        if self.overclaims:
            out.append("identity/intent wording is not allowed: " + ", ".join(self.overclaims))
        if self.missing:
            out.append("the answer states facts but cites no evidence")
        return out


def validate_answer(
    store: Store,
    text: str,
    *,
    retrieved: set[str] | None = None,
    require_citation: bool = True,
) -> CitationReport:
    """Check an analyst answer.

    `retrieved` holds "kind:id" strings returned by tools so far; when given, every citation must
    be in it. `require_citation` is False when no tool returned any evidence (the honest answer
    is then "insufficient evidence", which needs nothing to cite).
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
    report.overclaims = find_overclaims(text)
    report.missing = require_citation and not citations
    return report
