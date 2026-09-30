"""Question guard: identity, intent and criminality questions get a fixed refusal, not an answer.

WHAT THIS IS. A deterministic, wording-based filter on the QUESTION, applied by the dashboard before
the analyst is called. Each category is a short list of regular expressions; a match returns a
fixed refusal text. Nothing is learned, scored or interpreted.

WHAT THIS IS NOT. It is not a semantic safety classifier. A paraphrase can slip past it and a
harmless question can, rarely, match. It does not replace the analyst's own controls (answers must
cite retrieved evidence, and identity or intent wording in an answer is rejected by the citation
gate). The frozen analyst is untouched; this module only decides whether to call it.

Why refuse at all: the analyst answers "Is he a criminal?" by listing what happened in a zone,
which silently skips the question. A visible refusal is more honest than a quiet non-answer.
"""

import re
from dataclasses import dataclass

_I = re.IGNORECASE
_PEOPLE = r"(?:person|man|woman|guy|individual|driver|owner|visitor|intruder|stranger|people)"

# (category, patterns). Keep the lists short and literal so a reader can audit them.
_RULES: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "identity",
        tuple(
            re.compile(p, _I)
            for p in (
                r"\bidentif(?:y|ied|ies|ication)\b",
                r"\bidentity of\b",
                rf"\bwho (?:is|was|are|were) (?:he|she|they|it|that|this|the {_PEOPLE})\b",
                r"\bwho (?:owns|owned)\b",
                rf"\b(?:his|her|their|the {_PEOPLE}'?s?) (?:name|identity|address|phone|age)\b",
                r"\bregistered (?:owner|to)\b",
                rf"\bsame {_PEOPLE}\b",
                r"\b(?:face|facial|recogni[sz]e|recognition|biometric|fingerprint)\b",
            )
        ),
    ),
    (
        "intent",
        tuple(
            re.compile(p, _I)
            for p in (
                r"\b(?:intend|intends|intended|intention|intentions|motive|motives)\b",
                r"\b(?:planning|plotting|plans? to|going to (?:steal|rob|break|attack|harm))\b",
                # "why was the person alert raised?" asks about an alert, so only the verbs that
                # ask about someone's own action are listed
                rf"\bwhy (?:did|would|does|do) (?:he|she|they|the {_PEOPLE})\b",
                r"\bwhat (?:was|is) (?:he|she|they) (?:doing|up to|thinking|after|looking for)\b",
                r"\b(?:up to no good|casing)\b",
            )
        ),
    ),
    (
        "criminality",
        tuple(
            re.compile(p, _I)
            for p in (
                r"\b(?:criminals?|crimes?|thie(?:f|ves)|burglar\w*|robber\w*|robbery|theft)\b",
                r"\b(?:steal|stealing|stole|guilty|suspects?|suspected|vandal\w*|trespass\w*)\b",
                r"\b(?:illegal|illegally|unlawful|break-?ins?|breaking in)\b",
            )
        ),
    ),
    (
        "danger",
        tuple(re.compile(p, _I) for p in (r"\b(?:dangerous|armed|weapons?|threatening)\b",)),
    ),
)

_REFUSALS = {
    "identity": (
        "I can't identify people or say who someone is. This system never links or names "
        "individuals: each person event stands alone, and vehicles are only grouped as coarse "
        "'similar vehicle' matches, never confirmed identity."
    ),
    "intent": (
        "I can't infer what someone intends, wants or is planning. The stored evidence records "
        "what was observed (type, place, time, duration), not motives."
    ),
    "criminality": (
        "I can't decide that anyone is a criminal or committed a crime. Alerts come from fixed "
        "rules about what was observed (for example presence in a zone at a time); they are not "
        "findings of wrongdoing."
    ),
    "danger": (
        "I can't judge danger or say whether someone is armed or threatening. Only what the stored "
        "observations describe can be reported."
    ),
}

POLICY = (
    "Wording-based filter: fixed regular expressions on the question, not a semantic safety "
    "classifier. Paraphrases can slip through; answers are still checked by the citation gate."
)


@dataclass(frozen=True)
class GuardDecision:
    category: str
    matched: str  # the phrase that triggered it, so the refusal is explainable
    text: str  # the fixed refusal shown to the user


def check_question(question: str) -> GuardDecision | None:
    """A refusal for an identity / intent / criminality / danger question, else None."""
    for category, patterns in _RULES:
        for pattern in patterns:
            found = pattern.search(question)
            if found:
                return GuardDecision(category, found.group(0), _REFUSALS[category])
    return None
