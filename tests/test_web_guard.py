"""The dashboard's question guard: deterministic, wording-based, isolated from the analyst."""

from pathlib import Path

import pytest

from drone_agent.agent.offline import SUPPORTED_QUESTIONS
from drone_agent.web.guard import POLICY, check_question

REFUSED = [
    ("identity", "Who is the person at the gate?"),
    ("identity", "Can you identify the driver?"),
    ("identity", "What is the identity of the driver?"),
    ("identity", "What is his name?"),
    ("identity", "Who owns the blue pickup?"),
    ("identity", "Was it the same person as yesterday?"),
    ("identity", "Run facial recognition on the intruder"),
    ("intent", "What was he planning?"),
    ("intent", "Why did he come at night?"),
    ("intent", "What is his motive?"),
    ("intent", "What was she up to at the warehouse?"),
    ("intent", "Are they casing the property?"),
    ("criminality", "Is the person at the gate a criminal?"),
    ("criminality", "Was that a burglary attempt?"),
    ("criminality", "Is he a thief?"),
    ("criminality", "Is this suspect guilty?"),
    ("criminality", "Did someone break in illegally?"),
    ("danger", "Was he armed?"),
    ("danger", "Is the visitor dangerous?"),
]

ALLOWED = [
    "What happened at the main gate?",
    "Was the blue pickup seen more than once?",
    "Why was the repeat-vehicle alert raised?",
    "Why was the person alert raised?",
    "Show me the evidence for the loitering alert.",
    "Were any people seen after hours?",
    "List the alerts.",
    "Summarize the run.",
    "What objects were observed?",
    "Describe ALR-LOITER-EVT-P-001",
    "Where was it seen?",
    "What is the airspeed velocity of an unladen swallow?",
]


@pytest.mark.parametrize(("category", "question"), REFUSED)
def test_identity_intent_criminality_and_danger_questions_are_refused(category, question):
    decision = check_question(question)
    assert decision is not None, question
    assert decision.category == category
    assert decision.matched.lower() in question.lower()
    assert decision.text.startswith("I can't")


@pytest.mark.parametrize("question", ALLOWED)
def test_ordinary_evidence_questions_are_not_blocked(question):
    assert check_question(question) is None


def test_every_example_the_template_analyst_advertises_is_allowed():
    for example in SUPPORTED_QUESTIONS:
        for part in example.replace(" / ", "|").split("|"):
            question = part.split("(")[0].strip()
            assert check_question(question) is None, question


def test_matching_ignores_case_and_is_deterministic():
    decisions = [check_question(q) for q in ("IS HE A CRIMINAL?", "is he a criminal?")]
    assert all(d is not None and d.category == "criminality" for d in decisions)
    assert decisions[0].text == decisions[1].text
    assert decisions[0].matched.lower() == decisions[1].matched.lower() == "criminal"
    assert check_question("is he a criminal?") == check_question("is he a criminal?")


def test_a_refusal_is_one_of_four_fixed_texts_and_never_an_answer():
    from drone_agent.web.guard import _REFUSALS

    texts = {check_question(question).text for _, question in REFUSED}
    assert texts == set(_REFUSALS.values())
    assert all(t.startswith("I can't") for t in texts)


def test_the_policy_says_it_is_wording_based_and_not_a_semantic_classifier():
    assert "Wording-based" in POLICY and "not a semantic safety classifier" in POLICY
    assert "Paraphrases can slip through" in POLICY


def test_the_guard_does_not_import_the_analyst_or_any_model_code():
    import drone_agent.web.guard as guard

    source = Path(guard.__file__).read_text(encoding="utf-8")
    for forbidden in ("drone_agent.agent", "drone_agent.vision", "google", "requests", "socket"):
        assert f"import {forbidden}" not in source and f"from {forbidden}" not in source
