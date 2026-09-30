"""T03: a real citation must not make a false factual statement acceptable.

Every adversarial answer below cites evidence that EXISTS and WAS RETRIEVED, so the citation gate
alone accepts it (asserted on each one: that is the old failure mode). The factual check compares
the facts the sentence states with the evidence it cites and must reject it. The matching true
answers must still pass. Only structured facts are checked, never general truth.
"""

import json
from pathlib import Path

import pytest
from conftest import SCENARIO_DIR

from drone_agent.agent.build import build_planner_analyst
from drone_agent.agent.citations import validate_answer
from drone_agent.agent.factcheck import Verdict, check_facts, extract_claims
from drone_agent.agent.session import AnalystSession
from drone_agent.agent.tools import AnalystTools
from drone_agent.eval import grounding_exp as gx
from drone_agent.ingest import ingest_result
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store

LOITER_ALERT = "[alert:ALR-LOITER-EVT-P-001]"
P1, V1, LOITER, REPEAT, VEHICLE = gx.P1, gx.V1_EVIDENCE, gx.LOITER, gx.REPEAT, gx.VEHICLE
SIGNAL = ("get_alert", {"alert_id": "ALR-SIGNAL-20260929T183630Z"})
RESTRICTED = ("get_alert", {"alert_id": "ALR-RESTRICTED-EVT-P-003"})


@pytest.fixture(scope="module")
def store():
    with gx.evidence_store() as s:
        yield s


def gates(store, text, *prep):
    """(citation gate alone accepts, citation + factual gate accepts), both via the real session."""
    return (
        gx.gate_accepts(store, text, list(prep), facts=False)[0],
        gx.gate_accepts(store, text, list(prep), facts=True)[0],
    )


# ---- the old failure mode: valid citation, false fact -----------------------------------------
ADVERSARIAL = [
    # (what is wrong, text, evidence the answer retrieved)
    ("zone", "Right after midnight the pickup sat in the garage [event:EVT-V-001].", [V1]),
    ("zone", "The person was at the warehouse the whole time [event:EVT-P-001].", [P1]),
    ("time", "That person was first seen at 21:15 [event:EVT-P-001].", [P1]),
    ("time", "The loiterer showed up at 11:10 pm [event:EVT-P-001].", [P1]),
    ("count", "A similar blue pickup visited four times [entity:ENT-V-001].", [VEHICLE]),
    ("count", "The pickup appeared once [entity:ENT-V-001].", [VEHICLE]),
    ("count total", "Six alerts were raised in this run [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("count total", "There are 9 alerts stored [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("count filtered", "Two critical alerts fired [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("severity", "The loitering alert is of low severity [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("severity", "It was escalated to critical [alert:ALR-REPEAT-ENT-V-001].", [REPEAT]),
    ("status", "The loitering alert is resolved [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("status", "The alert was acknowledged by an operator [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("activity", "The person was sitting at the gate [event:EVT-P-001].", [P1]),
    ("activity", "The pickup drove away [event:EVT-V-001].", [V1]),
    ("duration", "The person loitered for 12 minutes [event:EVT-P-001].", [P1]),
    ("duration", "The person stayed 300 seconds [event:EVT-P-001].", [P1]),  # 300 s is a threshold
    ("rule", "This is the geofence alert [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("type", "EVT-V-001 is a person event [event:EVT-V-001].", [V1]),
]


@pytest.mark.parametrize(
    ("wrong", "text", "prep"),
    ADVERSARIAL,
    ids=[f"{w}-{i}" for i, (w, _, _) in enumerate(ADVERSARIAL)],
)
def test_a_valid_citation_does_not_launder_a_false_fact(store, wrong, text, prep):
    citation_only, production = gates(store, text, *prep)
    assert citation_only, "the premise of the test: the citation gate alone accepts this answer"
    assert not production, f"a false {wrong} statement was accepted: {text}"


TRUE_ANSWERS = [
    ("Right after midnight the pickup sat in the parking area [event:EVT-V-001].", [V1]),
    ("That person was first seen at 23:41 [event:EVT-P-001].", [P1]),
    ("The loiterer showed up at 11:41 pm at the main gate [event:EVT-P-001].", [P1]),
    ("A similar blue pickup visited three times [entity:ENT-V-001].", [VEHICLE]),
    ("The pickup appeared 3 separate times [entity:ENT-V-001].", [VEHICLE]),
    ("Seven alerts were raised in this run [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("One critical alert fired [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("The loitering alert is critical severity [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("It rose from medium to critical [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("The alert was escalated and remains open [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("The person was pacing at the gate [event:EVT-P-001].", [P1]),
    ("The person loitered for 8 minutes [event:EVT-P-001].", [P1]),
    ("The person stayed 480 seconds [event:EVT-P-001].", [P1]),
    ("It escalated to critical after 300 seconds [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("The alert warned after a 120 s threshold [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("This is the loitering alert [alert:ALR-LOITER-EVT-P-001].", [LOITER]),
    ("EVT-P-001 is a person event [event:EVT-P-001].", [P1]),
    ("Insufficient evidence: no stored observation answers that.", []),
]


@pytest.mark.parametrize(
    ("text", "prep"), TRUE_ANSWERS, ids=[str(i) for i in range(len(TRUE_ANSWERS))]
)
def test_true_answers_still_pass_the_full_gate(store, text, prep):
    assert gates(store, text, *prep) == (True, True)


# ---- supported, contradicted, unsupported ------------------------------------------------------
def verdicts(store, text):
    return {(f.claim.kind, f.verdict) for f in check_facts(store, text).findings}


def test_the_three_outcomes_are_distinguished(store):
    supported = verdicts(store, "The person was in the main gate [event:EVT-P-001].")
    assert ("zone", Verdict.SUPPORTED) in supported
    contradicted = verdicts(store, "The person was in the warehouse [event:EVT-P-001].")
    assert ("zone", Verdict.CONTRADICTED) in contradicted
    # the signal-loss alert has no event, so it records no activity and no object type: a claim
    # about either is neither true nor false of that evidence, it is simply not supported by it
    unsupported = verdicts(
        store, "The person was standing there [alert:ALR-SIGNAL-20260929T183630Z]."
    )
    assert ("activity", Verdict.UNSUPPORTED) in unsupported
    assert ("type", Verdict.UNSUPPORTED) in unsupported
    report = check_facts(
        store, "The person was standing there [alert:ALR-SIGNAL-20260929T183630Z]."
    )
    assert report.unsupported and not report.contradicted and not report.ok


def test_a_count_header_must_match_the_list_under_it(store):
    tools = AnalystTools(store)
    items = [f"- [alert:{a['ref'].split(':', 1)[1]}]" for a in tools.list_alerts()["alerts"][:3]]
    assert check_facts(store, "3 alerts were listed:\n" + "\n".join(items)).ok
    assert not check_facts(store, "4 alerts were listed:\n" + "\n".join(items)).ok


def test_counts_are_compared_with_the_store_not_with_what_was_retrieved(store):
    # the store holds 7 alerts; an answer that lists six and says "six alerts" is wrong
    tools = AnalystTools(store)
    six = [f"- [alert:{a['ref'].split(':', 1)[1]}]" for a in tools.list_alerts()["alerts"][:6]]
    text = "Six alerts were raised:\n" + "\n".join(six)
    assert not check_facts(store, "Six alerts were raised.").ok  # the claim on its own
    assert check_facts(store, text).ok  # header + its six items agree ...
    assert not check_facts(store, "There were six alerts in total, listed below.").ok  # ... total


# ---- recommendations ---------------------------------------------------------------------------
def test_a_recommendation_may_point_anywhere_but_its_premises_are_checked(store):
    ok = (
        "I recommend checking the driveway because the loitering alert is critical "
        "[alert:ALR-LOITER-EVT-P-001]."
    )
    assert check_facts(store, ok).ok
    bad = (
        "I recommend checking the main gate because the loitering alert is low severity "
        "[alert:ALR-LOITER-EVT-P-001]."
    )
    report = check_facts(store, bad)
    assert [f.claim.kind for f in report.contradicted] == ["severity"]
    assert report.contradicted[0].claim.recommendation


# ---- the session: correction, then fallback; refusals and safeguards unchanged -----------------
class _Script:
    name = "script"

    def __init__(self, *answers, prep=()):
        self.answers, self.prep, self.calls = list(answers), prep, 0

    def respond(self, system, history, question, tools, correction=None):
        for name, args in self.prep:
            getattr(tools, name)(**args)
        self.calls += 1
        return self.answers.pop(0)


def test_a_false_answer_gets_one_correction_and_the_corrected_answer_is_shown(store):
    llm = _Script(
        "The pickup was at the main gate [event:EVT-V-001].",
        "The pickup was in the parking area [event:EVT-V-001].",
        prep=[V1],
    )
    answer = AnalystSession(store, llm).ask("Where was the pickup?")
    assert answer.mode == "llm" and answer.retried and "parking area" in answer.text
    assert any("contradicted" in n and "zone" in n for n in answer.notes)


def test_two_false_answers_fall_back_to_the_deterministic_template(store):
    llm = _Script(
        "The pickup was at the main gate [event:EVT-V-001].",
        "It was in the garage [event:EVT-V-001].",
        prep=[V1],
    )
    answer = AnalystSession(store, llm).ask("Where was the blue pickup seen?")
    assert answer.mode == "fallback" and "main gate" not in answer.text.lower().split("[")[0]
    assert any("withheld" in n for n in answer.notes)


def test_the_truthful_refusal_is_still_accepted(store):
    answer = AnalystSession(
        store, _Script("Insufficient evidence: nothing is stored about that.")
    ).ask("What is the airspeed of a swallow?")
    assert answer.mode == "llm" and not answer.retried


def test_identity_and_citation_safeguards_are_unchanged(store):
    same = "This is the same person as before [event:EVT-P-001]."
    assert not validate_answer(store, same, retrieved={"event:EVT-P-001"}, sentence_level=True).ok
    unknown = "Loitering [evt:EVT-P-001]."
    assert not validate_answer(
        store, unknown, retrieved={"event:EVT-P-001"}, sentence_level=True
    ).ok
    nonexistent = "It was at the gate [event:EVT-Z-999]."
    assert not validate_answer(store, nonexistent, retrieved=set(), sentence_level=True).ok


def test_fact_checking_is_opt_in_at_the_validator_and_on_in_the_session(store):
    wrong = "The pickup was at the main gate [event:EVT-V-001]."
    retrieved = {"event:EVT-V-001"}
    assert validate_answer(store, wrong, retrieved=retrieved, sentence_level=True).ok
    report = validate_answer(
        store, wrong, retrieved=retrieved, sentence_level=True, fact_check=True
    )
    assert not report.ok and report.contradicted_facts
    assert "contradicted by the stored evidence" in " ".join(report.problems())


# ---- no false alarms on true, generated answers ------------------------------------------------
QUESTIONS = [
    "Give me the timeline",
    "Summarize the run",
    "Which alerts are still open?",
    "Which alerts were resolved?",
    "List the alerts",
    "List the critical alerts",
    "Show all events after midnight",
    "What happened at the main gate?",
    "What happened at the warehouse?",
    "Show all truck events",
    "Why was the loitering alert raised?",
    "What objects were observed?",
    "Was the blue pickup seen more than once?",
]


@pytest.mark.parametrize("scenario", sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml")))
def test_answers_built_from_tool_results_never_trip_the_fact_check(scenario, tmp_path):
    """Planner answers are rendered straight from tool output, so every one is true."""
    with Store.open(tmp_path / "s.db") as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / f"{scenario}.yaml")))
        build_events(s)
        analyze(s)
        for question in QUESTIONS:
            for llm in (build_planner_analyst(), None):
                answer = AnalystSession(s, llm, check_facts=False).ask(question)
                report = check_facts(s, answer.text)
                assert report.ok, (scenario, question, [f.describe() for f in report.failed])


def test_the_golden_offline_answers_are_all_factually_consistent(store):
    golden = json.loads((Path(__file__).parent / "golden_offline_answers.json").read_text())
    for question, text in golden.items():
        assert check_facts(store, text).ok, question


# ---- what the extractor reads, and what it deliberately ignores ---------------------------------
def kinds(sentence):
    return {c.kind for c in extract_claims(sentence)}


def test_the_extractor_reads_structured_forms_only():
    assert {"zone", "severity", "time", "duration"} <= kinds(
        "At 23:41 the critical alert for the main gate lasted 8 minutes."
    )
    assert kinds("The night was quiet and calm.") == set()  # prose without a checkable fact


def test_timestamps_offsets_and_ids_are_not_misread():
    claims = extract_claims(
        "- [alert:ALR-LOITER-EVT-P-001] medium/open: 2026-09-29 23:43:00+05:30 (UTC+05:30)"
    )
    times = [c for c in claims if c.kind == "time"]
    assert [c.values for c in times] == [((23, 43),)]  # no 05:30 from the offset
    assert "total" not in {c.kind for c in extract_claims("12:03 alert raised at the gate")}
    assert "rule" not in kinds("Alert ALR-LOITER-EVT-P-001 and event EVT-P-001.")  # ids, not words


def test_negation_and_return_forms_are_not_claims():
    assert "zone" not in kinds("It was not at the main gate [event:EVT-V-001].")
    assert "appearances" not in kinds("The pickup returned twice [entity:ENT-V-001].")  # 2 or 3?
    assert "status" in kinds("The alert is still open.") and "status" in kinds("It is unresolved.")


def test_number_words_and_units():
    durations = [
        c.values for c in extract_claims("It stayed twenty-five minutes.") if c.kind == "duration"
    ]
    assert durations == [((1500.0, True),)]
    seconds = [c.values for c in extract_claims("for 480 s") if c.kind == "duration"]
    assert seconds == [((480.0, False),)]
    total = [c.values for c in extract_claims("Showing 20 of 25 events") if c.kind == "total"]
    assert [(v[0][0], v[0][1]) for v in total] == [("event", 25)]  # only the total is a claim


def test_the_eval_layer_reuses_the_production_comparison():
    from drone_agent.agent import factcheck
    from drone_agent.eval import claims

    assert claims.check_claim is factcheck.check_claim
    assert claims.SUPPORTED_CLASSES == factcheck.SUPPORTED_CLASSES


def test_the_analyst_core_does_not_import_the_evaluation_layer_or_langchain():
    from drone_agent.agent import factcheck

    source = Path(factcheck.__file__).read_text(encoding="utf-8")
    assert "drone_agent.eval" not in source and "langchain" not in source.lower()
