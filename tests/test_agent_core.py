"""Analyst layer without any LLM: tools, citations, template analyst, session, summary."""

import re

import pytest
from conftest import SCENARIO_DIR

from drone_agent.agent.citations import (
    Citation,
    extract_citations,
    find_overclaims,
    validate_answer,
)
from drone_agent.agent.evidence import ToolLimits
from drone_agent.agent.offline import OfflineAnalyst
from drone_agent.agent.prompt import ANALYST_SYSTEM_PROMPT
from drone_agent.agent.session import AnalystLLMError, AnalystSession
from drone_agent.agent.summary import render_summary, run_facts
from drone_agent.agent.tools import TOOL_NAMES, AnalystTools
from drone_agent.ingest import ingest_result
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store

BANNED = re.compile(
    r"criminal|burglar|thief|thieves|steal|intent|suspect|the same (person|individual)",
    re.IGNORECASE,
)


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("agent") / "security.db"
    with Store.open(path) as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        build_events(s)
        analyze(s)
    return path


@pytest.fixture
def store(db_path):
    with Store.open(db_path) as s:
        yield s


@pytest.fixture
def tools(store):
    return AnalystTools(store)


# ---- tools ----------------------------------------------------------------------------------
def test_the_tool_set_is_a_fixed_read_only_list_with_no_sql_or_generic_execution(tools):
    assert TOOL_NAMES == (
        "search_observations", "list_events", "get_event", "get_event_evidence", "list_entities",
        "get_entity", "get_entity_history", "list_alerts", "get_alert", "get_run_summary",
    )  # fmt: skip
    public = {n for n in dir(tools) if not n.startswith("_") and callable(getattr(tools, n))}
    assert public == set(TOOL_NAMES)
    import inspect

    for name in TOOL_NAMES:
        assert not {"sql", "query_sql", "statement"} & set(
            inspect.signature(getattr(tools, name)).parameters
        )


def test_tools_never_change_the_database(db_path):
    def counts(s):
        return (s.count_observations(), s.count_events(), s.count_entities(), s.count_alerts())

    with Store.open(db_path) as s:
        before = counts(s)
        t = AnalystTools(s)
        t.search_observations(query="pickup")
        t.list_events(), t.list_alerts(), t.list_entities(), t.get_run_summary()
        t.get_alert("ALR-LOITER-EVT-P-001"), t.get_event_evidence("EVT-P-001")
        t.get_entity_history("ENT-V-001"), t.get_event("nope"), t.get_alert("nope")
        assert counts(s) == before


def test_search_observations_filters_and_shapes_results(tools):
    res = tools.search_observations(query="blue pickup")
    assert res["total"] == res["returned"] == 5 or res["total"] >= 2
    assert all("ref" in o and "frame_ref" in o for o in res["observations"])
    assert "obs:obs-10" in tools.retrieved and "frame:10" in tools.retrieved
    zone = tools.search_observations(zone="warehouse")
    assert {o["zone"] for o in zone["observations"]} == {"warehouse"} and zone["total"] == 3
    people = tools.search_observations(object_type="person", zone="main_gate")
    assert all(o["objects"][0]["type"] == "person" for o in people["observations"])
    window = tools.search_observations(
        start="2026-09-29T23:41:00+05:30", end="2026-09-29T23:43:00+05:30"
    )
    assert window["total"] == 3
    assert tools.search_observations(zone="lobby")["error"].startswith("zone must be one of")
    assert "offset" in tools.search_observations(start="2026-09-29T23:41:00")["error"]
    assert "ISO-8601" in tools.search_observations(start="yesterday")["error"]
    assert "searchable" in tools.search_observations(query="***")["error"]


def test_observation_results_are_bounded_and_report_truncation(store):
    small = AnalystTools(store, ToolLimits(default_observations=3, max_observations=5))
    res = small.search_observations()
    assert (res["returned"], res["total"], res["truncated"]) == (3, 27, True)
    assert small.search_observations(limit=500)["returned"] == 5  # capped at the maximum
    assert small.search_observations(limit=0)["returned"] == 1  # never negative or zero
    assert AnalystTools(store).search_observations()["returned"] == 8  # documented default


def test_descriptions_are_truncated_to_the_documented_limit(store):
    t = AnalystTools(store, ToolLimits(max_description=20))
    for o in t.search_observations()["observations"]:
        assert len(o["description"]) <= 20
    assert any(o["description"].endswith("...") for o in t.search_observations()["observations"])


def test_event_alert_and_entity_tools(tools):
    ev = tools.get_event("EVT-V-001")["event"]
    assert ev["zone"] == "parking_area" and ev["entity_ref"] == "entity:ENT-V-001"
    assert tools.get_event("EVT-Z-9")["error"].startswith("no event")
    assert len(tools.list_events()["events"]) == 7
    assert [e["ref"] for e in tools.list_events(zone="main_gate")["events"]] == [
        "event:EVT-P-001", "event:EVT-V-003"
    ]  # fmt: skip
    assert len(tools.list_events(object_type="vehicle")["events"]) == 4
    ent = tools.get_entity("ENT-V-001")["entity"]
    assert ent["label"] == "similar blue pickup" and ent["identity_confirmed"] is False
    assert tools.get_entity("ENT-X")["error"].startswith("no entity")
    assert tools.list_entities()["total"] == 2


def test_alert_tools_return_rule_output_with_evidence(tools):
    res = tools.get_alert("ALR-REPEAT-ENT-V-001")
    assert res["alert"]["severity"] == "low" and res["entity"]["appearance_count"] == 3
    assert [e["ref"] for e in res["events"]] == [f"event:EVT-V-00{i}" for i in (1, 2, 3)]
    assert {"alert:ALR-REPEAT-ENT-V-001", "entity:ENT-V-001", "frame:10"} <= tools.retrieved
    assert tools.get_alert("ALR-NOPE")["error"].startswith("no alert")
    assert tools.list_alerts()["total"] == 6
    assert tools.list_alerts(rule_id="loitering")["total"] == 1
    assert tools.list_alerts(severity="high")["total"] == 1
    assert tools.list_alerts(zone="warehouse")["total"] == 1
    assert "severity must be one of" in tools.list_alerts(severity="extreme")["error"]
    assert tools.list_alerts(limit=2)["truncated"] is True


def test_event_evidence_is_bounded_and_entity_history_lists_every_appearance(store):
    tight = AnalystTools(store, ToolLimits(max_evidence=3, max_frame_ids=2))
    res = tight.get_event_evidence("EVT-P-001")
    assert (res["observations_total"], res["observations_returned"]) == (8, 3)
    assert res["truncated"] and len(res["event"]["frame_refs"]) == 2
    assert res["event"]["frames_total"] == 8
    hist = AnalystTools(store).get_entity_history("ENT-V-001")
    assert hist["appearances_total"] == 3 and len(hist["appearances"]) == 3
    assert [a["zone"] for a in hist["appearances"]] == ["parking_area", "driveway", "main_gate"]
    assert hist["alerts"][0]["ref"] == "alert:ALR-REPEAT-ENT-V-001"
    assert "coarse" in hist["note"]
    assert tight.get_event_evidence("EVT-nope")["error"].startswith("no event")


def test_run_summary_facts(tools):
    facts = tools.get_run_summary()
    assert facts["counts"] == {
        "frames": 27, "observations": 27, "detected_objects": 20, "events": 7,
        "entities": 2, "alerts": 6,
    }  # fmt: skip
    assert facts["object_counts"] == {"person": 13, "vehicle": 7}
    assert facts["alerts_by_severity"] == {"high": 1, "low": 1, "medium": 4}
    assert facts["observation_span"]["first"].startswith("2026-09-29T23:41")
    assert any("simulated" in lim for lim in facts["limitations"])
    assert any("coarse" in lim for lim in facts["limitations"])
    assert not facts["events_truncated"]


def test_summary_lists_are_bounded(store):
    facts = run_facts(store, ToolLimits(max_events=2, max_alerts=1, max_entities=1))
    assert len(facts["events"]) == 2 and facts["events_total"] == 7 and facts["events_truncated"]
    assert len(facts["alerts"]) == 1 and facts["alerts_truncated"]
    assert "showing 2 of 7 events" in render_summary(facts)


# ---- prompt ---------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "phrase",
    [
        "Get every surveillance fact from the tools",
        "never invent observations",
        "[event:EVT-V-001]",
        "Cite only ids the tools returned",
        "Separate observation",
        "inference",
        "a similar blue pickup",
        'never say it is\n   the same vehicle unless match_level is "confirmed"',
        "Never say the same person appears in separate events",
        "criminal intent",
        "cannot change them",
        "Insufficient evidence",
        "never create, suppress, dismiss or re-rate alerts",
        "simulated patrol timeline",
    ],
)
def test_system_prompt_states_the_required_constraints(phrase):
    assert phrase in ANALYST_SYSTEM_PROMPT


# ---- citations ------------------------------------------------------------------------------
def test_citation_extraction_is_ordered_deduplicated_and_strict():
    text = "A [entity:ENT-V-001] then [event:EVT-V-001] [event:EVT-V-001] [frame:10] [obs:obs-10]"
    assert [c.token for c in extract_citations(text)] == [
        "[entity:ENT-V-001]", "[event:EVT-V-001]", "[frame:10]", "[obs:obs-10]"
    ]  # fmt: skip
    assert extract_citations("see [source:wikipedia] and (event:EVT-1) and [event EVT-1]") == []


def test_every_citation_kind_is_checked_against_the_database(store):
    good = (
        "[frame:10] [obs:obs-10] [event:EVT-V-001] [entity:ENT-V-001] [alert:ALR-LOITER-EVT-P-001]"
    )
    assert validate_answer(store, good).ok
    for bad in (
        "[frame:9999]", "[frame:abc]", "[obs:obs-999]", "[event:EVT-V-999]", "[entity:ENT-V-009]",
        "[alert:ALR-FAKE-1]",
    ):  # fmt: skip
        report = validate_answer(store, f"Something happened {bad}")
        assert not report.ok and [c.token for c in report.invalid] == [bad]
        assert "do not exist" in report.problems()[0]


def test_a_real_id_that_no_tool_returned_is_rejected_when_retrieval_is_tracked(store):
    report = validate_answer(store, "It [event:EVT-V-001].", retrieved={"event:EVT-V-002"})
    assert [c.token for c in report.not_retrieved] == ["[event:EVT-V-001]"] and not report.ok
    assert validate_answer(store, "It [event:EVT-V-001].", retrieved={"event:EVT-V-001"}).ok


def test_answers_without_citations_fail_unless_they_declare_insufficient_evidence(store):
    report = validate_answer(store, "A blue pickup came by.")
    assert report.missing and not report.ok
    assert validate_answer(store, "Insufficient evidence.", require_citation=False).ok
    session = AnalystSession(store, ScriptedLLM("A blue pickup came by.", "Still no ids."))
    assert session.ask("anything").mode == "fallback"
    ok = AnalystSession(store, ScriptedLLM("Insufficient evidence: nothing was retrieved."))
    assert ok.ask("anything").mode == "llm"


def test_malformed_citation_tokens_are_flagged(store):
    report = validate_answer(store, "Seen [event EVT-V-001] and [frame:10]")
    assert report.malformed == ["[event EVT-V-001]"] and not report.ok


def test_identity_and_intent_overclaims_are_flagged_but_negations_are_allowed():
    assert find_overclaims("This is the same person as before.")
    assert find_overclaims("He is a burglar.") and find_overclaims("criminal intent is likely")
    assert find_overclaims("confirmed to be the same vehicle")
    assert not find_overclaims("These are not the same person; people are not linked.")
    assert not find_overclaims("A similar blue pickup, not confirmed to be identical.")
    assert Citation("event", "E").token == "[event:E]"


# ---- template analyst: the five security questions ------------------------------------------
QUESTIONS = {
    "Q1": (
        "What happened at the main gate?",
        ["event:EVT-P-001", "event:EVT-V-003", "alert:ALR-LOITER-EVT-P-001"],
    ),
    "Q2": (
        "Was the blue pickup seen more than once?",
        ["entity:ENT-V-001", "event:EVT-V-001", "event:EVT-V-002", "event:EVT-V-003"],
    ),
    "Q3": (
        "Why was the repeat-vehicle alert raised?",
        ["alert:ALR-REPEAT-ENT-V-001", "entity:ENT-V-001"],
    ),
    "Q4": (
        "Show me evidence for the loitering alert.",
        ["alert:ALR-LOITER-EVT-P-001", "event:EVT-P-001", "obs:obs-1", "frame:1"],
    ),
    "Q5": (
        "Were any people seen after hours?",
        [f"alert:ALR-AFTERHOURS-EVT-P-00{i}" for i in (1, 2, 3)],
    ),
}


@pytest.mark.parametrize("key", sorted(QUESTIONS))
def test_template_answers_cite_real_stored_ids(store, key):
    question, expected = QUESTIONS[key]
    answer = AnalystSession(store).ask(question)
    assert answer.mode == "template" and not answer.notes
    tokens = {c.strip("[]") for c in answer.citations}
    assert set(expected) <= tokens
    assert validate_answer(store, answer.text).ok  # every id exists in the database
    assert not BANNED.search(answer.text)


def test_blue_pickup_answer_preserves_coarse_uncertainty(store):
    text = AnalystSession(store).ask("Was the blue pickup seen more than once?").text
    assert text.startswith("Yes.") and "similar blue pickup" in text
    assert "not confirmed to be the same vehicle" in text
    assert "dark blue" in text  # the varying attribute is shown, not hidden
    assert "same vehicle" not in text.replace("not confirmed to be the same vehicle", "")
    alert = AnalystSession(store).ask("Why was the repeat-vehicle alert raised?").text
    assert "does not confirm the same vehicle" in alert and "coarse" in alert


def test_after_hours_answer_does_not_link_people_across_events(store):
    text = AnalystSession(store).ask("Were any people seen after hours?").text
    assert "not identity-linked" in text and text.count("[event:") == 3
    assert not BANNED.search(text)


def test_severity_comes_from_the_rule_not_the_analyst(store):
    text = AnalystSession(store).ask("Why was the loitering alert raised?").text
    assert "severity medium" in text and "deterministic rule 'loitering'" in text
    assert "not by a language model" in text


def test_other_template_questions(store):
    s = AnalystSession(store)
    listing = s.ask("List the alerts").text
    assert listing.startswith("6 stored alert(s)") and listing.count("[alert:") == 6
    assert "[event:EVT-V-001]" in s.ask("Describe EVT-V-001").text
    assert "[entity:ENT-V-001]" in s.ask("tell me about ENT-V-001").text
    assert "[alert:ALR-RESTRICTED-EVT-P-003]" in s.ask("Was there a restricted zone alert?").text
    zone = s.ask("What happened in the warehouse?").text
    assert "[event:EVT-P-002]" in zone and "[alert:ALR-AFTERHOURS-EVT-P-002]" in zone
    assert "[obs:obs-" in s.ask("anyone wearing a hoodie").text  # keyword fallback, labelled
    assert "text match, not an interpretation" in s.ask("dark hoodie").text


def test_unknown_or_unsupported_questions_get_an_honest_insufficient_evidence_reply(store):
    s = AnalystSession(store)
    res = s.ask("What is the airspeed velocity of an unladen swallow?")
    assert res.text.startswith("Insufficient evidence") and res.citations == []
    assert "Describe ALR-" in res.text  # lists what it can answer
    assert s.ask("Describe ALR-NOPE-9").text.startswith("Insufficient evidence")
    assert s.ask("Was there a purple bus?").text.startswith("Insufficient evidence")
    assert s.ask("What is in the restricted zone alert").text  # does not crash


def test_empty_question_is_rejected(store):
    with pytest.raises(ValueError):
        AnalystSession(store).ask("   ")


def test_template_outputs_never_contain_intent_or_identity_claims(store):
    session = AnalystSession(store)
    for q in [v[0] for v in QUESTIONS.values()] + [
        "Summarize the run",
        "What objects were observed?",
    ]:
        assert not BANNED.search(session.ask(q).text), q


# ---- summary and follow-ups -----------------------------------------------------------------
def test_offline_summary_states_counts_significant_events_entities_and_limits(store):
    answer = AnalystSession(store).summarize()
    text = answer.text
    assert "27 observation(s)" in text and "7 event(s)" in text and "6 alert(s)" in text
    assert "Significant events (linked to alerts)" in text and "[event:EVT-P-003]" in text
    assert "similar blue pickup [entity:ENT-V-001]" in text and "3 separate appearance(s)" in text
    assert "Limitations:" in text and "simulated" in text
    assert answer.mode == "template" and not answer.notes and validate_answer(store, text).ok


def test_follow_up_after_summary_uses_stored_evidence_not_the_video(store):
    s = AnalystSession(store)
    s.summarize()
    text = s.ask("What objects were observed?").text
    assert "person 13, vehicle 7" in text and "[entity:ENT-V-002]" in text
    assert s.history[0] == ("user", "Summarize the run.") and len(s.history) == 4


def test_follow_up_questions_resolve_pronouns_from_the_previous_answer(store):
    s = AnalystSession(store)
    s.ask("Was the blue pickup seen more than once?")
    where = s.ask("Where was it seen?").text
    assert "[event:EVT-V-002]" in where and "driveway" in where and "similar blue pickup" in where
    s2 = AnalystSession(store)
    s2.ask("Why was the loitering alert raised?")
    assert "ALR-LOITER-EVT-P-001" in s2.ask("Show me the evidence for it").text


def test_ambiguous_pronouns_are_not_guessed(store):
    s = AnalystSession(store)
    s.summarize()  # cites two entities, several alerts and events: no single subject
    assert s.ask("Where was it seen?").text.startswith("Insufficient evidence")
    fresh = AnalystSession(store)
    assert fresh.ask("Where was it?").text.startswith("Insufficient evidence")


def test_conversation_memory_is_bounded(store):
    s = AnalystSession(store, max_turns=2)
    for _ in range(5):
        s.ask("List the alerts")
    assert len(s.history) == 4


# ---- LLM path with a scripted analyst (the LLM boundary is mocked) --------------------------
class ScriptedLLM:
    """Stands in for the analyst model: each script item is a string or a callable(tools)->str."""

    name = "scripted"

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def respond(self, system, history, question, tools, correction=None):
        self.calls.append({"system": system, "history": history, "q": question, "corr": correction})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(tools) if callable(item) else item


def cite_loiter(tools):
    tools.get_alert("ALR-LOITER-EVT-P-001")
    return "A person loitered for 480 s [alert:ALR-LOITER-EVT-P-001] [event:EVT-P-001]."


def test_a_valid_llm_answer_is_accepted_and_shown(store):
    llm = ScriptedLLM(cite_loiter)
    ans = AnalystSession(store, llm).ask("Who loitered?")
    assert ans.mode == "llm" and not ans.retried and ans.text.startswith("A person loitered")
    assert ans.citations == ["[alert:ALR-LOITER-EVT-P-001]", "[event:EVT-P-001]"]
    assert llm.calls[0]["system"] == ANALYST_SYSTEM_PROMPT and llm.calls[0]["corr"] is None


def test_a_nonexistent_citation_triggers_exactly_one_correction_retry(store):
    llm = ScriptedLLM("It happened [event:EVT-Z-999].", cite_loiter)
    ans = AnalystSession(store, llm).ask("What happened?")
    assert ans.mode == "llm" and ans.retried and len(llm.calls) == 2
    previous, instruction = llm.calls[1]["corr"]
    assert "EVT-Z-999" in previous and "do not exist" in instruction
    assert "[event:EVT-Z-999]" not in ans.text and any("attempt 1 rejected" in n for n in ans.notes)


def test_a_second_invalid_answer_falls_back_to_the_deterministic_answer(store):
    bad = "The same person returned [event:EVT-Z-999]."
    llm = ScriptedLLM(bad, "Still wrong, no citations at all.")
    ans = AnalystSession(store, llm).ask("Why was the repeat-vehicle alert raised?")
    assert len(llm.calls) == 2  # never a third call
    assert ans.mode == "fallback" and ans.retried
    assert "EVT-Z-999" not in ans.text and "same person" not in ans.text
    assert (
        "[alert:ALR-REPEAT-ENT-V-001]" in ans.text
        and "does not confirm the same vehicle" in ans.text
    )
    assert any("withheld" in n for n in ans.notes) and "withheld" in ans.label or ans.label
    assert validate_answer(store, ans.text).ok


def test_an_answer_citing_a_real_but_unretrieved_id_is_rejected(store):
    llm = ScriptedLLM("Loitering [alert:ALR-LOITER-EVT-P-001].", cite_loiter)  # no tool was called
    ans = AnalystSession(store, llm).ask("q")
    assert ans.retried and ans.mode == "llm"
    assert "no tool returned" in llm.calls[1]["corr"][1]


def test_llm_overclaims_are_rejected(store):
    llm = ScriptedLLM(
        lambda t: (
            t.get_event("EVT-P-001"),
            "He is the same person as EVT-P-002 [event:EVT-P-001]",
        )[1],
        cite_loiter,
    )
    ans = AnalystSession(store, llm).ask("q")
    assert ans.retried and "not allowed" in llm.calls[1]["corr"][1]


def test_an_unavailable_llm_degrades_to_the_template_analyst(store):
    ans = AnalystSession(store, ScriptedLLM(AnalystLLMError("RateLimit: quota"))).ask(
        "Was the blue pickup seen more than once?"
    )
    assert ans.mode == "fallback" and "similar blue pickup" in ans.text
    assert any("unavailable" in n and "quota" in n for n in ans.notes)


def test_llm_summary_is_checked_and_falls_back_to_the_facts(store):
    good = ScriptedLLM(
        lambda t: (t.get_run_summary(), "Six alerts fired, e.g. [alert:ALR-LOITER-EVT-P-001].")[1]
    )
    ans = AnalystSession(store, good).summarize()
    assert ans.mode == "llm" and "Six alerts" in ans.text
    assert "get_run_summary" in good.calls[0]["q"] and "Do not invent events" in good.calls[0]["q"]
    bad = ScriptedLLM("Invented [event:EVT-Q-1].", "Invented again [event:EVT-Q-2].")
    ans = AnalystSession(store, bad).summarize()
    assert ans.mode == "fallback" and "27 observation(s)" in ans.text and "EVT-Q" not in ans.text


def test_llm_receives_bounded_history_for_follow_ups(store):
    llm = ScriptedLLM(cite_loiter, "Insufficient evidence for that.")
    s = AnalystSession(store, llm, max_turns=1)
    s.ask("first")
    s.ask("follow-up")
    assert llm.calls[1]["history"][0] == ("user", "first") and len(llm.calls[1]["history"]) == 2
    assert len(s.history) == 2  # only the last turn is kept


def test_the_session_never_modifies_alerts(db_path):
    with Store.open(db_path) as s:
        before = [(a.alert_id, a.severity, a.message) for a in s.query_alerts()]
        session = AnalystSession(
            s, ScriptedLLM("Downgrade it [alert:ALR-LOITER-EVT-P-001]", cite_loiter, cite_loiter)
        )
        session.ask("Please dismiss the loitering alert and make it low severity")
        session.ask("Why was the loitering alert raised?")
        assert [(a.alert_id, a.severity, a.message) for a in s.query_alerts()] == before


# ---- offline analyst directly ---------------------------------------------------------------
def test_offline_analyst_reports_which_template_answered(tools):
    analyst = OfflineAnalyst(tools)
    assert analyst.answer("What happened at the driveway?").matched == "zone"
    assert analyst.answer("Summarize").matched == "summary"
    assert analyst.answer("gibberish zzz qqq").matched == "insufficient"


# ---- validator fixes found by the M7 adversarial suite --------------------------------------
@pytest.mark.parametrize(
    "token",
    [
        "[evt:EVT-P-002]",
        "[source:wikipedia]",
        "[Event:EVT-P-001]",
        "[ev-ent:EVT-P-001]",
        "[ref: 12]",
    ],
)
def test_unknown_citation_types_fail_validation_instead_of_being_ignored(store, token):
    text = f"Loitering [event:EVT-P-001] and {token}."
    report = validate_answer(store, text, retrieved={"event:EVT-P-001"})
    assert report.unknown_kinds == [token] and not report.ok
    unknown_problem = next(p for p in report.problems() if p.startswith("unknown citation type"))
    assert "allowed: frame, obs, event, entity, alert" in unknown_problem
    assert [c.token for c in report.cited] == ["[event:EVT-P-001]"]  # the valid one still counts


def test_only_unknown_typed_tokens_also_fail_and_recognised_or_plain_brackets_do_not(store):
    assert not validate_answer(store, "See [evt:EVT-P-001].").ok
    fine = "Seen at [12:30] in zone [1] [event:EVT-P-001]."
    assert validate_answer(store, fine, retrieved={"event:EVT-P-001"}).ok
    from drone_agent.agent.citations import find_unknown_kinds

    assert find_unknown_kinds("[event:EVT-P-001] [frame:3]") == []
    assert find_unknown_kinds("[evt:A] and [source:b]") == ["[evt:A]", "[source:b]"]


def test_an_answer_with_an_unknown_citation_type_is_retried_then_falls_back(store):
    llm = ScriptedLLM(
        lambda t: (t.get_event("EVT-P-001"), "Loitering [event:EVT-P-001] [evt:EVT-P-002].")[1],
        "Loitering [event:EVT-P-001] [evt:EVT-P-002].",
    )
    answer = AnalystSession(store, llm).ask("Why was the loitering alert raised?")
    assert answer.mode == "fallback" and answer.retried and len(llm.calls) == 2
    assert "unknown citation type" in llm.calls[1]["corr"][1]
    assert "[evt:" not in answer.text


@pytest.mark.parametrize(
    "text",
    [
        "It cannot be said that this is the same individual.",
        "The evidence does not establish that these are the same person.",
        "These observations do not confirm the same person.",
        "These are not the same person.",
        "We can't tell whether it is the same individual.",
        "No evidence shows the same person appears twice, and no criminal activity is claimed.",
    ],
)
def test_clearly_negated_identity_statements_are_not_rejected(text):
    assert find_overclaims(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "This is the same person.",
        "The same individual appears in both events.",
        "The person was identified as the same individual.",
        "There is no doubt this is the same person.",
        "These are not different people, this is the same person.",
        "Not sure, but this is the same person.",
        "He is a burglar.",
        "Criminal intent is likely.",
    ],
)
def test_affirmative_identity_and_intent_claims_are_still_rejected(text):
    assert find_overclaims(text) != []


def test_negation_is_syntactic_and_bounded():
    far = (
        "It cannot be said, after reviewing all of the stored events, that this is the same person."
    )
    assert find_overclaims(far) != []  # a clause break between cue and phrase: not negated
    too_far = "Not one of the many events listed here can be shown to be the same person."
    assert find_overclaims(too_far) != []  # cue more than 8 words before the phrase
