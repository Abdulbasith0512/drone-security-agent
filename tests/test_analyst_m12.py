"""M12: the analyst layer beyond the first LangChain adapter.

Covers the hardened citation gate (per-sentence, refusal, formatting-only repair), time-window
parsing, the tool-result size guard, the scripted planner running through the REAL tool-calling
loop, record/replay of model turns, the template analyst's telemetry/status/window/recurrence
answers, and the run trace. No network, no key, no live model.
"""

import json
from datetime import datetime, timedelta, timezone
from io import StringIO

import pytest
from conftest import SCENARIO_DIR
from langchain_core.messages import AIMessage
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.agent.build import build_planner_analyst, build_replay_analyst
from drone_agent.agent.citations import (
    claim_units,
    find_uncited_claims,
    is_pure_refusal,
    repair_citations,
    validate_answer,
)
from drone_agent.agent.config import AnalystConfigError
from drone_agent.agent.evidence import ToolLimits
from drone_agent.agent.llm import LangChainAnalyst
from drone_agent.agent.offline import OfflineAnalyst
from drone_agent.agent.replay import (
    CassetteFormatError,
    RecordingChatModel,
    ReplayChatModel,
    ReplayMiss,
)
from drone_agent.agent.session import AnalystSession
from drone_agent.agent.timeparse import parse_moment, parse_time_window
from drone_agent.agent.tools import AnalystTools
from drone_agent.ingest import ingest_result
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store

IST = timezone(timedelta(hours=5, minutes=30))


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("m12") / "security.db"
    with Store.open(path) as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        build_events(s)
        analyze(s)
    return path


@pytest.fixture
def store(db_path):
    with Store.open(db_path) as s:
        yield s


def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


class FakeChatModel:
    def __init__(self, *script):
        self.script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self.script.pop(0)


def call(name, args=None, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


# ---- the citation gate ----------------------------------------------------------------------
def test_one_valid_citation_cannot_launder_uncited_factual_sentences(store):
    text = (
        "The person loitered at the gate [event:EVT-P-001]. "
        "Two vehicles then arrived at the warehouse."
    )
    lenient = validate_answer(store, text, retrieved={"event:EVT-P-001"})
    strict = validate_answer(store, text, retrieved={"event:EVT-P-001"}, sentence_level=True)
    assert lenient.ok  # a citation is present, so the answer-level check passes
    assert not strict.ok
    assert strict.uncited_claims == ["Two vehicles then arrived at the warehouse."]
    assert "without their own citation" in " ".join(strict.problems())


def test_abbreviations_do_not_split_sentences_and_list_lead_ins_are_covered():
    assert claim_units("Use e.g. a fixed rule. Then stop.") == [
        "Use e.g. a fixed rule.",
        "Then stop.",
    ]
    listed = "Alerts raised by the rules:\n- [alert:ALR-A-1] a person alert was raised there"
    assert find_uncited_claims(listed) == []


def test_only_a_pure_refusal_needs_no_citation():
    assert is_pure_refusal("Insufficient evidence: nothing is stored for that.")
    assert not is_pure_refusal(
        "Insufficient evidence for the rest. The person was at the gate for 40 seconds."
    )
    assert not is_pure_refusal("The person was at the gate for 40 seconds.")


def test_repair_fixes_formatting_slips_but_never_substitutes_another_id():
    retrieved = {"event:EVT-V-001", "event:EVT-P-002"}
    fixed, applied = repair_citations("See [Event: evt-v-1] and [ event : EVT-P-002 ].", retrieved)
    assert fixed == "See [event:EVT-V-001] and [event:EVT-P-002]."
    assert len(applied) == 2
    # a neighbouring id is different evidence: it must stay invalid, not be "corrected"
    kept, none_applied = repair_citations("See [event:EVT-P-003].", retrieved)
    assert kept == "See [event:EVT-P-003]." and none_applied == []
    # an id that was not retrieved is never introduced either
    kept, none_applied = repair_citations("See [event:EVT-V-002].", retrieved)
    assert kept == "See [event:EVT-V-002]." and none_applied == []


def test_a_repaired_answer_is_accepted_and_reported(store):
    chat = FakeChatModel(
        call("get_event", {"event_id": "EVT-P-001"}),
        AIMessage(content="A person stood at the main gate [Event: evt-p-1]."),
    )
    answer = AnalystSession(store, LangChainAnalyst(chat, name="t")).ask("q")
    assert answer.mode == "llm"
    assert answer.text == "A person stood at the main gate [event:EVT-P-001]."
    assert answer.repairs and any("repaired" in n for n in answer.notes)


def test_a_fabricated_neighbouring_id_is_rejected_not_repaired(store):
    chat = FakeChatModel(
        call("get_event", {"event_id": "EVT-P-002"}),
        AIMessage(content="A person stood in the warehouse [event:EVT-P-003]."),
        AIMessage(content="A person stood in the warehouse [event:EVT-P-003]."),
    )
    answer = AnalystSession(store, LangChainAnalyst(chat, name="t")).ask("q")
    assert answer.mode == "fallback" and "EVT-P-003" not in answer.text.split("\n")[0]


# ---- time windows ---------------------------------------------------------------------------
FIRST = datetime(2026, 9, 29, 23, 40, tzinfo=IST)
LAST = datetime(2026, 9, 30, 0, 30, tzinfo=IST)


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("show events after midnight", datetime(2026, 9, 30, 0, 0, tzinfo=IST), None),
        (
            "what changed between 00:10 and 00:20?",
            datetime(2026, 9, 30, 0, 10, tzinfo=IST),
            datetime(2026, 9, 30, 0, 20, tzinfo=IST),
        ),
        ("anything before 12:05 AM", None, datetime(2026, 9, 30, 0, 5, tzinfo=IST)),
    ],
)
def test_time_windows_are_resolved_against_the_stored_run(question, start, end):
    assert parse_time_window(question, FIRST, LAST) == (start, end)


def test_a_clock_time_before_the_run_means_the_evening_that_contains_it():
    # "after 6 PM" on a run that starts at 23:40 is the previous 18:00: the whole run is inside
    start, end = parse_time_window("what happened after 6 PM?", FIRST, LAST)
    assert start == datetime(2026, 9, 29, 18, 0, tzinfo=IST) and end is None


def test_unreadable_times_return_none_instead_of_a_guess():
    assert parse_time_window("what happened after the truck left?", FIRST, LAST) is None
    assert parse_time_window("between 25:99 and 26:00", FIRST, LAST) is None


def test_a_moment_must_fall_inside_the_run():
    assert parse_moment("battery at 00:10?", FIRST, LAST) == datetime(
        2026, 9, 30, 0, 10, tzinfo=IST
    )
    assert parse_moment("battery at 05:00?", FIRST, LAST) is None
    assert parse_moment("battery level", FIRST, LAST) is None


# ---- tool results stay within the model's size cap ------------------------------------------
def test_the_run_summary_fits_the_tool_result_cap(store):
    text = json.dumps(AnalystTools(store, fit_results=True).summarize_run(), default=str)
    assert len(text) < 11500


def test_oversized_results_are_trimmed_with_totals_and_only_shown_refs_count_as_retrieved(store):
    tools = AnalystTools(store, ToolLimits(max_result_chars=1500), fit_results=True)
    result = tools.list_alerts()
    assert result["truncated"] and result["trimmed_to_fit"] == ["alerts"]
    assert result["total"] == 7 and result["returned"] == len(result["alerts"]) < 7
    assert len(json.dumps(result, default=str)) <= 1500
    shown = {a["ref"] for a in result["alerts"]}
    assert {r for r in tools.retrieved if r.startswith("alert:")} == shown


def test_the_size_guard_is_off_for_the_template_analyst(store):
    tools = AnalystTools(store, ToolLimits(max_result_chars=1500))
    result = tools.list_alerts()
    assert "trimmed_to_fit" not in result and len(result["alerts"]) == 7


def test_listings_are_compact_and_the_detail_lives_in_get_alert(store):
    tools = AnalystTools(store)
    listed = tools.list_alerts(rule_id="loitering")["alerts"][0]
    assert {"ref", "severity", "status", "reason", "event_refs"} <= set(listed)
    assert not {"ladder", "history", "rule_facts", "recommended_action"} & set(listed)
    full = tools.get_alert("ALR-LOITER-EVT-P-001")["alert"]
    assert [s["severity"] for s in full["ladder"]] == ["medium", "critical"]
    assert full["history"] and full["recommended_action"] and full["status"] == "resolved"


# ---- the scripted planner drives the real tool-calling loop ---------------------------------
QUESTIONS = [
    "Why was the loitering alert raised?",
    "Show me the evidence for ALR-RESTRICTED-EVT-P-003",
    "Which alerts are still open?",
    "Which alerts were resolved?",
    "List the critical alerts",
    "Show all truck events",
    "Was the person at the main gate previously detected?",
    "What was the drone battery and signal when the link was lost?",
    "Show all events after midnight",
    "What changed between 00:10 and 00:20?",
    "Give me the timeline",
    "Summarize the run",
    "What happened at the warehouse?",
]


@pytest.mark.parametrize("question", QUESTIONS)
def test_planner_answers_pass_the_strict_gate_without_falling_back(store, question):
    session = AnalystSession(store, build_planner_analyst())
    answer = session.ask(question)
    assert answer.mode == "llm", answer.notes
    assert answer.label.startswith("scripted-planner") and "NO LLM" in answer.label
    assert answer.citations and not answer.metrics.fallback
    assert answer.tool_trace and all(step["ok"] for step in answer.tool_trace)
    report = validate_answer(
        store, answer.text, retrieved=session.tools.retrieved, sentence_level=True
    )
    assert report.ok, report.problems()


def test_the_planner_says_so_when_it_cannot_answer_and_invents_nothing(store):
    answer = AnalystSession(store, build_planner_analyst()).ask(
        "What is the airspeed velocity of an unladen swallow?"
    )
    assert answer.mode != "llm" or "Insufficient evidence" in answer.text
    assert "swallow" not in answer.text.lower().replace("unladen swallow?", "")


def test_a_time_window_answer_lists_only_events_in_the_window(store):
    answer = AnalystSession(store, build_planner_analyst()).ask(
        "What changed between 00:10 and 00:20?"
    )
    assert "[event:EVT-P-002]" in answer.text and "[event:EVT-P-003]" in answer.text
    assert "[event:EVT-P-001]" not in answer.text and "[event:EVT-V-001]" not in answer.text
    assert "run_facts" not in answer.text and "Stored run:" not in answer.text


def test_a_type_filtered_window_does_not_add_unrelated_timeline_items(store):
    answer = AnalystSession(store, build_planner_analyst()).ask("Show truck events before 00:05")
    assert "Timeline" not in answer.text
    assert "[event:EVT-V-001]" in answer.text and "EVT-P-" not in answer.text


def test_the_planner_states_person_recurrence_only_as_a_labelled_weak_hint(store):
    answer = AnalystSession(store, build_planner_analyst()).ask(
        "Was the person at the main gate previously detected?"
    )
    text = answer.text.lower()
    assert "never identity-linked" in text or "weak hint" in text
    assert "same person" not in text.replace("not show it is the same person", "")


# ---- record and replay of model turns ---------------------------------------------------------
def scripted_gemini_turns():
    return FakeChatModel(
        call("get_alert", {"alert_id": "ALR-LOITER-EVT-P-001"}),
        AIMessage(
            content="The loitering alert was raised for a person [alert:ALR-LOITER-EVT-P-001]."
        ),
    )


def test_recorded_model_turns_replay_through_the_real_loop_without_a_model(store, tmp_path):
    cassette = tmp_path / "analyst.json"
    recorder = RecordingChatModel(scripted_gemini_turns(), cassette, label="gemini:test-model")
    live = AnalystSession(store, LangChainAnalyst(recorder, name="gemini:test-model")).ask(
        "Why was the loitering alert raised?"
    )
    recorder.flush()
    data = json.loads(cassette.read_text())
    assert data["format"] == "analyst-cassette-1" and data["recorded_with"] == "gemini:test-model"
    assert len(data["turns"]) == 2 and "key" in data["turns"][0]
    assert "tool_results" not in cassette.read_text()  # only the model's own turns are stored

    analyst = build_replay_analyst(str(cassette))
    replayed = AnalystSession(store, analyst).ask("Why was the loitering alert raised?")
    assert replayed.text == live.text and replayed.mode == "llm"
    assert replayed.label.startswith("REPLAYED") and analyst.chat_model.replayed == 2
    # the tools still ran against the real database
    assert [s["tool"] for s in replayed.tool_trace] == ["get_alert"]


def test_a_replay_miss_degrades_to_the_labelled_template_answer(store, tmp_path):
    cassette = tmp_path / "analyst.json"
    recorder = RecordingChatModel(scripted_gemini_turns(), cassette)
    AnalystSession(store, LangChainAnalyst(recorder)).ask("Why was the loitering alert raised?")
    recorder.flush()
    answer = AnalystSession(store, build_replay_analyst(str(cassette))).ask(
        "A question never asked"
    )
    assert answer.mode == "fallback" and answer.metrics.fallback
    assert any("unavailable" in n and "no recorded model turn" in n for n in answer.notes)


def test_a_replay_model_raises_a_clear_miss_and_a_bad_cassette_is_refused(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"format": "analyst-cassette-1", "turns": []}))
    with pytest.raises(ReplayMiss):
        ReplayChatModel(empty).invoke([])
    junk = tmp_path / "junk.json"
    junk.write_text("not json")
    with pytest.raises(CassetteFormatError):
        ReplayChatModel(junk)
    with pytest.raises(AnalystConfigError):
        build_replay_analyst(str(junk))


def test_recording_extends_an_existing_cassette(store, tmp_path):
    cassette = tmp_path / "analyst.json"
    first = RecordingChatModel(scripted_gemini_turns(), cassette)
    AnalystSession(store, LangChainAnalyst(first)).ask("Why was the loitering alert raised?")
    first.flush()
    second = RecordingChatModel(
        FakeChatModel(AIMessage(content="Insufficient evidence.")), cassette
    )
    AnalystSession(store, LangChainAnalyst(second)).ask("Anything about drones?")
    second.flush()
    assert len(json.loads(cassette.read_text())["turns"]) == 3


# ---- the template analyst: telemetry, status, windows, recurrence -----------------------------
@pytest.fixture
def offline(store):
    return OfflineAnalyst(AnalystTools(store))


def test_battery_at_a_time_is_a_telemetry_answer_not_an_alert_lookup(offline):
    answer = offline.answer("What was the battery level at 00:10?")
    assert answer.matched == "telemetry"
    assert (
        "battery" in answer.text
        and "(simulated telemetry)" in answer.text
        and "[tel:" in answer.text
    )


def test_a_time_outside_the_run_is_reported_not_guessed(offline):
    answer = offline.answer("What was the battery at 05:00?")
    assert answer.matched == "telemetry-outside" and "outside the stored run" in answer.text


def test_telemetry_without_a_time_reads_the_latest_sample(offline):
    answer = offline.answer("What is the drone altitude?")
    assert answer.text.startswith("Latest stored drone telemetry [tel:")


def test_a_safety_alert_answer_shows_status_action_and_the_drones_state(offline):
    answer = offline.answer("Was there a signal loss alert?")
    assert answer.matched == "alert-rule"
    for expected in (
        "Status: resolved",
        "Recommended action",
        "Triggered by drone telemetry, not by a camera event",
        "Drone telemetry at the alert [tel:",
    ):
        assert expected in answer.text


def test_alert_status_questions_filter_by_lifecycle(offline):
    unresolved = offline.answer("Which alerts are still open?").text
    assert "unresolved alert(s)" in unresolved and "/resolved" not in unresolved
    resolved = offline.answer("Which alerts were resolved?").text
    assert "/open" not in resolved and "ALR-LOITER-EVT-P-001" in resolved
    critical = offline.answer("show critical alerts").text
    assert "ALR-LOITER-EVT-P-001" in critical and "ALR-AFTERHOURS" not in critical


def test_the_escalation_ladder_is_shown_for_an_escalated_alert(offline):
    text = offline.answer("Show me the evidence for the loitering alert.").text
    assert "Escalation: medium at" in text and "critical at" in text


def test_a_time_window_question_lists_overlapping_events_and_alert_changes(offline):
    answer = offline.answer("What happened between 00:10 and 00:20?")
    assert answer.matched == "window" and "an event counts when it overlaps" in answer.text
    assert "[event:EVT-P-002]" in answer.text and "alert raised" in answer.text
    assert "[event:EVT-V-001]" not in answer.text


def test_a_window_with_nothing_in_it_says_so(offline):
    answer = offline.answer("What happened between 00:26 and 00:27 in the warehouse?")
    assert answer.matched == "window-none" and answer.text.startswith("Insufficient evidence")


def test_person_recurrence_is_only_ever_a_weak_labelled_hint(offline):
    answer = offline.answer("Was the person at the gate seen before?")
    assert answer.matched == "person-recurrence"
    assert "never identity-linked" in answer.text or "weak hint" in answer.text


def test_supported_questions_mention_the_new_shapes():
    from drone_agent.agent.offline import SUPPORTED_QUESTIONS

    joined = " ".join(SUPPORTED_QUESTIONS).lower()
    for word in ("battery", "open", "between", "seen before"):
        assert word in joined


# ---- provenance-aware limitations -------------------------------------------------------------
def test_scripted_runs_say_their_text_is_authored_not_perception(store):
    text = AnalystSession(store).summarize().text
    assert "scripted scenario text written by the author" in text
    assert "REPLAYED" not in text


# ---- the run trace ----------------------------------------------------------------------------
def test_answers_are_recorded_with_their_tool_calls_and_no_prompts(db_path, tmp_path):
    copy = tmp_path / "copy.db"
    copy.write_bytes(db_path.read_bytes())
    with Store.open(copy) as store:
        session = AnalystSession(store, build_planner_analyst(), record=True)
        session.ask("Which alerts are still open?")
        rows = store.list_agent_runs("analyst")
    assert len(rows) == 1
    row = rows[0]
    assert row["status"].startswith("ok") and "planner" in row["mode"]
    assert [t["tool"] for t in json.loads(json.dumps(row["trace"]))] == ["list_alerts"]
    assert "system" not in json.dumps(row).lower()  # no system prompt is stored


def test_the_cli_shows_every_tool_call_with_trace(db_path):
    code, text = run_cli(
        "ask", "--db", db_path, "--provider", "planner", "--trace", "Which alerts are still open?"
    )
    assert code == 0, text
    assert "scripted planner in the real tool-calling loop (NO LLM, no key)" in text
    assert "tool list_alerts(" in text and "citations verified" in text


def test_the_cli_replay_provider_needs_a_cassette(db_path):
    code, text = run_cli("ask", "--db", db_path, "--provider", "replay", "q")
    assert code != 0 and "--analyst-cassette" in text
