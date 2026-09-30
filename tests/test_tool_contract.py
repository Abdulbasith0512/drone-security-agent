"""T04: the tool names a model is told about must be the tools it is actually given.

The analyst prompt once told the model to call `get_run_summary`, a name that existed only as a
Python alias for the template analyst; the model-visible registry exports `summarize_run`. A model
that followed the prompt got an "unknown tool" error and could then return a refusal that looked
like "nothing is stored". These tests pin the contract in both directions and exercise the real
tool loop, so the two can no longer drift apart silently.
"""

import inspect
import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from drone_agent.agent import planner
from drone_agent.agent.build import build_planner_analyst
from drone_agent.agent.citations import CitationReport
from drone_agent.agent.llm import _DESCRIPTIONS, LangChainAnalyst, langchain_tools
from drone_agent.agent.prompt import ANALYST_SYSTEM_PROMPT, SUMMARY_REQUEST
from drone_agent.agent.session import _TOOL_FAILURE_CORRECTION, AnalystSession, _correction
from drone_agent.agent.tools import TOOL_NAMES, AnalystTools
from drone_agent.eval import grounding_exp as gx
from drone_agent.models import Zone

# Public methods of AnalystTools that the deterministic template analyst uses and a model is
# deliberately never offered (they are older or overlapping views). Adding a public method that is
# in neither this set nor TOOL_NAMES fails the registry test below, so a tool cannot be added
# without deciding whether the model sees it.
TEMPLATE_ONLY = {
    "search_observations",
    "list_events",
    "get_event_evidence",
    "get_entity",
    "get_entity_history",
    "trace",
}
# identifier-shaped words in model-facing text that are data fields, not tools
NOT_TOOLS = {"match_level"}
TOOL_LIKE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")
TOOL_VERB = re.compile(r"\b(?:get|list|search|query|summarize|telemetry|entity)_[a-z_]+\b")


@pytest.fixture(scope="module")
def store():
    with gx.evidence_store() as s:
        yield s


def public_methods(tools) -> set[str]:
    return {n for n in dir(tools) if not n.startswith("_") and callable(getattr(tools, n))}


def argument_names(tools) -> set[str]:
    return {p for name in TOOL_NAMES for p in inspect.signature(getattr(tools, name)).parameters}


# ---- the registry --------------------------------------------------------------------------------
def test_there_is_one_canonical_name_and_no_alias(store):
    tools = AnalystTools(store)
    assert "summarize_run" in TOOL_NAMES
    assert not hasattr(tools, "get_run_summary"), "an alias would hide a prompt/registry mismatch"


def test_every_public_tool_is_either_model_visible_or_explicitly_template_only(store):
    tools = AnalystTools(store)
    public = public_methods(tools)
    assert set(TOOL_NAMES) <= public  # every registered name is a real method
    assert public - set(TOOL_NAMES) == TEMPLATE_ONLY  # nothing is silently left out of the registry
    assert len(set(TOOL_NAMES)) == len(TOOL_NAMES)  # and no name is listed twice


def test_the_model_visible_registry_is_exactly_tool_names_with_matching_schemas(store):
    tools = AnalystTools(store)
    exposed = langchain_tools(tools)
    assert [t.name for t in exposed] == list(TOOL_NAMES)
    assert set(_DESCRIPTIONS) == set(TOOL_NAMES)  # no stale description, none missing
    for tool in exposed:
        assert tool.description and tool.description == _DESCRIPTIONS[tool.name]
        expected = set(inspect.signature(getattr(tools, tool.name)).parameters)
        assert set(tool.args) == expected, tool.name  # the model sees the real parameters


def test_the_chat_model_is_bound_to_exactly_the_registered_tools(store):
    class Recorder:
        bound = None

        def bind_tools(self, tools):
            Recorder.bound = [t.name for t in tools]
            return self

        def invoke(self, messages):
            return AIMessage(content="Insufficient evidence.")

    LangChainAnalyst(Recorder()).respond("s", [], "q", AnalystTools(store))
    assert Recorder.bound == list(TOOL_NAMES)


# ---- model-facing text only names registered tools ----------------------------------------
def model_facing_texts(store) -> dict[str, str]:
    tools = AnalystTools(store)
    return {
        "system prompt": ANALYST_SYSTEM_PROMPT,
        "summary request": SUMMARY_REQUEST,
        "tool-failure correction": _TOOL_FAILURE_CORRECTION,
        "citation correction": _correction(CitationReport()),
        **{f"description of {n}": d for n, d in _DESCRIPTIONS.items()},
        **{f"schema doc of {n}": (getattr(tools, n).__doc__ or "") for n in TOOL_NAMES},
    }


def test_prompts_and_descriptions_name_only_registered_tools(store):
    zones = {z.value for z in Zone}  # enum values the descriptions list, e.g. main_gate
    known = set(TOOL_NAMES) | argument_names(AnalystTools(store)) | NOT_TOOLS | zones
    for where, text in model_facing_texts(store).items():
        for token in TOOL_LIKE.findall(text):
            assert token in known, f"{where} mentions '{token}', which is not a registered tool"
        for token in TOOL_VERB.findall(text):  # anything that looks like a tool must BE one
            assert token in known, (where, token)


def test_the_tool_the_summary_request_names_is_registered_and_is_summarize_run():
    (named,) = re.findall(r"\bCall (\w+) first\b", SUMMARY_REQUEST)
    assert named == "summarize_run" and named in TOOL_NAMES


def test_a_nonexistent_tool_name_in_a_prompt_would_be_caught(store):
    """The check itself must be able to fail: the old prompt text is rejected by it."""
    old = SUMMARY_REQUEST.replace("summarize_run", "get_run_summary")
    assert "get_run_summary" not in TOOL_NAMES
    assert any(t not in TOOL_NAMES for t in TOOL_VERB.findall(old))


def test_the_planner_can_only_call_registered_tools(store):
    assert set(planner._RENDERERS) <= set(TOOL_NAMES)
    source = Path(planner.__file__).read_text(encoding="utf-8")
    literal_calls = set(re.findall(r'\(\s*"([a-z_]+)",\s*(?:\{|args|dict)', source))
    assert literal_calls <= set(TOOL_NAMES), literal_calls - set(TOOL_NAMES)
    questions = [
        "Why was the loitering alert raised?",
        "Which alerts are still open?",
        "Show all truck events",
        "What happened at the main gate?",
        "What was the drone battery when the link was lost?",
        "What changed between 00:10 and 00:20?",
        "Give me the timeline",
        "Summarize the run",
    ]
    for question in questions:
        session = AnalystSession(store, build_planner_analyst())
        answer = session.ask(question)
        used = {step["tool"] for step in answer.tool_trace}
        assert used <= set(TOOL_NAMES), (question, used - set(TOOL_NAMES))
        assert session.llm.stats.tool_errors == 0, question


# ---- summarize_run, end to end -------------------------------------------------------------------
def test_summarize_run_returns_the_stored_run_facts(store):
    result = AnalystTools(store).summarize_run()
    assert "error" not in result and result["observation_span"]
    assert result["counts"]["alerts"] == 7


class _Follow:
    """A chat model that does what the prompt tells it: call the named tool, then answer."""

    def __init__(self, tool_name, final):
        self.tool_name, self.final, self.seen, self.step = tool_name, final, [], 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        self.step += 1
        if self.step == 1:
            return AIMessage(
                content="", tool_calls=[{"name": self.tool_name, "args": {}, "id": "t1"}]
            )
        return AIMessage(content=self.final)


def test_a_model_that_follows_the_summary_prompt_reaches_a_real_tool(store):
    (name,) = re.findall(r"\bCall (\w+) first\b", SUMMARY_REQUEST)  # what the prompt says to call
    chat = _Follow(name, "Seven alerts were raised in this run [alert:ALR-LOITER-EVT-P-001].")
    llm = LangChainAnalyst(chat)
    answer = AnalystSession(store, llm).summarize()
    tool_messages = [m for m in chat.seen[-1] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1 and '"error"' not in tool_messages[0].content
    assert "observation_span" in tool_messages[0].content  # the tool really ran
    assert llm.stats.tools == ["summarize_run"] and llm.stats.tool_errors == 0
    assert answer.mode == "llm" and not answer.retried, answer.notes
    assert [s["tool"] for s in answer.tool_trace] == ["summarize_run"]


def test_the_planner_summarises_through_summarize_run_with_the_real_loop(store):
    llm = build_planner_analyst()
    answer = AnalystSession(store, llm).summarize()
    assert answer.mode == "llm" and answer.citations, answer.notes
    assert "summarize_run" in {s["tool"] for s in answer.tool_trace}
    assert all(s["ok"] for s in answer.tool_trace) and llm.stats.tool_errors == 0


def test_the_template_summary_uses_the_same_tool_and_contains_the_same_facts(store):
    answer = AnalystSession(store).summarize()
    assert answer.mode == "template" and "27 observation(s)" in answer.text


# ---- a failed tool call is not "no evidence" ---------------------------------------------------
def test_a_refusal_after_calling_a_nonexistent_tool_is_not_accepted_as_no_evidence(store):
    chat = _Follow("get_run_summary", "Insufficient evidence: nothing is stored.")
    llm = LangChainAnalyst(chat)
    answer = AnalystSession(store, llm).summarize()
    results = [m.content for m in chat.seen[1] if isinstance(m, ToolMessage)]
    assert "unknown tool 'get_run_summary'" in results[0]
    assert llm.stats.tool_errors >= 1
    assert answer.mode == "fallback" and "Insufficient evidence" not in answer.text
    assert "27 observation(s)" in answer.text  # the stored data was there all along
    assert any("does not show that the evidence is absent" in n for n in answer.notes)


def test_after_a_failed_lookup_a_successful_one_makes_the_answer_acceptable(store):
    script = iter(
        [
            AIMessage(content="", tool_calls=[{"name": "run_sql", "args": {}, "id": "a"}]),
            AIMessage(content="Insufficient evidence."),  # refused right after the failure
            AIMessage(content="", tool_calls=[{"name": "list_alerts", "args": {}, "id": "b"}]),
            AIMessage(content="Seven alerts were raised in this run [alert:ALR-LOITER-EVT-P-001]."),
        ]
    )

    class Scripted:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return next(script)

    answer = AnalystSession(store, LangChainAnalyst(Scripted())).ask("How many alerts?")
    assert answer.mode == "llm" and answer.retried and "Seven alerts" in answer.text


def test_genuine_insufficient_evidence_is_still_a_valid_refusal(store):
    script = iter(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_events", "args": {"query": "zebra"}, "id": "a"}],
            ),
            AIMessage(content="Insufficient evidence: no stored event mentions a zebra."),
        ]
    )

    class Scripted:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            return next(script)

    llm = LangChainAnalyst(Scripted())
    answer = AnalystSession(store, llm).ask("Was a zebra seen?")
    assert llm.stats.tool_calls == 1 and llm.stats.tool_errors == 0  # the lookup worked, found none
    assert answer.mode == "llm" and not answer.retried and "Insufficient evidence" in answer.text


def test_a_refusal_with_no_tool_calls_at_all_is_still_accepted(store):
    chat = _Follow("summarize_run", "x")
    chat.invoke = lambda messages: AIMessage(content="Insufficient evidence: out of scope.")
    answer = AnalystSession(store, LangChainAnalyst(chat)).ask("What is the airspeed of a swallow?")
    assert answer.mode == "llm" and not answer.retried
