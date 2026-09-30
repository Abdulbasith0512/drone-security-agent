"""LangChain adapter (with a fake chat model), the ask/summarize CLI, and the import boundary."""

import ast
import json
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from conftest import ROOT, SCENARIO_DIR
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from rich.console import Console

from drone_agent.__main__ import main
from drone_agent.agent.config import AnalystConfig, AnalystConfigError
from drone_agent.agent.llm import LangChainAnalyst, langchain_tools
from drone_agent.agent.session import AnalystLLMError, AnalystSession
from drone_agent.agent.tools import TOOL_NAMES, AnalystTools
from drone_agent.ingest import ingest_result
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store

SECRET = "TEST-PLACEHOLDER-NOT-A-REAL-KEY-0002"


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("llm") / "security.db"
    with Store.open(path) as s:
        ingest_result(s, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        build_events(s)
        analyze(s)
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for name in ("GEMINI_API_KEY", "ANALYST_MODEL", "GEMINI_MODEL"):
        monkeypatch.delenv(name, raising=False)


class FakeChatModel:
    """Implements the two methods the adapter uses; replies come from a script of AIMessages."""

    def __init__(self, *script):
        self.script = list(script)
        self.bound = None
        self.seen = []  # message lists passed to invoke

    def bind_tools(self, tools):
        self.bound = tools
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def call(name, args=None, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": call_id}])


# ---- configuration --------------------------------------------------------------------------
def test_analyst_model_is_configured_separately_from_the_vision_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    monkeypatch.setenv("GEMINI_MODEL", "vision-model")
    assert AnalystConfig.from_env(load_env_file=False).model != "vision-model"
    monkeypatch.setenv("ANALYST_MODEL", "analyst-model")
    assert AnalystConfig.from_env(load_env_file=False).model == "analyst-model"
    assert AnalystConfig.from_env("cli-model", load_env_file=False).model == "cli-model"
    assert SECRET not in repr(AnalystConfig.from_env(load_env_file=False))


def test_missing_key_is_a_concise_config_error():
    with pytest.raises(AnalystConfigError, match="GEMINI_API_KEY is not set"):
        AnalystConfig.from_env(load_env_file=False)


# ---- tools as LangChain tools ---------------------------------------------------------------
def test_langchain_tools_expose_exactly_the_read_only_tool_set(db_path):
    with Store.open(db_path) as store:
        tools = langchain_tools(AnalystTools(store))
        assert [t.name for t in tools] == list(TOOL_NAMES)
        search = next(t for t in tools if t.name == "search_frames")
        assert {"query", "zone", "object_type", "start", "end", "limit"} == set(search.args)
        out = json.loads(search.invoke({"zone": "warehouse"}))
        assert out["total"] == 3
        bad = json.loads(search.invoke({"zone": "lobby"}))
        assert "zone must be one of" in bad["error"]  # errors are returned, not raised


# ---- adapter loop ---------------------------------------------------------------------------
def test_adapter_runs_tool_calls_then_returns_the_final_text(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            call("get_alert", {"alert_id": "ALR-LOITER-EVT-P-001"}),
            AIMessage(
                content="Loitering for 480 s [alert:ALR-LOITER-EVT-P-001] [event:EVT-P-001]."
            ),
        )
        llm = LangChainAnalyst(chat)
        session = AnalystSession(store, llm)
        answer = session.ask("Why the loitering alert?")
        assert answer.mode == "llm" and answer.citations == [
            "[alert:ALR-LOITER-EVT-P-001]", "[event:EVT-P-001]"
        ]  # fmt: skip
        assert [t.name for t in chat.bound] == list(TOOL_NAMES)
        second_call = chat.seen[1]
        assert (
            isinstance(second_call[0], SystemMessage)
            and "Insufficient evidence" in second_call[0].content
        )
        tool_msg = second_call[-1]
        assert isinstance(tool_msg, ToolMessage) and tool_msg.tool_call_id == "c1"
        assert json.loads(tool_msg.content)["alert"]["severity"] == "critical"
        assert "alert:ALR-LOITER-EVT-P-001" in session.tools.retrieved


def test_adapter_sends_history_and_the_correction_turn(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            AIMessage(content="Wrong [event:EVT-Z-9]."),
            AIMessage(content="Insufficient evidence for that."),
        )
        session = AnalystSession(store, LangChainAnalyst(chat))
        session.history.extend([("user", "earlier q"), ("assistant", "earlier a")])
        answer = session.ask("new q")
        assert answer.retried and answer.mode == "llm"
        first, second = chat.seen
        assert [type(m) for m in first] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
        assert first[-1].content == "new q"
        assert isinstance(second[-2], AIMessage) and "EVT-Z-9" in second[-2].content
        assert "rejected by the evidence check" in second[-1].content


def test_unknown_tools_and_bad_arguments_are_reported_back_not_raised(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            AIMessage(content="", tool_calls=[
                {"name": "run_sql", "args": {"sql": "DROP TABLE alerts"}, "id": "a"},
                {"name": "get_alert", "args": {"wrong_arg": 1}, "id": "b"},
            ]),
            AIMessage(content="Insufficient evidence."),
        )  # fmt: skip
        answer = AnalystSession(store, LangChainAnalyst(chat)).ask("q")
        assert answer.mode == "llm"
        results = [m.content for m in chat.seen[1] if isinstance(m, ToolMessage)]
        assert "unknown tool 'run_sql'" in results[0] and "error" in results[1]
        assert store.count_alerts() == 7  # nothing was executed against the database


def test_tool_loop_is_bounded(db_path):
    with Store.open(db_path) as store:
        endless = FakeChatModel(*[call("get_run_summary", call_id=f"c{i}") for i in range(20)])
        with pytest.raises(AnalystLLMError, match="did not finish"):
            LangChainAnalyst(endless).respond("s", [], "q", AnalystTools(store))
        wide = FakeChatModel(
            AIMessage(content="", tool_calls=[
                {"name": "list_alerts", "args": {}, "id": f"x{i}"} for i in range(13)
            ])
        )  # fmt: skip
        with pytest.raises(AnalystLLMError, match="too many tool calls"):
            LangChainAnalyst(wide).respond("s", [], "q", AnalystTools(store))


def test_api_errors_become_a_fallback_without_leaking_the_key(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(RuntimeError(f"429 quota for key {SECRET}"))
        llm = LangChainAnalyst(chat)
        llm._secret = SECRET
        answer = AnalystSession(store, llm).ask("Was the blue pickup seen more than once?")
        assert answer.mode == "fallback" and "similar blue pickup" in answer.text
        assert any("RuntimeError" in n for n in answer.notes)
        assert SECRET not in answer.text and not any(SECRET in n for n in answer.notes)


def test_list_style_model_content_is_flattened(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            AIMessage(content=[{"type": "text", "text": "Insufficient "}, "evidence."])
        )
        text = LangChainAnalyst(chat).respond("s", [], "q", AnalystTools(store))
        assert text == "Insufficient evidence."


# ---- CLI ------------------------------------------------------------------------------------
def run_cli(*argv):
    out = StringIO()
    code = main([str(a) for a in argv], Console(file=out, width=200, force_terminal=False))
    return code, out.getvalue()


def test_cli_ask_offline_answers_with_citations_and_needs_no_key(db_path):
    code, text = run_cli("ask", "--db", db_path, "Was the blue pickup seen more than once?")
    assert code == 0, text
    assert "template answer (deterministic, no LLM)" in text
    assert "[entity:ENT-V-001]" in text and "Verified citations:" in text
    assert "similar blue pickup" in text and "GEMINI_API_KEY" not in text
    code, text = run_cli(
        "ask", "--db", db_path, "--offline", "Show me the evidence for the loitering alert."
    )
    assert code == 0 and "[obs:obs-1]" in text


def test_cli_ask_supports_follow_ups_in_one_session(db_path):
    code, text = run_cli(
        "ask", "--db", db_path, "Was the blue pickup seen more than once?",
        "--follow-up", "Where was it seen?",
    )  # fmt: skip
    assert code == 0
    assert "Q: Where was it seen?" in text and "driveway" in text


def test_cli_summarize_then_follow_up_on_objects(db_path):
    code, text = run_cli("summarize", "--db", db_path, "--follow-up", "What objects were observed?")
    assert code == 0
    assert "27 observation(s)" in text and "Limitations:" in text
    assert "Q: What objects were observed?" in text and "person 13, vehicle 7" in text


def test_cli_interactive_follow_ups(db_path, monkeypatch):
    answers = iter(["Where was it seen?", ""])
    console = Console(file=StringIO(), width=200, force_terminal=False)
    monkeypatch.setattr(console, "input", lambda prompt="": next(answers))
    code = main(
        ["ask", "--db", str(db_path), "Was the blue pickup seen more than once?", "-i"], console
    )
    assert code == 0 and "Where was it seen?" in console.file.getvalue()


def test_cli_errors_are_clear(db_path, tmp_path):
    assert run_cli("ask", "--db", db_path)[0] == 2  # no question
    code, text = run_cli("ask", "--db", tmp_path / "missing.db", "q")
    assert code == 2 and "database not found" in text and not (tmp_path / "missing.db").exists()
    code, text = run_cli("ask", "--db", db_path, "--provider", "gemini", "q")
    assert code == 2 and "GEMINI_API_KEY is not set" in text


def test_cli_gemini_provider_uses_the_langchain_analyst_and_shows_the_model(db_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    chat = FakeChatModel(
        call("get_alert", {"alert_id": "ALR-REPEAT-ENT-V-001"}),
        AIMessage(content="A similar blue pickup appeared 3 times [alert:ALR-REPEAT-ENT-V-001]."),
    )
    seen = {}

    def fake_build(config):
        seen["model"] = config.model
        return chat

    monkeypatch.setattr("drone_agent.agent.llm.build_chat_model", fake_build)
    code, text = run_cli(
        "ask", "--db", db_path, "--provider", "gemini", "--model", "m-analyst", "q"
    )
    assert code == 0, text
    assert seen["model"] == "m-analyst" and "Analyst model: m-analyst" in text
    assert "GEMINI_API_KEY configured: yes" in text and SECRET not in text
    assert "LLM answer, citations verified" in text


def test_cli_gemini_provider_falls_back_when_the_model_output_is_unsupported(db_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    chat = FakeChatModel(
        AIMessage(content="Made up [event:EVT-Z-1]."), AIMessage(content="No cites.")
    )
    monkeypatch.setattr("drone_agent.agent.llm.build_chat_model", lambda config: chat)
    code, text = run_cli(
        "ask", "--db", db_path, "--provider", "gemini", "Why was the repeat-vehicle alert raised?"
    )
    assert code == 0
    assert "deterministic fallback" in text and "EVT-Z-1" not in text.split("note:")[0]
    assert "[alert:ALR-REPEAT-ENT-V-001]" in text


# ---- import boundary ------------------------------------------------------------------------
CORE_PACKAGES = ("video", "vision", "pipeline", "rules", "store", "models", "telemetry")


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_core_packages_never_import_langchain_or_the_agent_layer():
    package = ROOT / "drone_agent"
    files = [f for name in CORE_PACKAGES for f in (package / name).rglob("*.py")]
    files += [package / n for n in ("clock.py", "ingest.py", "scenario.py", "simulate.py")]
    assert len(files) > 30
    for f in files:
        for module in _imports(f):
            assert not module.startswith("langchain"), f"{f.name} imports {module}"
            assert not module.startswith("drone_agent.agent"), f"{f.name} imports {module}"


def test_only_agent_llm_imports_langchain():
    # the model-facing modules of the analyst package: the live loop, the recorded-turn replay
    # and the scripted planner (all chat models); nothing else, and nothing outside agent/
    hits = [
        f.relative_to(ROOT).as_posix()
        for f in (ROOT / "drone_agent").rglob("*.py")
        if any(m.startswith("langchain") for m in _imports(f))
    ]
    assert sorted(set(hits)) == [
        "drone_agent/agent/llm.py",
        "drone_agent/agent/planner.py",
        "drone_agent/agent/replay.py",
    ]
    assert not any(
        "langgraph" in m for f in (ROOT / "drone_agent").rglob("*.py") for m in _imports(f)
    )


def test_the_pipeline_and_offline_analyst_work_without_langchain_installed():
    code = (
        "import sys\n"
        "sys.modules['langchain_core'] = None\n"  # makes any `import langchain_core` fail
        "sys.modules['langchain_google_genai'] = None\n"
        "import drone_agent.__main__, drone_agent.agent.session, drone_agent.agent.offline\n"
        "assert not any(m.startswith('langchain') and sys.modules[m] for m in sys.modules)\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    assert result.returncode == 0 and "ok" in result.stdout, result.stderr


def test_no_api_key_is_needed_for_the_deterministic_pipeline_and_offline_analyst(
    tmp_path, monkeypatch
):
    for name in ("GEMINI_API_KEY", "ANALYST_MODEL"):
        monkeypatch.delenv(name, raising=False)
    db = tmp_path / "x.db"
    assert run_cli("ingest", SCENARIO_DIR / "security_events.yaml", "--db", db)[0] == 0
    assert run_cli("build-events", "--db", db)[0] == 0
    assert run_cli("analyze", "--db", db)[0] == 0
    code, text = run_cli("ask", "--db", db, "Were any people seen after hours?")
    assert code == 0 and "[alert:ALR-AFTERHOURS-EVT-P-001]" in text


# ---- temperature and metrics ----------------------------------------------------------------
def test_no_unsupported_temperature_argument_is_passed_to_the_gemini_chat_model(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", Recorder)
    from drone_agent.agent.llm import build_chat_model

    build_chat_model(AnalystConfig(model="gemini-3.6-flash", api_key=SECRET))
    assert "temperature" not in captured
    assert captured["model"] == "gemini-3.6-flash" and captured["google_api_key"] == SECRET


def test_llm_answers_carry_metrics_outside_the_answer_text(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            call("get_alert", {"alert_id": "ALR-LOITER-EVT-P-001"}, "a"),
            call("get_event", {"event_id": "EVT-P-001"}, "b"),
            AIMessage(content="Loitering [alert:ALR-LOITER-EVT-P-001] [event:EVT-P-001]."),
        )
        answer = AnalystSession(store, LangChainAnalyst(chat, name="gemini:test")).ask("q")
    m = answer.metrics
    assert (m.model, m.rounds, m.tool_calls) == ("gemini:test", 3, 2)
    assert m.tools_used == ("get_alert", "get_event")
    assert (m.corrections, m.fallback) == (0, False) and m.latency_s >= 0
    line = m.line()
    for expected in ("model gemini:test", "model rounds 3", "tool calls 2", "correction attempts 0",
                     "fallback no", "total analyst latency"):  # fmt: skip
        assert expected in line
    assert "metrics" not in answer.text and "rounds" not in answer.text  # answer is unaffected
    assert answer.citations == ["[alert:ALR-LOITER-EVT-P-001]", "[event:EVT-P-001]"]


def test_metrics_count_the_correction_attempt_and_the_fallback(db_path):
    with Store.open(db_path) as store:
        retry = FakeChatModel(
            AIMessage(content="Wrong [event:EVT-Z-9]."),
            AIMessage(content="Insufficient evidence."),
        )
        m = AnalystSession(store, LangChainAnalyst(retry)).ask("q").metrics
        assert (m.rounds, m.corrections, m.fallback) == (2, 1, False)
        bad = FakeChatModel(AIMessage(content="No cites."), AIMessage(content="Still none."))
        answer = AnalystSession(store, LangChainAnalyst(bad)).ask("Summarize the run")
        assert (answer.mode, answer.metrics.rounds, answer.metrics.corrections) == (
            "fallback",
            2,
            1,
        )
        assert answer.metrics.fallback is True and "fallback yes" in answer.metrics.line()
        down = FakeChatModel(RuntimeError("boom"))
        m = AnalystSession(store, LangChainAnalyst(down)).ask("List the alerts").metrics
        assert m.fallback is True and m.rounds == 1


def test_stats_reset_between_questions(db_path):
    with Store.open(db_path) as store:
        chat = FakeChatModel(
            call("get_run_summary"),
            AIMessage(content="Insufficient evidence."),
            AIMessage(content="Insufficient evidence."),
        )
        session = AnalystSession(store, LangChainAnalyst(chat))
        assert session.ask("one").metrics.tool_calls == 1
        second = session.ask("two").metrics
        assert (second.rounds, second.tool_calls, second.tools_used) == (1, 0, ())


def test_offline_mode_has_no_llm_metrics(db_path):
    with Store.open(db_path) as store:
        assert AnalystSession(store).ask("List the alerts").metrics is None
    code, text = run_cli("ask", "--db", db_path, "List the alerts")
    assert code == 0 and "Analyst metrics" not in text and "model rounds" not in text
    code, text = run_cli("summarize", "--db", db_path)
    assert code == 0 and "Analyst metrics" not in text


def test_cli_prints_metrics_for_gemini_answers_without_secrets(db_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    chat = FakeChatModel(
        call("get_alert", {"alert_id": "ALR-REPEAT-ENT-V-001"}),
        AIMessage(content="A similar blue pickup appeared 3 times [alert:ALR-REPEAT-ENT-V-001]."),
    )
    monkeypatch.setattr("drone_agent.agent.llm.build_chat_model", lambda config: chat)
    code, text = run_cli("ask", "--db", db_path, "--provider", "gemini", "--model", "mm", "q")
    assert code == 0, text
    assert "Analyst metrics: model gemini:mm; model rounds 2; tool calls 1 (get_alert)" in text
    assert "correction attempts 0; fallback no; total analyst latency" in text
    assert SECRET not in text
