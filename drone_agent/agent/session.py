"""Analyst session: question -> (LLM or template) -> citation gate -> answer, with bounded memory.

The gate is the safety net: an LLM answer is shown only if every cited id exists in the database
AND was returned by a tool in this conversation, it cites something (or explicitly says
"insufficient evidence"), and it makes no identity/intent claims. One correction retry is allowed;
after that the deterministic template answer is returned instead of unsupported claims.
The session never writes to the store. No LangChain here (the LLM is a Protocol).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

from drone_agent.agent.citations import CitationReport, extract_citations, validate_answer
from drone_agent.agent.evidence import ToolLimits
from drone_agent.agent.offline import OfflineAnalyst
from drone_agent.agent.prompt import ANALYST_SYSTEM_PROMPT, SUMMARY_REQUEST
from drone_agent.agent.summary import render_summary
from drone_agent.agent.tools import AnalystTools
from drone_agent.store import Store

MAX_HISTORY_TURNS = 6  # question/answer pairs kept in memory; older ones are dropped


class AnalystLLMError(Exception):
    """The analyst model call failed (API error, tool-loop limit). Messages never hold the key."""


class AnalystLLM(Protocol):
    name: str

    def respond(
        self,
        system: str,
        history: list[tuple[str, str]],
        question: str,
        tools: AnalystTools,
        correction: tuple[str, str] | None = None,
    ) -> str:
        """Answer `question`, calling `tools` as needed. `correction` = (previous answer,
        instruction) when a first answer was rejected."""
        ...


@dataclass
class CallStats:
    """Counters an LLM adapter may keep (model rounds and tool executions). No content."""

    rounds: int = 0
    tool_calls: int = 0
    tools: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.rounds, self.tool_calls, self.tools = 0, 0, []


@dataclass(frozen=True)
class AnswerMetrics:
    """Observability for one LLM-backed answer. Deliberately holds no prompts, keys or reasoning."""

    model: str
    rounds: int | None  # model/API rounds (None if the adapter does not count them)
    tool_calls: int | None
    tools_used: tuple[str, ...]
    corrections: int  # correction attempts after a rejected answer (0 or 1)
    fallback: bool
    latency_s: float

    def line(self) -> str:
        rounds = "n/a" if self.rounds is None else self.rounds
        calls = "n/a" if self.tool_calls is None else self.tool_calls
        tools = f" ({', '.join(self.tools_used)})" if self.tools_used else ""
        return (
            f"model {self.model}; model rounds {rounds}; tool calls {calls}{tools}; "
            f"correction attempts {self.corrections}; fallback {'yes' if self.fallback else 'no'}; "
            f"total analyst latency {self.latency_s:.2f}s"
        )


@dataclass
class Answer:
    text: str
    mode: str  # "llm" | "template" | "fallback"
    citations: list[str] = field(default_factory=list)  # verified tokens like "[event:EVT-V-001]"
    retried: bool = False
    notes: list[str] = field(default_factory=list)
    metrics: AnswerMetrics | None = None  # only for LLM-backed answers; template mode has none

    @property
    def label(self) -> str:
        return {
            "llm": "LLM answer, citations verified",
            "template": "template answer (deterministic, no LLM)",
            "fallback": "deterministic fallback (LLM answer withheld or unavailable)",
        }[self.mode]


def _correction(report: CitationReport) -> str:
    return (
        "Your answer was rejected by the evidence check: "
        + "; ".join(report.problems())
        + ". Rewrite it using only ids returned by the tools (call them again if needed), cite "
        "every factual sentence as [kind:id], avoid identity or intent claims, or reply "
        "'Insufficient evidence'."
    )


class AnalystSession:
    def __init__(
        self,
        store: Store,
        llm: AnalystLLM | None = None,
        limits: ToolLimits | None = None,
        max_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        self.store = store
        self.llm = llm
        self.tools = AnalystTools(store, limits)
        self.offline = OfflineAnalyst(self.tools)
        self.history: deque[tuple[str, str]] = deque(maxlen=2 * max_turns)
        self.context: dict[str, str] = {}  # last entity/alert/event/zone, for template follow-ups

    # -- public -----------------------------------------------------------------------------
    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        if self.llm:
            answer = self._measured(lambda: self._llm_answer(question))
        else:
            answer = self._template(question, "template")
        self._remember(question, answer)
        return answer

    def summarize(self) -> Answer:
        if self.llm is None:
            text = render_summary(self.tools.get_run_summary())
            answer = self._checked_template(text, "template")
        else:
            answer = self._measured(
                lambda: self._llm_answer(
                    SUMMARY_REQUEST, fallback=lambda: render_summary(self.tools.get_run_summary())
                )
            )
        self._remember("Summarize the run.", answer)
        return answer

    # -- internals --------------------------------------------------------------------------
    def _measured(self, run) -> Answer:
        stats = getattr(self.llm, "stats", None)
        if isinstance(stats, CallStats):
            stats.reset()
        started = time.perf_counter()
        answer = run()
        answer.metrics = AnswerMetrics(
            model=getattr(self.llm, "name", "unknown"),
            rounds=stats.rounds if isinstance(stats, CallStats) else None,
            tool_calls=stats.tool_calls if isinstance(stats, CallStats) else None,
            tools_used=tuple(sorted(set(stats.tools))) if isinstance(stats, CallStats) else (),
            corrections=1 if answer.retried else 0,
            fallback=answer.mode == "fallback",
            latency_s=time.perf_counter() - started,
        )
        return answer

    def _template(self, question: str, mode: str, notes: list[str] | None = None) -> Answer:
        result = self.offline.answer(question, self.context)
        if result.subject:
            self.context[result.subject[0]] = result.subject[1]
        answer = self._checked_template(result.text, mode)
        answer.notes = list(notes or []) + answer.notes
        return answer

    def _checked_template(self, text: str, mode: str) -> Answer:
        report = validate_answer(
            self.store,
            text,
            retrieved=self.tools.retrieved,
            require_citation="insufficient evidence" not in text.lower(),
        )
        notes = (
            []
            if report.ok
            else ["template answer failed its own check: " + "; ".join(report.problems())]
        )
        return Answer(text, mode, [c.token for c in report.cited], notes=notes)

    def _llm_answer(self, question: str, fallback=None) -> Answer:
        assert self.llm is not None
        history = list(self.history)
        correction = None
        notes: list[str] = []
        for attempt in (1, 2):
            try:
                text = self.llm.respond(
                    ANALYST_SYSTEM_PROMPT, history, question, self.tools, correction
                )
            except AnalystLLMError as exc:
                notes.append(f"analyst model unavailable: {exc}")
                return self._fall_back(question, notes, fallback)
            report = validate_answer(
                self.store,
                text,
                retrieved=self.tools.retrieved,
                require_citation="insufficient evidence" not in text.lower(),
            )
            if report.ok:
                return Answer(
                    text, "llm", [c.token for c in report.cited], retried=attempt == 2, notes=notes
                )
            notes.append(f"attempt {attempt} rejected: " + "; ".join(report.problems()))
            correction = (text, _correction(report))
        notes.append("second answer also failed the evidence check; unsupported text withheld")
        answer = self._fall_back(question, notes, fallback)
        answer.retried = True
        return answer

    def _fall_back(self, question: str, notes: list[str], fallback) -> Answer:
        if fallback is not None:
            return self._checked_template_with_notes(fallback(), notes)
        return self._template(question, "fallback", notes)

    def _checked_template_with_notes(self, text: str, notes: list[str]) -> Answer:
        answer = self._checked_template(text, "fallback")
        answer.notes = notes + answer.notes
        return answer

    def _remember(self, question: str, answer: Answer) -> None:
        self.history.append(("user", question))
        self.history.append(("assistant", answer.text))
        # A follow-up such as "where was it seen?" resolves only when the answer named exactly one
        # subject of a kind; with several candidates the pronoun stays unresolved (never guessed).
        cited: dict[str, set[str]] = {}
        for c in extract_citations(answer.text):
            if c.kind in ("entity", "alert", "event"):
                cited.setdefault(c.kind, set()).add(c.ref)
        for kind, refs in cited.items():
            if len(refs) == 1:
                self.context[kind] = next(iter(refs))
