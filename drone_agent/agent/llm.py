"""LangChain adapter: the ONLY module that imports langchain.

A chat model that supports tool calling is given the read-only tools and runs a small, bounded
tool-calling loop (no LangGraph, no agent framework). It returns text only; the session then
verifies the citations before anything is shown. Install with `pip install -e ".[agent]"`.

The chat model is any object with `bind_tools(tools)` and `invoke(messages)`: the real Gemini
model (`build_chat_model`), a `ReplayChatModel` that replays recorded model turns, or the
deterministic `PlannerChatModel` (agent/replay.py). The loop, the tools, the caps and the citation
gate are identical for all of them; only where the model's messages come from differs.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from drone_agent.agent.config import AnalystConfig
from drone_agent.agent.session import AnalystLLMError, CallStats
from drone_agent.agent.tools import TOOL_NAMES, AnalystTools

MAX_TOOL_ROUNDS = 6  # model turns per answer that may request tools
MAX_TOOL_CALLS = 12  # total tool executions per answer
MAX_TOOL_RESULT_CHARS = 12000  # a bounded view should never reach this; guard anyway

_ZONES = "main_gate, driveway, parking_area, warehouse, restricted_zone, garage"
_DESCRIPTIONS = {
    "query_events": (
        f"List temporal events. Optional filters: zone ({_ZONES}), object_type (person, "
        "vehicle, other), start and end (ISO-8601 with UTC offset; events overlapping the range), "
        "entity_id, status (open or closed), limit."
    ),
    "search_events": (
        "Events whose evidence matches a keyword search such as 'truck', 'blue pickup' or "
        "'hoodie'. Optional zone and object_type. Returns the matched frame refs."
    ),
    "search_frames": (
        f"Search stored frame observations. Optional: query (keywords), zone ({_ZONES}), "
        "object_type, start and end (ISO-8601 with offset), limit. At most 15 results."
    ),
    "get_event": "Get one event by id, e.g. EVT-V-001 (status open/closed, confidence).",
    "get_frame": "Get one frame by number: its observation, telemetry, events and alerts.",
    "get_alert": (
        "Get one alert by id: severity, status (open/escalated/acknowledged/resolved), verdict, "
        "recommended action, escalation ladder, lifecycle history and rule facts."
    ),
    "get_evidence": (
        "The full evidence chain behind alert:ID, event:ID or entity:ID: rule facts, events, "
        "observations, frames and telemetry. Use this to explain WHY something was raised."
    ),
    "entity_history": (
        "Appearance history of a vehicle entity (ENT-V-...), or, for a person event id "
        "(EVT-P-...), earlier events in that zone with similar clothing. People are never "
        "identity-linked, so a person result is a weak hint, not a match."
    ),
    "list_entities": "List similar-vehicle entities (coarse attribute matches, not identities).",
    "list_alerts": (
        "List alerts raised by the deterministic rules; filter by rule_id, severity, zone, "
        "status (open, escalated, acknowledged, resolved), limit."
    ),
    "telemetry_at": (
        "The drone's telemetry (battery, signal, altitude, position, zone) at a time "
        "(ISO-8601 with offset): the latest sample at or before it. Telemetry is simulated."
    ),
    "get_timeline": (
        "A chronological list of events opening/closing and alert raised/escalated/resolved "
        "changes, optionally between start and end (ISO-8601 with offset)."
    ),
    "summarize_run": (
        "Structured facts about the whole stored run: counts, time span, zones, events, alerts "
        "and their statuses, entities and limitations."
    ),
}


def langchain_tools(tools: AnalystTools) -> list[StructuredTool]:
    """Wrap each read-only tool method; results are JSON text, errors are returned not raised."""

    def wrap(name: str) -> StructuredTool:
        method = getattr(tools, name)

        def run(**kwargs: Any) -> str:
            result = method(**kwargs)
            tools.trace(name, kwargs, result)
            text = json.dumps(result, default=str, ensure_ascii=True)
            if len(text) > MAX_TOOL_RESULT_CHARS:
                return json.dumps({"error": "result too large; narrow the query"})
            return text

        return StructuredTool.from_function(
            func=method_signature_proxy(method, run),
            name=name,
            description=_DESCRIPTIONS[name],
        )

    return [wrap(n) for n in TOOL_NAMES]


def method_signature_proxy(method: Any, run: Any) -> Any:
    """Give `run` the wrapped method's signature so LangChain builds the right argument schema."""
    import functools

    @functools.wraps(method)
    def proxy(*args: Any, **kwargs: Any) -> str:
        return run(*args, **kwargs)

    return proxy


def build_chat_model(config: AnalystConfig) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.model,
        google_api_key=config.api_key,
        timeout=config.timeout_s,
        max_retries=1,
    )  # no temperature: the Gemini 3.x chat models reject the argument


def _is_error(result: Any) -> bool:
    """Did a tool call fail? Tools report failure as a JSON object with an `error` key."""
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except ValueError:
        return False
    return isinstance(parsed, dict) and "error" in parsed


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


class LangChainAnalyst:
    """AnalystLLM implemented with a LangChain chat model that supports tool calling."""

    def __init__(self, chat_model: Any, name: str = "langchain") -> None:
        self.chat_model = chat_model
        self.name = name
        self.stats = CallStats()  # rounds / tool calls of the current answer (no content)
        self._secret: str | None = None

    @classmethod
    def from_config(cls, config: AnalystConfig) -> "LangChainAnalyst":
        analyst = cls(build_chat_model(config), name=f"gemini:{config.model}")
        analyst._secret = config.api_key
        return analyst

    def _scrub(self, text: str) -> str:
        return text.replace(self._secret, "[redacted]") if self._secret else text

    def respond(
        self,
        system: str,
        history: list[tuple[str, str]],
        question: str,
        tools: AnalystTools,
        correction: tuple[str, str] | None = None,
    ) -> str:
        lc_tools = langchain_tools(tools)
        by_name = {t.name: t for t in lc_tools}
        model = self.chat_model.bind_tools(lc_tools)
        messages: list[Any] = [SystemMessage(system)]
        messages += [(HumanMessage if r == "user" else AIMessage)(t) for r, t in history]
        messages.append(HumanMessage(question))
        if correction:
            messages += [AIMessage(correction[0]), HumanMessage(correction[1])]
        executed = 0
        for _ in range(MAX_TOOL_ROUNDS + 1):
            self.stats.rounds += 1
            try:
                reply = model.invoke(messages)
            except Exception as exc:  # network/auth/quota: report the class, never the key
                raise AnalystLLMError(
                    f"{type(exc).__name__}: {self._scrub(str(exc))[:200]}"
                ) from None
            messages.append(reply)
            calls = getattr(reply, "tool_calls", None) or []
            if not calls:
                return _flatten(reply.content)
            for call in calls:
                executed += 1
                if executed > MAX_TOOL_CALLS:
                    raise AnalystLLMError("too many tool calls in one answer")
                self.stats.tool_calls += 1
                tool = by_name.get(call["name"])
                if tool is not None:
                    self.stats.tools.append(call["name"])
                if tool is None:
                    result = json.dumps({"error": f"unknown tool '{call['name']}'"})
                else:
                    try:
                        result = tool.invoke(call.get("args") or {})
                    except Exception as exc:  # bad arguments from the model
                        result = json.dumps({"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                if _is_error(result):
                    self.stats.tool_errors += 1
                messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        raise AnalystLLMError("tool-calling loop did not finish")
