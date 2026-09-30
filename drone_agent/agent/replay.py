"""Record and replay the analyst model's turns, so the LangChain loop runs offline.

The analyst is a LangChain tool-calling loop: model turn -> tool calls -> tool results -> model turn
-> ... -> final text. Only the MODEL's turns need a network; everything else (the tools, their
results, the caps, the citation gate) is local and deterministic. So a real model's turns can be
recorded once and replayed with no key:

    RecordingChatModel(real_model, path)   wraps a real chat model and records every turn
    ReplayChatModel(path)                  returns the recorded turns; tools still run for real

A turn is keyed by the conversation so far (all user messages, plus how many model/tool messages
follow the last one), so replay only works when the tool results it sees are the same as when it
was recorded: a stale cassette against a changed database misses instead of answering wrongly, and
the session then falls back to the deterministic template analyst (labelled).

Nothing but the model's own text and tool-call requests is stored: no prompts, no tool results, no
credentials. Replayed answers are labelled REPLAYED wherever they are shown; they are recorded
model output, not a live call.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

FORMAT = "analyst-cassette-1"


class ReplayMiss(Exception):
    """No recorded model turn matches this conversation."""


class CassetteFormatError(ValueError):
    """The analyst cassette file is unreadable or malformed."""


def turn_key(messages: list[Any]) -> str:
    """Identify a model turn by the user's messages and how far the conversation has progressed."""
    humans = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
    last = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    step = len(messages) - 1 - last
    payload = json.dumps([humans, step], ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def _tool_calls(message: Any) -> list[dict]:
    return [
        {"name": c["name"], "args": c.get("args") or {}, "id": c["id"]}
        for c in (getattr(message, "tool_calls", None) or [])
    ]


class RecordingChatModel:
    """Wraps a chat model; behaves identically and records each reply for later replay."""

    def __init__(self, inner: Any, path: str | Path, label: str = "recorded") -> None:
        self.inner = inner
        self.path = Path(path)
        self.label = label
        self._bound: Any = inner
        self.turns: dict[str, dict] = {}
        if self.path.exists():  # extend an existing cassette instead of losing its turns
            self.turns = {t["key"]: t for t in load_cassette(self.path)["turns"]}

    def bind_tools(self, tools: Any) -> "RecordingChatModel":
        self._bound = self.inner.bind_tools(tools)
        return self

    def invoke(self, messages: list[Any]) -> Any:
        reply = self._bound.invoke(messages)
        first = next((str(m.content) for m in messages if isinstance(m, HumanMessage)), "")
        content = reply.content if isinstance(reply.content, str) else _text(reply.content)
        self.turns[turn_key(messages)] = {
            "key": turn_key(messages),
            "question": first[:80],
            "content": content,
            "tool_calls": _tool_calls(reply),
        }
        return reply

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": FORMAT, "recorded_with": self.label, "turns": list(self.turns.values())}
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _text(content: Any) -> str:
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def load_cassette(path: str | Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CassetteFormatError(f"cannot read analyst cassette {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise CassetteFormatError(f"{path} is not an {FORMAT} file")
    turns = data.get("turns")
    if not isinstance(turns, list) or not all(
        isinstance(t, dict) and {"key", "content", "tool_calls"} <= set(t) for t in turns
    ):
        raise CassetteFormatError(f"{path} has malformed turns")
    return data


class ReplayChatModel:
    """Returns recorded model turns; the tools around it still run against the real database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = load_cassette(self.path)
        self.recorded_with = str(data.get("recorded_with", "unknown"))
        self._turns = {t["key"]: t for t in data["turns"]}
        self.replayed = 0
        self.missed = 0

    def bind_tools(self, tools: Any) -> "ReplayChatModel":
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        turn = self._turns.get(turn_key(messages))
        if turn is None:
            self.missed += 1
            raise ReplayMiss(
                "no recorded model turn for this conversation (record one with "
                "--provider gemini --record-cassette FILE, or use --provider planner)"
            )
        self.replayed += 1
        return AIMessage(content=turn["content"], tool_calls=turn["tool_calls"])
