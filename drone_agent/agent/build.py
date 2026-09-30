"""Construct the analyst LLM lazily, so the CLI and offline mode never import LangChain."""

from drone_agent.agent.config import AnalystConfig, AnalystConfigError
from drone_agent.agent.session import AnalystLLM

_NEEDS_EXTRAS = (
    'the LangChain analyst needs the agent extras: pip install -e ".[agent]" (or --offline)'
)


def build_gemini_analyst(config: AnalystConfig, record_to: str | None = None) -> AnalystLLM:
    """The live Gemini analyst; `record_to` also records every model turn to a cassette file."""
    try:
        from drone_agent.agent.llm import LangChainAnalyst, build_chat_model
        from drone_agent.agent.replay import RecordingChatModel
    except ImportError:
        raise AnalystConfigError(_NEEDS_EXTRAS) from None
    if record_to is None:
        return LangChainAnalyst.from_config(config)
    recorder = RecordingChatModel(
        build_chat_model(config), record_to, label=f"gemini:{config.model}"
    )
    analyst = LangChainAnalyst(recorder, name=f"gemini:{config.model} (recording)")
    analyst._secret = config.api_key
    analyst.recorder = recorder  # the CLI flushes it after the answers
    return analyst


def build_planner_analyst() -> AnalystLLM:
    """The real LangChain loop and tools with a scripted model: no LLM, no key, no network."""
    try:
        from drone_agent.agent.llm import LangChainAnalyst
        from drone_agent.agent.planner import NAME, PlannerChatModel
    except ImportError:
        raise AnalystConfigError(_NEEDS_EXTRAS) from None
    return LangChainAnalyst(PlannerChatModel(), name=NAME)


def build_replay_analyst(path: str) -> AnalystLLM:
    """The real LangChain loop and tools, with model turns replayed from a recorded cassette."""
    try:
        from drone_agent.agent.llm import LangChainAnalyst
        from drone_agent.agent.replay import CassetteFormatError, ReplayChatModel
    except ImportError:
        raise AnalystConfigError(_NEEDS_EXTRAS) from None
    try:
        model = ReplayChatModel(path)
    except CassetteFormatError as exc:
        raise AnalystConfigError(str(exc)) from None
    return LangChainAnalyst(model, name=f"replay:{model.recorded_with}")
