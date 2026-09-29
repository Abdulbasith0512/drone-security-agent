"""Construct the analyst LLM lazily, so the CLI and offline mode never import LangChain."""

from drone_agent.agent.config import AnalystConfig, AnalystConfigError
from drone_agent.agent.session import AnalystLLM


def build_gemini_analyst(config: AnalystConfig) -> AnalystLLM:
    try:
        from drone_agent.agent.llm import LangChainAnalyst
    except ImportError:
        raise AnalystConfigError(
            'the Gemini analyst needs the agent extras: pip install -e ".[agent]" (or --offline)'
        ) from None
    return LangChainAnalyst.from_config(config)
