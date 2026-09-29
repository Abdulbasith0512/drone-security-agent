"""Analyst model configuration, separate from the vision model's (GEMINI_MODEL).

The analyst reuses the same credential variable (GEMINI_API_KEY) but its model comes from
ANALYST_MODEL (or --model), so the vision and analyst models can differ. No LangChain here.
"""

import os
from dataclasses import dataclass

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "ANALYST_MODEL"
DEFAULT_ANALYST_MODEL = "gemini-3.6-flash"


class AnalystConfigError(Exception):
    """Missing or unusable analyst configuration. Never contains the key."""


@dataclass(frozen=True)
class AnalystConfig:
    model: str
    api_key: str
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise AnalystConfigError(f"{API_KEY_ENV} is not set")
        if not self.model.strip():
            raise AnalystConfigError("analyst model name must not be empty")

    def __repr__(self) -> str:  # never show the key
        return f"AnalystConfig(model={self.model!r})"

    @classmethod
    def from_env(cls, model: str | None = None, load_env_file: bool = True) -> "AnalystConfig":
        """Model: `model` > ANALYST_MODEL > default. The key: GEMINI_API_KEY (env or root .env)."""
        if load_env_file:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True), override=False)
        chosen = model or os.environ.get(MODEL_ENV) or DEFAULT_ANALYST_MODEL
        return cls(model=chosen, api_key=os.environ.get(API_KEY_ENV, ""))
