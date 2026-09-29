"""Gemini adapter: a real FrameDescriber built on Google's official `google-genai` SDK.

Pipeline for one frame:  BGR ndarray -> (optional downscale) -> JPEG bytes -> Gemini with a JSON
schema -> response text -> `parse_provider_output` (our own strict Pydantic validation) ->
ProviderOutput.  The API's schema enforcement is a convenience, never the only validation.

What the model is asked: describe visible objects. It is NOT asked about suspicion, intent,
identity, alert severity, or whether two things are "the same". Those stay in the rules.

Confidence: Gemini gives no calibrated probability, so the schema does not request one. Objects
come back with `confidence = None`; the pipeline then applies an explicit, documented placeholder
(`PLACEHOLDER_CONFIDENCE`, an uncalibrated constant, NOT a model probability).

Secrets: the key comes only from the environment (GEMINI_API_KEY). It is never logged, stored in
cassettes or the database, or included in exception text (messages are scrubbed).
"""

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from drone_agent.vision.base import FrameInput
from drone_agent.vision.contract import (
    PROVIDER_INSTRUCTIONS,
    ProviderOutput,
    ProviderOutputError,
    parse_provider_output,
    provider_output_schema,
)

API_KEY_ENV = "GEMINI_API_KEY"
MODEL_ENV = "GEMINI_MODEL"
# Used only when neither --model nor GEMINI_MODEL is set (listed by the API as supporting
# generateContent when checked). Never a fallback for a model that fails.
DEFAULT_MODEL = "gemini-3.6-flash"
# Uncalibrated placeholder stored as `confidence` because the model reports none: NOT a probability.
PLACEHOLDER_CONFIDENCE = 0.5

_PROMPT_SUFFIX = """\
Exception to the instructions above: do NOT include a confidence field; it is not requested.
Reply only with JSON containing "description" and "objects"."""


class GeminiError(Exception):
    """Base class; messages never contain the API key."""


class GeminiConfigError(GeminiError):
    """Missing or unusable configuration (no API key, bad settings). Never retried."""


class GeminiAuthError(GeminiError):
    """The API rejected the key or permissions. Never retried."""


class GeminiModelError(GeminiError):
    """Model not found or not supported for this request. Never retried, no fallback model."""


class GeminiRequestError(GeminiError):
    """The API rejected the request itself (bad argument). Never retried."""


class GeminiRateLimitError(GeminiError):
    """Rate limit / quota still exceeded after the bounded retries."""


class GeminiTransientError(GeminiError):
    """Timeout, network or server error still failing after the bounded retries."""


class GeminiOutputError(GeminiError):
    """Malformed or empty model output still invalid after the bounded retry."""


@dataclass(frozen=True)
class GeminiConfig:
    model: str
    api_key: str
    max_dimension: int = 1024  # longest image side sent; larger frames are downscaled
    jpeg_quality: int = 85
    timeout_s: float = 60.0
    retry_delays: tuple[float, ...] = (2.0, 5.0)  # transient/rate-limit: len() retries at most
    malformed_retries: int = 1  # extra attempts after malformed model output

    def __post_init__(self) -> None:
        if not self.api_key:
            raise GeminiConfigError(f"{API_KEY_ENV} is not set")
        if not self.model.strip():
            raise GeminiConfigError("model name must not be empty")
        if self.max_dimension < 64:
            raise GeminiConfigError("max_dimension must be at least 64")
        if not 1 <= self.jpeg_quality <= 100:
            raise GeminiConfigError("jpeg_quality must be between 1 and 100")
        if self.timeout_s <= 0:
            raise GeminiConfigError("timeout_s must be positive")
        if self.malformed_retries < 0 or any(d < 0 for d in self.retry_delays):
            raise GeminiConfigError("retry settings must not be negative")

    def __repr__(self) -> str:  # never show the key
        return f"GeminiConfig(model={self.model!r}, max_dimension={self.max_dimension})"

    @classmethod
    def from_env(
        cls, model: str | None = None, load_env_file: bool = True, **overrides: Any
    ) -> "GeminiConfig":
        """Model: `model` argument > GEMINI_MODEL > DEFAULT_MODEL. The key: GEMINI_API_KEY.

        With `load_env_file`, a root `.env` is read (python-dotenv) without overriding variables
        that already exist in the process environment.
        """
        if load_env_file:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True), override=False)
        chosen = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        return cls(model=chosen, api_key=os.environ.get(API_KEY_ENV, ""), **overrides)


def prepare_image(image: Any, max_dimension: int, jpeg_quality: int) -> tuple[bytes, int, int]:
    """BGR uint8 frame -> (JPEG bytes, width, height), keeping aspect ratio.

    OpenCV frames are BGR and `cv2.imencode` expects BGR: it writes a normal JPEG whose colors are
    correct for any viewer. Converting to RGB first would swap red and blue in the file, so the
    channel order is handled by the encoder. Only downscales (never upscales); in memory only.
    """
    import cv2

    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
        raise ValueError("frame image must be a BGR array with 3 channels")
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest > max_dimension:
        scale = max_dimension / longest
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise ValueError("could not JPEG-encode the frame")
    return encoded.tobytes(), width, height


def response_schema() -> dict:
    """ProviderOutput's JSON Schema minus the confidence field (Gemini gives none)."""
    schema = provider_output_schema()
    schema["$defs"]["ProviderObject"]["properties"].pop("confidence", None)
    return schema


class GeminiDescriber:
    """FrameDescriber backed by the Gemini API. One `describe` = one frame = one API request
    (plus bounded retries)."""

    def __init__(
        self,
        config: GeminiConfig,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.name = f"gemini:{config.model}"
        self.sleep = sleep
        self.client = client or genai.Client(
            api_key=config.api_key,
            http_options=genai_types.HttpOptions(timeout=int(config.timeout_s * 1000)),
        )
        self.api_calls = 0  # every request sent, including retries
        self.retries = 0
        self.total_latency_s = 0.0
        self.last_latency_s: float | None = None
        self.last_image_size: tuple[int, int] | None = None
        self.last_jpeg_bytes: int | None = None

    def _scrub(self, text: str) -> str:
        return text.replace(self.config.api_key, "[redacted]")

    def _generate_config(self) -> Any:
        return genai_types.GenerateContentConfig(
            system_instruction=f"{PROVIDER_INSTRUCTIONS}\n{_PROMPT_SUFFIX}",
            response_mime_type="application/json",
            response_json_schema=response_schema(),
            temperature=0.0,
            automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
        )

    def describe(self, frame: FrameInput) -> ProviderOutput:
        jpeg, width, height = prepare_image(
            frame.image, self.config.max_dimension, self.config.jpeg_quality
        )
        self.last_image_size, self.last_jpeg_bytes = (width, height), len(jpeg)
        contents = [
            genai_types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
            "Describe this image.",
        ]
        malformed_left = self.config.malformed_retries
        while True:
            text = self._request_with_retries(contents)
            try:
                if not text or not text.strip():
                    raise ProviderOutputError("model returned no text (blocked or empty response)")
                return parse_provider_output(text)
            except ProviderOutputError as exc:
                if malformed_left <= 0:
                    raise GeminiOutputError(self._scrub(str(exc))) from None
                malformed_left -= 1
                self.retries += 1

    def _request_with_retries(self, contents: list[Any]) -> str | None:
        delays = iter(self.config.retry_delays)
        while True:
            started = time.perf_counter()
            self.api_calls += 1
            try:
                response = self.client.models.generate_content(
                    model=self.config.model, contents=contents, config=self._generate_config()
                )
            except Exception as exc:
                error = self._classify(exc)
                delay = next(delays, None)
                if not isinstance(error, GeminiRateLimitError | GeminiTransientError) or (
                    delay is None
                ):
                    raise error from None
                self.retries += 1
                self.sleep(delay)
                continue
            finally:
                self.last_latency_s = time.perf_counter() - started
                self.total_latency_s += self.last_latency_s
            return getattr(response, "text", None)

    def _classify(self, exc: Exception) -> GeminiError:
        detail = self._scrub(str(exc))[:300]
        kind = type(exc).__name__
        if isinstance(exc, genai_errors.APIError):
            code = exc.code
            if code in (401, 403) or (code == 400 and "api key" in detail.lower()):
                return GeminiAuthError(f"Gemini rejected the API key or permissions ({code})")
            if code == 404:
                return GeminiModelError(
                    f"model '{self.config.model}' was not found or is not supported ({code}); "
                    f"no fallback model is used. {detail}"
                )
            if code == 429:
                return GeminiRateLimitError(f"Gemini rate limit or quota exceeded ({code})")
            if code in (408, 500, 502, 503, 504):
                return GeminiTransientError(f"Gemini server/timeout error ({code})")
            return GeminiRequestError(f"Gemini rejected the request ({code}): {detail}")
        if isinstance(exc, httpx.TransportError | TimeoutError | ConnectionError):
            return GeminiTransientError(f"network error talking to Gemini ({kind})")
        return GeminiError(f"unexpected Gemini SDK error ({kind}): {detail}")
