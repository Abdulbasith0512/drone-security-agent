"""VisionProvider interface, provider output contract, scripted provider and cassette replay.

The real VLM adapter (M5B) will implement `FrameDescriber` (see contract.py); nothing else changes.
"""

from drone_agent.vision.base import FrameInput, VisionProvider
from drone_agent.vision.cassette import (
    Cassette,
    CassetteDescriber,
    CassetteError,
    CassetteFormatError,
    CassetteMismatch,
    CassetteMiss,
    CassetteMode,
)
from drone_agent.vision.contract import (
    PROVIDER_INSTRUCTIONS,
    DescriberProvider,
    FrameDescriber,
    ProviderObject,
    ProviderOutput,
    ProviderOutputError,
    observation_from_output,
    parse_provider_output,
    provider_output_schema,
)
from drone_agent.vision.scripted import ScriptedProvider, UnknownFrameError

__all__ = [
    "PROVIDER_INSTRUCTIONS",
    "Cassette",
    "CassetteDescriber",
    "CassetteError",
    "CassetteFormatError",
    "CassetteMismatch",
    "CassetteMiss",
    "CassetteMode",
    "DescriberProvider",
    "FrameDescriber",
    "FrameInput",
    "ProviderObject",
    "ProviderOutput",
    "ProviderOutputError",
    "ScriptedProvider",
    "UnknownFrameError",
    "VisionProvider",
    "observation_from_output",
    "parse_provider_output",
    "provider_output_schema",
]
