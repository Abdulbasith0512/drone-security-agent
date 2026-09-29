"""Small encoders shared by the store modules."""

import json
from datetime import UTC, datetime

_UTC_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"  # fixed width, so text order == time order


def utc_key(ts: datetime) -> str:
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.astimezone(UTC).strftime(_UTC_FORMAT)


def canonical_json(value: object) -> str:
    """Deterministic JSON (sorted keys, no whitespace, UTF-8 kept)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
