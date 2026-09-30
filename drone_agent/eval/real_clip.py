"""Evidence for ONE real clip sent through a real vision model (integration evidence, not accuracy).

Two facts make a real-model run inspectable later:

* the clip is identified by its bytes (size, SHA-256, decoded properties), and nothing else about
  its origin is claimed: where it came from and under what licence are recorded only if somebody
  actually knows, and otherwise the manifest says "not recorded";
* the model's answers are kept in a cassette, so the whole downstream pipeline can be replayed
  WITHOUT the model, the network or any credential, and must give the same structured result.

`replay_summary` is that replay, reduced to plain data: which frames were sampled, where they fall
in video time, what the gate decided, what the provider returned (as stored), and the events and
entities built from it. It is deterministic, so a recorded summary can be compared with a fresh
replay. There is no ground truth for the clip: nothing here measures whether a description is
right, only that the pipeline ran, reproduces, and what it produced.

This module never reads, prints or stores an API key. `scan_for_secrets` checks artifacts for one.
"""

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from drone_agent.models import Zone
from drone_agent.runtime.video import run_video_live
from drone_agent.store import Store
from drone_agent.video import SamplingConfig, VideoIngestConfig, VideoSource, single_zone_timeline
from drone_agent.vision import Cassette, CassetteDescriber, DescriberProvider
from drone_agent.vision.gemini import PLACEHOLDER_CONFIDENCE

MANIFEST_FORMAT = "drone-agent-real-clip-evidence/1"
# Patterns of credentials that must never appear in an artifact (generic, not a specific key).
_KEY_PATTERNS = (re.compile(r"AIza[0-9A-Za-z_\-]{30,}"), re.compile(r"\bsk-[A-Za-z0-9]{20,}"))


def clip_provenance(
    path: str | Path, *, source: str | None = None, licence: str | None = None
) -> dict:
    """What can be verified about the file. `source` and `licence` are recorded only if given."""
    path = Path(path)
    meta = VideoSource(path).metadata
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "decoded_sha256_matches_file": meta.sha256 == digest,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_seconds": round(meta.duration_seconds, 3),
        "source": source or "not recorded",
        "licence": licence or "not recorded",
    }


def replay_summary(
    clip: str | Path,
    cassette: str | Path,
    *,
    start: datetime,
    zone: Zone = Zone.MAIN_GATE,
    sample_every: float = 3.0,
) -> dict:
    """Replay a recorded cassette over the clip through the live pipeline, into plain data.

    No provider is called: every answer comes from the cassette. The store is in memory."""
    source = VideoSource(clip)
    recorded = Cassette.load(cassette)
    describer = CassetteDescriber(recorded)
    describer.check_video(source.metadata.sha256)
    # Gemini reports no confidence; the CLI stores the documented uncalibrated placeholder for a
    # cassette recorded by Gemini, and so does this replay (it is not a probability).
    default_confidence = (
        PLACEHOLDER_CONFIDENCE if recorded.data.provider.startswith("gemini") else None
    )
    timeline = single_zone_timeline(start, zone, source.metadata.duration_seconds)
    config = replace(VideoIngestConfig(), sampling=SamplingConfig(every_seconds=sample_every))
    with Store.open(":memory:") as store:
        live = run_video_live(
            source, DescriberProvider(describer, default_confidence), timeline, store, config=config
        )
        observations = store.query_observations()
        events, entities, alerts = store.query_events(), store.list_entities(), store.query_alerts()
    by_frame = {o.frame_id: o for o in observations}
    frames = []
    for step, cand in zip(live.steps, live.result.candidates, strict=True):
        obs = by_frame.get(cand.frame_index)
        frames.append(
            {
                "frame_index": cand.frame_index,
                "offset_s": cand.offset_seconds,
                "timestamp": obs.timestamp.isoformat() if obs else None,
                "gate_score": cand.gate_score,
                "gate_reason": cand.gate_reason,
                "action": cand.action,  # analyzed = sent to the provider; carried = propagated
                "error": cand.error,
                "observation": None
                if obs is None
                else {
                    "source": obs.source,
                    "provenance": obs.provenance.value if obs.provenance else None,
                    "description": obs.description,
                    "objects": [
                        {
                            "type": d.object_type.value,
                            "activity": d.activity,
                            "attributes": dict(sorted(d.attributes.items())),
                            "confidence": d.confidence,
                        }
                        for d in obs.detected_objects
                    ],
                },
                "after_step": {"events": step.events, "alerts": step.alerts},
            }
        )
    return {
        "sample_every_s": sample_every,
        "start": start.isoformat(),
        "zone": zone.value,
        "cassette_replays": describer.replayed,
        "cassette_content_mismatches": len(describer.content_mismatches),
        "frames": frames,
        "events": [
            {
                "id": e.event_id,
                "type": e.object_type.value,
                "frames": e.frame_ids,
                "duration_s": e.duration_seconds,
                "close_reason": e.close_reason.value,
                "entity": e.entity_id,
                "attributes": dict(sorted(e.attributes.items())),
            }
            for e in events
        ],
        "entities": [
            {
                "id": n.entity_id,
                "label": n.label,
                "appearances": n.appearance_count,
                "events": n.event_ids,
                "match_level": n.match_level.value,
            }
            for n in entities
        ],
        "alerts": [a.alert_id for a in alerts],
        "counts": {
            "sampled": len(live.steps),
            "provider_calls": live.result.provider_calls,
            "carried": live.result.carried,
            "failed": live.result.failed,
            "observations": len(observations),
            "objects": sum(len(o.detected_objects) for o in observations),
            "events": len(events),
            "entities": len(entities),
            "alerts": len(alerts),
        },
    }


_RECORD_ROWS = {
    "api_requests": re.compile(
        r"Gemini API requests\s*\|\s*(\d+) \(retries (\d+), total latency ([\d.]+)s\)"
    ),
    "provider_calls": re.compile(r"Provider calls\s*\|\s*(\d+), cassette replays (\d+)"),
    "provider": re.compile(r"Provider\s*\|\s*(\S+)"),
}


def parse_record_log(text: str, *, wall_seconds: float | None = None) -> dict:
    """The recording statistics the CLI printed (they are not stored in the cassette)."""
    out: dict = {"source_of_numbers": "printed summary of the recording run (not stored elsewhere)"}
    m = _RECORD_ROWS["api_requests"].search(text)
    if m:
        out |= {
            "api_requests": int(m.group(1)),
            "retries": int(m.group(2)),
            "total_api_latency_s": float(m.group(3)),
        }
    m = _RECORD_ROWS["provider_calls"].search(text)
    if m:
        out |= {"provider_calls": int(m.group(1)), "cassette_replays": int(m.group(2))}
    m = _RECORD_ROWS["provider"].search(text)
    if m:
        out["provider"] = m.group(1)
    out["failed_frames"] = len(re.findall(r"perception FAILED|PERCEPTION FAILURES", text))
    if wall_seconds is not None:
        out["wall_clock_seconds"] = wall_seconds
    return out


def scan_for_secrets(paths: list[str | Path], extra_secrets: list[str] | None = None) -> list[str]:
    """Names of files that contain anything shaped like an API key, or one of `extra_secrets`
    (for example the real configured key, which the caller supplies and this module never prints)."""
    leaked = []
    secrets = [s for s in (extra_secrets or []) if s]
    for path in map(Path, paths):
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        if any(p.search(text) for p in _KEY_PATTERNS) or any(s.encode() in data for s in secrets):
            leaked.append(path.name)
    return leaked


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
