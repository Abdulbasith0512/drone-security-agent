# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment: a docked-drone security analyst that turns video and
telemetry into events, deterministic alerts, searchable history and grounded Q&A.

> **Status: M5A (real-video pipeline, provider-neutral).** Scripted scenarios run end to end through alerts; real video runs through sampling, a frame-change gate and cassette replay. **No real vision model is integrated yet** (M5B); Q&A and summaries come later.

Design in one line: `FRAME -> OBSERVATION -> EVENT -> MEMORY -> REASONING`.

## Contents
- [docs/feature_spec.md](docs/feature_spec.md): value, requirements, assumptions, non-goals
- [docs/architecture.md](docs/architecture.md): pipeline and design
- [docs/ai_usage.md](docs/ai_usage.md): how AI tools were used, and what I changed or rejected

## Setup (Python 3.11+)
```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"       # add ".[agent]" for the LangChain Q&A layer (M6)
pytest
```
API keys, if any are ever used, are read from environment variables only (see `.env.example`). Never commit `.env`.

## Running the demo
No API key and no network are needed. Databases and demo files are written under `output/` (gitignored).

**Scripted scenarios (M1-M4):**
```
python -m drone_agent ingest   scenarios/security_events.yaml --db output/security.db
python -m drone_agent build-events --db output/security.db                      # observations -> events -> entities
python -m drone_agent analyze  --db output/security.db                          # deterministic rules -> alerts
python -m drone_agent alerts   --db output/security.db --show ALR-LOITER-EVT-P-001   # reason, facts, evidence frames
python -m drone_agent query    --db output/security.db --search "blue pickup"   # also --zone/--object/--start/--end/--explain
python -m drone_agent events   --db output/security.db --evidence EVT-P-001     # event -> observations -> frames
python -m drone_agent entities --db output/security.db --show ENT-V-001
```
On `security_events` this gives 7 events and 6 alerts (a person loitering at the gate for 480 s, three after-hours person alerts, one restricted-zone entry, and a
repeat-vehicle alert worded "a *similar* blue pickup was observed in 3 separate appearances"). `normal_day` gives no alerts. The rules are deterministic
(`config/security_rules.yaml`); no LLM decides, suppresses or changes an alert.

**Real video (M5A: no real vision model yet):**
```
python -m drone_agent make-demo-video --out-dir output/demo                      # tiny SYNTHETIC video + a canned demo cassette
python -m drone_agent ingest-video output/demo/demo.avi --db output/video.db \
       --provider cassette --cassette output/demo/demo.cassette.json --sample-every 2 --verbose
python -m drone_agent query --db output/video.db --search "blue pickup"          # ordinary observations
python -m drone_agent build-events --db output/video.db && python -m drone_agent analyze --db output/video.db
```
`ingest-video` decodes the video with OpenCV, samples a frame every N seconds, skips near-identical frames with a pixel-difference gate, and asks a
*provider* to describe the remaining ones. Today the only provider is `cassette` (replay of recorded responses), so the descriptions of the demo video are canned,
not produced by a model. The telemetry is SIMULATED: a start time and zone you supply (`--start`, `--zone`, or `--timeline FILE`), not read from the video.
Frames the gate skips are, by default, *carried forward*: the observation stored for them repeats the last accepted one and is labelled
`source = "<provider>+carried"`. That is temporal propagation of an earlier model answer, NOT a new model inference (`--skipped drop`
stores nothing for skipped frames instead). After ingestion the store, events, entities and rules run unchanged. See docs/architecture.md ("Real video ingestion") for the sampling, gate and cassette design.

`ingest` twice into one database is rejected and changes nothing (use a new `--db`); `build-events` and `analyze` replace their derived data (idempotent).

## Design decisions
_TBD (M8). See docs/architecture.md._

## Testing
_TBD (M7-M8)._

## AI tools and impact
_See docs/ai_usage.md; summary TBD (M8)._

## Repository safety
The confidential assignment PDF, `.env`, databases, caches and videos are gitignored.
