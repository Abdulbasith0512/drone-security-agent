# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment: a docked-drone security analyst that turns video and
telemetry into events, deterministic alerts, searchable history and grounded Q&A.

> **Status: M7 (evaluation, ablation and grounding-faithfulness harness).** Scripted scenarios run end to end through alerts; real video runs through sampling, a frame-change gate and cassette replay. **Gemini is available as an optional real vision model (M5B)**; Evidence-grounded Q&A and run summaries (offline template analyst, optional Gemini via LangChain) are in M6.

Design in one line: `FRAME -> OBSERVATION -> EVENT -> MEMORY -> REASONING`.

## Contents
- [docs/feature_spec.md](docs/feature_spec.md): value, requirements, assumptions, non-goals
- [docs/architecture.md](docs/architecture.md): pipeline and design
- [docs/evaluation.md](docs/evaluation.md): experiments, ablations, metrics, threats to validity
- [docs/ai_usage.md](docs/ai_usage.md): how AI tools were used, and what I changed or rejected

## Setup (Python 3.11+)
```
python -m venv .venv
.venv\Scripts\activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"       # add ".[agent]" for the LangChain Q&A layer (M6)
pytest
```
API keys are read from environment variables only, or from a local gitignored `.env` (see `.env.example`). Never commit `.env`.

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

**Real vision model (M5B, optional; needs `GEMINI_API_KEY`, and optionally `GEMINI_MODEL`, in `.env` or the environment):**
```
python -m drone_agent describe-frame VIDEO --time 4 --provider gemini            # ONE frame, ONE call, stores nothing
python -m drone_agent ingest-video VIDEO --db output/g.db --provider gemini --cassette-mode record --cassette output/g.cassette.json
python -m drone_agent ingest-video VIDEO --db output/g2.db --provider cassette --cassette output/g.cassette.json   # replay: zero API calls
```
Gemini only sees frames the gate accepts; skipped samples are carried forward (still not a new inference). The model reports no confidence, so objects carry an
UNCALIBRATED placeholder of 0.5. `--model` overrides `GEMINI_MODEL`; there is no fallback model. Telemetry and zones remain simulated. Cassettes and videos are never committed.
See docs/architecture.md ("Gemini provider").

**Analyst Q&A and summary (M6; works offline with no key):**
```
python -m drone_agent ask       --db output/security.db "Was the blue pickup seen more than once?"
python -m drone_agent ask       --db output/security.db "Why was the repeat-vehicle alert raised?" --follow-up "Where was it seen?"
python -m drone_agent summarize --db output/security.db --follow-up "What objects were observed?"
python -m drone_agent ask       --db output/security.db --provider gemini --model MODEL "What happened at the main gate?"   # needs .[agent] and a key
```
Answers cite stored evidence as `[event:EVT-V-001]`, `[alert:...]`, `[entity:...]`, `[obs:...]`, `[frame:N]`. The default (`--offline`) is a deterministic template analyst; `--provider gemini`
uses a LangChain tool-calling loop over read-only tools, and its answer is shown only if every citation exists in the database and was retrieved (one correction retry, then the template fallback).
The analyst model is `ANALYST_MODEL` (separate from the vision `GEMINI_MODEL`). The LLM only explains: alerts, severities, events and entities come from the deterministic rules. See docs/architecture.md ("Analyst layer").

`ingest` twice into one database is rejected and changes nothing (use a new `--db`); `build-events` and `analyze` replace their derived data (idempotent).

## Evaluation (M7)
```
python -m drone_agent evaluate --output output/eval                       # deterministic, offline, no key
python -m drone_agent evaluate --output output/eval --scale               # + timing benchmark (100/1k/10k observations)
```
Runs the frame-gate ablation, temporal compression, flicker-tolerance and entity-memory ablations (with negative controls), a labelled rule matrix, a citation-gate adversarial suite, a
citation-valid / semantically-unsupported probe with a prototype structured claim checker, and an optional scale benchmark. Writes `results.json`, `results.csv`, `ablation_table.csv` and `summary.md`
under `output/eval/` (gitignored). Every number comes from executable code. Results, setup, metrics and threats to validity are in [docs/evaluation.md](docs/evaluation.md); the rule metrics apply only to the controlled synthetic fixtures.

## Design decisions
_TBD (M8). See docs/architecture.md._

## Testing
_TBD (M7-M8)._

## AI tools and impact
_See docs/ai_usage.md; summary TBD (M8)._

## Repository safety
The confidential assignment PDF, `.env`, databases, caches and videos are gitignored.
