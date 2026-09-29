# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment: a docked-drone security analyst that turns video and
telemetry into events, deterministic alerts, searchable history and grounded Q&A.

> **Status: M3 (events and entity memory).** `simulate`, `ingest`, `query`, `build-events`, `events` and `entities` run; alerts, real video and Q&A arrive in later milestones.

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
No API key and no network are needed. Databases are written under `output/` (gitignored).
```
python -m drone_agent simulate scenarios/security_events.yaml          # scripted frames + telemetry sync
python -m drone_agent ingest   scenarios/security_events.yaml --db output/security.db
python -m drone_agent query --db output/security.db --zone main_gate
python -m drone_agent query --db output/security.db --search "blue pickup"
python -m drone_agent query --db output/security.db --zone main_gate --explain   # SQLite query plan

python -m drone_agent build-events --db output/security.db                      # observations -> events -> entities
python -m drone_agent events   --db output/security.db                          # list events
python -m drone_agent events   --db output/security.db --evidence EVT-P-001     # event -> observations -> frames
python -m drone_agent entities --db output/security.db --show ENT-V-001         # similar-vehicle entity and its events
```
On `security_events` the 8-minute gate loiterer becomes ONE event (`EVT-P-001`, frames 1-4 and 6-9, frame 5 is a
tolerated missed detection), and three separate blue-pickup sightings become three events grouped into one
*similar-vehicle* entity (`ENT-V-001`). People are never linked across events. This is temporal association of
detections, not object tracking or re-identification (see docs/architecture.md).
`query` filters are ANDed and can be combined with `--start/--end` (ISO-8601 with a UTC offset).
`ingest` twice into the same database is rejected and changes nothing (use a new `--db`); `build-events` twice
replaces the derived events (idempotent). Thresholds live in `config/events.yaml` (`--config` to override).
Alerts and Q&A arrive in later milestones.

## Design decisions
_TBD (M8). See docs/architecture.md._

## Testing
_TBD (M7-M8)._

## AI tools and impact
_See docs/ai_usage.md; summary TBD (M8)._

## Repository safety
The confidential assignment PDF, `.env`, databases, caches and videos are gitignored.
