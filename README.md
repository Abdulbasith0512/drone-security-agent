# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment: a docked-drone security analyst that turns video and
telemetry into events, deterministic alerts, searchable history and grounded Q&A.

> **Status: M4 (deterministic security rules and alerts).** `simulate`, `ingest`, `query`, `build-events`, `events`, `entities`, `analyze` and `alerts` run; real video/VLM, Q&A and summaries arrive in later milestones.

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
python -m drone_agent query --db output/security.db --search "blue pickup"   # also --zone/--object/--start/--end/--explain

python -m drone_agent build-events --db output/security.db                      # observations -> events -> entities
python -m drone_agent events   --db output/security.db --evidence EVT-P-001     # event -> observations -> frames
python -m drone_agent entities --db output/security.db --show ENT-V-001         # similar-vehicle entity and its events

python -m drone_agent analyze  --db output/security.db                          # deterministic rules -> alerts
python -m drone_agent alerts   --db output/security.db                          # list alerts
python -m drone_agent alerts   --db output/security.db --show ALR-LOITER-EVT-P-001   # reason, facts, evidence frames
```
On `security_events` the pipeline gives 7 events and 6 alerts: a person loitering at the gate for 480 s (frames 1-4 and 6-9;
frame 5 is a tolerated missed detection), three after-hours person alerts, one restricted-zone entry, and one repeat-vehicle alert
worded "a *similar* blue pickup was observed in 3 separate appearances" because entity matching is coarse (attributes only,
never identity). `normal_day` produces no alerts. Every alert lists its rule facts and traces to events and frames.

The rules are deterministic and configured in `config/security_rules.yaml` (thresholds, closed hours 22:00-06:00, restricted zones,
fixed severities; override with `analyze --rules`). Event thresholds are in `config/events.yaml`. VLM-derived observations feed
deterministic temporal rules; no LLM decides, suppresses or changes an alert. Re-running `analyze` or `build-events` replaces derived
data (idempotent); `ingest` twice into one database is rejected. Q&A and summaries arrive in later milestones.

## Design decisions
_TBD (M8). See docs/architecture.md._

## Testing
_TBD (M7-M8)._

## AI tools and impact
_See docs/ai_usage.md; summary TBD (M8)._

## Repository safety
The confidential assignment PDF, `.env`, databases, caches and videos are gitignored.
