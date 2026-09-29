# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment: a docked-drone security analyst that turns video and
telemetry into events, deterministic alerts, searchable history and grounded Q&A.

> **Status: M0 scaffold. Nothing runs yet.** Sections below are filled in as milestones land.

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
_TBD (M3 onward)._

## Design decisions
_TBD (M8). See docs/architecture.md._

## Testing
_TBD (M7-M8)._

## AI tools and impact
_See docs/ai_usage.md; summary TBD (M8)._

## Repository safety
The confidential assignment PDF, `.env`, databases, caches and videos are gitignored.
