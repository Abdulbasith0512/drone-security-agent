# Backend: where everything is

The backend is the Python package **[`drone_agent/`](../drone_agent/)** at the repository root. It was left in place on purpose: every import, test, command and document already uses that path, so moving it would risk breaking a verified system. This folder is the map.

## Run it
```
pip install -e ".[dev]"
python -m drone_agent -h          # the command-line tool (ingest, build-events, analyze, alerts, ask, run, scale, evaluate, ...)
python -m drone_agent.web --demo  # the dashboard server + the UI in ../frontend/ (opens on the bundled real clip)
pytest -q                         # the full offline test suite (1301 tests; the 39 browser tests need Chrome or Edge)
```

## The pipeline, stage by stage
`FRAME -> OBSERVATION -> EVENT -> ENTITY -> rules -> ALERT -> analyst` ("rules decide; the language model only explains").

| Stage | Code |
|---|---|
| Video, sampling, frame-change gate | [`drone_agent/video/`](../drone_agent/video/) (`source.py`, `sampling.py`, `gate.py`, `ingest.py`, `synthetic.py`) |
| Vision model boundary, recorded cassettes, Gemini adapter | [`drone_agent/vision/`](../drone_agent/vision/) (`base.py`, `contract.py`, `cassette.py`, `gemini.py`, `scripted.py`) |
| Simulated drone telemetry | [`drone_agent/telemetry/`](../drone_agent/telemetry/) (`simulator.py`, `sync.py`, `geo.py`) |
| Observations -> events, vehicle entities | [`drone_agent/pipeline/`](../drone_agent/pipeline/) (`aggregator.py`, `linker.py`, `recurrence.py`, `build.py`) |
| Deterministic security rules, alert lifecycle | [`drone_agent/rules/`](../drone_agent/rules/) (`loitering.py`, `after_hours.py`, `repeat_vehicle.py`, `restricted_zone.py`, `safety.py`, `lifecycle.py`, `engine.py`, `analyze.py`) |
| Streaming patrol engine, the shared downstream consumer (`consumer.py`: used by both the patrol engine and the progressive video driver `video.py`), live terminal view | [`drone_agent/runtime/`](../drone_agent/runtime/), [`drone_agent/agents/`](../drone_agent/agents/) (perception, sampling, context, reasoning) |
| SQLite + FTS5 store, schema and migrations | [`drone_agent/store/`](../drone_agent/store/) (`store.py`, `derived.py`, `schema.py`, `runs.py`) |
| AI analyst: tools, evidence, citation checks, offline and LLM modes | [`drone_agent/agent/`](../drone_agent/agent/) (`tools.py`, `evidence.py`, `citations.py`, `factcheck.py`, `session.py`, `offline.py`, `planner.py`, `replay.py`, `llm.py`) |
| Data models | [`drone_agent/models/`](../drone_agent/models/) |
| Evaluation harness, ground truth, scale benchmark, real-clip evidence | [`drone_agent/eval/`](../drone_agent/eval/) (`runner.py`, `ground_truth.py`, `scale_bench.py`, `real_clip.py`) |
| Command-line entry points | [`drone_agent/__main__.py`](../drone_agent/__main__.py), [`drone_agent/cli_run.py`](../drone_agent/cli_run.py) |
| **Dashboard server (the web backend)** | [`drone_agent/web/`](../drone_agent/web/) (`server.py`, `api.py`, `guard.py`, `registry.py`, `bundle.py`, `frames.py`, `__main__.py`) |

## Configuration, data, tests, docs
| What | Where |
|---|---|
| Rule thresholds and event/linking settings | [`config/security_rules.yaml`](../config/security_rules.yaml), [`config/events.yaml`](../config/events.yaml) |
| Scripted scenarios with ground-truth labels | [`scenarios/`](../scenarios/) |
| The bundled real clip, its recorded cassette and a SHA-256 manifest (the dashboard's default run; licensed for redistribution and hosting in this demo) | [`demo_assets/real_clip/`](../demo_assets/real_clip/) |
| Tests (all offline, no key needed; the browser tests need an installed Chrome or Edge and are skipped without one) | [`tests/`](../tests/) (the dashboard's are the `test_web_*.py` files) |
| Architecture, testing, evaluation, dashboard docs | [`docs/`](../docs/) |
| Scale benchmark results | [`bench/`](../bench/) |

## How the dashboard backend serves the frontend
`drone_agent/web/server.py` is a standard-library HTTP server bound to `127.0.0.1` by default (an opt-in public mode, `--host 0.0.0.0 --public-host HOSTNAME`, serves exactly one hostname; there is no authentication). It serves the four files in [`../frontend/`](../frontend/) from a fixed whitelist and exposes a read-only JSON API; `api.py` reads the run's SQLite database (opened read-only) and calls the existing `Store`, `AnalystTools`, `AnalystSession` and `VideoSource` code. `bundle.py` builds the `--demo` runs with the existing CLI commands: the bundled real clip replayed from its recorded cassette first, then the scripted and the synthetic run. No model, network or API key is used.
