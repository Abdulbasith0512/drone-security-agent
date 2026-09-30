# Testing

How the system was validated. Quantitative experiments (ablations, rule confusion matrices, scale timings) are in
[evaluation.md](evaluation.md); the one real-video run is in [real_video_experiment.md](real_video_experiment.md). This document covers the automated test suite
and how to verify a fresh clone.

## Running the tests
```
pip install -e ".[dev,agent]"     # the LangChain adapter tests need the `agent` extra
pytest -q                         # 963 tests are collected (pytest --collect-only)
ruff check .
ruff format --check .
```
The full suite runs in roughly 30 to 90 seconds on the development machine (the web tests bind localhost sockets, so a sandbox that blocks them fails those tests, not the application). No test needs a Gemini key or a network connection.

## Principles
- **Offline and deterministic.** Time comes from a simulated clock, telemetry from a seeded simulator, and scenario data from YAML. Real-video tests generate a small synthetic MJPG video
  in a temporary directory; no video, database or cassette is committed.
- **Mocked model boundaries.** The Gemini SDK is replaced by a fake client (`tests/test_gemini.py`); the analyst LLM is a scripted stand-in or a fake chat model that returns real LangChain
  messages (`tests/test_agent_core.py`, `tests/test_agent_llm.py`). Environment loading is patched so a developer's real `.env` never reaches a test.
- **Isolated state.** Every database test uses its own temporary SQLite file.
- **Golden regression.** `tests/golden_offline_answers.json` holds offline analyst answers recorded from the code before M8a; `tests/test_agent_truncation.py` requires the non-truncated ones
  to stay byte-identical.
- **Mutation checks during development.** At several milestones the code was deliberately broken in a handful of ways to confirm the tests notice (recorded in [ai_usage.md](ai_usage.md)).
  These are a development practice, not part of the suite.

## Test organisation
| File(s) | Layer | Tests | What it verifies |
|---|---|---|---|
| `test_models.py` | domain models | 24 | strict, immutable Pydantic models: ranges, timezone-aware timestamps, unknown fields rejected, event/entity/alert validation (entities are vehicle-only) |
| `test_clock.py` | simulation clock | 3 | deterministic offsets, crossing midnight into the next date |
| `test_telemetry_simulator.py`, `test_sync.py` | telemetry | 23 | fixed-interval samples, seeded determinism, patrol across midnight, dropouts, nearest-prior join, stale-telemetry boundary |
| `test_scenario.py`, `test_simulate.py` | scenarios, scripted provider | 20 | valid and malformed scenario files, ordering, validated observations, determinism, the `simulate` command |
| `test_store.py`, `test_store_v4.py`, `test_store_v5.py` | SQLite + FTS5 (the frame index) | 64 | schema and foreign keys, round trips, timezone-safe ordering, filters by time/zone/object, FTS ranking, safe search text, in-place migrations to v5 |
| `test_ingest.py` | ingest + query CLI | 10 | exact round trip, atomic ingest, CLI queries by time, object, zone and keyword |
| `test_aggregator.py`, `test_context_engine.py`, `test_build_events.py` | events and context | 57 | events with evidence and duration, missed-detection tolerance, gap splitting, zone change, midnight crossing, incremental context equals batch, events/entities CLI |
| `test_linker.py`, `test_linker_scoring.py`, `test_recurrence.py` | entity memory | 37 | coarse vehicle grouping and scoring, people never linked, plates, recurrence ("returned three times") |
| `test_rules.py`, `test_analyze.py` | camera rules and alerts | 84 | loitering threshold and escalation (inclusive), after-hours window edges and midnight crossing, repeat vehicle, restricted zone, severity, de-duplication, alert persistence and evidence tracing |
| `test_safety_rules.py`, `test_lifecycle.py` | emergencies: drone-safety rules and alert lifecycle | 29 | low battery (warning, critical, hysteresis), degraded/lost signal and telemetry dropout, altitude envelope, geofence, escalation of the same alert, operator acknowledge/resolve, illegal transitions rejected |
| `test_runtime.py`, `test_sampling.py`, `test_perception.py`, `test_cli_run.py` | progressive runtime (dynamic inputs) | 67 | tick ordering, pacing, alerts persisted while the run is in progress, streaming equals batch, adaptive sampling reasons, scene changes that are activity-only or count-only, perception timeout/fallback, the `run` and `scenarios` commands |
| `test_ground_truth.py`, `test_sampler_noise_eval.py`, `test_eval.py`, `test_scale_bench.py` | evaluation harness | 83 | ground-truth scoring (precision, recall, F1 boundaries, latency), perception-noise and sampler experiments, determinism, scale benchmark mechanics |
| `test_video.py`, `test_cassette.py`, `test_video_ingest.py`, `test_gemini.py` | real-video pipeline and VLM adapter | 129 | decoding, sampling, frame gate, cassette record/replay, strict provider output, video to observations to events and alerts, Gemini adapter with a fake SDK client |
| `test_agent_core.py`, `test_agent_llm.py`, `test_agent_truncation.py`, `test_analyst_m12.py` | analyst (LangChain agent) | 186 | bounded read-only tools, citation gate, retry then fallback, prompt and tool-registry agreement, tool-calling loop bounds, truncation notices, planner and replay |
| `test_web_*.py` | dashboard server | 138 | read-only JSON API, localhost guard, static bundle, frame images |
| `test_scaffold.py`, `test_repo_layout.py` | repository rules | 9 | package version, core never imports LangChain, repository layout |
| **Total** | | **963** | |

Counts are from `pytest --collect-only -q`; re-run it after adding tests.

## Dynamic inputs and emergency responses
The assignment asks for test cases for dynamic inputs and emergency responses. They are covered at three levels, all offline:

| Situation | Scenario / test | What is asserted |
|---|---|---|
| Person loiters at the main gate at midnight | `scenarios/loiter_midnight.yaml`, `test_ground_truth.py`, `test_rules.py` | a warning alert at 120 s, the same alert critical at 300 s, an after-hours alert at the first sighting, inclusive thresholds and the 22:00-06:00 window across midnight |
| Intruder reaches a restricted zone, or walks to the garage | `security_events`, `intruder_garage`, `test_rules.py` | a high-severity restricted-zone alert with the event and frames as evidence; after-hours alerts follow a person who moves from the gate to the garage |
| A vehicle keeps coming back | `repeat_pickup`, `test_ground_truth.py`, `test_recurrence.py`, `test_linker.py` | three separate appearances of a similar blue pickup raise one low-severity repeat-vehicle alert and the red sedan distractor does not join it; coarse links are never called the same vehicle |
| Benign activity must stay silent | `normal_day`, `benign_delivery` | zero alerts |
| Detector flickers | `flicker_noise`, `test_aggregator.py` | two missed detections do not split one visit, and the visit still raises exactly one alert |
| Battery falls | `low_battery`, `test_safety_rules.py` | warning at 20%, critical at 10%, resolved only after recovery past the hysteresis band |
| Link degrades, fails or drops out | `signal_loss`, `test_safety_rules.py` | degraded then lost escalates one alert; a telemetry gap longer than 30 s is a lost link |
| Drone leaves the geofence | `geofence_violation`, `test_safety_rules.py` | high then critical, then resolved when back inside, with no camera event involved |
| Scene changes while sampling saves model calls | `test_runtime.py`, `test_sampling.py` | activity-only and count-only changes are analysed, identical frames are carried forward and labelled, every decision has a reason |
| Alert raised mid-run | `test_runtime.py`, `test_lifecycle.py` | alerts are persisted and emitted before the run finishes; streaming and batch agree on the final alerts |
| Operator response | `test_lifecycle.py`, `test_cli_run.py` | acknowledge then resolve is recorded with who and when; illegal actions are refused |

Each scenario file carries an `expect:` block written before the system was run against it; `run` and `evaluate` compare the output with it (precision, recall, F1, severity ladder, timing, detection latency).

## What kind of coverage this is
- **Unit tests** cover models, clock, telemetry, aggregation, linking, rules, the tools and the citation validator.
- **Integration tests** run whole slices: scenario to SQLite to events to alerts, and video to observations to alerts, including the CLI entry points and their error paths.
- **Boundary tests** check the seams: the SDK and LLM boundaries with fakes, the import rules between packages, and that errors never contain a key.
- **Regression tests** pin known facts: the shipped `security_events` scenario yields 7 events, 2 vehicle entities (1 with several appearances) and 7 alerts (including a `signal_loss` alert from a telemetry dropout), and `normal_day` yields none;
  they also pin the byte-exact offline analyst answers mentioned above.
- **Evaluation tests** check that the harness computes what it claims and reproduces itself. They deliberately assert no timing thresholds and no tuned quality numbers.

## What the test suite does not cover
- **Live Gemini behaviour.** Live calls were made only as separate, individually approved runs; their results are described in [ai_usage.md](ai_usage.md) and
  [real_video_experiment.md](real_video_experiment.md), not asserted by tests.
- **Perception accuracy or field performance.** There is no ground-truth annotation of real footage, and the rule metrics apply only to controlled synthetic scenarios.
- **Semantic truth of analyst prose.** The citation gate is tested for what it checks (ids exist and were retrieved, citations present, unknown types and selected wording rejected); its limits are
  measured in [evaluation.md](evaluation.md), not hidden by the tests.

## Clean-clone verification
Run this procedure before submitting, in a fresh directory, to confirm the repository works without any local state (no `.env`, no `output/`, no assignment PDF):
```
git clone <path-or-url-of-the-repository> final-check && cd final-check
python -m venv .venv
.venv/Scripts/python -m pip install -U pip                       # use .venv/bin/python on Linux and macOS
.venv/Scripts/python -m pip install -e ".[dev,agent]"
.venv/Scripts/python -m pytest -q                                # expect every collected test to pass
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
.venv/Scripts/python -m drone_agent ingest scenarios/security_events.yaml --db output/security.db
.venv/Scripts/python -m drone_agent build-events --db output/security.db   # expect 7 events, 2 vehicle entities, 1 with several appearances
.venv/Scripts/python -m drone_agent analyze --db output/security.db        # expect 7 alerts
.venv/Scripts/python -m drone_agent ask --db output/security.db --offline "Was the blue pickup seen more than once?"
.venv/Scripts/python -m drone_agent evaluate --output output/eval          # offline; writes results under output/eval
git status --short                                                # expect clean: generated files are gitignored
```
The pipeline and offline analyst work with `pip install -e ".[dev]"` alone; the `agent` extra is needed for the LangChain adapter tests and for `--provider gemini`.
