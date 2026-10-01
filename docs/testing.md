# Testing

How the system was validated. Quantitative experiments (ablations, rule confusion matrices, scale timings) are in
[evaluation.md](evaluation.md); the real-video experiments are in [real_video_experiment.md](real_video_experiment.md). This document covers the automated test suite
and how to verify a fresh clone.

## Running the tests
```
pip install -e ".[dev,agent]"     # the LangChain adapter tests need the `agent` extra
pytest -q                         # 1301 tests are collected (pytest --collect-only)
ruff check .
ruff format --check .
```
The full suite runs in roughly 3 to 4 minutes on the development machine (the web tests bind localhost sockets, so a sandbox that blocks them fails those tests, not the application; the browser tests start a headless Chrome or Edge). No test needs a Gemini key or a network connection.

## Principles
- **Offline and deterministic.** Time comes from a simulated clock, telemetry from a seeded simulator, and scenario data from YAML. Real-video tests generate a small synthetic MJPG video
  in a temporary directory; no generated video, database or cassette is committed. The one committed video is the bundled real clip and its recorded cassette in `demo_assets/real_clip/`, which the tests check against
  their SHA-256 manifest and never modify.
- **Mocked model boundaries.** The Gemini SDK is replaced by a fake client (`tests/test_gemini.py`); the analyst LLM is a scripted stand-in or a fake chat model that returns real LangChain
  messages (`tests/test_agent_core.py`, `tests/test_agent_llm.py`). Environment loading is patched so a developer's real `.env` never reaches a test.
- **Isolated state.** Every database test uses its own temporary SQLite file.
- **A real browser for the UI.** `tests/test_web_ui_browser.py` drives an installed headless Chrome or Edge over the DevTools protocol (`tests/cdp_browser.py`, built on the `websockets` package that `google-genai` already requires; no Node and
  no new dependency). It compares what the page shows with what the API returns, and is skipped when no browser is installed. `tests/test_web_ui.py` holds the static checks that need no browser.
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
| `test_store.py`, `test_store_v4.py`, `test_store_v5.py` | SQLite + FTS5 (the frame index) | 64 | schema and foreign keys, round trips, timezone-safe ordering, filters by time/zone/object, FTS ranking, safe search text, in-place migrations |
| `test_ingest.py` | ingest + query CLI | 10 | exact round trip, atomic ingest, CLI queries by time, object, zone and keyword |
| `test_aggregator.py`, `test_context_engine.py`, `test_build_events.py` | events and context | 57 | events with evidence and duration, missed-detection tolerance, gap splitting, zone change, midnight crossing, incremental context equals batch, events/entities CLI |
| `test_departure.py` | observed departure (schema v6) | 12 | a close reason is not a departure, `departure_observed_at` is set only when the zone was seen without the object, a loitering alert resolves only then (or by an operator), older rows are not reinterpreted |
| `test_linker.py`, `test_linker_scoring.py`, `test_recurrence.py` | entity memory | 37 | coarse vehicle grouping and scoring, people never linked, plates, recurrence ("returned three times") |
| `test_rules.py`, `test_analyze.py` | camera rules and alerts | 84 | loitering threshold and escalation (inclusive), after-hours window edges and midnight crossing, repeat vehicle, restricted zone, severity, de-duplication, alert persistence and evidence tracing |
| `test_safety_rules.py`, `test_lifecycle.py` | emergencies: drone-safety rules and alert lifecycle | 29 | low battery (warning, critical, hysteresis), degraded/lost signal and telemetry dropout, altitude envelope, geofence, escalation of the same alert, operator acknowledge/resolve, illegal transitions rejected |
| `test_runtime.py`, `test_sampling.py`, `test_perception.py`, `test_cli_run.py` | progressive runtime (dynamic inputs) | 69 | tick ordering, pacing, alerts persisted while the run is in progress, streaming equals batch, adaptive sampling reasons, scene changes that are activity-only or count-only, perception timeout/fallback, the `run` and `scenarios` commands |
| `test_ground_truth.py`, `test_eval_metrics.py`, `test_sampler_noise_eval.py`, `test_eval.py`, `test_scale_bench.py` | evaluation harness | 124 | ground-truth scoring (precision, recall, F1 boundaries, latency, label-satisfied versus exact, one-to-one maximum-cardinality matching), perception-noise and sampler experiments, determinism, scale benchmark mechanics |
| `test_video.py`, `test_cassette.py`, `test_video_ingest.py`, `test_gemini.py` | real-video pipeline and VLM adapter | 129 | decoding, sampling, frame gate, cassette record/replay, strict provider output, video to observations to events and alerts, Gemini adapter with a fake SDK client |
| `test_video_live.py`, `test_video_resilience.py`, `test_real_clip_evidence.py` | progressive video, failure handling, real-clip evidence | 56 | the progressive path reaches the same events, entities and alerts as the batch path, the carry bound, explicit failed outcomes with classified reasons, timeouts and `--fail-fast`, and (when the local experiment files exist) the real-clip replay evidence |
| `test_agent_core.py`, `test_agent_llm.py`, `test_agent_truncation.py`, `test_analyst_m12.py` | analyst (LangChain agent) | 186 | bounded read-only tools, citation gate, retry then fallback, prompt and tool-registry agreement, tool-calling loop bounds, truncation notices, planner and replay |
| `test_factcheck.py`, `test_tool_contract.py` | analyst grounding | 79 | the factual-claim check (supported, contradicted, unsupported) with the false-positive cases found while building it, one tool name for the run summary, a refusal after a failed tool call is not accepted |
| `test_web_api.py`, `test_web_bundle.py`, `test_web_guard.py`, `test_web_server.py` | dashboard server | 138 | read-only JSON API, localhost guard, static bundle, frame images, the question guard |
| `test_web_public.py`, `test_web_real_demo.py` | public mode and the bundled real demo | 61 | the opt-in public hostname (everything else still refused), the Render command through the CLI, the real clip and cassette checked against their manifest and replayed with no key and no network, the expected replay result |
| `test_web_ui.py`, `test_web_ui_browser.py` | dashboard UI | 87 | static checks (required ids, no inline styles, WCAG AA colour contrast, no hardcoded run results), then a real headless browser: on-screen counts equal the API for all three runs, event, frame and alert interactions, run switching, the analyst, error and loading states, keyboard use, and no console errors, broken images or horizontal overflow at five viewport sizes (the 39 browser tests are skipped without Chrome or Edge) |
| `test_scaffold.py`, `test_repo_layout.py` | repository rules | 9 | package version, core never imports LangChain, repository layout |
| **Total** | | **1301** | |

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
  the bundled real clip replays to 9 observations (8 replayed, 1 carried forward), 33 objects, 28 events, 12 entities and 0 alerts; they also pin the byte-exact offline analyst answers mentioned above.
- **Evaluation tests** check that the harness computes what it claims and reproduces itself. They deliberately assert no timing thresholds and no tuned quality numbers.

## What the test suite does not cover
- **Live Gemini behaviour.** Live calls were made only as separate, individually approved runs; their results are described in [ai_usage.md](ai_usage.md) and
  [real_video_experiment.md](real_video_experiment.md), not asserted by tests.
- **Perception accuracy or field performance.** There is no ground-truth annotation of real footage, and the rule metrics apply only to controlled synthetic scenarios.
- **Semantic truth of analyst prose.** The citation gate and the factual-claim check are tested for what they check (ids exist and were retrieved, citations present, unknown types and selected wording rejected, stated zones, types, times,
  durations, counts and severities agree with the cited evidence); claims outside those forms are not checked, and the limits are measured in [evaluation.md](evaluation.md), not hidden by the tests.
- **Hosted performance and other browsers.** The browser tests use one locally installed Chromium-based browser; nothing measures a hosted instance, and frame decoding cost on a small host is not measured.

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
.venv/Scripts/python -m drone_agent.web --demo                             # opens on the bundled real clip at http://127.0.0.1:8765 (Ctrl+C to stop)
git status --short                                                # expect clean: generated files are gitignored
```
The pipeline and offline analyst work with `pip install -e ".[dev]"` alone; the `agent` extra is needed for the LangChain adapter tests and for `--provider gemini`. A clean clone includes `demo_assets/real_clip/`, so the
dashboard demo needs no local files; the 39 browser tests run only where Chrome or Edge is installed and are skipped otherwise.
