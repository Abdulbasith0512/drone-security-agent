# Testing

How the system was validated. Quantitative experiments (ablations, rule confusion matrices, scale timings) are in
[evaluation.md](evaluation.md); the one real-video run is in [real_video_experiment.md](real_video_experiment.md). This document covers the automated test suite
and how to verify a fresh clone.

## Running the tests
```
pip install -e ".[dev,agent]"     # the LangChain adapter tests need the `agent` extra
pytest -q                         # at the time of this submission, 538 tests were collected (pytest --collect-only)
ruff check .
ruff format --check .
```
The full suite ran in roughly 40-45 seconds on the development machine. No test needs a Gemini key or a network connection.

## Principles
- **Offline and deterministic.** Time comes from a simulated clock, telemetry from a seeded simulator, and scenario data from YAML. Real-video tests generate a small synthetic MJPG video
  in a temporary directory; no video, database or cassette is committed.
- **Mocked model boundaries.** The Gemini SDK is replaced by a fake client (`tests/test_gemini.py`); the analyst LLM is a scripted stand-in or a fake chat model that returns real LangChain
  messages (`tests/test_agent_core.py`, `tests/test_agent_llm.py`). Environment loading is patched so a developer's real `.env` never reaches a test.
- **Isolated state.** Every database test uses its own temporary SQLite file.
- **Golden regression.** `tests/golden_offline_answers.json` holds 14 offline analyst answers recorded from the code before M8a; `tests/test_agent_truncation.py` requires the non-truncated ones
  to stay byte-identical.
- **Mutation checks during development.** At several milestones the code was deliberately broken in a handful of ways to confirm the tests notice (recorded in [ai_usage.md](ai_usage.md)).
  These are a development practice, not part of the suite.

## Test organisation
| File | Layer | Tests | What it verifies |
|---|---|---|---|
| `test_models.py` | domain models | 24 | strict, immutable Pydantic models: ranges, timezone-aware timestamps, unknown fields rejected, event/entity/alert validation (entities are vehicle-only) |
| `test_clock.py` | simulation clock | 3 | deterministic offsets, crossing midnight into the next date |
| `test_telemetry_simulator.py` | telemetry | 7 | fixed-interval samples, seeded determinism, patrol coverage across midnight, dropouts |
| `test_sync.py` | telemetry sync | 7 | nearest-prior join, stale-telemetry flag boundary |
| `test_scenario.py` | scenarios | 13 | valid and malformed scenario files, ordering and duplicate frames, the two shipped scenarios contain the evidence later stages need |
| `test_simulate.py` | scripted provider and simulation | 7 | validated observations, determinism, synchronisation, the CLI `simulate` command and its error output |
| `test_store.py` | SQLite + FTS5 | 35 | schema and foreign keys, round trips, timezone-safe ordering, filters, FTS ranking, safe handling of search text, missing FTS5 reported |
| `test_ingest.py` | ingest + query CLI | 10 | exact round trip of both scenarios, atomic ingest (duplicate rejected, failure leaves nothing), CLI queries and error messages |
| `test_aggregator.py` | event aggregation | 26 | events with evidence and duration, missed-detection tolerance, gap splitting, zone change, attribute compatibility, midnight crossing, ordering, determinism |
| `test_linker.py` | entity memory | 9 | coarse vehicle grouping, missing attributes never guessed, overlapping events, people never linked, explicit and conflicting plates |
| `test_build_events.py` | events and entities CLI | 24 | the shipped scenarios' expected events and entities, the evidence chain down to frames, atomic and idempotent rebuilds, in-place schema upgrades, events/entities CLI |
| `test_rules.py` | deterministic rules | 53 | loitering threshold (inclusive), after-hours window edges and midnight crossing, repeat vehicle, restricted zone, severity, de-duplication, overlap policy |
| `test_analyze.py` | rules persistence + alerts CLI | 24 | alert persistence, evidence tracing, idempotent re-analysis, the expected alerts for the shipped scenarios |
| `test_video.py` | video source, sampling, gate | 30 | unusable files and containers, sampling plans (integer and fractional FPS), decode failures, gate behaviour, timeline mapping |
| `test_cassette.py` | provider contract and cassettes | 37 | strict provider-output validation, cassette round trip, replay, misses, record and live modes, safe file contents |
| `test_video_ingest.py` | real-video pipeline + CLI | 24 | video to observations to events and alerts, carry-forward versus drop, CLI success and failure paths, M1-M4 regression facts |
| `test_gemini.py` | Gemini adapter | 38 | key handling, image encoding, structured output, bounded retries, error classification, no key in errors, record/replay wiring |
| `test_agent_core.py` | analyst tools, citations, sessions | 79 | bounded read-only tools, citation validation (nonexistent, unretrieved, malformed, unknown-type, missing), overclaim wording, retry then fallback, template answers, summaries, follow-ups |
| `test_agent_llm.py` | LangChain adapter + CLI | 26 | tool-calling loop bounds, tool errors, metrics, no key leaks, `ask`/`summarize` CLI, import boundary (core never imports LangChain) |
| `test_agent_truncation.py` | analyst truncation (M8a) | 29 | every capped list states what was cut, compound people-and-vehicles questions, no notice when nothing is cut, golden equality |
| `test_eval.py` | evaluation harness | 31 | determinism, metric maths and zero denominators, experiment mechanics, serialization, `evaluate` CLI, no network or key |
| `test_scaffold.py` | repository rules | 2 | package version; the core does not mention or import LangChain |
| **Total** | | **538** | |

## What kind of coverage this is
- **Unit tests** cover models, clock, telemetry, aggregation, linking, rules, the tools and the citation validator.
- **Integration tests** run whole slices: scenario to SQLite to events to alerts, and video to observations to alerts, including the CLI entry points and their error paths.
- **Boundary tests** check the seams: the SDK and LLM boundaries with fakes, the import rules between packages, and that errors never contain a key.
- **Regression tests** pin known facts: the shipped `security_events` scenario yields 7 events, 2 vehicle entities (1 with several appearances) and 6 alerts, and `normal_day` yields none;
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
.venv/Scripts/python -m drone_agent analyze --db output/security.db        # expect 6 alerts
.venv/Scripts/python -m drone_agent ask --db output/security.db --offline "Was the blue pickup seen more than once?"
.venv/Scripts/python -m drone_agent evaluate --output output/eval          # offline; writes results under output/eval
git status --short                                                # expect clean: generated files are gitignored
```
The pipeline and offline analyst work with `pip install -e ".[dev]"` alone; the `agent` extra is needed for the LangChain adapter tests and for `--provider gemini`.
