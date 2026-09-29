# AI Usage Log

Honest record of where AI tools were used, what I changed or rejected, and how each result was
validated. Updated at the end of every milestone.

## M0: Assignment analysis and architecture planning
**Tool:** Claude Code (Claude Sonnet 5.5), used in plan mode before any code.

**What the AI did**
- Read the assignment PDF and extracted deliverables, assessment criteria and ambiguities
  (e.g. text-simulated frames vs. VLM descriptions, LangChain mentioned only in submission
  details, "scalability" scored but never defined).
- Proposed the architecture, schema, models, aggregation logic, rules, milestones and a
  requirement traceability matrix.
- Noticed the assignment PDF is marked confidential, so it was gitignored before the first commit.

**Major architecture suggestions (from AI)**
- Entity re-identification via coarse attribute signatures, for "entered twice" questions.
- Deterministic rules decide, LLM only explains, with a citation check on generated IDs.
- Record/replay ("cassette") VLM cache so demos and tests run without an API key.
- Frame-diff gating to reduce VLM calls, measured as a scalability metric.
- Confining LangChain to the Q&A layer.

**Scope reductions I requested (human decisions)**
- No assumption of Anthropic API access; VLM stays behind a generic `VisionProvider`, starting with scripted/mock providers.
- SQLite + FTS5 only: dropped embeddings, CLIP, fastembed, vector DBs and rank fusion from the MVP.
- Cut the AI's six proposed rules to four: loitering, after-hours presence, repeat vehicle, restricted-zone entry.
- No frontend; CLI/Rich only. Git branch `main`.
- Dropped 1M-row benchmarking in favour of practical prototype metrics.
- Tightened the plan to about 26 hours to leave a submission buffer (deadline Oct 1, 3:06 PM IST).
- Rejected two plan revisions that did not yet reflect these cuts, until they did.

**Postponed:** embeddings/CLIP/vector search, battery/geofence/crowd/stationary rules, web UI, additional
VLM providers, YOLO/tracking, Docker/CI, LangGraph, face recognition, multi-drone.

**How decisions will be validated**
- Scenario-based golden tests with known expected events and alerts (including a benign
  case that must not alert, and a midnight-rollover boundary).
- Precision/recall of events and alerts against scripted ground truth (M7).
- Measured scalability: observations processed, events generated, frames-to-events compression, processing time, query latency, VLM calls saved.
- A test that the core pipeline does not import LangChain, and `git ls-files` checks that the PDF, `.env` and databases are not tracked.
- Every AI-generated component is reviewed and logged here with my changes.

**Also in M0:** Claude Code generated the `.gitignore`, package scaffold, `pyproject.toml` and these docs.
Validated by `git check-ignore` on the PDF, `git ls-files`, and a passing scaffold smoke test.

## M1: Domain models, simulated telemetry, ScriptedProvider
**Tool:** Claude Code (Claude Sonnet 5.5). I gave a detailed M1 spec; the AI implemented it.

**What the AI generated**
- Pydantic v2 models (Telemetry, TelemetryAssociation, DetectedObject, Observation, minimal Event/Alert
  schemas), `SimulationClock`, the telemetry simulator, `TelemetrySynchronizer`, the `VisionProvider`
  protocol, `ScriptedProvider`, scenario loader, Rich CLI (`simulate`), two YAML scenarios and 60 tests.

**Design decisions (AI proposed, I approved by approving the plan and milestone spec)**
- Models are frozen and reject unknown fields; timestamps use `AwareDatetime`; confidence is 0..1 with NaN rejected.
- The clock is stateless (`start + offset`), so midnight crossing is plain datetime arithmetic and never reads the wall clock.
- Nearest-prior sync uses `bisect`; age equal to the threshold is fresh, strictly greater is stale;
  a frame before the first sample has no association.
- The scenario validator cross-checks that each frame's zone equals the drone's patrol zone at that
  offset. This encodes the assumption "object zone = drone zone" and catches scenario typos.
- Telemetry randomness is drawn for every sample before dropouts are applied, so a dropout never changes other samples.
- `security_events.yaml` documents its ground truth in a header comment (loiter 8 min with a missed
  detection, repeat blue pickup in three sightings incl. a "dark blue" variant, brief after-hours person,
  restricted-zone entry, a white sedan distractor, one stale-telemetry frame).

**Assumptions:** closed hours are 22:00-06:00 (used by M4, not enforced in M1); coordinates are synthetic
values near (0, 0); attribute values must be strings (a YAML number such as `plate: 1234` is rejected on purpose).

**Problems found and fixed during validation (honest log)**
- 12 tests errored with `PermissionError` on the system temp dir (`pytest-of-Admin`). I set
  `--basetemp=.pytest_tmp` in `pyproject.toml` and gitignored it.
- Ruff's default rule set flagged 17 style issues. I auto-fixed import order, then pinned an explicit
  rule selection (E, F, I, UP, B, SIM) rather than chase stylistic rules, and fixed the 4 real findings by hand.

**Validation:** `pytest -v` (60 passed), `ruff check .` (clean), `ruff format --check .` (clean), and a manual
run of `python -m drone_agent simulate scenarios/security_events.yaml`. Tests need no network or API key.
