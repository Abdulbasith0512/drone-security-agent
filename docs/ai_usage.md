# AI Usage Log

Honest record of where AI tools were used, what I changed or rejected, and how each result was
validated. Updated at the end of every milestone.

## Current status summary (M0-M8b)
**Author:** Abdul Basith Syed.

**Development tool:** Claude Code (Claude Sonnet 5.5) generated the code, tests, evaluation harness and documentation from my milestone specifications.
Work proceeded one milestone at a time, each committed separately (commits listed below), and each entry records what was generated, the design decisions and how the result
was validated. My contribution was the requirements and scope cuts, the design decisions recorded in each entry below, and the corrections I directed. **Runtime models:** Gemini
(`gemini-3.6-flash` in the recorded runs) is a runtime component (an optional vision model and an optional analyst backend), not a development assistant.

| Milestone | Commit | Subject |
|---|---|---|
| M0 | `d1c675d` (`.gitignore` first: `77330a6`) | scaffold, feature spec, architecture plan |
| M1 | `ce851c9` | domain models, simulated telemetry, scripted provider, scenarios |
| M2 | `67322ed` | SQLite persistence, FTS5 search, ingest and query CLI |
| M3 | `f269130` | temporal events and coarse vehicle entity memory |
| M4 | `fc2c4d3` | deterministic rules and evidence-backed alerts |
| M5A | `6c70037` | provider-neutral real-video pipeline, frame gate, cassette replay |
| M5B | `8e7a3c4` | Gemini frame describer, record/replay wiring |
| M6 | `3883630` | grounded analyst with validated evidence citations |
| M7 | `560c7c3` (tag `m7-final-baseline`) | evaluation, ablations and grounding analysis |
| M8a | `0ae7414` | explicit truncation in the offline analyst |
| M8b | `7589c2c` | real-video experiment documentation |

**How AI output was validated** (details in each entry): offline deterministic tests (the SDK boundary is mocked; no test needs a key), mutation checks that break the
code to confirm the tests notice, real runs of Gemini only as separate, individually approved steps, and honest problem logs. **Decisions and corrections recorded below** include:
rejecting plan revisions until my scope cuts were reflected (M0), choosing carry-forward for gate-skipped frames (M5A), reporting the gate and citation-suite results without tuning them (M7),
and fixing two citation-validator behaviours after review while stating that this makes the suite's accuracy non-independent (M7).

**Accidental live Gemini verification during M8d.** While checking the README's demo commands I ran two Gemini-backed commands, one analyst query (`ask --provider gemini`) and one
frame-description operation (`describe-frame`), intending to confirm the missing-key error message. The locally configured `.env` supplied a key, so they executed as live calls.
The exact number of API requests is unknown and was not recorded. Neither operation writes output; no tracked repository file and no real-video artifact was modified by them, and
no further live calls were made. They were an accidental verification of the error path, not part of a new experiment, and no result from them is used anywhere in this project.

**Not claimed anywhere in this project:** perception accuracy on real video, security-detection performance in the field, that a coarse vehicle match proves physical identity, that
citation validation establishes semantic entailment, or reliability from the few live Gemini runs.

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

## M2: SQLite persistence, FTS5 search, ingestion and query CLI
**Tool:** Claude Code (Claude Sonnet 5.5), from my detailed M2 spec.

**What the AI generated:** the schema, the `Store` API (writes, reads, filters, transactions/savepoints),
the FTS5 query sanitiser, `ingest_result`, the `ingest`/`query` CLI commands, 45 new tests, and the
architecture/README updates.

**Decisions and where I constrained it (my spec):** stdlib `sqlite3` only, no ORM; FTS5 without embeddings;
report rather than fall back if FTS5 is missing; Pydantic models kept at the boundary; either idempotent or
atomic-reject ingestion; events/alerts tables postponed.

**Design decisions the AI made (please review):**
- Reject duplicates rather than upsert, because an upsert can silently replace stored evidence. This makes `frame_id`
  a global key, so a database holds one run.
- `timestamp` (ISO with offset) plus `ts_utc` (fixed-width UTC) columns, because comparing ISO strings with
  different offsets orders them wrongly.
- Attributes as canonical JSON, not an attribute table (open-ended keys, read back whole, text search covered by FTS5).
- The `frames` table has no caption column since observations are 1:1 with frames and the description lives once.
- FTS5 `porter unicode61` tokenizer, with search words individually quoted and ANDed so user text can never be
  parsed as FTS5 syntax. Consequence: stemming means "parking" also matches "parked".
- `query_observations` and `explain_query` share one query builder, so the plan shown is the plan that runs.

**Problems found during validation (honest log)**
- Three of my first-draft tests were wrong, not the code: I miscounted `main_gate` frames (12, not 17) and person
  frames (13), and expected "parking" not to match a "parked" vehicle (porter stemming makes it match).
  `*` alone correctly raises "no searchable words". I corrected the tests to the real, intended behaviour.
- The object-type query plan is `SCAN o USING INDEX idx_observations_ts` plus a covering-index probe per row.
  The index is used, but the query is still linear in observation count. I documented that instead of claiming otherwise.
- Text search results are ranked by BM25, so they are not chronological (visible in the demo output).
- Review of my own CLI code found a real bug: error messages were printed with Rich markup enabled, so an error
  that echoes user input such as `[/oops]` (e.g. a bad `zone` value in a scenario) would crash with `MarkupError`.
  I added `escape()` and a regression test. My first version of that test passed even without the fix, so I
  rewrote it and confirmed it fails (`MarkupError`) without the fix and passes with it.

**Validation performed:** `pytest -v` (105 passed), `ruff check .` and `ruff format --check .` clean, end-to-end
`ingest` of `security_events.yaml`, the three query demos, `EXPLAIN QUERY PLAN` inspection, a raw-SQLite check of row counts,
`PRAGMA foreign_key_check` (no violations) and `PRAGMA integrity_check` (ok). Tests use isolated temp databases, no network and no API key.

## M3: Temporal events and coarse vehicle entity memory
**Tool:** Claude Code (Claude Sonnet 5.5), from my detailed M3 spec.

**What the AI generated:** the extended `Event` model and new `Entity` model, `EventAggregator`, attribute
normalisation, `EntityLinker`, the YAML/pydantic configuration, schema v2 with an in-place migration, event/entity
persistence, `build_events`, the `build-events`/`events`/`entities` CLI commands, 58 new tests, and these docs.

**Model changes (please review):** `Event` was extended from 6 fields to also carry `attributes`, `activities`,
`description`, `frame_ids`, `close_reason` and `entity_id` (`duration_seconds` and `evidence_count` are properties,
not stored fields). This changed one M1 test. `Entity` is new, vehicle-only by validation, and has `match_level`
(`coarse_match` / `confirmed`) so the data model can say "similar vehicle" versus "confirmed vehicle".

**Design decisions the AI made (please review)**
- **Deviation from my spec:** I asked for events to close "when incompatible attributes appear". The AI did NOT
  implement that literally: an incompatible detection starts a separate event but leaves the existing one open,
  because closing it would truncate an event whenever two different people stand together. Incompatible attributes
  still produce separate events, and this is tested.
- Two flicker knobs, `max_gap_seconds` (180) and `max_missed_frames` (2); missed frames only count in the event's own zone.
- Zone change always closes an event, even if the drone returns within the gap. This is conservative: it can split one
  activity into two events but never merges across a zone the drone left.
- Event ids are assigned when an event opens (`EVT-<P|V|O>-nnn`), so they follow first-seen order and do not depend on closing order.
- Entity linking requires equal normalised `color` and `body_type`; a missing attribute never links. Overlapping
  events are never the same vehicle. A plate is used only if the input supplied one.
- `build-events` replaces derived data in one transaction (idempotent), while `ingest` still rejects duplicates.
- Schema migrations are additive and versioned instead of failing on an M2 database.

**Problems found and fixed during validation (honest log)**
- Two M1/M2 tests failed because of the intended model/schema changes (`Event` fields, `STATEMENTS` renamed to
  `MIGRATIONS`); I updated them rather than the code.
- The first draft of the linker had convoluted plate handling; I rewrote it as three explicit rules before testing.
- A review found that table cells were still parsed as Rich markup, so a VLM description containing `[/x]` would
  crash the CLI. Cells now use `rich.text.Text`; `test_cli_prints_markup_like_descriptions_literally` covers it.
- All new tests passed first time, which is not proof they can fail. I ran three deliberate mutations (no colour
  normalisation, people linked, no flicker tolerance) and each was caught by the suite; the code was then restored.
- I predicted the `security_events` outcome before running it (7 events, 2 entities); the real implementation produced
  exactly that, and the metrics reported in the CLI come from the run, not from the tests' expectations.
- No scenario ends with an event still open (the last event is closed by a zone change), so `scenario_end` is only
  exercised by unit tests, not by the shipped scenarios.

**How behaviour was validated:** unit tests for every boundary (adjacent frames, missed frame, long gap, missed-frame
limit, zone change, incompatible attributes, midnight, flush, ordering, determinism, normalisation, config); linker
tests (blue/dark-blue merge, sedan separation, missing attributes, overlap, people never linked, plate cases);
persistence tests (round trip, foreign keys, atomic replace, idempotent rebuild, migration, newer-version refusal);
scenario tests against the M1 ground truth; CLI tests; and a raw-SQL integrity inspection (`PRAGMA integrity_check`
ok, `foreign_key_check` empty, evidence rows equal detections, no evidence outside an event's span or zone).

## M4: Deterministic security rules and alerts
**Tool:** Claude Code (Claude Sonnet 5.5), from my detailed M4 spec.

**What the AI generated:** the redesigned `Alert` model and `Severity` levels, the rule configuration models and
`config/security_rules.yaml`, four rule functions, `RuleEngine`, `analyze`, schema v3 with alert persistence, the
`analyze`/`alerts` CLI commands, 80 new tests (244 total), and these docs. No LLM is called anywhere in M4, by design.

**Model change (please review):** `Severity` was replaced by `low/medium/high/critical` and the M1 `Alert` placeholder
was redesigned (see docs/architecture.md). One M1 test changed accordingly.

**Design decisions the AI made (please review)**
- **Overlap policy: rules coexist.** I recommended (and you allowed) independent rules with dedup only on identical
  (rule, subject). So a night-time loiterer gets two alerts. I did not add cross-rule suppression: it would couple the rules
  and hide information. Cost: more alerts per situation (6 alerts for 4 situations in `security_events`).
- **After-hours defaults to people only.** With vehicles included the same pickup would raise four extra after-hours
  alerts on top of the repeat-vehicle alert. Vehicles are one config line away, and this is documented as a false-negative mode.
- **Threshold boundary is inclusive** (`duration >= threshold`), `triggered_at` is computed from event timestamps
  (first_seen + threshold for loitering), and the after-hours window is `[from, until)` with overlap semantics for events
  that straddle a boundary.
- **Timezone handling is explicit**: closed hours use the timestamp's own offset unless `utc_offset` is configured; the system clock is never used.
- **Alerts are replaced, not appended**, on each `analyze`; `build-events` discards alerts because they cite events it replaces.
- Alert ids are `ALR-<RULE>-<event or entity id>` so identity, not a counter, decides duplicates.

**Results were not tuned.** I ran the rules on both scenarios with the defaults and reported what came out:
`security_events` -> 6 alerts (after-hours 3, loitering 1, repeat vehicle 1, restricted zone 1); `normal_day` -> 0 alerts.
I did not adjust thresholds or scenarios to reach a number, and nothing is special-cased per scenario.

**Problems found during validation (honest log)**
- PyYAML reads an unquoted `22:00` as the integer 1320 (base-60). I found this while designing the config and made the loader
  reject unquoted times with a clear message; the shipped YAML quotes them and a test covers it.
- Two existing tests failed because of intended changes (schema version 2 -> 3, `Alert`/`Severity`); I updated them and
  added a real M3 -> v3 migration test.
- One of my new tests expected the wrong alert order (I forgot loitering triggers at 23:46, after the two 23:41 alerts).
  The engine's `(triggered_at, rule order)` sort was right; I fixed the test.
- All new tests passed quickly, so I ran eight deliberate mutations (loitering `>` vs `>=`, after-hours end inclusive,
  start exclusive, previous-evening window dropped, repeat-vehicle threshold off by one, overclaiming "same vehicle
  returned" wording, engine de-duplication removed, restricted-zone check removed). Every one was caught by the suite;
  the code was restored afterwards.
- `alerts` printed "run `analyze` first?" on `normal_day` even after a correct analysis found nothing; I reworded it.
- I made these edits with file tools after a long shell command failed to parse; nothing was executed from the failed command.

**How behaviour was validated:** rule unit tests at every boundary you listed (below/at/above threshold, 21:59:59 / 22:00:00 /
23:59:59 / 00:00:00 / 05:59:59 / 06:00:00 / midday, straddling events, non-midnight windows, UTC vs configured offsets,
1/2/3/5 appearances, coarse vs confirmed wording, a different vehicle, allowed vs restricted zone, multi-frame event), config
validation (13 invalid files), engine determinism/dedup/overlap, database round trip against an independent engine run, evidence
chain to frames, foreign-key and CHECK enforcement, atomic failed replace, idempotent re-analysis, config-change replacement,
alert discard on rebuild, `security_events` and `normal_day` end to end, CLI tests, and a raw-SQL integrity inspection
(`integrity_check` ok, `foreign_key_check` empty, every alert has events and evidence frames, `triggered_at` inside its span).

## M5A: Real video ingestion, frame gate, provider contract and cassette replay
**Tool:** Claude Code (Claude Sonnet 5.5), from my detailed M5A spec. Per my instruction no real VLM is chosen or implemented, and this
milestone was committed as its own checkpoint before any real-provider work.

**What the AI generated:** the `video/` package (validated source, deterministic sampler, frame-change gate, simulated timeline, ingest
orchestration, synthetic test video), the `vision/` contract (`FrameDescriber`, strict `ProviderOutput` schema, `DescriberProvider`) and the
cassette (file format, replay/record/live modes), the `ingest-video` and `make-demo-video` commands, a generalised `ingest_observations`, and 91 new tests (335 total).

**Design decisions the AI made (please review)**
- **Carry-forward of gate-skipped frames (default) is an addition beyond my spec.** My spec only said the gate avoids sending frames to a VLM.
  The AI noticed that dropped frames leave long gaps that the M3 aggregator turns into split events (a still person becomes one
  zero-duration event; there is a test showing it), so skipped frames are represented by a labelled copy of the last accepted observation
  (`source = "<provider>+carried"`) and the gate resets on zone changes. `--skipped drop` gives the literal behaviour.
  **Decision (mine, after review): carry-forward stays the default; `--skipped drop` stays as the explicit alternative.** A carried
  observation is temporal propagation of the last accepted model observation, NOT a new model inference, and is documented and labelled as such.
- **A real provider implements a smaller interface** (`describe(frame) -> ProviderOutput`), not `analyze() -> Observation`. The pipeline, not the
  model, assigns zone, time and ids. `VisionProvider` and `ScriptedProvider` are unchanged; `FrameInput` gained optional video fields.
- **The provider output schema is strict** (unknown keys and wrong types are errors). Missing confidence is an error unless an explicit
  `default_confidence` is configured; nothing is invented silently. This may be brittle against a real model and could need a repair step in M5B.
- **Cassettes are keyed by video file hash + frame index**, not pixels, so they survive different OpenCV builds; a pixel-level content hash is recorded
  and only reported on mismatch. The cassette in `output/` is canned (written by the tool), labelled as such, not a model's output.
- **The gate default (threshold 6.0 on a 64x36 greyscale copy) is an uncalibrated starting point.** `frame_id` = source frame index, so one database holds one video.
  Frames are decoded sequentially (simple, codec-safe) rather than by seeking. `numpy` was added as an explicit dependency.

**Problems found and fixed during validation (honest log)**
- The first synthetic scenes were too subtle: scene changes scored 1.7 and 3.2 against the default threshold of 6.0, so the gate would have skipped real
  changes. I did NOT tune the default to a toy video; I made the scenes large and high-contrast (scores now 13-26 versus exactly 0.0 for identical frames).
- Three of my first-draft tests assumed wrong behaviour: a "cassette miss" test never missed because the gate correctly skipped the extra samples (it now uses `--no-gate`
  for the miss and checks the gated run separately), and I miscounted the simulated telemetry samples (3, not 22).
- Mutation testing: I broke the code 8 ways (gate `>` versus `>=`, gate comparing with the last seen frame, truncating instead of rounding frame indexes, carrying
  across zone changes, a cassette miss silently returning text, silently inventing confidence, ignoring extra provider keys, skipping the first-frame decode check).
  One mutation (first-frame decode) was NOT caught at first, because my corrupt-file tests fail earlier at "cannot open"; I added stubbed-OpenCV tests
  (bad FPS, no frames, undecodable first frame) and re-ran it. Another mutation did not apply because of a shell multi-line `sed`; I redid it with a script. All 8 are now caught.
- A long mutation loop exceeded the tool timeout and ran in the background; I did not edit code while it ran and confirmed the restored files match before continuing.

**How it was tested:** offline and deterministic (the video is generated in the tests, never committed): metadata, missing/empty/garbage/truncated files,
stubbed unusable containers, sampling plans (integer, 12.5 and 29.97 fps, short video, end of video, tiny intervals, `max_samples`), timestamps, decode failures and early stream end,
gate behaviour (first frame, identical, changed, drift, boundary, disabled, reset, scene changes on the synthetic video), timeline mapping including midnight, provider
output validation (13 malformed shapes), observation construction, cassette round trip/determinism/safe contents/invalid files, replay/miss/mismatch, record and live modes
with a FAKE describer (no real model), stable frame identity, video ingest into SQLite, unchanged events/entities/alerts from video observations, CLI success and failure paths (a failure
leaves no database), and an M1-M4 regression check (7 events, 2 entities, 1 multi-appearance entity, 6 alerts; `normal_day` 0 alerts).

## M5B: Gemini FrameDescriber (single-frame smoke test, record/replay wiring)
**Tool:** Claude Code (Claude Sonnet 5.5), from my M5B spec.

**What the AI generated:** `vision/gemini.py` (config, image preparation, error classes, retry policy, structured-output request), `read_frame_at` (decode one frame),
the `describe-frame` command, `--provider gemini` / `--cassette-mode` / `--model` on `ingest-video` (with cassette flush-on-failure and resume), `.env` loading, and 38 offline tests.

**Design decisions the AI made (please review)**
- No manual BGR->RGB conversion: `cv2.imencode` wants BGR, so converting first would swap colors in the JPEG. I deviated from the literal spec for that reason and added a test.
- Gemini is not asked for confidence; the schema drops the field and a documented placeholder (0.5, uncalibrated) fills it downstream. The shared contract and M5A cassette tests are unchanged.
- The prompt is the shared `PROVIDER_INSTRUCTIONS` plus a one-line exception; I did not edit the shared text.
- Retries are bounded: 2 for rate limit/transient, 1 for malformed output; auth, bad model and bad requests never retry; errors are scrubbed of the key and not chained.

**Validation:** all unit tests mock the SDK (no network). Live checks I ran with my own key: one frame of the synthetic demo video through `describe-frame`, then a record run and a zero-call replay of the
same video with equivalent observations. The frames are synthetic rectangles, so the descriptions show the plumbing works, not that the model recognises real scenes.
**Honest notes:** the record run made 5 requests for 4 accepted frames (one retry, cause not recorded in the summary). Latency was several seconds per call. Real footage has not been evaluated yet.

## M6: Analyst Q&A and summary layer
**Tool:** Claude Code (Claude Sonnet 5.5), from my M6 spec.

**What the AI generated:** the `drone_agent/agent/` package (read-only tools with explicit limits, citation parser/validator, template analyst, session with one retry and fallback, run summary,
LangChain/Gemini tool-calling adapter, system prompt, analyst config), the `ask` and `summarize` commands, four small read-only store helpers (counts, span, per-zone/source counts) and 83 offline tests (456 in total), plus small call metrics (model, rounds, tool calls, correction attempts, fallback, latency) reported outside the answer text.

**Design decisions the AI made (please review)**
- LangChain is used only for the chat-model tool-calling loop in one file; a hand-written bounded loop replaces an agent framework, so there is no LangGraph and the offline path needs no extras.
- Validation is stricter than "ids exist": a cited id must also have been returned by a tool in this conversation, and an answer with facts but no citation (and no "Insufficient evidence") is rejected.
  After one correction retry the deterministic answer replaces the model's text; the same fallback covers API outages.
- Pronoun follow-ups for the template analyst resolve only when the previous answer had a single subject; after a summary ("it" is ambiguous) they return "insufficient evidence" instead of guessing.
- `--offline` is the default provider so no command spends API quota unless `--provider gemini` is given.

**Problems found and fixed during validation (honest log)**
- The first template "no such alert" reply failed my own citation gate (a factual claim without citations); it now says "Insufficient evidence".
- I had guessed the restricted-zone rule id as `restricted_zone`; the real id is `restricted_zone_entry`, so that question returned "no alert" until a test caught it.
- Citing every id in an answer to resolve "it" picked the last entity after a summary (wrong); replaced by the single-subject rule above.
- A scaffold test that greps core sources for the word "langchain" caught my CLI wiring; the lazy construction moved into `agent/build.py`.
- Mutation checks: I broke the code 9 ways (skip the existence check, skip the missing-citation check, skip the retrieved-set check, allow a third attempt, accept invalid answers, remove the observation cap,
  drop the coarse-vehicle wording, resolve pronouns with several candidates, remove the tool-call cap). All are caught by the tests (one mutation first did not apply because the string differed; redone by hand).

**How it was tested:** all offline. The analyst LLM boundary is faked (a scripted analyst for the session; a fake chat model that returns real LangChain messages for the adapter). Covered: each tool, filters, bounds and
truncation flags, read-only behaviour, prompt constraints, citation extraction/rejection of nonexistent, unretrieved, malformed and missing citations, one retry then fallback, no identity or intent wording in template output,
coarse vehicle wording, sessions and follow-ups, summaries, the import boundary (source scan plus a subprocess with LangChain blocked), and the five `security_events` questions.
**Known limitation:** the citation gate checks that ids are real and retrieved, not that the prose around them is true. Real model behaviour will be assessed only after the first live call.

**Live analyst validation (Gemini, `gemini-3.6-flash`, three approved runs of one question; a tiny sample, no reliability statistics implied)**
- Grounded LLM answers with verified citations were produced. The first run was accepted; its call metrics were not yet instrumented.
- One run hit a transient `ReadTimeout` at the configured 60-second boundary. The system withheld the unavailable LLM path and returned the deterministic, evidence-backed fallback, labelled as such,
  with the reason in the notes. I do not treat this as evidence of a systematic Gemini problem, and the timeout value was left unchanged.
- A subsequent run succeeded in 13.02 s: 3 model rounds, 4 tool calls (4 unique tools: `get_entity_history`, `list_entities`, `list_events`, `search_observations`), 0 correction attempts, fallback = no, citations verified.
- Removing the unsupported `temperature` argument from the LangChain chat model made the "fixed sampling defaults" warnings disappear (a test checks that it is not passed).

**Semantic-grounding limitation (an explicit M7 evaluation target)**
Citation validation verifies that cited ids exist, that they were actually retrieved by a tool, that required citations are present, and that selected identity/intent wording is rejected.
It does NOT prove semantic entailment between each natural-language claim and the evidence it cites. Observed live example: the model wrote "vehicle tracking", although the system does not perform
persistent vehicle tracking; it performs coarse attribute-based vehicle grouping. Wording inside a list item can also be less carefully qualified than the surrounding uncertainty statement
(e.g. "a blue pickup" without "similar"). I deliberately did not add a one-off phrase ban for "vehicle tracking": that would be brittle and would hide the general issue. M7 should measure it.

## M7: Evaluation, ablation and grounding-faithfulness harness
**Tool:** Claude Code (Claude Sonnet 5.5), from my M7 spec. No live API call was made.

**What the AI generated:** the `drone_agent/eval/` package (generated frame-gate fixture, pipeline and rule experiments, citation adversarial suite, semantic-faithfulness probe, eval-only structured claim checker,
scale benchmark, runner with an ablation table and JSON/CSV/Markdown output), the `evaluate` command, `docs/evaluation.md` and offline tests (31 for the harness plus 22 regression test cases for the two validator fixes; 509 in total).

**Design decisions the AI made (please review)**
- Ablations live in the harness (`entities=[]`, other `max_missed_frames`, gate threshold 0) so production code is unchanged. The one production-side change is a ruff per-file ignore for long prose strings in `eval/`.
- The gate fixture has exact ground truth and includes two illumination-only steps and two small objects on purpose, to show pixel change is not semantic importance. Thresholds were picked from the observed score distribution; the 6.0 default was not tuned.
- The citation suite runs the real `AnalystSession` with a scripted LLM boundary, so it measures the actual gate. Rule cells are count based (TP/FP/FN per alert, one TN per empty cell) and zero denominators print N/A.
- The semantic probe answers are authored from the evidence model, so its pass percentage characterises a failure boundary rather than estimating a rate. The claim checker only accepts structured claims; I did not attempt free-text extraction.

**Results I did not expect or did not like (kept, not tuned)**
- At the default threshold the gate missed the small and the medium object appearances and accepted both illumination-only steps.
- The citation gate got two of its 29 adversarial decisions wrong on the first run: an unknown citation kind (`[evt:...]`) was invisible, and a negation further than two words before "the same individual" was rejected. After review I fixed exactly these two validator behaviours (unknown citation-shaped tokens now fail; negation is recognised within the same clause up to 8 words before the phrase, with a `doubt` exception) and re-ran the whole evaluation without touching fixtures or methodology. The suite's accuracy is therefore no longer independent evidence, because the validator was corrected against the same cases.
- Every one of the authored citation-valid but unsupported answers passed the gate together with the true controls; the checker fixes only the classes it was built for.
- Entity linking is clearly superlinear at 50,000 observations (probably the per-comparison recomputation of a group's last-seen time). I did not change the linker.

**Problems found and fixed during validation (honest log)**
- A mutation run (8 mutations of the harness, all caught except one that turned out to be a no-op mutation) left stale bytecode: a restored file had the same size and mtime second as the mutated one, so Python imported the mutated `.pyc` and the CLI failed with a KeyError. Clearing `__pycache__` fixed it; I re-ran the whole suite afterwards.
- Two of my patch scripts wrote real newlines into string literals (shell escaping); I fixed them by editing the files directly.

**How it was tested:** determinism (two full runs compared), metric math and zero denominators, gate mechanics, flicker sweep boundary, entity ablation and controls, rule matrix consistency, adversarial categories and internal consistency, probe well-formedness (every citation exists, is retrieved and well formed),
claim checker verdicts, generated observations, scale smoke run and exponent math, JSON/CSV serialization, the CLI, no-network and no-key operation, and the import boundary.

## M8a: Explicit truncation in the offline analyst
**Tool:** Claude Code (Claude Sonnet 5.5), from my M8a spec, after the first real-video (Gemini) run exposed silent omissions.

**What the AI changed:** `agent/offline.py` and `agent/summary.py` only. Every capped list now states what was cut (`showing N of M ..., earliest first; K not shown`), vehicle answers show full history for 3 entities and list the rest briefly, zone answers disclose that observations are a 3-item sample, and a general people-and-vehicles question is answered with both categories. Event aggregation, entity linking, rules, the VLM pipeline and the M7 evaluation are untouched.

**Process:** tests first (29 new tests; 13 failed before the change for exactly the expected reasons: missing notices and compound questions routed to vehicles only), plus 14 byte-for-byte golden answers recorded from the unchanged baseline for non-truncated questions, so unchanged answers are proven unchanged. Fixtures are generated (no video, no API).

**Results:** all M8a tests, the full suite and ruff pass; the M7 evaluation re-run is identical to the frozen baseline in every functional experiment and the ablation table (only timings vary). On the real-clip analysis copy the answers now say "showing 20 of 25 events, earliest first; 5 not shown", "full history for 3 of 11 similar-vehicle entities; 8 more are listed briefly", and report both people and vehicles.

**Honest notes**
- Design choice: a completeness note is shown only when something was cut, so small answers are unchanged.
- The vehicle search still only sees the first 20 entities the tool returns; the answer now says so instead of hiding it. Paging the tool is not implemented.
- I found that opening ANY database with `Store.open` rewrites its file bytes even for read-only questions (schema initialisation). My three demo questions therefore changed the bytes of `output/real/clip_analysis.db`; I verified its tables are logically identical to a fresh rebuild from the untouched `clip.db`, and the video, cassette and `clip.db` are byte-identical. This existing M2 behaviour is not changed here.
- The citation gate still cannot detect omission; the notes are what protects the reader.

## M8b: Real-video experiment write-up (documentation only)
**Tool:** Claude Code (Claude Sonnet 5.5), from my M8b spec. No production code, dependencies, `.env` or real-video artifacts were changed, and no API call was made.

**What the AI wrote:** `docs/real_video_experiment.md` plus a short README section, from the measured results of the completed run. Numbers that could be re-derived were re-checked from the artifacts (clip properties, cassette entries, observation/object/event/entity/alert counts, confidence values, source labels, gate accepted/skipped counts through a free replay).

**Honest notes**
- The figures 8 API calls, 0 retries and 76.1 s total API latency come from the record run's printed summary as provided; latency and retries are not stored in the cassette or database, so I could not re-derive them. The document says so.
- Verifying the replay command showed that replayed observations are identical to the recorded run except the `source` label (`cassette` instead of `gemini:gemini-3.6-flash`); this is documented rather than changed.
- The document deliberately separates what the run validates (end-to-end integration on real video) from what it does not (perception accuracy, security performance).

