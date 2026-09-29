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
