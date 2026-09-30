# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment, by Abdul Basith Syed: a docked-drone security analyst that turns video and simulated telemetry into
events, deterministic alerts, searchable history and evidence-cited answers.

Design in one line: `FRAME -> OBSERVATION -> EVENT -> MEMORY -> REASONING`. **Rules decide; the LLM only explains.**

## Contents
- [docs/feature_spec.md](docs/feature_spec.md): value, requirements, assumptions, non-goals
- [docs/architecture.md](docs/architecture.md): architecture diagram and design rationale, layer by layer
- [docs/testing.md](docs/testing.md): how the system was tested, and clean-clone verification
- [docs/evaluation.md](docs/evaluation.md): controlled experiments, ablations, metrics, threats to validity
- [docs/real_video_experiment.md](docs/real_video_experiment.md): one real-video Gemini run through the whole pipeline, its findings and limits
- [docs/ai_usage.md](docs/ai_usage.md): how AI tools were used, and what I changed or rejected

## Setup (Python 3.11+)
```
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,agent]"
pytest -q
```
`.[dev,agent]` installs the test tools and the LangChain analyst adapter. **The core pipeline and the offline analyst work without any Gemini credentials**
(`.[dev]` alone is enough to run them; the `agent` extra is needed for the LangChain adapter tests and for `--provider gemini`). The offline demos below need no key and no network.

**Gemini (optional, configured separately).** Copy `.env.example` to `.env` (gitignored) and set `GEMINI_API_KEY`; optionally set `GEMINI_MODEL` (vision) and `ANALYST_MODEL`
(analyst), or pass `--model`. If a model variable is unset a code default is used; there is no fallback model. The key is read from the environment or `.env` only, never printed or stored,
and Gemini is called only by `describe-frame`, `ingest-video --provider gemini` and `ask/summarize --provider gemini`. Never commit `.env`.

## Demo commands
Generated files (databases, cassettes, results) go under `output/`, which is gitignored.

**1. Scripted scenario, end to end (offline):**
```
python -m drone_agent ingest scenarios/security_events.yaml --db output/security.db
python -m drone_agent build-events --db output/security.db                      # observations -> events -> similar-vehicle entities
python -m drone_agent analyze      --db output/security.db                      # deterministic rules -> alerts
python -m drone_agent alerts       --db output/security.db --show ALR-LOITER-EVT-P-001   # reason, rule facts, evidence frames
python -m drone_agent query        --db output/security.db --search "blue pickup"        # also --zone --object --start --end --explain
python -m drone_agent events       --db output/security.db --evidence EVT-P-001
```
On `security_events` this gives 7 events and 6 alerts (a person loitering at the gate for 480 s, three after-hours person alerts, one restricted-zone entry, and a repeat-vehicle alert
worded "a *similar* blue pickup was observed in 3 separate appearances"); `normal_day` gives none. Ingesting twice into one database is rejected and changes nothing.

**2. Analyst questions, summary and follow-ups (offline template analyst, no key):**
```
python -m drone_agent ask --db output/security.db "What happened at the main gate?"
python -m drone_agent ask --db output/security.db "Why was the repeat-vehicle alert raised?" --follow-up "Where was it seen?"
python -m drone_agent ask --db output/security.db "Show me the evidence for the loitering alert."
python -m drone_agent summarize --db output/security.db --follow-up "What objects were observed?"
```
Answers cite stored evidence as `[event:EVT-V-001]`, `[alert:...]`, `[entity:...]`, `[obs:...]`, `[frame:N]`, and say when a list was cut ("showing N of M ..."). `--provider gemini` (needs `.[agent]` and a key) uses a LangChain
tool-calling loop over the same read-only tools; its answer is shown only if every citation exists and was retrieved (one correction retry, then the template answer).

**3. Real-video pipeline with a cassette (offline):**
```
python -m drone_agent make-demo-video --out-dir output/demo       # tiny SYNTHETIC video + a canned demo cassette (not a vision model)
python -m drone_agent ingest-video output/demo/demo.avi --db output/video.db --provider cassette --cassette output/demo/demo.cassette.json --sample-every 2 --verbose
python -m drone_agent build-events --db output/video.db && python -m drone_agent analyze --db output/video.db
```
For your own clip with Gemini: `python -m drone_agent describe-frame VIDEO --time 4` (one frame, one call), then `ingest-video VIDEO --db ... --provider gemini --cassette-mode record --cassette ...`.
Replaying that cassette (`--provider cassette --cassette ...`) needs no key and makes zero API calls. Exact commands and findings for the recorded clip are in [docs/real_video_experiment.md](docs/real_video_experiment.md).

**4. Evaluation (offline, deterministic, no key):**
```
python -m drone_agent evaluate --output output/eval                      # add --scale [--scale-sizes 100,1000,10000] for the timing benchmark
```

## Architecture overview
Scripted text frames or a real video (sampled, pixel-gated, described by Gemini or a cassette) become **observations** joined to simulated telemetry, indexed in **SQLite + FTS5**. An
**event aggregator** merges repeated detections into events with duration; an **entity linker** groups separate vehicle events into coarse "similar vehicle" entities; four **deterministic rules**
turn events into alerts with evidence ids; a **read-only analyst** (offline template, or LangChain + Gemini behind a citation gate) answers questions and summarises the run. An offline
**evaluation harness** measures the pieces. The full diagram, stage table and package layout are in [docs/architecture.md](docs/architecture.md).

## Real versus simulated
| Real | Simulated or scripted |
|---|---|
| The video file, its pixels, frame sampling and gate scores | Telemetry (position, altitude, battery, zone) from a seeded simulator or a timeline you supply |
| Gemini's descriptions when `--provider gemini` is used | The zone and wall-clock time assigned to each frame (a video carries none) |
| SQLite index, events, entities, alerts (computed by the code) | Scripted scenario frames: text descriptions I authored, used as deterministic ground truth for tests |
| Timings, measured on the development machine | "Real-time": streaming aggregation on a simulated clock, not a live feed |

## Requirements to evidence
| Assignment item | Where it is met |
|---|---|
| Process simulated telemetry and video frames | `simulate`, `ingest` (scripted frames + seeded telemetry); `ingest-video` (real video); `drone_agent/telemetry`, `drone_agent/video` |
| Identify objects/events and log them with context | observations with zone, time and telemetry; events with first/last seen and duration; `events --evidence`; Gemini frame descriptions ([real-video experiment](docs/real_video_experiment.md)) |
| Alerts from predefined rules | four deterministic rules in `config/security_rules.yaml`; `analyze`, `alerts --show` |
| Frame-by-frame indexing (cross-domain) | SQLite `frames`/`observations` with FTS5; `query` by time, zone, object or keyword ("show all truck events" style queries) |
| Feature spec / architecture / design artifacts | [feature_spec.md](docs/feature_spec.md), [architecture.md](docs/architecture.md) (Mermaid diagram) |
| AI-generated components and AI usage | [ai_usage.md](docs/ai_usage.md) |
| Test cases | [testing.md](docs/testing.md), `pytest`, [evaluation.md](docs/evaluation.md) |
| Sample output: logs, alert, queryable frames | demo commands 1 and 2 above |
| VLM script, LangChain agent, context management | `drone_agent/vision/gemini.py`; `drone_agent/agent/` (bounded session memory, tools, citation gate) |
| Bonus: video summary, follow-up questions | `summarize`; `ask --follow-up` and `--interactive` |

## Measured results at a glance
Numbers come from the executable harness and the recorded run; the documents explain each and its limits. Controlled synthetic results and the real-video run are different kinds of evidence.

| Area | Result | Kind of evidence |
|---|---|---|
| Test suite | At the time of this submission, 538 tests pass (all offline) | automated tests ([testing.md](docs/testing.md)) |
| `security_events` scenario | 27 observations, 20 detections, 7 events (ratio 2.86), 2 vehicle entities, 6 alerts; `normal_day`: 0 alerts | controlled scripted scenario |
| Frame gate (default threshold 6.0) | 5 of 40 fixture frames sent (87.5% fewer calls) but only 2 of 4 meaningful changes detected; both illumination-only changes accepted | controlled synthetic frames |
| Flicker tolerance | one undetected frame: 2 events and 0 loitering alerts without tolerance, 1 event and 1 alert with the default | controlled scenario |
| Entity memory | 1 repeat-vehicle alert with linking, 0 without; negative controls show a false-link limitation | controlled scenario |
| Rules | 21 labelled synthetic scenarios all matched their labels (labels and rules share an author) | controlled; not real-world precision/recall |
| Citation gate | 29 of 29 correct on the authored adversarial fixture set. This is not independent evidence of general citation reliability: two validator fixes were made after this same suite exposed them, so the fixtures informed the corrections | controlled adversarial suite |
| Semantic faithfulness probe | all 22 answers with valid citations but unsupported claims were accepted by the gate; an eval-only structured checker catches 13 of them by construction and 9 still pass | controlled probe |
| Scale (one machine) | up to 50,000 generated observations; entity linking is superlinear at that size | timings on the development machine |
| Real video | one 24.68 s clip: 8 Gemini calls, 9 observations, 33 objects, 25 events, 11 coarse vehicle entities, 0 alerts | integration run; **no perception ground truth** |

The real clip validates end-to-end integration on real video. It is a traffic intersection, not security footage; zero alerts is expected because none of the rules could apply given
its length and the simulated time and zone, so it says nothing about detection accuracy.

## Design decisions
- **Rules decide, the LLM explains.** Alerts, severities, events and entities come from deterministic code; the analyst can only read them, and cannot create, suppress or re-rate an alert.
- **Events, not frames.** Repeated detections become events with duration (temporal association, not tracking); separate vehicle events are grouped only as a coarse "similar vehicle" and never claim identity.
- **SQLite + FTS5 as the frame index.** Zero infrastructure and enough for time, zone, class and keyword queries; embeddings and vector search were deliberately left out of scope.
- **A provider-neutral vision boundary with cassettes.** Real models implement one small interface; recorded responses replay with no key, so demos and tests are reproducible. Gemini is optional.
- **A frame-change gate with carry-forward.** Near-identical frames are not sent to the model; their observation repeats the last answer and is labelled as carried (propagation, not a new inference).
- **LangChain only where needed.** It drives the analyst's tool loop in `agent/llm.py`; everything else, including the offline analyst, is plain Python. A citation gate checks ids, and a template answer is the fallback.
- **Nothing silent.** Simulated telemetry, placeholder confidence (0.5) and truncated lists are labelled wherever they appear.
- **Measure, do not tune.** The evaluation harness is offline and deterministic, and negative results were kept rather than tuned away.

## Limitations
- Telemetry, zone and time are simulated; the "real-time" behaviour is streaming on a simulated clock.
- The real-video run is one short clip with no ground truth, so it does not measure perception accuracy or security performance.
- The gate is pixel-based, not semantic: at the default threshold it missed small meaningful changes on the fixture.
- Free-text VLM attributes fragment events in real output, and coarse vehicle matching can link different vehicles; neither is identity.
- Citation validation checks that cited ids exist and were retrieved, not that the surrounding claim is true; unsupported answers with valid citations pass it.
- Rule precision/recall apply only to controlled synthetic scenarios. Timings come from one machine.
- Only a few live Gemini runs were made; no reliability claim is made from them.

## Testing
At the time of this submission, 538 automated tests pass, all offline (SDK and LLM boundaries are mocked; no test needs a key). See [docs/testing.md](docs/testing.md) for the organisation, what is and is not covered, and the clean-clone verification steps.

## AI tools and impact
Claude Code (Claude Sonnet 5.5) generated the code, tests, evaluation harness and documentation from my milestone specifications; I set the scope, made and recorded the design decisions, and directed the
corrections. Gemini is a runtime component, not a development assistant. The milestone-by-milestone record of what was generated, what I changed or rejected, and how each
result was validated is in [docs/ai_usage.md](docs/ai_usage.md).

## Repository safety
The confidential assignment PDF, `.env`, databases, cassettes, videos, evaluation results and caches are gitignored; `.env.example` holds placeholders only.
