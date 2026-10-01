# Drone Security Analyst Agent

Prototype for the FlytBase AI Engineer assignment, by Abdul Basith Syed: a docked-drone security analyst that turns video and simulated telemetry into
events, deterministic alerts, searchable history and evidence-cited answers.

Design in one line: `FRAME -> OBSERVATION -> EVENT -> MEMORY -> REASONING`. **Rules decide; the LLM only explains.**

## Contents
- [Quick start](#quick-start) and [What the prototype does](#what-the-prototype-does)
- [Dashboard](#dashboard-evaluator-ui), the [real-video demo](#real-video-demo) and [public hosting](#public-hosting-opt-in)
- [Command-line demos](#command-line-demos), [architecture](#architecture-overview), [results](#measured-results-at-a-glance), [limitations](#limitations), [testing](#testing)
- [docs/feature_spec.md](docs/feature_spec.md): value, requirements, assumptions, non-goals
- [docs/architecture.md](docs/architecture.md): architecture diagram and design rationale, layer by layer
- [docs/testing.md](docs/testing.md): how the system was tested, and clean-clone verification
- [docs/evaluation.md](docs/evaluation.md): controlled experiments, ablations, metrics, threats to validity
- [docs/real_video_experiment.md](docs/real_video_experiment.md): a real-video Gemini run through the whole pipeline, its findings and limits
- [docs/ai_usage.md](docs/ai_usage.md): how AI tools were used, and what I changed or rejected
- [docs/dashboard.md](docs/dashboard.md): the evaluator dashboard: API, what feeds each panel, safety limits
- [frontend/README.md](frontend/README.md) and [backend/README.md](backend/README.md): where the UI and the backend code live

## Quick start
Python 3.11 or newer.
```
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,agent]"
pytest -q                         # all offline; no key needed
ruff check .                      # lint
ruff format --check .             # formatting
python -m drone_agent.web --demo  # the dashboard, at http://127.0.0.1:8765
```
`.[dev,agent]` installs the test tools and the LangChain analyst adapter. **The core pipeline, the offline analyst, the dashboard and the real-video replay work without any Gemini
credentials** (`.[dev]` alone is enough to run them; the `agent` extra is needed for the LangChain adapter tests and for `--provider gemini`).

**Gemini (optional, configured separately).** Copy `.env.example` to `.env` (gitignored) and set `GEMINI_API_KEY`; optionally set `GEMINI_MODEL` (vision) and `ANALYST_MODEL`
(analyst), or pass `--model`. If a model variable is unset a code default is used; there is no fallback model. The key is read from the environment or `.env` only, never printed or stored,
and Gemini is called only by `describe-frame`, `ingest-video --provider gemini` and `ask/summarize --provider gemini`. The dashboard never reads it. Never commit `.env`.

## What the prototype does
- **Scripted scenarios and real video** become observations joined to simulated drone telemetry and indexed in SQLite + FTS5.
- **Events and entities:** repeated detections merge into events with duration; separate vehicle events are grouped only as coarse "similar vehicle" entities, never as identity.
- **Eight deterministic rules** (four over camera events, four over drone telemetry) raise alerts with evidence ids, severity ladders and a lifecycle (raised, escalated, acknowledged, resolved).
- **Progressive processing:** `run` (scripted scenarios) and `ingest-video --live` (video) consume frames one at a time and save each alert as it fires.
- **A read-only analyst** answers questions and summarises the run, citing stored evidence; follow-ups are supported.
- **An offline evaluation harness** measures the pieces, with a separate opt-in scale benchmark.
- **A read-only dashboard** shows the whole pipeline, with the real video as the default run.

## Dashboard (evaluator UI)
```
python -m drone_agent.web --demo          # builds the demo runs into output/dashboard/, then serves http://127.0.0.1:8765
```
`--demo` builds **three runs** with the existing CLI commands and opens on the first:

| Run | Kind label | What it is |
|---|---|---|
| `real_clip` (opens first) | **REAL VIDEO** | The bundled real clip. Real frames; descriptions replayed from a recorded Gemini cassette. See [Real-video demo](#real-video-demo). |
| `security_events` | **SCRIPTED** | The scripted scenario: authored text frames, no image. Deterministic test data, with alerts. |
| `synthetic_video` | **SYNTHETIC DEMO** | A tiny synthetic video of shapes with canned descriptions from a cassette. Real pixels, not a vision model. |

The page shows: a video viewer with zone, run-kind and time overlays; an observation timeline and strip (each observation marked NEW or CARRIED); an investigation sidebar (current observation, event
and evidence); run totals; an event timeline; an alert center (a run with no alerts shows "0 ALERTS" as a result, not an error); entity memory; the AI analyst with cited evidence; and the pipeline
`Video -> Observations -> Event memory -> Entity memory -> Security rules -> Alerts -> AI analyst`. Zone, time and telemetry are labelled as simulated wherever they appear.

It is **offline and read-only, with no model call and no API key**, and binds to localhost by default. To add your own run, declare what it is (the kind is never guessed):
`python -m drone_agent.web --run NAME=DB,kind=scripted|synthetic-demo|real-video[,video=PATH]`. No new dependencies and no Node/npm: a Python standard-library server plus plain HTML/CSS/JS.
The API and what feeds each panel are in [docs/dashboard.md](docs/dashboard.md).

### Where the frontend and backend are
| | Location |
|---|---|
| **Frontend** (the UI: `index.html`, `app.js`, `app.css`) | [`frontend/`](frontend/), see [frontend/README.md](frontend/README.md) |
| **Backend** (pipeline, rules, store, analyst, CLI) | the Python package [`drone_agent/`](drone_agent/), mapped file by file in [backend/README.md](backend/README.md) |
| **Dashboard server** (serves the frontend, read-only JSON API) | [`drone_agent/web/`](drone_agent/web/) |

```
frontend/    UI (plain HTML/CSS/JS)          backend/    guide to the backend code
drone_agent/ Python backend + dashboard server (web/)     tests/  docs/  config/  scenarios/  bench/  demo_assets/
```

## Real-video demo
The dashboard's default run uses a real video bundled with the repository in [`demo_assets/real_clip/`](demo_assets/real_clip/): the clip, the cassette of recorded vision-model answers,
and a manifest that pins both by SHA-256 (the dashboard refuses to start the run if either file has been altered).

| Fact | Value |
|---|---|
| Clip | `clip.mp4`, 1920x1080, 25 fps, 617 frames, 24.68 s, 10,797,538 bytes |
| Recorded answers | a Gemini 3.6 Flash cassette (`clip.cassette.json`) |
| Replay result | 9 observations (8 replayed and 1 carried forward), 33 objects, 28 events, 12 entities, 0 alerts |

- The video is licensed for redistribution and for hosting in this demo. The bundled replay needs neither the original source URL nor Gemini credentials.
- The frames shown are the clip's real pixels. The descriptions are **replayed from the cassette: no model is called and no API key is used**, and the result is the same every time.
- The start time, the zone and the telemetry are **simulated** (a video carries none of them). There is **no ground truth** for this clip, so nothing here measures whether a description is correct.
- The 0 alerts is what the deterministic rules produced for this footage, not a statement about detection accuracy.

Replay it from the command line:
```
python -m drone_agent ingest-video demo_assets/real_clip/clip.mp4 --db output/real_clip.db --provider cassette --cassette demo_assets/real_clip/clip.cassette.json --sample-every 3 --start 2026-09-29T14:00:00+05:30 --zone main_gate
python -m drone_agent build-events --db output/real_clip.db && python -m drone_agent analyze --db output/real_clip.db
```
Add `--live` to `ingest-video` to process it progressively (no separate `build-events` or `analyze` step). An earlier real-video experiment is written up in
[docs/real_video_experiment.md](docs/real_video_experiment.md).

## Public hosting (opt-in)
The dashboard binds to localhost unless told otherwise. To serve it from a host that gives you a port and a public hostname (for example a Render web service):
```
pip install -e .
python -m drone_agent.web --demo --host 0.0.0.0 --port $PORT --public-host $RENDER_EXTERNAL_HOSTNAME
```
- Public hosting is **opt-in**: `--host 0.0.0.0` is accepted only together with `--public-host`, and the server then accepts exactly that hostname (and its https origin). Any other Host or Origin is still refused.
- **There is no authentication.** The data is read-only demo data and no API key is needed or read.
- No deployed URL is stated in this repository.

## Command-line demos
Generated files (databases, cassettes, results) go under `output/`, which is gitignored.

**1. Scripted scenario, end to end (offline):**
```
python -m drone_agent ingest scenarios/security_events.yaml --db output/security.db
python -m drone_agent build-events --db output/security.db                      # observations -> events -> similar-vehicle entities
python -m drone_agent analyze      --db output/security.db                      # deterministic rules -> alerts
python -m drone_agent alerts       --db output/security.db --show ALR-LOITER-EVT-P-001   # reason, rule facts, evidence frames
python -m drone_agent query        --db output/security.db --search "blue pickup"        # also --zone --object --start --end --explain
python -m drone_agent events       --db output/security.db --evidence EVT-P-001
python -m drone_agent alert-action ALR-LOITER-EVT-P-001 acknowledge --db output/security.db --reason "checked"   # operator action, recorded in the alert's history
```
On `security_events` this gives 27 observations, 20 detections and 7 events (ratio 2.86), 2 vehicle entities and 7 alerts (a person loitering at the gate for 480 s that escalates from warning to critical,
three after-hours person alerts, one restricted-zone entry, a repeat-vehicle alert worded "a *similar* blue pickup was observed in 3 separate appearances", and a drone-safety `signal_loss` alert from a
telemetry dropout); `normal_day` gives none. Ingesting twice into one database is rejected and changes nothing.

**1b. The same scenario as a live, progressive run (offline):** frames and telemetry are consumed one tick at a time, and each alert is saved and printed the moment it fires, not after the run.
```
python -m drone_agent scenarios                                           # the ten shipped scenarios and what each exercises
python -m drone_agent run --scenario security_events --speed 0 --no-live  # add --speed 5 for paced playback; omit --no-live for the live dashboard
python -m drone_agent run --scenario loiter_midnight --speed 0 --no-live  # person at the gate across midnight: warning at 2 min, critical at 5 min
python -m drone_agent run --scenario low_battery --speed 0 --no-live      # emergency: drone-safety alerts from telemetry alone (also signal_loss, geofence_violation)
```
Each run prints the alert lifecycle (raised, escalated, acknowledged, resolved), the recommended action stored with every alert, and a comparison with the scenario's ground-truth labels.

**2. Analyst questions, summary and follow-ups (offline template analyst, no key):**
```
python -m drone_agent ask --db output/security.db "What happened at the main gate?"
python -m drone_agent ask --db output/security.db "Why was the repeat-vehicle alert raised?" --follow-up "Where was it seen?"
python -m drone_agent ask --db output/security.db "Show me the evidence for the loitering alert."
python -m drone_agent summarize --db output/security.db --follow-up "What objects were observed?"
```
Answers cite stored evidence as `[event:EVT-V-001]`, `[alert:...]`, `[entity:...]`, `[obs:...]`, `[frame:N]`, and say when a list was cut ("showing N of M ..."). `--provider gemini` (needs `.[agent]` and a key) uses a LangChain
tool-calling loop over the same read-only tools; its answer is shown only if every citation exists and was retrieved and the factual claims it states agree with the cited evidence (one correction retry, then the template answer).

**3. Video with a cassette (offline):**
```
python -m drone_agent make-demo-video --out-dir output/demo       # tiny SYNTHETIC video + a canned demo cassette (not a vision model)
python -m drone_agent ingest-video output/demo/demo.avi --db output/video.db --provider cassette --cassette output/demo/demo.cassette.json --sample-every 2 --verbose
python -m drone_agent build-events --db output/video.db && python -m drone_agent analyze --db output/video.db
```
For your own clip with Gemini: `python -m drone_agent describe-frame VIDEO --time 4` (one frame, one call), then `ingest-video VIDEO --db ... --provider gemini --cassette-mode record --cassette ...`.
Replaying that cassette (`--provider cassette --cassette ...`) needs no key and makes zero API calls. The bundled real clip is replayed the same way (see [Real-video demo](#real-video-demo)).

**4. Evaluation (offline, deterministic, no key):**
```
python -m drone_agent evaluate --output output/eval                      # add --noise for the perception-noise sweep, --scale for the timing benchmark
python -m drone_agent scale --sizes 10000,100000                         # the opt-in scale benchmark on its own (minutes at 1M)
```

## Architecture overview
Scripted text frames or a real video (sampled, pixel-gated, described by Gemini or a cassette) become **observations** joined to simulated telemetry, indexed in **SQLite + FTS5**. Both the scripted patrol
engine and the progressive video driver feed the same downstream consumer, so the two paths share one pipeline. An **event aggregator** merges repeated detections into events with duration; an **entity linker**
groups separate vehicle events into coarse "similar vehicle" entities; eight **deterministic rules** turn events and telemetry into alerts with evidence ids, severity ladders and a lifecycle (a loitering alert
resolves only when the person's departure was actually observed); a **read-only analyst** (offline template, or LangChain + Gemini behind a citation gate and a factual-claim check) answers questions and
summarises the run. An offline **evaluation harness** measures the pieces, and a read-only **dashboard** (`drone_agent/web/` and `frontend/`) shows the stored runs. The full diagram, stage table and package layout
are in [docs/architecture.md](docs/architecture.md).

## Real versus simulated
| Real | Simulated or scripted |
|---|---|
| The video file, its pixels, frame sampling and gate scores | Telemetry (position, altitude, battery, zone) from a seeded simulator or a timeline you supply |
| Gemini's descriptions when `--provider gemini` is used, and the recorded answers in the bundled cassette | The zone and wall-clock time assigned to each frame (a video carries none) |
| SQLite index, events, entities, alerts (computed by the code) | Scripted scenario frames: text descriptions I authored, used as deterministic ground truth for tests |
| The bundled clip's pixels; its descriptions are a recorded replay, not a live call | "Real-time": streaming processing on a simulated clock, not a live feed |
| Timings, measured on the development machine | The synthetic demo video and its canned cassette (not a vision model) |

## Requirements to evidence
| Assignment item | Where it is met |
|---|---|
| Process simulated telemetry and video frames | `simulate`, `ingest`, `run` (scripted frames + seeded telemetry, batch or progressive); `ingest-video` (real video, batch or `--live`); `drone_agent/telemetry`, `drone_agent/video`, `drone_agent/runtime` |
| Identify objects/events and log them with context | observations with zone, time and telemetry; events with first/last seen and duration; `events --evidence`; Gemini frame descriptions (bundled cassette, [real-video demo](#real-video-demo)) |
| Alerts from predefined rules | eight deterministic rules in `config/security_rules.yaml` (loitering, after-hours presence, repeat vehicle, restricted-zone entry, low battery, signal loss, altitude anomaly, geofence violation); `analyze`, `alerts --show`; `run` raises them progressively |
| Frame-by-frame indexing (cross-domain) | SQLite `frames`/`observations` with FTS5; `query` by time, zone, object or keyword ("show all truck events" style queries) |
| Feature spec / architecture / design artifacts | [feature_spec.md](docs/feature_spec.md), [architecture.md](docs/architecture.md) (Mermaid diagram) |
| AI-generated components and AI usage | [ai_usage.md](docs/ai_usage.md) |
| Test cases | [testing.md](docs/testing.md), `pytest`, [evaluation.md](docs/evaluation.md) |
| Sample output: logs, alert, queryable frames | the command-line demos above, and the dashboard |
| VLM script, LangChain agent, context management | `drone_agent/vision/gemini.py`; `drone_agent/agent/` (bounded session memory, tools, citation gate) |
| Bonus: video summary, follow-up questions | `summarize`; `ask --follow-up` and `--interactive`; the dashboard analyst |

## Measured results at a glance
Numbers come from the executable harness and the test suite; the documents explain each and its limits. Controlled synthetic results and the real-video replay are different kinds of evidence.

| Area | Result | Kind of evidence |
|---|---|---|
| Test suite | 1301 tests, all passing (39 are real-browser tests that need Chrome or Edge and are skipped without one) | automated tests |
| `security_events` scenario | 27 observations, 20 detections, 7 events (ratio 2.86), 2 vehicle entities, 7 alerts; `normal_day`: 0 alerts | controlled scripted scenario |
| Frame gate (default threshold 6.0) | 5 of 40 fixture frames sent (87.5% fewer calls) but only 2 of 4 meaningful changes detected | controlled synthetic frames |
| Flicker tolerance | one undetected frame: 2 events and 2 loitering alerts without tolerance, 1 event and 1 alert with the default | controlled scenario |
| Entity memory | 1 repeat-vehicle alert with linking, 0 without; the negative controls include one false link | controlled scenario |
| Rules | all 22 labelled synthetic scenarios agree with their labels in alert counts (labels and rules share an author) | controlled; not real-world precision/recall |
| Ground-truth scenarios | 10 of 10 satisfy their authored labels and 9 of 10 match them exactly (alert precision, recall and F1 are 1.0 on author-written labels) | controlled; labels and perception are scripted |
| Citation gate | 29 of 29 adversarial cases decided as expected on the authored fixture set; not independent evidence of general citation reliability, since validator fixes followed this same suite | controlled adversarial suite |
| Semantic faithfulness probe | the citation check alone accepts all 22 unsupported answers; with the factual-claim check the production gate rejects 13 and still accepts 9 | controlled probe |
| Scale (one machine) | 10,000 and 100,000 generated observations in [bench/scale_report.md](bench/scale_report.md); entity linking scales superlinearly between them | timings on the development machine |
| Real video | one 24.68 s clip replayed from a recorded cassette: 9 observations (8 replayed, 1 carried), 33 objects, 28 events, 12 entities, 0 alerts | integration replay; **no ground truth** |

The real clip shows the pipeline running end to end on real footage. It is a traffic intersection, not security footage; the zone and time are simulated, and 0 alerts says nothing about detection accuracy.

## Design decisions
- **Rules decide, the LLM explains.** Alerts, severities, events and entities come from deterministic code; the analyst can only read them, and cannot create, suppress or re-rate an alert.
- **Events, not frames.** Repeated detections become events with duration (temporal association, not tracking); separate vehicle events are grouped only as a coarse "similar vehicle" and never claim identity.
- **SQLite + FTS5 as the frame index.** Zero infrastructure and enough for time, zone, class and keyword queries; embeddings and vector search were deliberately left out of scope.
- **A provider-neutral vision boundary with cassettes.** Real models implement one small interface; recorded responses replay with no key, so demos and tests are reproducible. Gemini is optional.
- **A frame-change gate with carry-forward.** Near-identical frames are not sent to the model; their observation repeats the last answer, is labelled as carried (propagation, not a new inference), and is never carried longer than a fixed bound.
- **One downstream pipeline.** The scripted engine and the progressive video driver feed the same consumer, so alerts are produced by the same code on both paths.
- **Observed departure.** A loitering alert is resolved only when the person's departure was observed (or by an operator); the end of observation coverage alone does not resolve it.
- **LangChain only where needed.** It drives the analyst's tool loop in `agent/llm.py`; everything else, including the offline analyst, is plain Python. A citation gate checks ids and a factual-claim check compares stated facts with the cited evidence; a template answer is the fallback.
- **Nothing silent.** Simulated telemetry, placeholder confidence (0.5) and truncated lists are labelled wherever they appear.
- **Measure, do not tune.** The evaluation harness is offline and deterministic, and negative results were kept rather than tuned away.

## Limitations
- Telemetry, zone and time are simulated; "real-time" means streaming processing on a simulated clock, not a live feed.
- The real-video run is one short clip with no ground truth, so it does not measure perception accuracy or security performance. The demo replays recorded Gemini answers; no Gemini inference happens at demo time.
- The gate is pixel-based, not semantic: at the default threshold it missed small meaningful changes on the fixture.
- Free-text VLM attributes fragment events in real output, and coarse vehicle matching can link different vehicles; neither is identity.
- Citation validation checks that cited ids exist and were retrieved; the factual-claim check covers only the claim types it can read out of the answer text. Unsupported answers outside those types still pass (9 of 22 in the probe).
- Rule and alert precision/recall apply only to controlled synthetic scenarios with author-written labels. Timings come from one machine.
- The dashboard has no authentication and is a demonstration tool, not a production system. Video frames are decoded on demand (up to about a second per frame on the development machine; not measured on a hosted instance).
- The browser tests need an installed Chrome or Edge and are skipped otherwise.
- Only a few live Gemini runs were made; no reliability claim is made from them.

## Testing
1301 automated tests pass, all offline (SDK and LLM boundaries are mocked; no test needs a key). 39 of them drive the dashboard in a real headless Chrome or Edge and are skipped if neither is installed.
```
pytest -q
ruff check .
ruff format --check .
```
See [docs/testing.md](docs/testing.md) for the test organisation and the clean-clone verification steps.

## AI tools and impact
Claude Code (Claude Sonnet 5.5) generated the code, tests, evaluation harness and documentation from my milestone specifications; I set the scope, made and recorded the design decisions, and directed the
corrections. Gemini is a runtime component, not a development assistant. [docs/ai_usage.md](docs/ai_usage.md) records, milestone by milestone, what was generated, what I changed or rejected, and how each
result was validated (M0 to M13 and the first audit). The later phases (audit remediation, the progressive video pipeline, the opt-in hosting mode, the bundled real-video demo and the dashboard redesign) were
done the same way and are visible as separate commits in the git history.

## Demo videos
Demo video links will be added before submission.

## Repository safety
- The confidential assignment PDF, `.env`, databases, generated outputs (`output/`), caches and all other videos are gitignored; `.env.example` holds placeholders only. No key is stored in the repository.
- The one bundled exception is `demo_assets/real_clip/` (the licensed clip, its cassette and a manifest), pinned by SHA-256.
