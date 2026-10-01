# Evaluator dashboard

A small read-only web UI over the existing backend (localhost by default). It adds **no detection logic**: every number, event, alert and
answer is produced by the existing pipeline and read back through `Store`, `AnalystTools` and `AnalystSession`.
It is offline (no model, no network, no API key) and read-only (databases are opened with SQLite `mode=ro`).

```
python -m drone_agent.web --demo                       # builds three demo runs (real clip first), serves http://127.0.0.1:8765
python -m drone_agent.web --run NAME=DB,kind=KIND[,video=PATH]   # register your own run (repeatable)
```
`KIND` is `scripted`, `synthetic-demo` or `real-video`. It is **declared, never inferred**: a replayed recording and canned demo text
are both stored with the source label `cassette`, so the stored label cannot tell them apart.

## Architecture
```
browser (frontend/index.html, app.js, app.css)         presentation only: fetch, lay out, cross-link
   |  JSON over http (127.0.0.1 by default)
drone_agent/web/server.py     stdlib http.server: fixed routes, static whitelist, Host/Origin checks, body caps, CSP
drone_agent/web/api.py        thin adapter: calls existing code and reshapes it for JSON
   |
Store | AnalystTools | AnalystSession | run_facts | VideoSource | rules config     (existing, unmodified)
```
The UI files live in the top-level [`frontend/`](../frontend/) folder; a map of the backend code is in [`backend/README.md`](../backend/README.md).
Other modules: `registry.py` (declared run kinds), `bundle.py` (`--demo`: runs the existing CLI commands in-process), `frames.py` (JPEG of a
video frame), `guard.py` (question filter, below), `__main__.py` (entry point).

## What the page shows and what feeds it
The page opens on the first run in `GET /api/runs` (the bundled real clip in `--demo`); the run selector lists it as the primary demo and the others as secondary runs, and `#run=NAME` in the address opens a run directly.
Every number, time, description, alert and answer on the page is read from the API; the page holds no results of its own.

| Part of the page | Shows | Existing code behind it |
|---|---|---|
| Header | the product name; status chips: SYSTEM ONLINE (the API answered), the run's kind label, a replay chip derived from each observation's declared provenance (**VLM replay** for a recorded model run, **Canned replay** for canned text, **Scripted text**), and REPRODUCIBLE (shown only when the data is read back from stored output) | `Run` kind, `observation.provenance` |
| Data strip | the kind label, the honest one-sentence description of the data, the run name, the simulated time range and the video's size, frame rate and length; "About this data" adds the zone, object and source breakdowns and the backend's stated limitations | `run_facts`, run registry |
| Video viewer | the real frame of the selected observation (a text card for a scripted run, which has no pixels), with overlays: zone, run kind, current and total video time, frame number, and EVENT DETECTED or OBSERVATION (plus CARRIED FORWARD when the description is a repeat); a timeline of the observations with previous and next, also operable with the arrow keys | `Store.query_observations`, `VideoSource` / `read_frame_at` |
| Totals | observations, events, entities, alerts and VLM frames (frames a recorded model described; "n/a" for a run with no vision model) | `run_facts` |
| Observation strip | a scrolling strip of the sampled frames, each marked NEW or CARRIED, with zone, object counts and a short description | `Store.query_observations` |
| Investigation | the current observation (time, zone, objects, description), the current event (type, start, end, duration, entity, alerts, or "No security rule triggered for this event.") and the evidence behind it (frame, timestamp, source, the event's observations) | `event_view`, `observation_view`, `get_event_observations` |
| Event timeline | per-zone lanes with overlapping events stacked in rows, alert diamonds coloured and labelled by severity, a lane for telemetry alerts that have no camera event, and a table of the events | `Store.query_events` |
| Alert center | one card per alert (severity, rule, status, zone, entity, time) that expands to why it was raised, the numbers the rule compared, the evidence chain and its frames, with "View evidence"; a run without alerts shows "0 ALERTS" as a neutral result; the rules in force | `Store.query_alerts`, rules config |
| Entity memory | vehicles grouped by similar appearance, with appearances, first and last seen, and a reminder that a match is never a confirmed identity | `Store.list_entities` |
| AI analyst | a question box, suggested questions, and answers shown as ANSWER and EVIDENCE / CITATIONS (clickable, grouped by kind), with the answer's label and how it was produced | `AnalystSession(store, llm=None).ask()` (the deterministic template analyst and its citation check) |
| System pipeline | Video, then the backend's six stages with their counts, each linking to its part of the page | `run_facts`, rules config |

Selecting an event, an alert or an observation moves the viewer to its frame and highlights the related items elsewhere on the page; citations in answers open the cited item. Failures are shown where they happen (a panel whose
request failed shows its error with a Retry button while the rest of the page keeps working). The page uses one dark theme, system fonts, keyboard-operable controls with visible focus rings, and a layout that adapts from desktop to phone.

## Real, synthetic and carried-forward data
* **SCRIPTED**: text frames written by the author in a scenario file; no image exists. Deterministic test data, not perception.
* **SYNTHETIC DEMO**: a tiny synthetic video of shapes. The pixels are that file's real frames; the descriptions are canned text replayed from a cassette,
  not the output of any vision model.
* **REAL VIDEO**: a real video whose descriptions came from a vision model earlier and are stored (nothing is called now). `--demo` builds it first, from the clip bundled in
  `demo_assets/real_clip/` (licensed for redistribution and hosting in this demo) and its recorded cassette; no original video URL and no Gemini credentials are needed. Its result is 9 observations
  (8 replayed, 1 carried forward), 33 objects, 28 events, 12 entities and 0 alerts, and there is no ground truth for it. You can register another real run with `--run`.
* Zone, time and telemetry are simulated in every kind, and the page says so.
* **NEW OBSERVATION** means a description was produced for that frame; **CARRIED FORWARD** (`+carried`) repeats the last accepted description for a similar frame and is *not* a new inference.
* Analyst answers are labelled `template answer (deterministic, no LLM)` with a note that no language model, network or Gemini call was involved. Nothing on the page is presented as live Gemini output.

## Identity / intent / criminality guard
The frozen template analyst answers "Is he a criminal?" by listing what happened in a zone, silently skipping the question. `guard.py` therefore
intercepts such questions **before** the analyst is called and shows a fixed, labelled refusal (identity, intent, criminality, danger).
It is a **wording-based filter** (short lists of regular expressions), deterministic and unit-tested. It is **not a semantic safety classifier**:
paraphrases can slip through and a harmless question can, rarely, match. The analyst's own controls are unchanged: answers must cite retrieved evidence,
and identity or intent wording in an answer is rejected by the citation gate.

## API (all read-only except `ask`, which does not write either)
`GET /api/runs`, `/api/rules`, `/api/runs/{run}/overview`, `/events[?zone=&type=]`, `/events/{id}`, `/entities`, `/alerts`, `/alerts/{id}`,
`/observations[?zone=&q=&limit=]`, `/frames/{n}.jpg[?w=]` (only for a run with a registered video), `/summary`; `POST /api/runs/{run}/ask`
with `{"question": "...", "session_id": "..."}`. Follow-ups work through a server-side session (a plain history and context per session id, one run per session, at most 64 kept).

## Safety and limits
* Binds to loopback by default; other hosts are refused. Public hosting is opt-in: `--host 0.0.0.0` is accepted only together with `--public-host HOSTNAME`, and the server then accepts exactly that hostname (and its https origin) and refuses every other Host or Origin. Checks the `Host` and `Origin` headers, serves only a fixed list of static files, accepts only integer frame ids, caps request bodies (8 KiB) and query strings, and sends a strict Content-Security-Policy (no inline script or style, no external origin).
* No authentication and not a production system: this is an evaluation tool. In the opt-in public mode it serves read-only demo data without authentication.
* The page loads nothing from the network; there is no CDN, Node or build step.
* Video frames are decoded on demand from registered videos only; the frame index of an observation is its source video's frame index. Decoding a frame reads the video up to it, so the first request for a
  late frame takes longer (up to about a second on the development machine; not measured on a hosted instance). The page asks for one picture size, through a small queue, so each frame is decoded once.
* Limits: a small in-memory cache; per-request database open; the demo runs are small, so the page lists everything (large runs are capped by the backend's own limits).
