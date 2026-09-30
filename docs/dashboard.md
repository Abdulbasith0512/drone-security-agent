# Evaluator dashboard

A small localhost web UI over the existing backend. It adds **no detection logic**: every number, event, alert and
answer is produced by the existing pipeline and read back through `Store`, `AnalystTools` and `AnalystSession`.
It is offline (no model, no network, no API key) and read-only (databases are opened with SQLite `mode=ro`).

```
python -m drone_agent.web --demo                       # builds the demo runs, serves http://127.0.0.1:8765
python -m drone_agent.web --run NAME=DB,kind=KIND[,video=PATH]   # register your own run (repeatable)
```
`KIND` is `scripted`, `synthetic-demo` or `real-video`. It is **declared, never inferred**: a replayed recording and canned demo text
are both stored with the source label `cassette`, so the stored label cannot tell them apart.

## Architecture
```
browser (static/index.html, app.js, app.css)          presentation only: fetch, lay out, cross-link
   |  JSON over http://127.0.0.1
drone_agent/web/server.py     stdlib http.server: fixed routes, static whitelist, Host/Origin checks, body caps, CSP
drone_agent/web/api.py        thin adapter: calls existing code and reshapes it for JSON
   |
Store | AnalystTools | AnalystSession | run_facts | VideoSource | rules config     (existing, unmodified)
```
Other modules: `registry.py` (declared run kinds), `bundle.py` (`--demo`: runs the existing CLI commands in-process), `frames.py` (JPEG of a
video frame), `guard.py` (question filter, below), `__main__.py` (entry point).

## Panels and what feeds them
| Panel | Shows | Existing code behind it |
|---|---|---|
| Pipeline strip | the six stages with live counts; each scrolls to its panel | `agent.summary.run_facts`, rules config |
| Security overview | observations, frames, events, alerts, vehicle entities; severity, rule, zone, object and source breakdowns; limitations; run summary | `run_facts`, `render_summary` |
| Event timeline | per-zone timeline (SVG) plus a table: time, type, subject, zone, duration, event id, alerts, frames; event evidence | `Store.query_events`, `event_view`, `get_event_observations` |
| Alert investigation | rule, severity, zone, alert/event/frame ids, why it was raised, the numbers the rule compared, the evidence chain, the rules in force | `Store.query_alerts`, `AnalystTools.get_alert`, `alert_view`, rules config |
| AI analyst | question box, grounded answer, clickable citations, follow-ups, refusals | `AnalystSession(store, llm=None).ask()` (the deterministic template analyst and its citation check) |
| Video / observations | the real frame for each observation (video runs), description, stored source, carried-forward flag, simulated telemetry, linked events | `Store.query_observations`, `observation_view`, `VideoSource` / `read_frame_at` |

Selecting an event, alert or observation highlights the related items in the other panels; citations in answers open the cited item.

## Real, synthetic and carried-forward data
* **SCRIPTED**: text frames written by the author in a scenario file; no image exists. Deterministic test data, not perception.
* **SYNTHETIC DEMO**: a tiny synthetic video of shapes. The pixels are that file's real frames; the descriptions are canned text replayed from a cassette,
  not the output of any vision model.
* **REAL VIDEO**: one recorded real-video experiment; its descriptions came from a vision model earlier and are stored (nothing is called now).
  The clip is gitignored, so it is not part of a fresh clone; register it with `--run` where it exists.
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
* Binds to loopback only; other hosts are refused. Checks the `Host` and `Origin` headers, serves only a fixed list of static files, accepts only integer frame ids, caps request bodies (8 KiB) and query strings, and sends a strict Content-Security-Policy (no inline script or style, no external origin).
* No authentication and no production deployment: this is a local evaluation tool.
* The page loads nothing from the network; there is no CDN, Node or build step.
* Video frames are decoded on demand from registered videos only; the frame index of an observation is its source video's frame index.
* Limits: a small in-memory cache; per-request database open; the demo runs are small, so the page lists everything (large runs are capped by the backend's own limits).
