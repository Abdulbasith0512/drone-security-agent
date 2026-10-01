# Frontend: the evaluator dashboard UI

Plain HTML, CSS and JavaScript. **No React, no Node/npm, no build step, no CDN.** The page loads nothing from the network.

| File | What it is |
|---|---|
| `index.html` | the page structure: header with status chips and the run selector, the data-kind strip, the video investigation panel with its sidebar, the event timeline, the alert center, entity memory, the analyst and the pipeline |
| `app.js` | all page logic: fetches JSON from the backend, keeps one selection state (observation, event, alert) and draws every part of the page, including the timeline (SVG) and the icons |
| `app.css` | the styling: one dark theme built on CSS variables (colour is used for status and severity only; object types are told apart by shape and text), with breakpoints for tablet and phone |
| `favicon.svg` | the tab icon |

The backend serves exactly these four files from a fixed whitelist, so the UI stays in these three files plus the icon.

## How it runs
The backend serves these files from this folder. Start it from the repository root:

```
python -m drone_agent.web --demo        # then open http://127.0.0.1:8765
```
It opens on the first run the API lists (the bundled real clip in `--demo`); the run selector switches runs, and `#run=NAME` in the address opens one directly.

## What it talks to
Only the backend's read-only JSON API on the same address (`/api/runs`, `/api/rules`, `/api/runs/{run}/overview`, `/events`, `/alerts`, `/entities`, `/observations`, `/frames/{n}.jpg`, `/summary`, and `POST /api/runs/{run}/ask`).
The endpoint list and what feeds each part of the page are in [../docs/dashboard.md](../docs/dashboard.md); the server code is in [../drone_agent/web/](../drone_agent/web/).

## What it does NOT do
It contains **no detection logic**: no rules, thresholds, event merging, entity linking or security decisions, and no result of any particular run is written into the page. Every number, time, description, alert and answer is read from the API and only displayed. Tests check this
(`tests/test_web_server.py`: no rule ids or thresholds in the page, no external URLs, no inline scripts; `tests/test_web_ui.py`: no run results hardcoded, no inline styles, no data URLs).

## Constraints it is built under
- The server sends a strict Content-Security-Policy (`default-src 'none'`; scripts, styles, images and connections from the same origin only). So there are no inline styles or scripts, no `data:` images and no web fonts: icons are drawn with the DOM and the fonts are the system fonts.
- Pictures use one size and are fetched through a short queue (the viewer's frame first, thumbnails as they scroll into view), because the server decodes a video frame by reading the video up to it.
- Colours are checked for WCAG AA contrast by a test; controls are real buttons with visible focus rings and keyboard operation; motion is reduced when the visitor asks for it.

## Honest labelling
Real, synthetic and carried-forward data are labelled on the page (**SCRIPTED**, **SYNTHETIC DEMO**, **REAL VIDEO**; **NEW** versus **CARRIED FORWARD**), the replay chip follows each observation's declared provenance (a synthetic demo is never called a vision model), zone, time and telemetry are marked as simulated, a run
with no alerts shows "0 ALERTS" as a result and not an error, and analyst answers show how they were produced (the offline template analyst, with no model involved).

## Tests
`tests/test_web_ui.py` (static checks) and `tests/test_web_ui_browser.py` (a real headless Chrome or Edge, skipped when none is installed) check this folder; see [../docs/testing.md](../docs/testing.md).
