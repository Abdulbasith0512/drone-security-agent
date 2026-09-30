# Frontend: the evaluator dashboard UI

Plain HTML, CSS and JavaScript. **No React, no Node/npm, no build step, no CDN.** The page loads nothing from the network.

| File | What it is |
|---|---|
| `index.html` | page layout: the pipeline strip and the five panels |
| `app.js` | all page logic: fetches JSON from the backend and draws the overview, the timeline (SVG), the alert investigation, the analyst chat and the video/observation viewer |
| `app.css` | styling, with light and dark themes |
| `favicon.svg` | the tab icon |

## How it runs
The backend serves these files from this folder. Start it from the repository root:

```
python -m drone_agent.web --demo        # then open http://127.0.0.1:8765
```

## What it talks to
Only the backend's read-only JSON API on the same localhost address (`/api/runs`, `/api/runs/{run}/overview`, `/events`, `/alerts`, `/observations`, `/frames/{n}.jpg`, `/summary`, and `POST /api/runs/{run}/ask`). The endpoint list and what feeds each panel are in [../docs/dashboard.md](../docs/dashboard.md); the server code is in [../drone_agent/web/](../drone_agent/web/).

## What it does NOT do
It contains **no detection logic**: no rules, thresholds, event aggregation, entity linking or security decisions. Every number, event, alert and answer is computed by the backend and only displayed here. Tests check this (`tests/test_web_server.py`: no rule ids or thresholds in the page, no external URLs, no inline scripts).

Real, synthetic and carried-forward data are labelled on the page (**SCRIPTED**, **SYNTHETIC DEMO**, **REAL VIDEO**; **NEW OBSERVATION** versus **CARRIED FORWARD**), and analyst answers are labelled as offline template answers with no model involved.
