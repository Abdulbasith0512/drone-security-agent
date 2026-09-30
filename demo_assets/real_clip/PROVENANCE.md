# Real-video demo assets

These files make `python -m drone_agent.web --demo` open on a real video. The dashboard shows the
real frames of the clip; the descriptions come from a recorded vision-model run and are replayed
from a file. No model is called, no network is used and no API key is needed.

## The clip

| | |
|---|---|
| File | `clip.mp4`, the original file, unmodified (not trimmed, re-encoded or re-muxed) |
| Size | 10,797,538 bytes |
| SHA-256 | `d7e3ceeba55902bee5f6c9efbe68c8799266e699c7c5cb2d1a7180bcc7027ea2` |
| Video | H.264, 1920x1080, 25 fps, 617 frames, 24.68 s |
| Licence | The project owner states that this clip is licensed for redistribution and for hosting in this demo. |
| Source, author, licence name, licence URL, retrieval date | Not recorded in this repository. Nothing is asserted about them here. |

The container carries no GPS, author or camera tags; its only metadata is the FFmpeg/x264 encoder
tag. `manifest.json` pins the exact bytes by SHA-256, and the dashboard refuses to start the run if
the clip does not match.

## The cassette

`clip.cassette.json` holds the answers a vision model (`gemini:gemini-3.6-flash`) gave for 8 frames
of this clip when they were recorded (frames 0, 75, 150, 225, 300, 375, 450 and 600). It is keyed to
the clip's SHA-256 and the frame index, so it replays only against this exact file. It contains
descriptions and detected objects only: no images, no key and no local paths.

## What the demo does and does not claim

- Frames are sampled every 3 seconds; one near-identical frame (frame 525) reuses the previous
  answer and is labelled `cassette+carried`.
- The start time (`2026-09-29T14:00:00+05:30`), the zone (`main_gate`) and all telemetry are
  simulated; a video carries none of them.
- Confidence is the uncalibrated placeholder 0.5, because the model reports none.
- There is no ground truth for this clip: nothing here measures whether a description is correct.
  The run shows that the pipeline works end to end on real footage and gives the same result every
  time, not that the model is accurate.
- The deterministic rules raise no alert on this footage, so the Alerts panel is empty. That is the
  real result, not a fault. The scripted and synthetic runs remain in the dashboard for alerts.
