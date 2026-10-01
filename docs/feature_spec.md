# Feature Spec: Drone Security Analyst Agent

## Value proposition for property owners
A docked drone patrols a fixed property every day, but nobody watches hours of footage. The
Analyst Agent turns drone video and telemetry into a short, trustworthy list of things that
matter: **immediate alerts** ("person loitering at main gate, 00:01"), a **searchable history**
("show all truck events", "did the blue pickup come back today?") and a **one-line summary** of the
patrol. Owners get automated monitoring, faster response to real threats, and evidence
(event and frame IDs) behind every claim, so alerts can be checked instead of trusted blindly.

## Key requirements
1. **Understand time, not just frames.** Consecutive frames of the same person or vehicle are merged into
   one *event* with first-seen, last-seen and duration, using drone telemetry (zone, timestamp)
   for context. Repeat visits by a similar vehicle are remembered across the day.
2. **Deterministic, evidence-backed alerts in real time.** A rule engine decides what is an alert
   (camera rules: loitering, after-hours presence, repeat vehicle, restricted-zone entry; drone-safety
   rules over telemetry: low battery, signal loss, altitude anomaly, geofence violation). Every alert cites the
   event, frame or telemetry evidence that triggered it, carries a severity and a recommended action. For a
   simulated patrol (`run`) and for a video processed with `ingest-video --live`, alerts are raised while the run is in progress; a video
   ingested in batch gets its alerts from the separate `build-events` and `analyze` steps. The LLM may explain an alert but never decides it.
3. **Queryable history with grounded answers.** Observations and events are indexed by time, class
   and zone (SQLite + FTS5). The analyst answers follow-up questions and summarises the video,
   citing evidence IDs. Every cited ID must exist and have been retrieved from the stored run, the facts an answer states in checkable forms (zones,
   types, activities, times, durations, counts, severities) must agree with the evidence it cites, and an answer that
   fails these checks is replaced by a template answer built from the stored facts. (The checks do not verify every sentence: a claim
   outside the forms they can read out of the text is not checked; see the limitations in the README.)

## Assumptions
- One fixed property, one drone, one patrol per run. The property has six zones: `main_gate`, `driveway`,
  `parking_area`, `warehouse`, `restricted_zone` and `garage`; a scenario or timeline YAML assigns a zone to each patrol leg.
- An object's zone is the drone's telemetry zone at frame time (the camera faces where the drone is).
- Video timestamps are mapped onto a simulated wall clock so time-of-day rules (midnight) work on any clip.
- VLM output is noisy. Vehicle make/model is low-confidence; matching uses coarse attributes
  (class, colour, body type) and answers say "similar vehicle".
- No face recognition; only descriptive attributes are stored.
- Simulated text-frame scenarios are the deterministic ground truth for tests; real video is
  processed through the same `VisionProvider` interface.
- The prototype is offline/replay ("real-time" = streaming ingest on a simulated clock).
- No API access is assumed: the core pipeline, the tests and the offline demos need no key. Gemini (official SDK) is an optional real vision model and analyst backend; keys come from environment variables (or a local gitignored `.env`) only.

## MVP scope
- Simulated frames + telemetry -> Observation -> SQLite index -> temporal Events -> coarse entity memory.
- Eight rules: loitering, after-hours presence, repeat vehicle, restricted-zone entry (camera) and low battery, signal loss, altitude anomaly, geofence violation (telemetry).
- A progressive patrol run (`run`) that processes telemetry and frames tick by tick and alerts immediately, with an alert lifecycle (open, escalated, acknowledged, resolved).
- Real video through OpenCV and a real VLM behind `VisionProvider` (M5), processed in batch or progressively (`ingest-video --live`) through the same downstream pipeline as a patrol.
- LangChain analyst Q&A with citations and a video summary (M6), with a template fallback when no LLM is configured.
- A read-only evaluator dashboard that opens on a real video bundled with the repository (licensed for redistribution and hosting in this demo), replayed from a recorded cassette with no model call and no key; scripted and synthetic runs are kept as secondary runs. An opt-in public hosting mode serves one hostname, with no authentication.
- CLI/Rich console, pytest suite, scalability measurements, docs, report, voiceover demo.

## Non-goals (postponed)
Embeddings, CLIP, vector databases, rank fusion; crowd/stationary-object rules; YOLO or multi-object tracking;
multi-drone; real MAVLink/RTSP integration; face recognition; Docker/CI; Kafka/Redis/Kubernetes/Neo4j;
LangGraph/multi-agent. (A read-only evaluator dashboard (local by default, with an opt-in public mode), battery/geofence rules and a 10K/100K-row benchmark
were originally postponed and have since been built; a 1M-row run is runnable but has not been measured.)
