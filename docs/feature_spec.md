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
   (loitering, after-hours presence, repeat vehicle, restricted-zone entry). Every alert cites the event
   and frame IDs that triggered it. The LLM may explain an alert but never decides it.
3. **Queryable history with grounded answers.** Observations and events are indexed by time, class
   and zone (SQLite + FTS5). The analyst answers follow-up questions and summarises the video,
   citing evidence IDs and refusing to invent unsupported facts.

## Assumptions
- One fixed property, one drone, one patrol per run; zones (gate, garage, driveway, perimeter) come
  from a config file.
- An object's zone is the drone's telemetry zone at frame time (the camera faces where the drone is).
- Video timestamps are mapped onto a simulated wall clock so time-of-day rules (midnight) work on any clip.
- VLM output is noisy. Vehicle make/model is low-confidence; matching uses coarse attributes
  (class, colour, body type) and answers say "similar vehicle".
- No face recognition; only descriptive attributes are stored.
- Simulated text-frame scenarios are the deterministic ground truth for tests; real video is
  processed through the same `VisionProvider` interface.
- The prototype is offline/replay ("real-time" = streaming ingest on a simulated clock).
- No API access is assumed; the real VLM/LLM provider is chosen later (M5/M6). Keys come from environment variables only.

## MVP scope
- Simulated frames + telemetry -> Observation -> SQLite index -> temporal Events -> coarse entity memory.
- Four rules: loitering, after-hours presence, repeat vehicle, restricted-zone entry.
- Real video through OpenCV and a real VLM behind `VisionProvider` (M5).
- LangChain analyst Q&A with citations and a video summary (M6), with a template fallback when no LLM is configured.
- CLI/Rich console, pytest suite, scalability measurements, docs, report, voiceover demo.

## Non-goals (postponed)
Web frontend or dashboard; embeddings, CLIP, vector databases, rank fusion; battery/geofence/crowd/stationary
rules; YOLO or multi-object tracking; multi-drone; real MAVLink/RTSP integration; face recognition;
Docker/CI; Kafka/Redis/Kubernetes/Neo4j; LangGraph/multi-agent; 1M-row benchmarking.
