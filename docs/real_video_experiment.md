# Real-video experiment (Gemini through the full pipeline)

This document records ONE experiment: a real video clip sent through
`video -> sampling -> frame gate -> Gemini -> observation -> event -> entity -> rules -> analyst`,
with real vision-model output. Every number below was measured on that single clip. Nothing here is a
performance claim beyond it.

## 1. Purpose and how it differs from the controlled evaluation
| | Controlled evaluation (M7, `security_events` and the labelled fixtures) | Real-video experiment (this document) |
|---|---|---|
| Perception | scripted detections, perfect by construction | real Gemini descriptions of real frames |
| Ground truth | authored, so rule precision/recall and ablations are measurable | none; perception was not scored |
| Question answered | do the deterministic rules, aggregation and gates behave as designed? | does the pipeline run end to end on real footage, and what breaks? |
| Telemetry, zone, time | simulated | simulated (a video carries none) |

The controlled experiments provide the rule-performance measurements (see docs/evaluation.md). This
experiment only closes the gap "scripted detections do not exercise a real vision model".

## 2. Input
| Property | Value |
|---|---|
| File | `output/real/clip.mp4` (local, gitignored, never committed) |
| Size | 10,797,538 bytes |
| Video | H.264 (High), MP4, 1920x1080, 25 fps constant, 617 frames, 24.68 s |
| Audio | none |
| Content | a high-angle view of a street intersection with traffic and pedestrians (a public scene, not fixed-property security footage) |
| Identity used by the cassette | sha256 prefix `d7e3ceeba55902be` plus frame index |

The provenance and licence of the clip are not recorded in the repository; state them wherever the clip is shown.

## 3. Sampling and gating
- Sampling: one frame every 3 s, giving 9 candidates at offsets 0, 3, ..., 24 s (frame indexes 0, 75, ..., 600); all decoded, none failed.
- Gate: mean absolute difference of 64x36 greyscale copies against the last accepted frame, threshold 6.0 (the unchanged default).
  Result: **8 accepted, 1 skipped**. The first frame is always accepted. Scores of the other samples (computed offline with the same gate before recording; the record run's 8 accepted and 1 skipped match) were 22.27 (3 s), 8.76 (6 s), 10.25 (9 s), 7.76 (12 s), 8.05 (15 s),
  10.76 (18 s), **5.36 (21 s, skipped)** and 8.08 (24 s). The scene changes a lot between samples (almost every score is above the threshold), so the gate saved one call here.
- The skipped sample (21 s, frame 525) was **carried forward**: its observation repeats frame 450's answer and is labelled `gemini:gemini-3.6-flash+carried`.
  **A carried observation is temporal propagation of an earlier answer, not a new model inference.**

## 4. Gemini recording
- Model `gemini-3.6-flash`; provider `--provider gemini --cassette-mode record`; structured output validated by the strict `ProviderOutput` contract.
- **8 API calls, 0 retries, 76.1 s total API latency** (about 9.5 s per call), as printed in the record run's summary. Latency is not stored in the cassette or database, so these two figures
  cannot be re-derived from the saved artifacts.
- The cassette holds 8 validated responses (`output/real/clip.gemini.cassette.json`, gitignored). It contains descriptions and objects only, no images, key or absolute paths.
- Stored result: **9 observations** (8 Gemini, 1 carried) and **33 detected objects** (17 person, 16 vehicle). Three of the 33 belong to the carried observation, so the 8 Gemini responses
  contained 30 objects.
- **Telemetry (SIMULATED):** 3 telemetry samples, not read from the video. Start time `2026-09-29T14:00:00+05:30` and zone `main_gate` were supplied on the command line and apply to the whole clip.
- **Confidence:** Gemini reports none, so every stored confidence is the documented uncalibrated placeholder 0.5. It is not a probability.

Objects reported per sample (frame, offset, people / vehicles): 0 (0 s) 2 / 3; 75 (3 s) 3 / 1; 150 (6 s) 3 / 2; 225 (9 s) 3 / 1; 300 (12 s) 1 / 2; 375 (15 s) 1 / 2; 450 (18 s) 1 / 2;
525 (21 s, carried) 1 / 2; 600 (24 s) 2 / 1.

## 5. Downstream results (deterministic stages, no LLM)
| Stage | Result |
|---|---|
| Events | **25** (13 person, 12 vehicle); 33 detections give a compression ratio of 1.32 (the scripted `security_events` scenario gives 2.86) |
| Multi-frame events | only **5** of 25: EVT-P-001 (frames 0, 75, 300, 375; 15 s), EVT-V-009 (300, 375; 3 s), EVT-V-010 (375, 450, 525; 6 s), EVT-P-011 (450, 525; 3 s), EVT-V-011 (450, 525; 3 s) |
| Single-frame events | 20, each lasting 0 s |
| Vehicle entities | **11** coarse "similar ..." groups; 10 have one appearance and 1 (ENT-V-002, a similar white van) has two |
| Alerts | **0** |

**Why zero alerts is expected, and what it does NOT mean.** None of the four camera rules could fire on this input (the run was recorded before the telemetry-safety rules existed, and its telemetry is simulated), for reasons unrelated to what the footage shows:
loitering needs an event of at least 300 s in `main_gate` (the clip lasts 24.68 s and the longest event is 15 s); after-hours presence does not apply because the simulated start is 14:00
(closed hours are 22:00-06:00); the simulated zone is not restricted; and repeat-vehicle needs three separate appearances (the maximum here is two). Zero alerts is therefore not a
security-detection result, not a specificity measurement and not evidence that nothing noteworthy happened. Any alert on this clip would equally be a consequence of the simulated time and
zone, not of the footage.

## 6. Findings from the real output
1. **Event fragmentation from free-text attributes.** Person events require the `clothing` text to match after lower-casing. Gemini's wording varies across frames and objects
   ("dark clothing", "dark coat", "dark coat, jeans", "dark jacket and jeans", "dark suit", "dark jacket, dark trousers"), so most detections start their own event. A frame can also hold up to three
   people and three vehicles, and the observations carry no position, so which object is which cannot be inferred; an event that is not matched for 3 consecutive sampled frames (9 s) closes.
   Result: 20 of 25 events are single-frame.
2. **Inconsistent VLM attributes.** For example, frame 300 lists a white hatchback and a blue vehicle with no body type, frame 375 a blue hatchback and frame 600 a black hatchback; colour and
   body type are not stable descriptions of the same vehicle across samples.
3. **Coarse vehicle-linking limitation.** ENT-V-002 links EVT-V-002 (frame 0, "white van") with EVT-V-010 (frames 375-525) because both normalise to white van. The gap between them is 15 s, and
   the frame-375 description calls it "a white ambulance" while its attribute says `van`. Whether it is one vehicle cannot be determined from this data. The system labels it a coarse, unconfirmed match;
   the case illustrates the false-link risk documented in the M7 threats.
4. **Carried observation.** Frame 525's objects are a copy of frame 450's. Three events (EVT-V-010, EVT-P-011, EVT-V-011) extend over frame 525, so their durations of 6 s, 3 s and 3 s include propagation,
   not a new detection.
5. **No perception ground truth.** Gemini's descriptions were not compared with a human annotation, so this experiment says nothing about how accurate they are.

## 7. Analyst findings (offline template analyst)
Asking the offline analyst about this run exposed silent omission:

| Question | Before M8a | After M8a (commit 0ae7414) |
|---|---|---|
| "What people and vehicles were observed?" | routed to the vehicle intent: showed 3 of 11 vehicle entities, no people, no note | answers both categories: counts (person 17, vehicle 16), "Events by type: person 13, vehicle 12", events and entities, with the limits stated |
| "What happened in the main_gate?" | header "25 event(s)" but only 20 listed, no note | "(showing 20 of 25 events, earliest first; 5 not shown)" and "(showing 3 of 9 observations as a sample)" |
| "What objects were observed?" | the same 20 of 25 events, no note | the same "showing 20 of 25 events" note |
| "Were any vehicles observed more than once?" | 3 of 11 entities, no note | "(showing full history for 3 of 11 similar-vehicle entities; 8 more are listed briefly below)" plus one line per remaining entity |

The run summary already said "showing 20 of 25 events" before M8a; its entity list now carries a note when it is capped. Non-truncated answers are unchanged (checked byte for byte against the baseline).
The vehicle search still only sees the first 20 entities the tool returns, and now says so. The citation gate cannot detect omission, so the notes are what protects the reader.

## 8. Limitations
- A traffic intersection, not fixed-property security footage; the rules were designed for a different scenario.
- 24.68 s and 9 sampled frames: too short for loitering or repeat-vehicle behaviour and too small for any rate.
- Time, zone and telemetry are simulated; the footage contains none of them.
- Confidence is a constant placeholder.
- No ground truth: perception quality was not measured.
- Vehicle identity is coarse: "similar", never confirmed. Gemini's descriptions can be wrong or inconsistent between frames.
- One clip, one model, one run: nothing here supports reliability or accuracy claims.

## 9. Reproducibility
The video, cassette and databases live under `output/real/`, which is gitignored; they are not part of the repository, and no key or secret appears in this document.

Record (the only step that calls Gemini; needs `GEMINI_API_KEY` in the environment or `.env`):
```
python -m drone_agent ingest-video output/real/clip.mp4 --db output/real/clip.db --provider gemini --cassette-mode record --cassette output/real/clip.gemini.cassette.json --sample-every 3 --start 2026-09-29T14:00:00+05:30 --zone main_gate --verbose
```
Replay (free, no key, no network): same video bytes, same `--sample-every 3`, a NEW database path:
```
python -m drone_agent ingest-video output/real/clip.mp4 --db output/real/clip_replay.db --provider cassette --cassette output/real/clip.gemini.cassette.json --sample-every 3 --start 2026-09-29T14:00:00+05:30 --zone main_gate
```
The replay was executed for this document: 8 cassette replays, 9 observations and 33 objects, with observations and detected objects identical to the recorded run except the `source` label, which reads
`cassette` / `cassette+carried` instead of `gemini:gemini-3.6-flash` / `...+carried`.

Inspect offline on a COPY of the database (opening a database with the CLI rewrites its file, and `build-events` and `analyze` write derived tables):
```
python -m drone_agent build-events --db output/real/clip_analysis.db
python -m drone_agent events   --db output/real/clip_analysis.db
python -m drone_agent entities --db output/real/clip_analysis.db
python -m drone_agent analyze  --db output/real/clip_analysis.db
python -m drone_agent ask --db output/real/clip_analysis.db --offline "What people and vehicles were observed?"
python -m drone_agent summarize --db output/real/clip_analysis.db --offline
```
Any change to the clip (trimming, re-encoding) invalidates the cassette, because it is keyed to the file's sha256 and the frame index.

## 10. Interpretation
- **What this validates:** the pipeline runs end to end on real video with real Gemini output: decoding, sampling, the gate, structured output and its validation, the cassette (record and free replay),
  persistence, events, entities, the rules and the analyst all worked on data they had never seen, and the failures that appeared (fragmentation, attribute variance, silent truncation) were visible and diagnosable.
- **What it does NOT establish:** real-world perception accuracy, security-detection performance, alert precision or recall, or that the event and entity model works on crowded scenes. The zero alerts and the
  25 events are outputs of these settings on this clip, not quality scores.
- **Where the performance numbers come from:** the controlled `security_events` and labelled-scenario experiments (docs/evaluation.md) provide the rule-performance measurements, with their own stated limitations.
