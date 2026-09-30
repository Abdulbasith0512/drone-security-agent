# Evaluation, ablation and grounding-faithfulness harness (M7)

This document explains how the architecture was evaluated and where it fails. The harness is
executable (`python -m drone_agent evaluate`), deterministic, offline, and needs no API key: no
number below is typed by hand. **Measured results, interpretation and limitations are kept apart**
for every experiment.

## How to reproduce
```
python -m drone_agent evaluate --output output/eval                    # functional experiments
python -m drone_agent evaluate --output output/eval --scale --scale-sizes 100,1000,10000,50000
```
Outputs (gitignored): `results.json` (everything), `results.csv` (all experiment rows),
`ablation_table.csv`, `summary.md`. The snapshot in the last section of this file was produced by
the second command; timings will differ on another machine, everything else must not.

## Qualification that applies to everything below
Controlled synthetic fixtures only. Rule precision/recall here measure agreement with labels written
by the same author, on scripted detections, and say nothing about real-world security performance
or vision-model perception accuracy. The evaluation makes no live model or network call.

## Research questions and experiments
| RQ | Question | Experiment (module) | Method |
|---|---|---|---|
| RQ1 | Vision-model calls avoided by the frame gate, and what it misses | `gate_exp` | 40 generated frames with exact ground-truth transitions (large, medium and small object changes, illumination-only steps), real `FrameGate` at 8 thresholds |
| RQ2 | Compression of repeated evidence into events | `pipeline_exp.run_compression` | production aggregator on `security_events`, `normal_day` and a controlled 60-frame dwell; evidence-retention invariants |
| RQ3 | Flicker robustness | `pipeline_exp.run_flicker` | `max_missed_frames` 0 (ablated) vs the default, all else equal; the loitering episode with one undetected frame; a 0-4 missed-frame gap sweep |
| RQ4 | What entity memory adds | `pipeline_exp.run_entity_memory` | events with coarse linking vs `entities=[]` (harness only); seven vehicle-linking negative and positive controls |
| RQ5 | Rule correctness without an LLM | `rules_exp` | 22 labelled scenarios x the 4 camera rules; count-based TP/FP/TN/FN; N/A for zero denominators (the 4 telemetry safety rules are scored by the ground-truth experiment, RQ8, and their own tests) |
| RQ6 | Citation gate | `grounding_exp.run_adversarial` | 29 adversarial analyst outputs through the real `AnalystSession` (scripted LLM boundary) |
| RQ7 | Semantic faithfulness | `grounding_exp.run_semantic_probe`, `claims` | answers with valid citations but unsupported claims; then an eval-only structured claim checker |
| RQ8 | Streaming alerts against author-written ground truth | `ground_truth` | 10 scenario files with `expect:` blocks run headless through the streaming engine; alert precision, recall and F1 (2TP / (2TP + FP + FN), so TP = 0 with misses scores 0, not N/A), severity ladder, timing and detection latency |
| RQ9 | Adaptive sampling versus a pixel gate | `sampler_exp` | generated streams with known meaningful changes; calls saved and changes missed, against a pixel gate and a fixed-refresh baseline |
| RQ10 | Perception noise | `noise_exp` (`evaluate --noise`) | how precision, recall and latency degrade as scripted detections are dropped, mislabelled or delayed |
| - | Scalability | `scale` | generated observations, per-stage median timings |

## Metrics
- **Frame gate:** accepted/skipped candidate frames (a proxy for vision-model calls), call reduction %, and recall over
  meaningful transitions (a transition counts as detected when the transition frame itself is accepted). Thresholds are 0, one just above
  the measured noise floor, values bracketing the small and medium object scores, the default, and two higher ones, chosen from the
  observed score distribution and not tuned; the default stays 6.0.
- **Compression ratio:** detections per event. It is descriptive, not an accuracy measure.
- **Rules:** each (scenario, rule) pair is a cell. Matched alerts are TP, surplus alerts FP, missing alerts FN, and an empty-vs-empty cell is one TN.
  Precision = TP/(TP+FP), recall = TP/(TP+FN); a zero denominator is reported as N/A.
- **Citation suite:** decision accuracy on this fixture set only (accept/reject compared with the label).
- **Citation-valid / semantically-unsupported probe:** the share of authored answers, whose citations all exist, were retrieved and are well
  formed but whose claims are false or overreaching, that the production gate still accepts. This is a characterisation of a failure boundary, not hallucination-detection accuracy.
- **Claim-evidence checker (prototype, eval-only):** exact comparison of structured claims (event zone, type, activity, duration, start time,
  alert severity and rule, entity appearance count) with stored evidence. It does not read prose and does not establish semantic entailment.

## Findings that matter (qualitative; numbers are in the generated tables)
- **Gate:** it removes most calls on a mostly static sequence, but it is a pixel test. At the default threshold it misses meaningful changes that are small in
  pixels (the small and medium objects) while accepting illumination-only changes, and a missed change is silently absorbed. A low threshold just above the noise floor recovers
  them on this clean fixture at a small cost in calls; whether that holds on real footage is untested, and the default was intentionally not tuned to this fixture.
- **Aggregation and flicker tolerance:** repeated detections become a handful of bounded events with every detection retained as evidence. One undetected frame splits a
  continuous episode when tolerance is 0 and, because loitering is a duration test, removes the loitering alert; the tolerance is bounded (a longer gap still splits).
- **Entity memory:** the repeat-vehicle rule and "was it seen more than once" depend on it. The controls also show its limit: two genuinely different vehicles with the same coarse attributes are linked
  (a false link), which is why every wording says "similar" and never identity.
- **Rules:** all labelled cells agree with the documented behaviour. That is a check of boundaries (inclusive threshold, closed-window edges, midnight crossing,
  missing-evidence non-link), not evidence of field performance.
- **Citation gate:** it rejects nonexistent, unretrieved, missing and phrase-listed overclaims. The first run of this suite exposed two wrong decisions: an unknown citation type such as `[evt:...]` was silently ignored (a false accept)
  and a negation more than two words before "the same individual" was not recognised (a false reject). Both were fixed in the validator (unknown citation-shaped tokens now fail validation; negation is recognised within the same clause up to 8 words before
  the phrase, syntactically only) and the fixtures were left unchanged. Because the validator was corrected against these same cases, the accuracy in the tables below is no longer an independent measure; unit tests cover the fixed behaviours separately.
- **Semantic faithfulness:** the gate cannot tell a true answer from a false one when the citations are valid: every authored unsupported answer passed, alongside the true controls. The narrow structured checker
  catches the claim classes it covers (by construction) and nothing else; capability overclaims such as "tracking", identity wording outside the phrase list, intent inference and claims that the
  analyst created an alert remain unchecked. General semantic entailment is not attempted and remains an open problem for this system.
- **Scale:** event aggregation grows close to linearly, rule evaluation and the list queries a little above linear, and the SQLite keyword and filtered queries stay in milliseconds (exponents are in the scale table). Entity linking is clearly superlinear at the largest sizes; reading the code, the linker recomputes each candidate
  group's last-seen time (a maximum over the group's events) for every comparison, which would explain it. That is an observation, not a fix: M1-M6 behaviour was not changed to improve benchmark numbers.

## Threats to validity
1. Synthetic scenarios are not real surveillance distributions (no weather, crowds, shadows, camera motion, long-tail behaviour).
2. Scripted detections do not measure vision-model perception accuracy; a real model's misses, hallucinated objects and attribute drift are not represented.
3. MAD frame gating is pixel-based, not semantic; the fixture has a smooth background with clean dark rectangles, so real thresholds and noise floors will differ.
4. Coarse vehicle matching can merge visually similar vehicles (shown by a control) and split one vehicle described with different colours.
5. Controlled rule precision/recall does not imply field performance; labels and rules share an author and therefore share blind spots.
6. Citation validity does not guarantee semantic entailment; the probe answers are authored from the evidence model, not sampled from a real model, so the pass rate is a boundary, not a rate.
7. The small live Gemini analyst sample (three approved runs of one question) cannot support any reliability claim; one run timed out and fell back, one wording flaw ("vehicle tracking") passed the gate.
8. Timings come from one machine and one session, use in-process medians and single SQLite ingest runs, and exclude vision-model latency.
9. The citation validator was corrected after this adversarial suite exposed two wrong decisions, so its accuracy on the suite is not independent evidence; the two fixes are narrow and syntactic and do not address semantic faithfulness.
10. The eval-only claim checker is validated on claims written to fit its classes, so its perfect catch rate on them says nothing about generalisation.

## Results snapshot (generated; regenerate to refresh)

Generated by `python -m drone_agent evaluate` (harness v2).

- Controlled synthetic fixtures only: these results do not estimate real-world security precision, recall or perception accuracy.
- Scripted detections do not measure vision-model perception accuracy.
- No live model or network call is made by the evaluation.
- Timing results (scale) are measurements on one machine, not universal claims.

### Ablation table

| component | metric | baseline_setting | baseline | ablated_setting | ablated | effect | interpretation |
|---|---|---|---|---|---|---|---|
| Frame gate | vision-model calls per 40 candidate frames (proxy) | gate threshold 6 (default) | 5 | gate off (threshold 0) | 40 | -35 calls (87.5% fewer) | fewer calls, paid for by the transitions listed in the next row |
| Frame gate | meaningful scene transitions detected | gate threshold 6 (default) | 2/4 | gate off (threshold 0) | 4/4 | -2 transitions | missed: small object appears, medium object appears; pixel change is not semantic importance |
| Temporal aggregation | evidence units (security_events) | events (aggregation on) | 7 | raw frame-level detections | 20 | -13 units (ratio 2.857) | repeated detections become bounded events; every detection stays linked as evidence (True) |
| Flicker tolerance | events from one loitering episode | max_missed_frames=2 (default) | 1 | max_missed_frames=0 | 2 | +1 | one undetected frame splits one continuous episode |
| Flicker tolerance | loitering alerts from that episode | max_missed_frames=2 (default) | 1 | max_missed_frames=0 | 2 | +1 | one undetected frame splits the episode into fragments that each alert at the warning level instead of one escalated alert |
| Flicker tolerance | highest loitering severity from that episode | max_missed_frames=2 (default) | critical | max_missed_frames=0 | medium | critical -> medium | the split fragments never reach the critical threshold, so the escalation is lost |
| Adaptive sampling | meaningful changes detected within 5 frames (generated streams) | adaptive (pixel + telemetry + event watch + refresh) | 0.7556 | pixel gate only (previous default) | 0.4444 | -0.311 | the pixel gate misses 31 small and 19 medium changes that the adaptive sampler mostly catches |
| Adaptive sampling | model calls per 600-frame stream | adaptive | 36.5 | pixel gate only (previous default) | 11.375 | -25.1 calls | recall is bought with calls: never quote the saving without the miss rate in the row above |
| Adaptive sampling | model calls per detected change | adaptive | 4.29 | pixel gate + refresh every 15 frames | 5.69 | +1.40 | against the fair cheap baseline (a fixed refresh), not a straw man |
| Streaming alerts with ground truth | alert F1 (perfect scripted perception) | 10 scenarios, 18 expected alerts | 1.0 | n/a | n/a | latency median 0.0 s, max 10.0 s | 1.0 by construction (author-written labels, perfect input); the noise sweep shows how it degrades |
| Entity memory | repeat-vehicle alerts (security_events) | coarse linking on (default) | 1 | each vehicle event independent | 0 | -1 | linking is what lets the repeat rule and 'seen again' questions work; false links exist (1 in the controls) |
| Entity memory | multi-appearance vehicle entities | coarse linking on (default) | 1 | each vehicle event independent | 0 | -1 | linking is what lets the repeat rule and 'seen again' questions work; false links exist (1 in the controls) |
| Citation validation | invalid outputs accepted (of 19 adversarial) | citation gate on | 0 | no gate | 19 | -19 | gate decision accuracy on this fixture set: 1.0; misses: none |
| Claim-consistency prototype (eval-only) | semantically unsupported answers accepted (of 22) | citation gate + structured claim checker | 9 | citation gate alone (production) | 22 | -13 | covers only structured claim classes; remaining passes: boundary_overclaim, capability_overclaim, identity_evasion, intent_evasion |

### Frame-gate threshold ablation

**Research question.** RQ1: how many vision-model calls does the gate avoid, and what meaningful transitions does it miss?

**Setup.** 40 generated 320x180 candidate frames with seeded sensor noise (sigma 1.5); 6 known transitions (4 meaningful object changes of size 20.8%, 0.44%, 2.8% and 20.8% of the frame, 2 illumination-only steps). The real FrameGate runs at thresholds [0.0, 0.5, 1.0, 3.0, 4.0, 6.0, 12.0, 24.0]; no model is called.

**Measured result**

| threshold | accepted | skipped | call_reduction_pct | meaningful_detected | meaningful_missed | meaningful_recall | irrelevant_accepted |
|---|---|---|---|---|---|---|---|
| 0.0 | 40 | 0 | 0.0 | 4 | 0 | 1.0 | 2 |
| 0.5 | 7 | 33 | 82.5 | 4 | 0 | 1.0 | 2 |
| 1.0 | 6 | 34 | 85.0 | 3 | 1 | 0.75 | 2 |
| 3.0 | 6 | 34 | 85.0 | 3 | 1 | 0.75 | 2 |
| 4.0 | 5 | 35 | 87.5 | 2 | 2 | 0.5 | 2 |
| 6.0 | 5 | 35 | 87.5 | 2 | 2 | 0.5 | 2 |
| 12.0 | 4 | 36 | 90.0 | 2 | 2 | 0.5 | 1 |
| 24.0 | 3 | 37 | 92.5 | 1 | 3 | 0.25 | 1 |

**Interpretation**

- Threshold 0 sends all 40 of 40 candidate frames to the vision model; the default 6 sends 5 (87.5% fewer calls) while detecting 2 of 4 meaningful transitions (recall 0.500).
- Missed at the default threshold: small object appears, medium object appears. A missed change is absorbed silently: the reference frame is not updated, so the vision model is never asked about it.
- Pixel change and semantic importance are not equivalent: the small meaningful object scores 0.952 while an irrelevant illumination step scores 6.196 (+8) and 19.908 (+25); 2 of 2 illumination-only changes are accepted (wasted calls) at the default threshold.
- Static-frame noise peaks at score 0.388, so thresholds above that separate 'no change' from any real change.

**Limitations**

- One synthetic 40-frame sequence with a flat background and dark rectangles; real scenes have far richer pixel statistics (shadows, foliage, weather, camera shake).
- MAD on a 64x36 greyscale copy is pixel-based, not semantic; a small person far from the camera can fall below any threshold that also suppresses illumination flicker.
- Thresholds were chosen from the observed score distribution (see transition_scores); the default 6.0 was NOT tuned to this fixture.
- Transition 'detected' means the transition frame itself was accepted.

### Temporal compression: observations to events

**Research question.** RQ2: how much does aggregation compress repeated evidence while preserving temporal structure?

**Setup.** The production EventAggregator (default config) over three fixtures: the security_events and normal_day scenarios and a controlled 60-frame dwell.

**Measured result**

| fixture | observations | detected_objects | events | compression_ratio |
|---|---|---|---|---|
| security_events | 27 | 20 | 7 | 2.857 |
| normal_day | 11 | 11 | 5 | 2.2 |
| long_dwell_controlled | 60 | 90 | 2 | 45.0 |

**Interpretation**

- security_events: 20 detections became 7 events (ratio 2.857); every detection is still linked to an event as evidence (True) and every event span equals the span of its evidence (True).
- long_dwell_controlled: 90 repeated frame-level detections collapsed into 2 events (ratio 45.000) lasting [590.0, 290.0] s: repetitive evidence became bounded temporal units.
- The ratio only describes how much repetition was merged; it says nothing about whether the events are correct (that is measured separately by the rule and flicker experiments).

**Limitations**

- Scripted detections are perfect and repetitive by construction; real detector output is noisier, so real compression and split behaviour will differ.
- Compression ratio is not accuracy. A wrong merge also raises it.

### Flicker-tolerance ablation (max_missed_frames)

**Research question.** RQ3: does max_missed_frames stop one missed detection from splitting a continuous event?

**Setup.** Same observations and every other setting identical; only max_missed_frames varies ([0, 2]; default 2). Fixtures: the security_events loiterer (frame 5 has no detection), the whole scenario, and a generated gap sweep (0-4 consecutive missed frames).

**Measured result**

| fixture | max_missed_frames | person_events | durations_s | loitering_alerts | loitering_severities |
|---|---|---|---|---|---|
| gap=0 | 0 | 1 | [420.0] | 1 | ['critical'] |
| gap=0 | 2 | 1 | [420.0] | 1 | ['critical'] |
| gap=1 | 0 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |
| gap=1 | 2 | 1 | [480.0] | 1 | ['critical'] |
| gap=2 | 0 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |
| gap=2 | 2 | 1 | [540.0] | 1 | ['critical'] |
| gap=3 | 0 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |
| gap=3 | 2 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |
| gap=4 | 0 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |
| gap=4 | 2 | 2 | [180.0, 180.0] | 2 | ['medium', 'medium'] |

**Interpretation**

- security_events frames 1-9 (one continuous loitering episode, one undetected frame): max_missed_frames=2 gives 1 event(s) [480.0] s and 1 loitering alert(s) ['critical']; max_missed_frames=0 gives 2 event(s) [180.0, 180.0] s and 2 loitering alert(s) ['medium', 'medium'].
- Loitering is decided from an event's duration, so a split changes downstream security reasoning: fragments that each fall under a threshold lose that alert, and here each fragment is long enough to warn but too short to reach the critical rung.
- Gap sweep (default setting): a gap of up to 2 consecutive missed frame(s) kept one event; a gap of 3 split it (the tolerance is bounded on purpose, so a person who really left is not merged forever).

**Limitations**

- Missed frames are inserted by hand into scripted data; real detector misses are correlated (occlusion, lighting) and come in longer runs.
- The result depends on frame spacing (here 60 s): tolerance counts frames, while max_gap_seconds counts time, and either can close an event.
- Tolerance can also bridge a person leaving and a different, similar person arriving; events never claim identity.

### Coarse vehicle entity memory ablation and negative controls

**Research question.** RQ4: what does coarse vehicle linking add compared with treating every vehicle event independently?

**Setup.** production events with EntityLinker (baseline) versus entities=[] (ablation, done in the harness, production untouched), plus seven vehicle-linking controls checked against the linker's documented contract and against the fixture's ground truth.

**Measured result**

| setting | vehicle_events | vehicle_entities | multi_appearance_entities | repeat_vehicle_alerts | actual_linked | contract_respected | false_link |
|---|---|---|---|---|---|---|---|
| linking on | 4 | 2 | 1 | 1 |  |  |  |
| linking off | 4 | 0 | 0 | 0 |  |  |  |
| control: different colour |  |  |  |  | False | True | False |
| control: different body type |  |  |  |  | False | True | False |
| control: missing body type |  |  |  |  | False | True | False |
| control: conflicting explicit plates |  |  |  |  | False | True | False |
| control: two real pickups, same coarse attributes (false-link limitation) |  |  |  |  | True | True | True |
| control: same plate, different colour (explicit plate evidence) |  |  |  |  | True | True | False |
| control: overlapping in time (cannot be one vehicle) |  |  |  |  | False | True | False |

**Interpretation**

- security_events: with coarse linking 4 vehicle events form 2 entities (1 with several appearances) and 1 repeat-vehicle alert(s); without linking there are 0 entities and 0 repeat-vehicle alert(s).
- The 3 pickup sightings (blue, dark blue, blue; zones parking_area, driveway, main_gate) map to entities ['ENT-V-001'] with linking and to ['no entity'] without it: the capability gained is answering 'was a similar vehicle seen more than once' and firing the repeat rule.
- Negative controls: the documented contract was respected in 7 of 7 cases. Coarse matching can still merge genuinely different vehicles with identical coarse attributes (false links in this fixture set: ['two real pickups, same coarse attributes (false-link limitation)']); this is an inherent limit of attribute-only matching, which is why the wording is 'similar vehicle' and never identity.

**Limitations**

- This is coarse attribute grouping of separate events. It is not re-identification and not tracking; there is no appearance model.
- Ground truth 'same vehicle' is authored in the fixture; real colour descriptions from a vision model vary with lighting, which can cause both false links and missed links.
- Only seven controls: they check the documented contract, not the false-link rate.

### Rule evaluation on a labelled scenario matrix

**Research question.** RQ5: do the deterministic rules produce the expected alerts on controlled labelled scenarios without an LLM?

**Setup.** 22 authored scenarios with expected alert counts per rule, run through the production aggregator, linker and rule engine (default configs). security_events and normal_day are reported separately as development fixtures.

**Measured result**

| rule | tp | fp | tn | fn | precision | recall |
|---|---|---|---|---|---|---|
| loitering | 4 | 0 | 18 | 0 | 1.0 | 1.0 |
| after_hours_presence | 5 | 0 | 17 | 0 | 1.0 | 1.0 |
| repeat_vehicle | 1 | 0 | 21 | 0 | 1.0 | 1.0 |
| restricted_zone_entry | 3 | 0 | 19 | 0 | 1.0 | 1.0 |

**Interpretation**

- On the 22 labelled scenarios (88 scenario-rule cells) the rules matched the labels exactly in 22 scenarios; mismatching scenarios: none.
- Per rule (TP/FP/TN/FN, precision, recall): loitering 4/0/18/0 P=1.000 R=1.000; after_hours_presence 5/0/17/0 P=1.000 R=1.000; repeat_vehicle 1/0/21/0 P=1.000 R=1.000; restricted_zone_entry 3/0/19/0 P=1.000 R=1.000. N/A means a zero denominator.
- Rules are deterministic thresholds over events, so they need no LLM; agreement with the labels mainly confirms the implemented boundaries (inclusive 300 s, closed-window start inclusive and end exclusive, midnight crossing, missing-evidence non-link).
- Controlled synthetic fixtures only: these results do not estimate real-world security precision, recall or perception accuracy.

**Limitations**

- Controlled synthetic fixtures only: these results do not estimate real-world security precision, recall or perception accuracy.
- The labels were written by the same author as the rules, so agreement is not independent validation; it checks the documented boundaries, not usefulness.
- Scripted detections are perfect; upstream perception errors are not measured here.

### Citation-gate adversarial suite

**Research question.** RQ6: does the citation gate reject nonexistent, unretrieved, malformed, missing and prohibited-overclaim outputs?

**Setup.** 29 authored analyst outputs over 10 categories, run through the production AnalystSession (scripted LLM boundary, tool calls performed first) against the security_events evidence.

**Measured result**

| category | case | expected | actual | correct | reason |
|---|---|---|---|---|---|
| valid_retrieved | alert and event | accept | accept | True |  |
| valid_retrieved | entity history | accept | accept | True |  |
| valid_retrieved | frame and observation | accept | accept | True |  |
| nonexistent | unknown event | reject | reject | True | attempt 1 rejected: cited ids that do not exist: [event:EVT-Z-999] |
| nonexistent | unknown alert | reject | reject | True | attempt 1 rejected: cited ids that do not exist: [alert:ALR-FAKE-1] |
| nonexistent | unknown frame | reject | reject | True | attempt 1 rejected: cited ids that do not exist: [frame:9999] |
| nonexistent | unknown entity | reject | reject | True | attempt 1 rejected: cited ids that do not exist: [entity:ENT-V-009] |
| nonexistent | unknown observation | reject | reject | True | attempt 1 rejected: cited ids that do not exist: [obs:obs-999] |
| real_not_retrieved | event never fetched | reject | reject | True | attempt 1 rejected: cited ids that no tool returned in this conversation: [event:EVT-V-001] |
| real_not_retrieved | alert never fetched | reject | reject | True | attempt 1 rejected: cited ids that no tool returned in this conversation: [alert:ALR-RESTRICTED-EVT-P-003] |
| real_not_retrieved | frame outside retrieved evidence | reject | reject | True | attempt 1 rejected: cited ids that no tool returned in this conversation: [frame:20] |
| malformed | space instead of colon | accept | accept | True |  |
| malformed | unclosed bracket | reject | reject | True | attempt 1 rejected: the answer states facts but cites no evidence |
| malformed | valid plus repairable token | accept | accept | True |  |
| malformed | valid plus unknown citation kind | reject | reject | True | attempt 1 rejected: unknown citation type (allowed: frame, obs, event, entity, alert, tel): [evt:EVT-P-002] |
| no_citation | factual claim without citation | reject | reject | True | attempt 1 rejected: the answer states facts but cites no evidence; factual sentences without their own citation: "A person stood at the main gate for eight minu |
| no_citation | alert claim without citation | reject | reject | True | attempt 1 rejected: the answer states facts but cites no evidence; factual sentences without their own citation: "The repeat-vehicle alert has low severity." |
| no_citation | vehicle claim without citation | reject | reject | True | attempt 1 rejected: the answer states facts but cites no evidence; factual sentences without their own citation: "A blue pickup was seen three times." |
| insufficient_evidence | declares insufficient evidence | accept | accept | True |  |
| identity_claim | same person | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: the same person |
| identity_claim | same individual | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: The same individual |
| identity_claim | confirmed same vehicle | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: Confirmed to be the same vehicle |
| intent_claim | burglar | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: burglar |
| intent_claim | criminal intent | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: Criminal |
| intent_claim | committing a crime | reject | reject | True | attempt 1 rejected: identity/intent wording is not allowed: committing a crime |
| negated_identity | not the same person | accept | accept | True |  |
| negated_identity | cannot say same individual | accept | accept | True |  |
| uncertainty_vehicle | similar, coarse match | accept | accept | True |  |
| uncertainty_vehicle | alert wording | accept | accept | True |  |

**Interpretation**

- The gate made the expected accept/reject decision in 29 of 29 adversarial cases (1.000 on THIS fixture set only). Invalid outputs accepted: 0 of 19 (none); valid outputs rejected: 0 of 10 (none).
- Without the gate every one of the 19 invalid outputs would have been shown.

**Limitations**

- Accuracy is over this hand-built fixture set, not a sample of real model outputs.
- The first run of this suite (before any validator change) made two wrong decisions: an unknown citation type such as [evt:...] was ignored (false accept) and a negation more than two words before 'the same individual' was not recognised (false reject). Both validator behaviours were then fixed and the fixtures were left unchanged, so accuracy on this suite is no longer an independent measure of the gate: the validator was corrected against these same cases.
- The gate checks form, existence and retrieval, not whether the claims are true (see the semantic-faithfulness probe).

### Citation-valid / semantically-unsupported probe (+ claim-evidence checker)

**Research question.** RQ7: which unsupported claims still pass despite valid citations, and how much does a narrow structured claim check add?

**Setup.** 27 authored answers over the security_events evidence: 5 true controls and 22 unsupported answers (13 with a structured form, 9 not expressible). Each is run through the production gate; the eval-only checker then verifies the structured claims against the store by exact comparison.

**Measured result**

| category | case | claims_supported_by_evidence | gate_accepts | structured_claims | checker_verdicts | gate_plus_checker_accepts |
|---|---|---|---|---|---|---|
| control_true | correct zone | True | True | 1 | supported | True |
| control_true | correct duration | True | True | 1 | supported | True |
| control_true | correct severity | True | True | 1 | supported | True |
| control_true | correct appearance count | True | True | 1 | supported | True |
| control_true | correct type and activity | True | True | 2 | supported,supported | True |
| wrong_zone | vehicle event at the wrong zone | False | True | 1 | contradicted | False |
| wrong_zone | person event in the restricted zone | False | True | 1 | contradicted | False |
| wrong_duration | loitering stretched | False | True | 1 | contradicted | False |
| wrong_duration | short vehicle event inflated | False | True | 1 | contradicted | False |
| wrong_severity | repeat alert escalated | False | True | 1 | contradicted | False |
| wrong_severity | analyst claims to upgrade | False | True | 1 | contradicted | False |
| wrong_rule | alert attributed to another rule | False | True | 1 | contradicted | False |
| wrong_count | overcounted appearances | False | True | 1 | contradicted | False |
| wrong_count | undercounted appearances | False | True | 1 | contradicted | False |
| wrong_type | person described as a vehicle | False | True | 1 | contradicted | False |
| wrong_activity | person running | False | True | 1 | contradicted | False |
| wrong_activity | vehicle driving | False | True | 1 | contradicted | False |
| wrong_time | wrong start time | False | True | 1 | contradicted | False |
| capability_overclaim | persistent tracking | False | True | 0 | none | True |
| capability_overclaim | vehicle tracking wording | False | True | 0 | none | True |
| identity_evasion | very same pickup | False | True | 0 | none | True |
| identity_evasion | identical truck | False | True | 0 | none | True |
| identity_evasion | that same guy | False | True | 0 | none | True |
| identity_evasion | person from earlier | False | True | 0 | none | True |
| intent_evasion | casing the property | False | True | 0 | none | True |
| intent_evasion | planning a break-in | False | True | 0 | none | True |
| boundary_overclaim | analyst claims it made the alert | False | True | 0 | none | True |

**Interpretation**

- Citation-valid / semantically-unsupported probe: 22 of 22 answers whose citations all exist, were retrieved and are well formed, but whose claims are false or overreaching, were accepted by the gate (100.0%). All 5 of 5 true control answers were accepted too, so the gate cannot tell them apart.
- Structured claim checker (prototype): of the 13 unsupported cases expressible as structured claims it caught 13 (catch rate 1.000); unsupported answers still accepted by gate + checker: 9 of 22, all in claim classes the checker does not cover (['boundary_overclaim', 'capability_overclaim', 'identity_evasion', 'intent_evasion']). True controls rejected by the checker: 0.
- The expressible cases were authored to fit the checker's classes, so its catch rate is true by design, not evidence that it generalises. This is a measurement of the gate's failure boundary, not a hallucination detector: the checker verifies pre-structured claims and does not read prose.

**Limitations**

- The unsupported answers are authored from the evidence model; they are not a sample of real model errors, so the pass percentage is a boundary characterisation, not a rate.
- The checker receives structured claims; it does not extract them from text. Free-text extraction (and general semantic entailment) is not attempted.
- Covered claim classes: event_zone, event_type, event_activity, event_duration_s, event_start_hhmm, alert_severity, alert_rule, entity_appearance_count. Tracking, identity wording, intent inference and boundary claims are outside them and remain unchecked.
- The checker lives in the evaluation layer and is not part of the production analyst.

### Ground-truth scenarios: precision, recall, F1, severity and detection latency

**Research question.** RQ8: does a streaming run reproduce the events, alerts, severity ladders and timing that each scenario's author wrote down in advance?

**Setup.** 10 scenario files with machine-readable `expect:` blocks, each run headless through the streaming engine with the default rules and the adaptive sampler on, then matched one-to-one to its labels (drone_agent/eval/ground_truth.py).

**Measured result**

| scenario | expected_alerts | tp | fp | fn | precision | recall | f1 | forbidden_hits | events | exact |
|---|---|---|---|---|---|---|---|---|---|---|
| benign_delivery | 0 | 0 | 0 | 0 | N/A | N/A | N/A | 0 | 3/3 | True |
| flicker_noise | 1 | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 1/1 | True |
| geofence_violation | 1 | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 0/0 | True |
| intruder_garage | 2 | 2 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 2/2 | True |
| loiter_midnight | 2 | 2 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 1/1 | True |
| low_battery | 1 | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 0/0 | True |
| normal_day | 0 | 0 | 0 | 0 | N/A | N/A | N/A | 0 | 1/1 | True |
| repeat_pickup | 1 | 1 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 4/4 | True |
| security_events | 7 | 7 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 7/7 | True |
| signal_loss | 3 | 3 | 0 | 0 | 1.0 | 1.0 | 1.0 | 0 | 0/0 | True |

**Interpretation**

- 10 of 10 scenarios matched their ground truth exactly. Micro-averaged over 18 expected alerts: precision 1.000, recall 1.000, F1 1.000 (18 TP, 0 FP, 0 FN); forbidden alerts raised: 0.
- Of the matched alerts, the severity ladder was right in 1.000 and every rung landed within the tolerance in 1.000; the final lifecycle status was right in 1.000.
- Alert-detection latency (simulated seconds from the condition becoming true to the alert being raised): median 0.0 s, mean 1.1 s, p95 10.0 s, max 10.0 s over 18 alerts. It is bounded by frame spacing (60 s) for camera alerts and by the 10 s telemetry tick for safety alerts.
- Events recalled 1.000, vehicle entities 1.000.

**Limitations**

- Controlled synthetic fixtures only: these results do not estimate real-world security precision, recall or perception accuracy.
- The labels were written by the same author as the rules, so agreement mainly shows that the implementation matches that author's intent; it is not independent evidence.
- Scripted detections are perfect; upstream perception errors are the subject of the noise experiment, not of this one.
- Latency is measured in simulated time on scripted data with 60 s frame spacing.

### Adaptive sampling versus a pixel gate: calls saved AND changes missed

**Research question.** RQ9: does combining pixel change, telemetry change, event state and a staleness bound detect more meaningful changes per model call than a pixel gate, including a pixel gate with a fixed refresh?

**Setup.** 8 seeded 600-frame streams (frames 2 s apart, 160x90, noise sigma 1.5) with objects of three sizes appearing and leaving, illumination steps, and a 30% telemetry-cue rate; the real FrameGate pixel score, the real AdaptiveSampler; a change counts as detected if a frame within 5 frames was sent. Configurations fixed in advance: adaptive = watch 20 s, forced refresh 120 s.

**Measured result**

| policy | family | calls_mean | avoided_pct_mean | recall | mean_lag_frames | calls_per_detected | missed_small | missed_medium | missed_large | meaningful_changes |
|---|---|---|---|---|---|---|---|---|---|---|
| every frame (gate off) | send everything | 600.0 | 0.0 | 1.0 | 0.0 | 53.33 | 0 | 0 | 0 | 90 |
| pixel gate, threshold 6 (previous default) | pixel only | 11.375 | 98.088 | 0.4444 | 0.203 | 2.27 | 31 | 19 | 0 | 90 |
| pixel gate, threshold 3 | pixel only, more sensitive | 12.125 | 97.963 | 0.5111 | 0.181 | 2.11 | 31 | 13 | 0 | 90 |
| pixel gate + refresh every 15 frames | pixel + periodic | 45.5 | 92.425 | 0.7111 | 0.805 | 5.69 | 13 | 13 | 0 | 90 |
| pixel gate + refresh every 30 frames | pixel + periodic | 26.125 | 95.65 | 0.5778 | 0.566 | 4.02 | 23 | 15 | 0 | 90 |
| adaptive (pixel + telemetry + event watch + refresh) | adaptive | 36.5 | 93.912 | 0.7556 | 0.361 | 4.29 | 16 | 6 | 0 | 90 |
| adaptive without telemetry signal | ablation | 31.75 | 94.7 | 0.6111 | 1.054 | 4.62 | 22 | 13 | 0 | 90 |
| adaptive without event watch | ablation | 17.625 | 97.075 | 0.7 | 0.258 | 2.24 | 19 | 8 | 0 | 90 |
| adaptive without forced refresh | ablation | 32.5 | 94.588 | 0.7333 | 0.32 | 3.94 | 18 | 6 | 0 | 90 |

**Interpretation**

- On 8 generated 600-frame streams (90 meaningful changes in total): the previous pixel-only default sent 11.4 frames per stream (98.1% avoided) and detected 0.444 of the meaningful changes within 5 frames; it missed 31 small, 19 medium and 0 large ones.
- The adaptive sampler sent 36.5 frames (93.9% avoided) and detected 0.756, mean lag 0.36 frames, 4.29 calls per detected change.
- The honest baseline is pixel gate plus a fixed refresh: every 15 frames it sent 45.5 frames and detected 0.711 (5.69 calls per detected change).
- Ablations remove one adaptive signal at a time (telemetry, event watch, forced refresh): without telemetry signal 0.611 recall at 31.8 calls; without event watch 0.700 recall at 17.6 calls; without forced refresh 0.733 recall at 32.5 calls.

**Limitations**

- Controlled synthetic fixtures only: these results do not estimate real-world security precision, recall or perception accuracy.
- The share of meaningful changes that come with a telemetry cue (30%) is an assumption; with none, the telemetry signal contributes nothing (see the ablation).
- Generated scenes are flat backgrounds with dark rectangles; real scenes have far richer pixel statistics, so absolute recall and call counts will differ.
- Detection means a model call was made in time, not that the model would have described the change correctly (that is perception accuracy, which is not measured here).

### Scale benchmark (generated observations)

**Research question.** Assignment scalability question: how do processing and query costs grow with the number of observations?

**Setup.** Deterministic generated observations at sizes [100, 1000, 10000, 50000]; each pipeline stage timed 5 times (median), SQLite queries 20 times (median).

**Measured result**

| observations | stage | ms |
|---|---|---|
| 100 | event_aggregation | 0.901 |
| 100 | entity_linking | 0.146 |
| 100 | rule_evaluation | 0.131 |
| 100 | query_fts_keyword_top10 | 0.097 |
| 100 | query_filter_zone_type_window | 0.818 |
| 100 | query_count_matching_zone | 0.088 |
| 100 | query_list_events_by_zone | 0.282 |
| 100 | query_list_alerts | 0.359 |
| 100 | sqlite_ingest | 19.782 |
| 100 | sqlite_persist_derived | 15.743 |
| 1000 | event_aggregation | 14.483 |
| 1000 | entity_linking | 2.788 |
| 1000 | rule_evaluation | 1.598 |
| 1000 | query_fts_keyword_top10 | 0.724 |
| 1000 | query_filter_zone_type_window | 1.551 |
| 1000 | query_count_matching_zone | 0.413 |
| 1000 | query_list_events_by_zone | 1.731 |
| 1000 | query_list_alerts | 3.075 |
| 1000 | sqlite_ingest | 246.523 |
| 1000 | sqlite_persist_derived | 36.001 |
| 10000 | event_aggregation | 143.559 |
| 10000 | entity_linking | 60.701 |
| 10000 | rule_evaluation | 31.04 |
| 10000 | query_fts_keyword_top10 | 1.254 |
| 10000 | query_filter_zone_type_window | 2.108 |
| 10000 | query_count_matching_zone | 2.572 |
| 10000 | query_list_events_by_zone | 15.832 |
| 10000 | query_list_alerts | 48.709 |
| 10000 | sqlite_ingest | 2298.009 |
| 10000 | sqlite_persist_derived | 253.199 |
| 50000 | event_aggregation | 732.383 |
| 50000 | entity_linking | 1242.491 |
| 50000 | rule_evaluation | 183.465 |
| 50000 | query_fts_keyword_top10 | 4.144 |
| 50000 | query_filter_zone_type_window | 1.538 |
| 50000 | query_count_matching_zone | 22.169 |
| 50000 | query_list_events_by_zone | 110.172 |
| 50000 | query_list_alerts | 317.248 |
| 50000 | sqlite_ingest | 11790.594 |
| 50000 | sqlite_persist_derived | 914.648 |

**Interpretation**

- Event aggregation took a median 0.901 ms for 100 observations and 732.383 ms for 50000 (812.9x time for 500x data).
- Empirical scaling exponent between the two largest sizes (time ~ size^k; k=1 is linear): event_aggregation 1.01, entity_linking 1.88, rule_evaluation 1.10, query_fts_keyword_top10 0.74, query_filter_zone_type_window -0.20, query_count_matching_zone 1.34, query_list_events_by_zone 1.21, query_list_alerts 1.16. Superlinear here (k>1.3): ['entity_linking', 'query_count_matching_zone'].
- Times are medians of repeated in-process runs on this machine; the SQLite ingest and derived-table writes are single runs.

**Limitations**

- One machine, one run session; no claim about other hardware, larger data, concurrent access or the vision model (its latency dominates a real deployment).
- Synthetic observations with a small attribute vocabulary; linking cost depends on the number of distinct vehicle attribute combinations.
- In-memory stages exclude reading rows back from SQLite.

**Environment (scale benchmark)**

- python: 3.11.9
- platform: Windows-10-10.0.22631-SP0
- processor: Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
- cpu_count: 14
- sqlite_version: 3.45.1
- note: measurements on this machine only; not a universal performance claim

