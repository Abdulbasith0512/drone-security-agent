# Scale benchmark

Machine: Windows-10-10.0.22631-SP0, Python 3.11.9, 14 logical CPUs, SQLite 3.45.1. measurements on this machine only; not a universal performance claim.

## Pipeline (real code paths, generated observations)

| | 10,000 | 100,000 |
|---|---:|---:|
| events | 1,385 | 13,889 |
| entities | 49 | 498 |
| alerts | 565 | 6,204 |
| frames per event | 7.22 | 7.2 |
| ingest_sqlite (s) | 1.039 | 10.419 |
| aggregate (s) | 0.058 | 0.651 |
| link (s) | 0.011 | 0.518 |
| rules (s) | 0.015 | 0.339 |
| persist_derived (s) | 0.214 | 1.224 |
| ingest throughput (obs/s) | 9,621 | 9,598 |
| DB size (MiB) | 7.3 | 73.52 |
| DB bytes / observation | 765 | 771 |
| peak memory (MiB) | 52.9 | 117.8 |

## Query latency (ms; p50 / p95)

| operation | 10,000 | 100,000 |
|---|---:|---:|
| store_count_zone | 0.043 / 0.097 | 1.12 / 2.014 |
| store_filter_zone_window | 0.225 / 0.521 | 0.253 / 0.553 |
| store_fts_keyword_top10 | 0.85 / 10.141 | 10.467 / 110.793 |
| tool_get_frame | 0.214 / 0.499 | 0.277 / 1.336 |
| tool_get_timeline_hour | 0.701 / 1.086 | 4.826 / 11.45 |
| tool_list_alerts_all | 1.034 / 3.78 | 1.181 / 2.802 |
| tool_list_alerts_rule | 0.764 / 1.286 | 0.888 / 1.96 |
| tool_query_events_window | 0.699 / 1.574 | 4.121 / 11.779 |
| tool_query_events_zone | 0.726 / 1.08 | 4.146 / 8.619 |
| tool_search_events | 7.173 / 15.659 | 16.238 / 121.711 |
| tool_summarize_run | 11.807 / 20.918 | 125.089 / 140.896 |
| tool_telemetry_at | 0.022 / 0.034 | 0.025 / 0.038 |

## VLM calls saved (SIMULATED score stream)

| | 10,000 | 100,000 |
|---|---:|---:|
| frames | 10,000 | 100,000 |
| model calls | 2,792 | 27,854 |
| calls avoided (%) | 72.1 | 72.1 |
| real scene changes | 1,928 | 19,523 |

## Empirical scaling exponent between the two largest sizes (time ~ size^k)

k = 1 is linear; well above 1 is superlinear.

- query.store_count_zone.p50: 1.42
- query.store_filter_zone_window.p50: 0.05
- query.store_fts_keyword_top10.p50: 1.09
- query.tool_get_frame.p50: 0.11
- query.tool_get_timeline_hour.p50: 0.84
- query.tool_list_alerts_all.p50: 0.06
- query.tool_list_alerts_rule.p50: 0.07
- query.tool_query_events_window.p50: 0.77
- query.tool_query_events_zone.p50: 0.76
- query.tool_search_events.p50: 0.35
- query.tool_summarize_run.p50: 1.03
- query.tool_telemetry_at.p50: 0.06
- seconds.aggregate: 1.05
- seconds.generate: 1.1
- seconds.ingest_sqlite: 1.0
- seconds.link: 1.67
- seconds.persist_derived: 0.76
- seconds.rules: 1.35
- seconds.sampler: 1.06
- seconds.telemetry: 1.06

## Qualifications

- Generated observations with a small attribute vocabulary and a plate fleet; not real footage, and no vision model is involved (its latency would dominate a deployment).
- One machine, one run per size; timings vary run to run. Counts are deterministic.
- The sampler figures come from a SIMULATED scene-score stream: they estimate call savings if scores looked like this, not measured perception.
- SQLite on a local disk, one writer; no concurrent access, no network filesystem.
