"""Evaluator dashboard: a small localhost web UI over the existing store, analyst and video code.

The UI is read-only and adds no detection logic: every number, event, alert and answer is produced
by the existing pipeline and read back through `Store`, `AnalystTools` and `AnalystSession`.
Start it with `python -m drone_agent.web --demo`.
"""
