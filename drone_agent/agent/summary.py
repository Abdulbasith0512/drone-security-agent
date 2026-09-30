"""Run summary: deterministic facts first, prose second.

`run_facts` gathers structured, bounded facts from the store. `render_summary` turns them into
plain text with citations (the offline summary). With an LLM, the model may rewrite these same
facts as prose, but only the facts below may appear in it. No LangChain here.
"""

from drone_agent.agent.evidence import ToolLimits, alert_view, entity_view, event_view
from drone_agent.store import Store


def run_facts(store: Store, limits: ToolLimits | None = None) -> dict:
    """Bounded, structured facts about the whole run. Only the shown rows are loaded: totals and
    breakdowns come from SQL counts, so the cost does not grow with the size of the run."""
    limits = limits or ToolLimits()
    span = store.observation_span()
    sources = store.observation_counts_by("source")
    provenance = store.observation_counts_by("provenance")
    events = store.query_events(limit=limits.max_events)
    alerts = store.query_alerts(limit=limits.max_alerts)
    entities = store.list_entities(limit=limits.max_entities)
    events_total, alerts_total = store.count_events(), store.count_alerts()
    entities_total = store.count_entities()
    alert_ids_by_event = store.alert_ids_by_event([ev.event_id for ev in events])

    event_views = []
    for ev in events:
        view = event_view(ev, limits, detail=False)
        view["alert_refs"] = [f"alert:{a}" for a in alert_ids_by_event.get(ev.event_id, [])]
        event_views.append(view)

    facts = {
        "counts": {
            "frames": store.count_frames(),
            "observations": store.count_observations(),
            "detected_objects": store.count_detected_objects(),
            "events": events_total,
            "entities": entities_total,
            "alerts": alerts_total,
        },
        "observation_span": (
            {"first": span[0].isoformat(), "last": span[1].isoformat()} if span else None
        ),
        "zones": [
            {"zone": z, "observations": n} for z, n in store.observation_counts_by("zone").items()
        ],
        "object_counts": store.detected_object_counts(),
        "sources": sources,
        "provenance": provenance,
        "events": event_views,
        "events_total": events_total,
        "events_truncated": events_total > len(events),
        "alerts": [alert_view(a, limits, detail=False) for a in alerts],
        "alerts_total": alerts_total,
        "alerts_truncated": alerts_total > len(alerts),
        "alerts_by_severity": store.alert_counts_by("severity"),
        "alerts_by_rule": store.alert_counts_by("rule"),
        "alerts_by_status": store.alert_counts_by("status"),
        "entities": [entity_view(e) for e in entities],
        "entities_total": entities_total,
    }
    facts["limitations"] = limitations(
        store, sources, provenance, events_total, alerts_total, entities_total
    )
    return facts


def limitations(
    store: Store,
    sources: dict[str, int],
    provenance: dict[str, int],
    events: int,
    alerts: int,
    entities: int,
) -> list[str]:
    out = ["Zones, timestamps and telemetry are simulated; they are not read from the video."]
    carried = sum(n for s, n in sources.items() if s.endswith("+carried"))
    if carried:
        out.append(
            f"{carried} observation(s) are carried forward from an earlier accepted frame "
            "(temporal propagation), not new model inferences."
        )
    live, replayed = provenance.get("live", 0), provenance.get("replayed", 0)
    if live or replayed or any(s.startswith("gemini") for s in sources):
        out.append(
            "Descriptions come from a vision model and can be wrong; stored confidence values "
            "are an uncalibrated placeholder, not probabilities."
        )
    if replayed:
        out.append(
            f"{replayed} observation(s) are REPLAYED from a recorded model response: no model "
            "was called in this run."
        )
    if provenance.get("canned"):
        out.append(
            f"{provenance['canned']} observation(s) are canned demo text, not the output of "
            "any vision model."
        )
    if provenance.get("scripted"):
        out.append(
            f"{provenance['scripted']} observation(s) are scripted scenario text written by the "
            "author, not perception output; scores on them measure the rules, not a model."
        )
    if entities:
        out.append(
            'Vehicle matches are coarse attribute matches ("similar"), not confirmed identity; '
            "people are never linked across events."
        )
    if store.count_observations() and not events:
        out.append("No events are stored (build-events may not have been run).")
    elif events and not alerts:
        out.append("No alerts are stored (none triggered, or analyze has not been run).")
    return out


def _times(first: str, last: str) -> str:
    return f"{first[:19].replace('T', ' ')} to {last[:19].replace('T', ' ')}"


def render_summary(facts: dict) -> str:
    """Deterministic plain-text summary; every id is cited as [kind:id]."""
    c = facts["counts"]
    lines = [
        f"Stored run: {c['observations']} observation(s) over {c['frames']} frame(s), "
        f"{c['events']} event(s), {c['entities']} vehicle entity(ies), {c['alerts']} alert(s)."
    ]
    if facts["observation_span"]:
        span = facts["observation_span"]
        lines.append(f"Observation time span: {_times(span['first'], span['last'])}.")
    if facts["zones"]:
        lines.append(
            "Observations by zone: "
            + ", ".join(f"{z['zone']} {z['observations']}" for z in facts["zones"])
            + "."
        )
    if facts["object_counts"]:
        lines.append(
            "Detected objects: "
            + ", ".join(f"{k} {v}" for k, v in facts["object_counts"].items())
            + "."
        )
    if facts["alerts"]:
        sev = ", ".join(f"{k} {v}" for k, v in facts["alerts_by_severity"].items())
        lines.append(f"Alerts (raised by deterministic rules; severity: {sev}):")
        for a in facts["alerts"]:
            events = " ".join(f"[{e}]" for e in a["event_refs"])
            lines.append(
                f"- [{a['ref']}] {a['severity']}/{a['status']}: {a['title']} ({a['zone']}, "
                f"triggered {a['triggered_at'][:19].replace('T', ' ')}) {events}".rstrip()
            )
        if facts["alerts_truncated"]:
            lines.append(f"  (showing {len(facts['alerts'])} of {facts['alerts_total']} alerts)")
    flagged = [e for e in facts["events"] if e["alert_refs"]]
    if flagged:
        lines.append("Significant events (linked to alerts):")
        for e in flagged:
            lines.append(_event_line(e))
    others = [e for e in facts["events"] if not e["alert_refs"]]
    if others:
        lines.append("Other events:")
        for e in others:
            lines.append(_event_line(e))
    if facts["events_truncated"]:
        lines.append(f"  (showing {len(facts['events'])} of {facts['events_total']} events)")
    for ent in facts["entities"]:
        events = " ".join(f"[{e}]" for e in ent["event_refs"])
        lines.append(
            f"Vehicle memory: a {ent['label']} [{ent['ref']}] in {ent['appearance_count']} "
            f"separate appearance(s) {events}. This is a coarse attribute match, not confirmed "
            "to be the same vehicle."
        )
    if facts["entities_total"] > len(facts["entities"]):
        shown, total = len(facts["entities"]), facts["entities_total"]
        lines.append(
            f"  (showing {shown} of {total} vehicle entities, earliest first; "
            f"{total - shown} not shown)"
        )
    lines.append("Limitations: " + " ".join(facts["limitations"]))
    return "\n".join(lines)


def _event_line(e: dict) -> str:
    attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
    detail = f" ({attrs})" if attrs else ""
    return (
        f"- [{e['ref']}] {e['type']}{detail} in {e['zone']}, "
        f"{_times(e['first_seen'], e['last_seen'])}, {e['observation_count']} observation(s)"
    )
