"""A scripted tool-calling "model" (NO LLM) for the analyst's LangChain loop.

Purpose: let an evaluator, a test or a demo run the REAL analyst machinery offline: the LangChain
loop, the typed read-only tools, the caps and the citation gate, with no API key. Only the model's
part is scripted. It behaves like a chat model (`bind_tools`, `invoke(messages)`):

    step 0..n  plan tool calls from the question with plain rules, reading earlier tool results
    last step  compose the answer from those tool results

Everything in the answer is copied from tool results and every factual line carries the citation
of the evidence it came from, so it passes the same gate a real model's text must pass. It is
labelled "scripted planner, no LLM" wherever it is shown. It understands a bounded set of question
shapes (alerts and why, events, vehicles, a person seen before, zones, telemetry, timelines, time
windows, the run summary); anything else gets an honest "Insufficient evidence".
"""

import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from drone_agent.agent.summary import render_summary
from drone_agent.agent.timeparse import parse_time_window
from drone_agent.models import Zone

NAME = "planner (scripted, no LLM)"

_ALERT_ID = re.compile(r"\bALR-[A-Z0-9-]+\b", re.IGNORECASE)
_EVENT_ID = re.compile(r"\bEVT-[PVO]-\d+\b", re.IGNORECASE)
_ENTITY_ID = re.compile(r"\bENT-V-\d+\b", re.IGNORECASE)
_ZONES = (
    (r"main[ _]?gate|\bgate\b", "main_gate"),
    (r"driveway", "driveway"),
    (r"parking", "parking_area"),
    (r"warehouse", "warehouse"),
    (r"restricted", "restricted_zone"),
    (r"garage", "garage"),
)
_RULES = (
    (r"loiter", "loitering"),
    (r"after[- ]?hours|overnight|at night", "after_hours_presence"),
    (r"repeat", "repeat_vehicle"),
    (r"restricted[- ]zone (alert|entry)|intrusion|intruder", "restricted_zone_entry"),
    (r"battery", "low_battery"),
    (r"signal|link|outage|dropout", "signal_loss"),
    (r"altitude", "altitude_anomaly"),
    (r"geofence|fence", "geofence_violation"),
)
_STOP_TEXT = (
    "a an the of in at on to for and or is was were are be been did do does any some show me "
    "what which who when where why how there it its this that those they them with from "
    "about all list find tell give seen see saw more than once again events event"
)
_STOP = frozenset(re.split(r"\s+", _STOP_TEXT))
MAX_PLAN_CALLS = 3


def _t(iso: str) -> str:
    return iso[:19].replace("T", " ") + iso[19:]


def _cite(*refs: str) -> str:
    return " ".join(f"[{r}]" for r in refs if r)


def _pct(value: float | None) -> str:
    return "not reported" if value is None else f"{value:.0f}%"


def _flat(text: object) -> str:
    """Free text as ONE sentence: inner sentence breaks become semicolons, no final period.

    The citation gate checks every sentence, so text copied from the evidence must not add
    sentence breaks of its own after the citation that covers it.
    """
    return re.sub(r"[.!?]+\s+", "; ", str(text).strip()).rstrip(".!? ")


def _line(text: object, *refs: str) -> str:
    """One sentence that carries its own citation (placed before the final period)."""
    return f"{_flat(text)} {_cite(*refs)}."


def _parse(message: ToolMessage) -> dict:
    try:
        data = json.loads(message.content if isinstance(message.content, str) else "{}")
    except json.JSONDecodeError:
        return {"error": "unreadable tool result"}
    return data if isinstance(data, dict) else {"error": "unexpected tool result"}


class PlannerChatModel:
    """Deterministic stand-in for a tool-calling chat model. Not an LLM."""

    name = NAME

    def __init__(self) -> None:
        self._tools: set[str] = set()

    def bind_tools(self, tools: Any) -> "PlannerChatModel":
        self._tools = {getattr(t, "name", str(t)) for t in tools}
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        question_index = max(
            (
                i
                for i, m in enumerate(messages)
                if isinstance(m, HumanMessage)
                and not str(m.content).startswith("Your answer was rejected")
            ),
            default=-1,
        )
        if question_index < 0:
            return AIMessage(content="Insufficient evidence: there is no question to answer.")
        if any(
            isinstance(m, HumanMessage) and str(m.content).startswith("Your answer was rejected")
            for m in messages[question_index:]
        ):
            return AIMessage(
                content="Insufficient evidence: the scripted planner cannot produce a verifiable "
                "answer to that question."
            )
        question = str(messages[question_index].content)
        done: list[tuple[str, dict, dict]] = []
        pending: dict[str, tuple[str, dict]] = {}
        for m in messages[question_index + 1 :]:
            if isinstance(m, AIMessage):
                for call in getattr(m, "tool_calls", None) or []:
                    pending[call["id"]] = (call["name"], call.get("args") or {})
            elif isinstance(m, ToolMessage) and m.tool_call_id in pending:
                name, args = pending[m.tool_call_id]
                done.append((name, args, _parse(m)))
        calls = plan(question, done) if len(done) < 8 else []
        if calls:
            return AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": args, "id": f"plan-{len(done)}-{i}"}
                    for i, (name, args) in enumerate(calls[:MAX_PLAN_CALLS])
                ],
            )
        return AIMessage(content=compose(question, done))


# ---- planning ---------------------------------------------------------------------------------
def _zone_in(q: str) -> str | None:
    return next((z for pattern, z in _ZONES if re.search(pattern, q)), None)


def _rule_in(q: str) -> str | None:
    return next((r for pattern, r in _RULES if re.search(pattern, q)), None)


_SAFETY = ("low_battery", "signal_loss", "altitude_anomaly", "geofence_violation")


def _safety_rules_in(q: str) -> list[str]:
    """Every safety rule the question mentions, in the order of the rule table."""
    return [r for pattern, r in _RULES if r in _SAFETY and re.search(pattern, q)]


def _object_type(q: str) -> str | None:
    if re.search(r"\b(person|people|man|woman|pedestrian|visitor|intruder|courier)\b", q):
        return "person"
    if re.search(r"\b(vehicle|truck|pickup|car|van|sedan|suv|lorry)s?\b", q):
        return "vehicle"
    return None


def _ran(done: list[tuple[str, dict, dict]], name: str) -> dict | None:
    return next((r for n, _, r in done if n == name), None)


def _first_alert_ids(done: list[tuple[str, dict, dict]], limit: int = 2) -> list[str]:
    listing = _ran(done, "list_alerts") or {}
    return [a["ref"] for a in listing.get("alerts", [])[:limit]]


def _keywords(q: str) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2 and w not in _STOP]
    return " ".join(words[:4])


# "after 6 pm", "between 00:10 and 00:20", "after midnight": a question about a time window
_WINDOW = re.compile(r"\b(after|before|between|since|until|from)\b.*\d|midnight|noon")


def plan(question: str, done: list[tuple[str, dict, dict]]) -> list[tuple[str, dict]]:
    """The next tool calls, given what has been retrieved so far ([] = ready to answer)."""
    q = question.lower()
    called = {n for n, _, _ in done}
    alert_id = _ALERT_ID.search(question)
    event_id = _EVENT_ID.search(question)
    entity_id = _ENTITY_ID.search(question)
    zone, rule, kind = _zone_in(q), _rule_in(q), _object_type(q)

    if alert_id:
        return (
            []
            if "get_evidence" in called
            else [("get_evidence", {"ref_id": f"alert:{alert_id.group(0).upper()}"})]
        )
    if entity_id:
        return (
            []
            if "entity_history" in called
            else [("entity_history", {"id": entity_id.group(0).upper()})]
        )
    if event_id:
        ident = event_id.group(0).upper()
        if "entity_history" in called or "get_evidence" in called:
            return []
        recurrence = re.search(r"before|earlier|previous|again|seen|same", q)
        if ident.startswith("EVT-P-") and recurrence:
            return [("entity_history", {"id": ident})]
        return [("get_evidence", {"ref_id": f"event:{ident}"})]

    if _WINDOW.search(q):
        if "summarize_run" not in called:
            return [("summarize_run", {})]
        span = (_ran(done, "summarize_run") or {}).get("observation_span")
        if span and "query_events" not in called:
            window = parse_time_window(
                question,
                datetime.fromisoformat(span["first"]),
                datetime.fromisoformat(span["last"]),
            )
            if window:
                start, end = window
                args: dict = {}
                if start:
                    args["start"] = start.isoformat()
                if end:
                    args["end"] = end.isoformat()
                events = {
                    **args,
                    **({"zone": zone} if zone else {}),
                    **({"object_type": kind} if kind else {}),
                }
                # the timeline has no type/zone filter, so it would add unrelated items to a
                # question that names one; it is only asked for when the window is the whole ask
                if zone or kind:
                    return [("query_events", events)]
                return [("query_events", events), ("get_timeline", args)]
        return []

    if re.search(r"\b(why|explain|reason|how come|what triggered)\b", q) and (rule or "alert" in q):
        if "list_alerts" not in called:
            return [
                (
                    "list_alerts",
                    {**({"rule_id": rule} if rule else {}), **({"zone": zone} if zone else {})},
                )
            ]
        if "get_evidence" not in called:
            return [("get_evidence", {"ref_id": r}) for r in _first_alert_ids(done)]
        return []

    safety_rules = _safety_rules_in(q)
    if safety_rules:
        if "list_alerts" not in called:
            return [("list_alerts", {"rule_id": r}) for r in safety_rules]
        first = next(
            (a for n, _, r in done if n == "list_alerts" for a in r.get("alerts", [])), None
        )
        if first and "telemetry_at" not in called:
            return [("telemetry_at", {"time": first["triggered_at"]})]
        return []

    if re.search(r"\b(previous\w*|earlier|before|again|seen)\b", q) and kind == "person":
        if "query_events" not in called:
            return [("query_events", {"object_type": "person", **({"zone": zone} if zone else {})})]
        events = (_ran(done, "query_events") or {}).get("events", [])
        if events and "entity_history" not in called:
            latest = events[-1]["ref"].split(":", 1)[1]
            return [("entity_history", {"id": latest})]
        return []

    if re.search(r"\b(timeline|sequence|chronolog|order of events)\b", q):
        return [] if "get_timeline" in called else [("get_timeline", {})]
    if re.search(r"\b(summar\w*|overview|patrol|what happened|whole run|status)\b", q) and not zone:
        return [] if "summarize_run" in called else [("summarize_run", {})]
    if re.search(r"\b(objects?|what was (seen|observed)|what did)\b", q) and not zone:
        return [] if "summarize_run" in called else [("summarize_run", {})]
    if re.search(r"\balerts?\b|\b(critical|escalated|resolved|acknowledged|open)\b", q):
        if "list_alerts" in called:
            return []
        args = {}
        if zone:
            args["zone"] = zone
        if rule:
            args["rule_id"] = rule
        if re.search(r"\bcritical\b", q):
            args["severity"] = "critical"
        return [("list_alerts", args)]
    if zone:
        wanted = [("query_events", {"zone": zone}), ("list_alerts", {"zone": zone})]
        return [] if "query_events" in called else wanted
    if kind or re.search(r"truck|pickup|blue|red|white|hoodie|jacket|sedan|van", q):
        if "search_events" in called:
            return []
        return [("search_events", {"query": _keywords(q) or kind or "person"})]
    if "search_events" in called or "summarize_run" in called:
        return []
    words = _keywords(q)
    return [("search_events", {"query": words})] if words else [("summarize_run", {})]


# ---- composing ----------------------------------------------------------------------------------
def compose(question: str, done: list[tuple[str, dict, dict]]) -> str:
    """The answer, written only from tool results. Every factual line cites its evidence."""
    lines: list[str] = []
    errors: list[str] = []
    windowed = bool(_WINDOW.search(question.lower())) and len(done) > 1
    for name, args, result in done:
        if "error" in result:
            errors.append(str(result["error"]))
            continue
        if windowed and name == "summarize_run":
            continue  # only fetched for the run's time span; the window answer is the events
        renderer = _RENDERERS.get(name)
        if renderer is None:
            continue
        lines.extend(renderer(result, args, question))
    lines = _dedupe(lines)
    if not lines:
        reason = errors[0] if errors else "the stored evidence has nothing matching that question"
        return f"Insufficient evidence: {reason}."
    return "\n".join(lines)


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _event_line(e: dict) -> str:
    attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
    frames = _cite(*e["frame_refs"][:6])
    more = " ..." if e["frames_total"] > len(e["frame_refs"]) else ""
    return (
        f"- {_cite(e['ref'])} {e['type']}" + (f" ({attrs})" if attrs else "")
        + f" in {e['zone']}, {_t(e['first_seen'])} to {_t(e['last_seen'])}"
        f" ({e['duration_seconds']:.0f}s, {e['status']}); {e['observation_count']} observation(s) "
        f"{frames}{more}"
    )  # fmt: skip


def _alert_lines(chain: dict) -> list[str]:
    a = chain["alert"]
    ref = a["ref"]
    facts = "; ".join(f"{k}={v}" for k, v in a["rule_facts"].items())
    lines = [
        f"{_cite(ref)} {a['title']}: severity {a['severity']}, status {a['status']}, verdict "
        f"{a['verdict']}, confidence {a['confidence']} (raised by deterministic rule "
        f"'{a['rule_id']}', not by a language model), triggered {_t(a['triggered_at'])}.",
        _line(f"Why it was raised: {a['reason']}", ref),
        _line(f"Rule facts compared: {facts}", ref),
    ]
    if len(a["ladder"]) > 1:
        steps = " then ".join(
            f"{s['level']} ({s['severity']}) at {_t(s['reached_at'])}" for s in a["ladder"]
        )
        lines.append(_line(f"Escalation ladder: {steps}", ref))
    if a["history"]:
        events = ", ".join(
            f"{h['kind']} by {h['actor']} at {_t(h['effective_at'])}" for h in a["history"]
        )
        lines.append(_line(f"Lifecycle of the alert: {events}", ref))
    events_cited = _cite(*a["event_refs"])
    frames = _cite(*a["frame_refs"])
    if events_cited:
        lines.append(f"Triggering events {events_cited} and frames {frames}.")
    for sample in chain.get("telemetry", []):
        lines.append(
            f"Telemetry evidence {_cite(sample['telemetry_ref'])}: {_t(sample['time'])} battery "
            f"{sample['battery_percent']:.0f}%, signal {_pct(sample['signal_percent'])}, altitude "
            f"{sample['altitude_m']:.0f} m (simulated)."
        )
    for block in chain.get("events", []):
        for o in block["observations"][:3]:
            note = "; carried forward, not a new inference" if o["carried_forward"] else ""
            lines.append(
                f"Evidence {_cite(o['ref'], o['frame_ref'])} {_t(o['time'])} {o['zone']}: "
                f"{_flat(o['description'])} (provenance {o['provenance'] or 'unknown'}{note})."
            )
    lines.append(
        _line(
            "Recommended action (a fixed table by rule and severity, not a model): "
            f"{a['recommended_action']}",
            ref,
        )
    )
    return lines


def _r_evidence(result: dict, args: dict, question: str) -> list[str]:
    if "alert" in result and "steps" in result:
        return _alert_lines(result)
    if "event" in result:  # get_evidence on an event, or get_event_evidence
        e = result["event"]
        lines = [_event_line(e)[2:]]
        for o in result.get("observations", [])[:4]:
            lines.append(
                f"Evidence {_cite(o['ref'], o['frame_ref'])} {_t(o['time'])} {o['zone']}: "
                f"{_flat(o['description'])}."
            )
        return lines
    return _r_entity(result, args, question)


def _r_entity(result: dict, args: dict, question: str) -> list[str]:
    if "similar_earlier_events" in result:
        e = result["event"]
        hints = result["similar_earlier_events"]
        head = f"Person event {_cite(e['ref'])} in {e['zone']}:"
        if not hints:
            return [
                f"{head} no earlier person event in this zone has similar described clothing "
                f"{_cite(e['ref'])}; people are never identity-linked."
            ]
        lines = [
            f"{head} {len(hints)} earlier event(s) in the same zone have similar described "
            f"clothing (a weak hint, score at most {result['hint_score_cap']}; this does not "
            f"show it is the same person):"
        ]
        lines += [
            f"- {_cite(h['ref'])} {_flat(h['why'])} (hint score {h['hint_score']})" for h in hints
        ]
        return lines
    if "entity" in result:
        ent = result["entity"]
        score = ent.get("likelihood_same_vehicle")
        likely = (
            f", likelihood {score:.2f} (an uncalibrated heuristic)" if score is not None else ""
        )
        lines = [
            f"A {ent['label']} {_cite(ent['ref'])} was observed in "
            f"{result.get('appearances_total', ent['appearance_count'])} separate appearance(s) "
            f"(a coarse attribute match{likely}; not confirmed to be the same vehicle):"
        ]
        lines += [_event_line(e) for e in result.get("appearances", [])]
        for alert in result.get("alerts", []):
            lines.append(
                f"Related alert {_cite(alert['ref'])} {alert['severity']}: {_flat(alert['title'])}."
            )
        return lines
    return []


def _r_events(result: dict, args: dict, question: str) -> list[str]:
    events = result.get("events", [])
    if not events:
        return []
    head = f"{result['total']} matching event(s):"
    lines = [head, *[_event_line(e) for e in events]]
    if result.get("truncated"):
        lines.append(f"(showing {result['returned']} of {result['total']} events)")
    return lines


def _r_alerts(result: dict, args: dict, question: str) -> list[str]:
    alerts = result.get("alerts", [])
    if not alerts:
        return []
    q = question.lower()
    if re.search(r"\b(open|active|unresolved)\b", q):
        alerts = [a for a in alerts if a["status"] != "resolved"]
    elif re.search(r"\bresolved\b", q):
        alerts = [a for a in alerts if a["status"] == "resolved"]
    if not alerts:
        return []
    lines = [f"{len(alerts)} alert(s) raised by the deterministic rules:"]
    for a in alerts:
        lines.append(
            f"- {_cite(a['ref'])} {a['severity']}/{a['status']}: {a['title']} "
            f"({a['zone']}, {_t(a['triggered_at'])}) {_cite(*a['event_refs'])}".rstrip()
        )
    return lines


def _r_telemetry(result: dict, args: dict, question: str) -> list[str]:
    t = result["telemetry"]
    return [
        f"Drone telemetry {_cite(t['telemetry_ref'])} at {_t(t['time'])}: battery "
        f"{t['battery_percent']:.0f}%, signal {_pct(t['signal_percent'])}, altitude "
        f"{t['altitude_m']:.0f} m, zone {t['zone']} (simulated telemetry)."
    ]


def _r_timeline(result: dict, args: dict, question: str) -> list[str]:
    items = result.get("timeline", [])
    if not items:
        return []
    lines = [f"Timeline ({result['total']} item(s), earliest first):"]
    for item in items:
        ref = item.get("event_ref") or item.get("alert_ref")
        lines.append(f"- {_t(item['time'])} {item['kind']}: {item['summary']} {_cite(ref)}")
    if result.get("truncated"):
        lines.append(f"(showing {result['returned']} of {result['total']} items)")
    return lines


def _r_frames(result: dict, args: dict, question: str) -> list[str]:
    found = result.get("observations", [])
    if not found:
        return []
    lines = [f"{result['total']} matching frame observation(s):"]
    lines += [
        f"- {_cite(o['ref'], o['frame_ref'])} {_t(o['time'])} {o['zone']}: {o['description']}"
        for o in found
    ]
    return lines


def _r_frame(result: dict, args: dict, question: str) -> list[str]:
    o = result["observation"]
    return [
        f"Frame {_cite(o['frame_ref'], o['ref'])} {_t(o['time'])} {o['zone']}: {o['description']}"
    ]


# Aggregate lines of the run summary (counts, span, per-zone and per-type totals) are facts about
# the whole store, not about any one alert/event/entity, so no citation can honestly back them.
# The planner leaves them out rather than attach an unrelated citation; the template summary
# (`summarize`) has them.
_UNCITABLE_SUMMARY_LINES = (
    "Stored run:",
    "Observation time span:",
    "Observations by zone:",
    "Detected objects:",
)


def _r_summary(result: dict, args: dict, question: str) -> list[str]:
    lines = [
        line
        for line in render_summary(result).splitlines()
        if not line.startswith((*_UNCITABLE_SUMMARY_LINES, "Limitations:"))
    ]
    # The store's fixed limitation notes are disclosures, not claims about the evidence; one per
    # line under the "Limitation:" prefix, which the citation gate treats as a note.
    lines += [f"Limitation: {note}" for note in result.get("limitations", [])]
    return lines


_RENDERERS = {
    "get_evidence": _r_evidence,
    "get_alert": lambda r, a, q: _alert_lines({"alert": r["alert"]}) if "alert" in r else [],
    "entity_history": _r_entity,
    "query_events": _r_events,
    "search_events": _r_events,
    "get_event": lambda r, a, q: [_event_line(r["event"])[2:]],
    "list_alerts": _r_alerts,
    "telemetry_at": _r_telemetry,
    "get_timeline": _r_timeline,
    "search_frames": _r_frames,
    "get_frame": _r_frame,
    "summarize_run": _r_summary,
    "list_entities": lambda r, a, q: [
        f"- {_cite(e['ref'])} {e['label']}, {e['appearance_count']} appearance(s)"
        for e in r.get("entities", [])
    ],
}

__all__ = ["NAME", "PlannerChatModel", "compose", "plan", "Zone"]
