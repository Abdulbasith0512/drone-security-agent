"""Deterministic template analyst: works with no API key and no network.

It recognises a limited set of question shapes (alerts, events, entities, zones, objects, run
summary, keyword search) with plain pattern matching and answers ONLY by reading stored evidence
through `AnalystTools`, citing every id it mentions. It is not a language model and does not try to
understand arbitrary questions; unknown ones get an honest "insufficient evidence" reply.
Its purposes: offline evaluation, API-outage fallback and reproducible tests/demos.
No LangChain here.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from drone_agent.agent.summary import render_summary
from drone_agent.agent.timeparse import parse_moment, parse_time_window
from drone_agent.agent.tools import AnalystTools

SUPPORTED_QUESTIONS = (
    "What happened at the main gate? (or driveway, parking, warehouse, restricted zone)",
    "Was the blue pickup seen more than once?",
    "Why was the repeat-vehicle alert raised? (also loitering, after-hours, restricted-zone)",
    "Was there a low-battery / signal-loss / altitude / geofence alert?",
    "What was the battery level at 00:10?",
    "Which alerts are still open? (or resolved, escalated, critical)",
    "What happened between 00:10 and 00:20? (or after midnight, before 6 PM)",
    "Show me the evidence for the loitering alert.",
    "Were any people seen after hours? / Was the person at the gate seen before?",
    "List the alerts. / Summarize the run. / What objects were observed?",
    "Describe ALR-..., EVT-..., or ENT-... by id.",
)

_ZONES = (
    (r"main[ _]?gate|\bgate\b", "main_gate"),
    (r"driveway", "driveway"),
    (r"parking", "parking_area"),
    (r"warehouse", "warehouse"),
    (r"restricted", "restricted_zone"),
)
_WORDS = re.compile(r"[a-z]+")
_STOP = frozenset(
    _WORDS.findall(
        "a an the of in at on to for and or is was were are be been did do does any some show me "
        "what which who when where why how there it its this that those they them with from about "
        "seen see saw more than once again tell give list"
    )
)
_COLORS = frozenset(
    _WORDS.findall("blue red white black green yellow silver grey gray orange brown dark")
)
_PEOPLE = re.compile(r"\b(people|persons?|pedestrians?|humans?)\b")
DETAIL_ENTITIES = 3  # vehicle entities answered with their full appearance history
_VEHICLE = re.compile(r"\b(vehicle|pickup|truck|car|van|suv|lorry|bike|motorcycle)s?\b")
_ID = re.compile(r"\b(ALR-[A-Z]+-[A-Z0-9-]+|EVT-[PVO]-\d+|ENT-V-\d+)\b", re.IGNORECASE)


@dataclass
class OfflineAnswer:
    text: str
    subject: tuple[str, str] | None = None  # ("entity"|"event"|"alert"|"zone", id) for follow-ups
    matched: str = "none"  # which template answered (for tests and display)
    notes: list[str] = field(default_factory=list)


def _t(iso: str) -> str:
    return iso[:19].replace("T", " ") + iso[19:]


def _cites(refs: list[str]) -> str:
    return " ".join(f"[{r}]" for r in refs)


def _telemetry_line(t: dict, label: str) -> str:
    signal = "not reported" if t["signal_percent"] is None else f"{t['signal_percent']:.0f}%"
    stale = " (stale sample)" if t.get("stale") else ""
    return (
        f"{label} [{t['telemetry_ref']}] {_t(t['time'])}: battery {t['battery_percent']:.0f}%, "
        f"signal {signal}, altitude {t['altitude_m']:.0f} m, zone {t['zone']}{stale} "
        "(simulated telemetry)."
    )


def _not_shown(shown: int, total: int, noun: str) -> str | None:
    """A visible note when a list was cut (None when everything is shown). Lists are always in
    time order, so the omitted items are the latest ones."""
    if shown >= total:
        return None
    return f"(showing {shown} of {total} {noun}, earliest first; {total - shown} not shown)"


def _note_lines(*notes: str | None) -> list[str]:
    return [n for n in notes if n]


def _zone_in(q: str) -> str | None:
    for pattern, zone in _ZONES:
        if re.search(pattern, q):
            return zone
    return None


# A safety topic only means "the alert" when the question is about an alert or a fault; otherwise
# "the battery at 00:10" is a telemetry question.
_FAULT_WORDS = re.compile(
    r"alert|\bwhy\b|raised|trigger|warn|lost|loss|violat|breach|anomal|dropout|\blow\b"
)
_SAFETY_RULES = (
    (r"battery", "low_battery"),
    (r"signal|link|dropout|connection", "signal_loss"),
    (r"altitude", "altitude_anomaly"),
    (r"geofence|perimeter|boundary", "geofence_violation"),
)
_TELEMETRY = re.compile(r"\b(battery|signal|altitude|telemetry|link)\b")
_TIME_WINDOW = re.compile(r"\b(after|before|between|since|until|from)\b.*\d|midnight|noon")
_STATUS_WORDS = (
    (r"\b(open|active|unresolved|still)\b", "unresolved"),
    (r"\bresolved\b", "resolved"),
    (r"\bescalated\b", "escalated"),
    (r"\backnowledged\b", "acknowledged"),
)
_RECURRENCE = re.compile(r"\b(previous\w*|earlier|before|again|seen|same)\b")


def _rule_in(q: str) -> str | None:
    if "loiter" in q:
        return "loitering"
    if re.search(r"after[- ]?hours|overnight|at night", q):
        return "after_hours_presence"
    if "repeat" in q:
        return "repeat_vehicle"
    if "restricted" in q and re.search(r"alert|entry|entered|enter|breach", q):
        return "restricted_zone_entry"
    if _FAULT_WORDS.search(q):
        for pattern, rule in _SAFETY_RULES:
            if re.search(pattern, q):
                return rule
    return None


def _event_line(e: dict) -> str:
    attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
    detail = f" ({attrs})" if attrs else ""
    acts = f", activity {', '.join(e['activities'])}" if e["activities"] else ""
    return (
        f"[{e['ref']}] {e['type']}{detail} in {e['zone']}, {_t(e['first_seen'])} to "
        f"{_t(e['last_seen'])}{acts}; {e['observation_count']} observation(s) "
        f"{_cites(e['frame_refs'])}" + (" ..." if e["frames_total"] > len(e["frame_refs"]) else "")
    )


class OfflineAnalyst:
    def __init__(self, tools: AnalystTools) -> None:
        self.tools = tools

    # -- entry point ------------------------------------------------------------------------
    def answer(self, question: str, context: dict | None = None) -> OfflineAnswer:
        context = context or {}
        q = question.lower().strip()
        for match in _ID.finditer(question):
            return self._by_id(match.group(1).upper(), q)
        rule = _rule_in(q)
        zone = _zone_in(q) if rule != "restricted_zone_entry" else None
        if _TELEMETRY.search(q) and not rule:
            return self._telemetry(question)
        if _TIME_WINDOW.search(q) and not rule:
            found = self._window(question, zone, q)
            if found:
                return found
        if re.search(r"\bpeople|person|anyone|someone|human", q) and rule == "after_hours_presence":
            return self._alerts_for_rule(rule, q, people=True)
        if rule:
            return self._alerts_for_rule(rule, q)
        if re.search(r"\balerts?\b", q):
            return self._alert_list(q)
        if _PEOPLE.search(q) and _RECURRENCE.search(q) and not _VEHICLE.search(q):
            return self._person_recurrence(zone)
        if _VEHICLE.search(q):
            if _PEOPLE.search(q) and not set(re.findall(r"[a-z]+", q)) & _COLORS:
                # people AND vehicles asked about in general: answer both, never just one
                return self._zone(zone) if zone else self._objects(compound=True)
            return self._vehicle(q)
        if zone:
            return self._zone(zone)
        if re.search(
            r"\bobjects?\b|what was (seen|observed|in the video)|what.*(seen|observed)", q
        ):
            return self._objects()
        if re.search(r"summar|overview|metric|statistic|how many|what happened|run\b", q):
            return self._summary()
        follow = self._follow_up(q, context)
        if follow:
            return follow
        return self._keyword_search(q)

    # -- by id ------------------------------------------------------------------------------
    def _by_id(self, ident: str, q: str) -> OfflineAnswer:
        if ident.startswith("ALR-"):
            return self._alert(ident, evidence=True)
        if ident.startswith("EVT-"):
            return self._event(ident)
        return self._entity(ident)

    # -- alerts -----------------------------------------------------------------------------
    def _alert(self, alert_id: str, evidence: bool) -> OfflineAnswer:
        res = self.tools.get_alert(alert_id)
        if "error" in res:
            return OfflineAnswer(f"Insufficient evidence: {res['error']}.", matched="alert-missing")
        a = res["alert"]
        facts = "; ".join(f"{k}={v}" for k, v in a["rule_facts"].items())
        lines = [
            f"[{a['ref']}] {a['title']}: severity {a['severity']} (assigned by deterministic rule "
            f"'{a['rule_id']}', not by a language model), zone {a['zone']}, triggered "
            f"{_t(a['triggered_at'])}.",
            f"Why it was raised: {a['reason']}",
            f"Rule facts compared: {facts}.",
            self._lifecycle_line(a),
            f"Recommended action (a fixed table entry for this rule, not model-written): "
            f"{a['recommended_action']}",
            self._trigger_line(a),
        ]
        for stamp in a.get("telemetry_times", [])[:1]:  # a safety alert: show the drone's state
            sample = self.tools.telemetry_at(stamp)
            if "telemetry" in sample:
                lines.append(_telemetry_line(sample["telemetry"], "Drone telemetry at the alert"))
        if evidence:
            for ev in res["events"]:
                lines.append(self._evidence_block(ev["ref"].split(":", 1)[1]))
        return OfflineAnswer("\n".join(lines), ("alert", alert_id), "alert")

    @staticmethod
    def _trigger_line(a: dict) -> str:
        if not a["event_refs"] and not a["frame_refs"]:
            return "Triggered by drone telemetry, not by a camera event (see the telemetry below)."
        line = f"Triggering events: {_cites(a['event_refs'])}"
        if a["entity_ref"]:
            line += f"; entity {_cites([a['entity_ref']])}"
        line += f". Frames {_cites(a['frame_refs'])}"
        if a["frames_total"] > len(a["frame_refs"]):
            line += f" (first {len(a['frame_refs'])} of {a['frames_total']})"
        return line + "."

    @staticmethod
    def _lifecycle_line(a: dict) -> str:
        parts = [f"Status: {a['status']}"]
        if a.get("verdict"):
            parts.append(f"verdict {a['verdict']}")
        if a.get("confidence") is not None:
            parts.append(f"confidence {a['confidence']:.2f} (a rule-derived score)")
        if a.get("detection_latency_seconds"):  # only when the alert was raised late
            parts.append(
                f"raised {a['detection_latency_seconds']:.0f}s after the condition first held"
            )
        line = ", ".join(parts) + "."
        ladder = a.get("ladder") or []
        if len(ladder) > 1:
            steps = " -> ".join(f"{s['severity']} at {_t(s['reached_at'])}" for s in ladder)
            line += f" Escalation: {steps}."
        if a.get("resolved_at"):
            line += f" Resolved at {_t(a['resolved_at'])}."
        return line

    def _evidence_block(self, event_id: str, show: int = 5) -> str:
        res = self.tools.get_event_evidence(event_id)
        if "error" in res:
            return f"No evidence found for {event_id}."
        lines = [f"Evidence for {_event_line(res['event'])}:"]
        for o in res["observations"][:show]:
            lines.append(
                f"  - [{o['ref']}] [{o['frame_ref']}] {_t(o['time'])} {o['zone']}: "
                f"{o['description']}"
                + (" (carried forward, not a new inference)" if o["carried_forward"] else "")
            )
        if res["observations_total"] > show:
            lines.append(f"  ... {res['observations_total'] - show} more observation(s) not shown.")
        return "\n".join(lines)

    def _alerts_for_rule(self, rule: str, q: str, people: bool = False) -> OfflineAnswer:
        res = self.tools.list_alerts(rule_id=rule)
        if not res["alerts"]:
            what = "after-hours presence" if people else f"'{rule}'"
            return OfflineAnswer(
                f"Insufficient evidence: no {what} alert is stored, so the evidence does not "
                "show one. "
                "(Alerts are only created by the deterministic rules.)",
                matched="rule-none",
            )
        evidence = bool(re.search(r"evidence|frames?|proof|show", q))
        if people:
            lines = [
                f"Yes, the rules stored {res['total']} after-hours presence alert(s) for people "
                "(observed outside the configured allowed hours):"
            ]
            for a in res["alerts"]:
                lines.append(
                    f"- [{a['ref']}] {a['zone']}, triggered {_t(a['triggered_at'])}: {a['reason']} "
                    f"{_cites(a['event_refs'])}"
                )
            lines.append(
                "These are separate person events; people are not identity-linked across events."
            )
            return OfflineAnswer(
                "\n".join(lines), ("alert", res["alerts"][0]["ref"].split(":", 1)[1]), "after-hours"
            )
        parts = []
        subject = None
        for a in res["alerts"][:5]:
            alert_id = a["ref"].split(":", 1)[1]
            ans = self._alert(alert_id, evidence=evidence)
            parts.append(ans.text)
            subject = subject or ans.subject
        if res["total"] > 5:
            parts.append(f"({res['total'] - 5} more alert(s) for this rule not shown.)")
        return OfflineAnswer("\n\n".join(parts), subject, "alert-rule")

    def _alert_list(self, q: str = "") -> OfflineAnswer:
        res = self.tools.list_alerts(severity="critical" if re.search(r"\bcritical\b", q) else None)
        status = next((name for pattern, name in _STATUS_WORDS if re.search(pattern, q)), None)
        alerts = res["alerts"]
        if status == "unresolved":
            alerts = [a for a in alerts if a["status"] != "resolved"]
        elif status:
            alerts = [a for a in alerts if a["status"] == status]
        if not alerts:
            what = "matching " if (status or "critical" in q) else ""
            return OfflineAnswer(
                f"Insufficient evidence: no {what}alerts are stored.", matched="alerts-none"
            )
        noun = {"unresolved": "unresolved ", None: ""}.get(status, f"{status} ")
        if "critical" in q:
            noun += "critical "
        lines = [f"{len(alerts)} stored {noun}alert(s) (raised by deterministic rules):"]
        for a in alerts:
            lines.append(
                f"- [{a['ref']}] {a['severity']}/{a['status']}: {a['title']} ({a['zone']}, "
                f"{_t(a['triggered_at'])}) {_cites(a['event_refs'])}".rstrip()
            )
        if res["truncated"]:
            lines.append(f"(showing {res['returned']} of {res['total']})")
        return OfflineAnswer("\n".join(lines), None, "alert-list")

    # -- telemetry, time windows, person recurrence -----------------------------------------
    def _span(self) -> tuple[datetime, datetime] | None:
        span = self.tools.summarize_run().get("observation_span")
        if not span:
            return None
        return datetime.fromisoformat(span["first"]), datetime.fromisoformat(span["last"])

    def _telemetry(self, question: str) -> OfflineAnswer:
        span = self._span()
        if span is None:
            return OfflineAnswer(
                "Insufficient evidence: no observations are stored, so there is no run to read "
                "telemetry for.",
                matched="telemetry-none",
            )
        moment = parse_moment(question, *span)
        if re.search(r"\bat\s+\d|\baround\s+\d|midnight|noon", question.lower()) and moment is None:
            return OfflineAnswer(
                f"Insufficient evidence: that time is outside the stored run "
                f"({_t(span[0].isoformat())} to {_t(span[1].isoformat())}).",
                matched="telemetry-outside",
            )
        label = "Drone telemetry at that time" if moment else "Latest stored drone telemetry"
        res = self.tools.telemetry_at((moment or span[1]).isoformat())
        if "error" in res:
            return OfflineAnswer(
                f"Insufficient evidence: {res['error']}.", matched="telemetry-none"
            )
        return OfflineAnswer(_telemetry_line(res["telemetry"], label), None, "telemetry")

    def _window(self, question: str, zone: str | None, q: str) -> OfflineAnswer | None:
        span = self._span()
        window = parse_time_window(question, *span) if span else None
        if window is None:
            return None
        start, end = window
        args = {**({"start": start.isoformat()} if start else {})}
        args.update({"end": end.isoformat()} if end else {})
        kind = "person" if _PEOPLE.search(q) else "vehicle" if _VEHICLE.search(q) else None
        events = self.tools.query_events(**args, zone=zone, object_type=kind)
        lo, hi = (
            _t(start.isoformat()) if start else "the run start",
            (_t(end.isoformat()) if end else "the run end"),
        )
        if not events["events"]:
            return OfflineAnswer(
                f"Insufficient evidence: no matching events are stored between {lo} and {hi}.",
                matched="window-none",
            )
        lines = [
            f"{events['total']} event(s) between {lo} and {hi} (an event counts when it overlaps):"
        ]
        lines.extend(f"- {_event_line(e)}" for e in events["events"])
        lines += _note_lines(_not_shown(len(events["events"]), events["total"], "events"))
        if not (zone or kind):  # the timeline has no type/zone filter
            timeline = self.tools.get_timeline(**args)
            for item in timeline.get("timeline", []):
                if item["kind"].startswith("alert_"):
                    ref = item.get("alert_ref") or item.get("event_ref")
                    lines.append(
                        f"- {_t(item['time'])} {item['kind'].replace('_', ' ')}: "
                        f"{item['summary']} [{ref}]"
                    )
        return OfflineAnswer("\n".join(lines), None, "window")

    def _person_recurrence(self, zone: str | None) -> OfflineAnswer:
        """People are never identity-linked; this reports only a weak, labelled similarity hint."""
        listing = self.tools.query_events(object_type="person", zone=zone)
        if not listing["events"]:
            return OfflineAnswer(
                "Insufficient evidence: no person events are stored"
                + (f" for {zone}." if zone else "."),
                matched="person-none",
            )
        latest = listing["events"][-1]
        res = self.tools.entity_history(latest["ref"].split(":", 1)[1])
        if "error" in res:
            return OfflineAnswer(f"Insufficient evidence: {res['error']}.", matched="person-none")
        event = res["event"]
        hints = res["similar_earlier_events"]
        lead = f"Latest person event [{event['ref']}] in {event['zone']}"
        if not hints:
            text = (
                f"{lead}: no earlier person event in that zone has similar described clothing "
                "[" + event["ref"] + "]. People are never identity-linked, so this does not show "
                "the person is new either."
            )
        else:
            lines = [
                f"{lead}: {len(hints)} earlier event(s) in the same zone have similar described "
                f"clothing (a weak hint, score at most {res['hint_score_cap']}; this does not "
                "show it is the same person):"
            ]
            lines += [f"- [{h['ref']}] {h['why']} (hint score {h['hint_score']})" for h in hints]
            text = "\n".join(lines)
        return OfflineAnswer(text, ("event", event["ref"].split(":", 1)[1]), "person-recurrence")

    # -- events / entities / zones ----------------------------------------------------------
    def _event(self, event_id: str) -> OfflineAnswer:
        res = self.tools.get_event_evidence(event_id)
        if "error" in res:
            return OfflineAnswer(f"Insufficient evidence: {res['error']}.", matched="event-missing")
        return OfflineAnswer(self._evidence_block(event_id), ("event", event_id), "event")

    def _entity(self, entity_id: str) -> OfflineAnswer:
        res = self.tools.get_entity_history(entity_id)
        if "error" in res:
            return OfflineAnswer(
                f"Insufficient evidence: {res['error']}.", matched="entity-missing"
            )
        ent = res["entity"]
        lines = [self._entity_lead(ent, res["appearances_total"])]
        for e in res["appearances"]:
            lines.append(f"- {_event_line(e)}")
        lines += _note_lines(
            _not_shown(len(res["appearances"]), res["appearances_total"], "appearances")
        )
        for a in res["alerts"]:
            lines.append(f"Related alert: [{a['ref']}] {a['severity']}: {a['title']}.")
        return OfflineAnswer("\n".join(lines), ("entity", entity_id), "entity")

    @staticmethod
    def _entity_lead(ent: dict, count: int) -> str:
        if ent["identity_confirmed"]:
            kind = "identity supported by explicit plate evidence"
        else:
            kind = "a coarse attribute match; not confirmed to be the same vehicle"
        return (
            f"A {ent['label']} [{ent['ref']}] was observed in {count} separate appearance(s) "
            f"({kind}):"
        )

    def _vehicle(self, q: str) -> OfflineAnswer:
        listing = self.tools.list_entities()
        entities, total = listing["entities"], listing["total"]
        words = set(re.findall(r"[a-z]+", q))
        scored = []
        for ent in entities:
            vals = {w for v in ent["attributes"].values() for w in re.findall(r"[a-z]+", v.lower())}
            scored.append((len(vals & words - _STOP), ent))
        best = max((s for s, _ in scored), default=0)
        chosen = [e for s, e in scored if s == best and (best > 0 or not words & _COLORS)]
        if not chosen:
            return self._vehicle_events(q)
        many = bool(re.search(r"more than once|again|repeat|multiple|how many|several|times", q))
        lines = []
        for ent in chosen[:DETAIL_ENTITIES]:
            res = self.tools.get_entity_history(ent["ref"].split(":", 1)[1])
            count = res["appearances_total"]
            head = ""
            if many:
                head = "Yes. " if count > 1 else "No, only one appearance is stored. "
            lines.append(head + self._entity_lead(res["entity"], count))
            lines.extend(f"- {_event_line(e)}" for e in res["appearances"])
            lines.extend(
                f"Related alert: [{a['ref']}] {a['severity']}: {a['title']}." for a in res["alerts"]
            )
        rest = chosen[DETAIL_ENTITIES:]
        if rest:  # every matching entity stays visible: full history for a few, a line for the rest
            lines.append(
                f"(showing full history for {DETAIL_ENTITIES} of {len(chosen)} similar-vehicle "
                f"entities; {len(rest)} more are listed briefly below)"
            )
            for ent in rest:
                lines.append(
                    f"- [{ent['ref']}] {ent['label']}, {ent['appearance_count']} appearance(s) "
                    f"{_cites(ent['event_refs'])}"
                )
        note = _not_shown(len(entities), total, "similar-vehicle entities")
        if note:
            lines.append(note)
            if best > 0 or words & _COLORS:  # the search itself only saw the listed entities
                lines.append("(the search covered only the entities listed by the tool)")
        return OfflineAnswer(
            "\n".join(lines), ("entity", chosen[0]["ref"].split(":", 1)[1]), "vehicle"
        )

    def _vehicle_events(self, q: str) -> OfflineAnswer:
        res = self.tools.list_events(object_type="vehicle")
        if not res["events"]:
            return OfflineAnswer(
                "Insufficient evidence: no vehicle events are stored for that description.",
                matched="vehicle-none",
            )
        lines = ["No similar-vehicle entity matches the description. Vehicle events stored:"]
        lines.extend(f"- {_event_line(e)}" for e in res["events"])
        lines += _note_lines(_not_shown(len(res["events"]), res["total"], "vehicle events"))
        return OfflineAnswer("\n".join(lines), None, "vehicle-events")

    def _zone(self, zone: str) -> OfflineAnswer:
        events = self.tools.list_events(zone=zone)
        alerts = self.tools.list_alerts(zone=zone)
        obs = self.tools.search_observations(zone=zone, limit=3)
        if not events["events"] and not obs["total"]:
            return OfflineAnswer(
                f"Insufficient evidence: nothing is stored for {zone}.", matched="zone-none"
            )
        lines = [f"In {zone}: {obs['total']} stored observation(s) and {events['total']} event(s)."]
        lines.extend(f"- {_event_line(e)}" for e in events["events"])
        lines += _note_lines(_not_shown(len(events["events"]), events["total"], "events"))
        for a in alerts["alerts"]:
            lines.append(
                f"Alert [{a['ref']}] {a['severity']}: {a['title']} (rule '{a['rule_id']}') "
                f"{_cites(a['event_refs'])}"
            )
        lines += _note_lines(_not_shown(len(alerts["alerts"]), alerts["total"], "alerts"))
        for o in obs["observations"]:
            lines.append(f"Observation [{o['ref']}] [{o['frame_ref']}]: {o['description']}")
        if obs["truncated"]:  # the observation lines are a sample, not the full list
            lines.append(f"(showing {obs['returned']} of {obs['total']} observations as a sample)")
        return OfflineAnswer("\n".join(lines), ("zone", zone), "zone")

    # -- run level --------------------------------------------------------------------------
    def _summary(self) -> OfflineAnswer:
        return OfflineAnswer(render_summary(self.tools.summarize_run()), None, "summary")

    def _objects(self, compound: bool = False) -> OfflineAnswer:
        facts = self.tools.summarize_run()
        if not facts["counts"]["observations"]:
            return OfflineAnswer(
                "Insufficient evidence: no observations are stored.", matched="objects-none"
            )
        counts = ", ".join(f"{k} {v}" for k, v in facts["object_counts"].items()) or "none"
        lines = [
            f"Objects detected across {facts['counts']['observations']} stored observation(s): "
            f"{counts}."
        ]
        carried = sum(n for s, n in facts["sources"].items() if s.endswith("+carried"))
        if carried:
            lines.append(f"Counts include {carried} carried-forward (repeated) observation(s).")
        if compound:  # people AND vehicles: state both categories explicitly, from full totals
            by_type = []
            for kind in ("person", "vehicle", "other"):
                total = self.tools.list_events(object_type=kind)["total"]
                if total:
                    by_type.append(f"{kind} {total}")
            if by_type:
                lines.append("Events by type: " + ", ".join(by_type) + ".")
        for e in facts["events"]:
            attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
            lines.append(
                f"- [{e['ref']}] {e['type']}"
                + (f" ({attrs})" if attrs else "")
                + f" in {e['zone']}, {_t(e['first_seen'])}"
            )
        lines += _note_lines(_not_shown(len(facts["events"]), facts["events_total"], "events"))
        for ent in facts["entities"]:
            lines.append(f"- similar-vehicle memory [{ent['ref']}]: {ent['label']} (coarse match)")
        lines += _note_lines(
            _not_shown(len(facts["entities"]), facts["entities_total"], "similar-vehicle entities")
        )
        return OfflineAnswer("\n".join(lines), None, "objects")

    # -- follow-ups and fallback ------------------------------------------------------------
    def _follow_up(self, q: str, context: dict) -> OfflineAnswer | None:
        if not context or not re.search(r"\b(it|they|them|that|those|this|there|where|when)\b", q):
            return None
        for kind in ("entity", "alert", "event", "zone"):
            if kind in context:
                ident = context[kind]
                if kind == "entity":
                    return self._entity(ident)
                if kind == "alert":
                    return self._alert(ident, evidence=bool(re.search(r"evidence|frame", q)))
                if kind == "event":
                    return self._event(ident)
                return self._zone(ident)
        return None

    def _keyword_search(self, q: str) -> OfflineAnswer:
        words = [w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2 and w not in _STOP]
        for candidate in ([" ".join(words)] if words else []) + words[:3]:
            res = self.tools.search_observations(query=candidate, limit=5)
            if res.get("observations"):
                lines = [
                    f"Keyword search for '{candidate}' found {res['total']} stored observation(s) "
                    "(a text match, not an interpretation of your question):"
                ]
                lines.extend(
                    f"- [{o['ref']}] [{o['frame_ref']}] {_t(o['time'])} {o['zone']}: "
                    f"{o['description']}"
                    for o in res["observations"]
                )
                if res["truncated"]:
                    lines.append(f"(showing {res['returned']} of {res['total']})")
                return OfflineAnswer("\n".join(lines), None, "keyword")
        return OfflineAnswer(
            "Insufficient evidence: the template analyst could not match that question to stored "
            "evidence. It can answer:\n" + "\n".join(f"- {s}" for s in SUPPORTED_QUESTIONS),
            matched="insufficient",
        )
