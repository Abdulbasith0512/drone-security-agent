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

from drone_agent.agent.summary import render_summary
from drone_agent.agent.tools import AnalystTools

SUPPORTED_QUESTIONS = (
    "What happened at the main gate? (or driveway, parking, warehouse, restricted zone)",
    "Was the blue pickup seen more than once?",
    "Why was the repeat-vehicle alert raised? (also loitering, after-hours, restricted-zone)",
    "Show me the evidence for the loitering alert.",
    "Were any people seen after hours?",
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


def _zone_in(q: str) -> str | None:
    for pattern, zone in _ZONES:
        if re.search(pattern, q):
            return zone
    return None


def _rule_in(q: str) -> str | None:
    if "loiter" in q:
        return "loitering"
    if re.search(r"after[- ]?hours|overnight|at night", q):
        return "after_hours_presence"
    if "repeat" in q:
        return "repeat_vehicle"
    if "restricted" in q and re.search(r"alert|entry|entered|enter|breach", q):
        return "restricted_zone_entry"
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
        if re.search(r"\bpeople|person|anyone|someone|human", q) and rule == "after_hours_presence":
            return self._alerts_for_rule(rule, q, people=True)
        if rule:
            return self._alerts_for_rule(rule, q)
        if re.search(r"\balerts?\b", q):
            return self._alert_list()
        if _VEHICLE.search(q):
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
            f"Triggering events: {_cites(a['event_refs'])}"
            + (f"; entity {_cites([a['entity_ref']])}" if a["entity_ref"] else "")
            + f". Frames {_cites(a['frame_refs'])}"
            + (
                f" (first {len(a['frame_refs'])} of {a['frames_total']})"
                if a["frames_total"] > len(a["frame_refs"])
                else ""
            )
            + ".",
        ]
        if evidence:
            for ev in res["events"]:
                lines.append(self._evidence_block(ev["ref"].split(":", 1)[1]))
        return OfflineAnswer("\n".join(lines), ("alert", alert_id), "alert")

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

    def _alert_list(self) -> OfflineAnswer:
        res = self.tools.list_alerts()
        if not res["alerts"]:
            return OfflineAnswer(
                "Insufficient evidence: no alerts are stored.", matched="alerts-none"
            )
        lines = [f"{res['total']} stored alert(s) (raised by deterministic rules):"]
        for a in res["alerts"]:
            lines.append(
                f"- [{a['ref']}] {a['severity']}: {a['title']} ({a['zone']}, "
                f"{_t(a['triggered_at'])}) {_cites(a['event_refs'])}"
            )
        if res["truncated"]:
            lines.append(f"(showing {res['returned']} of {res['total']})")
        return OfflineAnswer("\n".join(lines), None, "alert-list")

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
        entities = self.tools.list_entities()["entities"]
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
        for ent in chosen[:3]:
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
        for a in alerts["alerts"]:
            lines.append(
                f"Alert [{a['ref']}] {a['severity']}: {a['title']} (rule '{a['rule_id']}') "
                f"{_cites(a['event_refs'])}"
            )
        for o in obs["observations"]:
            lines.append(f"Observation [{o['ref']}] [{o['frame_ref']}]: {o['description']}")
        return OfflineAnswer("\n".join(lines), ("zone", zone), "zone")

    # -- run level --------------------------------------------------------------------------
    def _summary(self) -> OfflineAnswer:
        return OfflineAnswer(render_summary(self.tools.get_run_summary()), None, "summary")

    def _objects(self) -> OfflineAnswer:
        facts = self.tools.get_run_summary()
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
        for e in facts["events"]:
            attrs = ", ".join(f"{k}={v}" for k, v in e["attributes"].items())
            lines.append(
                f"- [{e['ref']}] {e['type']}"
                + (f" ({attrs})" if attrs else "")
                + f" in {e['zone']}, {_t(e['first_seen'])}"
            )
        for ent in facts["entities"]:
            lines.append(f"- similar-vehicle memory [{ent['ref']}]: {ent['label']} (coarse match)")
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
