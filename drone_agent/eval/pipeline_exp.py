"""Experiments 2-4 (RQ2-RQ4): temporal compression, flicker tolerance, entity memory."""

from datetime import datetime

from drone_agent.eval.base import IST, ExperimentResult, Timeline, det, fmt
from drone_agent.eval.harness import run_pipeline, scenario_observations
from drone_agent.models import CloseReason, Event, ObjectType, Observation, Severity, Zone
from drone_agent.pipeline import EntityLinker, EventConfig
from drone_agent.pipeline.config import AggregationSettings

DAY = datetime(2026, 9, 29, 12, 0, tzinfo=IST)  # daytime: after-hours never fires
NIGHT = datetime(2026, 9, 29, 23, 30, tzinfo=IST)


# ---- Experiment 2: temporal compression -----------------------------------------------------
def long_dwell_fixture() -> list[Observation]:
    """One person for 60 frames (10 s apart) and a parked pickup for 30 of them, same zone."""
    tl = Timeline(DAY)
    for i in range(60):
        objects = [det("person", "standing", clothing="dark jacket")]
        if 10 <= i < 40:
            objects.append(det("vehicle", "parked", color="blue", body_type="pickup"))
        tl.add(i * 10, Zone.MAIN_GATE, *objects)
    return tl.observations


def _compression(name: str, observations: list[Observation]) -> dict:
    events, _, _ = run_pipeline(observations)
    detections = sum(len(o.detected_objects) for o in observations)
    by_id = {o.observation_id: o for o in observations}
    retained = sum(e.evidence_count for e in events)
    spans_ok = all(
        e.first_seen == min(by_id[i].timestamp for i in e.observation_ids)
        and e.last_seen == max(by_id[i].timestamp for i in e.observation_ids)
        for e in events
    )
    durations = [e.duration_seconds for e in events]
    return {
        "fixture": name,
        "observations": len(observations),
        "detected_objects": detections,
        "events": len(events),
        "compression_ratio": None if not events else round(detections / len(events), 3),
        "observations_per_event": None if not events else round(len(observations) / len(events), 3),
        "event_durations_s": durations,
        "evidence_per_event": [e.evidence_count for e in events],
        "detections_retained_as_evidence": retained,
        "all_detections_have_evidence": retained == detections,
        "event_spans_match_evidence": spans_ok,
        "zero_duration_events": sum(1 for d in durations if d == 0),
    }


def run_compression() -> ExperimentResult:
    fixtures = {
        "security_events": scenario_observations("security_events"),
        "normal_day": scenario_observations("normal_day"),
        "long_dwell_controlled": long_dwell_fixture(),
    }
    results = [_compression(n, o) for n, o in fixtures.items()]
    rows = [
        {
            "experiment": "compression",
            "fixture": r["fixture"],
            "observations": r["observations"],
            "detected_objects": r["detected_objects"],
            "events": r["events"],
            "compression_ratio": r["compression_ratio"],
        }
        for r in results
    ]
    dwell = results[2]
    sec = results[0]
    interpretation = [
        f"security_events: {sec['detected_objects']} detections became {sec['events']} events "
        f"(ratio {fmt(sec['compression_ratio'])}); every detection is still linked to an event "
        f"as evidence ({sec['all_detections_have_evidence']}) and every event span equals the span "
        f"of its evidence ({sec['event_spans_match_evidence']}).",
        f"long_dwell_controlled: {dwell['detected_objects']} repeated frame-level detections "
        f"collapsed into {dwell['events']} events (ratio {fmt(dwell['compression_ratio'])}) "
        f"lasting {dwell['event_durations_s']} s: repetitive evidence became bounded temporal units.",
        "The ratio only describes how much repetition was merged; it says nothing about whether the "
        "events are correct (that is measured separately by the rule and flicker experiments).",
    ]
    return ExperimentResult(
        key="compression",
        title="Temporal compression: observations to events",
        research_question="RQ2: how much does aggregation compress repeated evidence while "
        "preserving temporal structure?",
        definition="The production EventAggregator (default config) over three fixtures: the "
        "security_events and normal_day scenarios and a controlled 60-frame dwell.",
        measured={"fixtures": results},
        rows=rows,
        interpretation=interpretation,
        limitations=[
            "Scripted detections are perfect and repetitive by construction; real detector output "
            "is noisier, so real compression and split behaviour will differ.",
            "Compression ratio is not accuracy. A wrong merge also raises it.",
        ],
    )


# ---- Experiment 3: flicker tolerance --------------------------------------------------------
def _config(max_missed: int) -> EventConfig:
    return EventConfig(aggregation=AggregationSettings(max_missed_frames=max_missed))


def _flicker_summary(observations: list[Observation], max_missed: int) -> dict:
    events, _, alerts = run_pipeline(observations, event_config=_config(max_missed))
    person = [e for e in events if e.object_type is ObjectType.PERSON]
    return {
        "max_missed_frames": max_missed,
        "person_events": len(person),
        "durations_s": [e.duration_seconds for e in person],
        "close_reasons": [e.close_reason.value for e in person],
        "loitering_alerts": sum(1 for a in alerts if a.rule_id == "loitering"),
        "loitering_severities": sorted(
            (a.severity.value for a in alerts if a.rule_id == "loitering"),
            key=lambda v: Severity(v).rank,
        ),
    }


def gap_fixture(missed: int) -> list[Observation]:
    """A person seen for 4 frames, `missed` frames with no detection, then 4 more frames (60 s
    apart, main gate, daytime). One continuous episode lasts (7 + missed) minutes; each half
    alone lasts 3 minutes: above the 2-minute loitering warning but below the 5-minute critical
    threshold, so a split turns one CRITICAL alert into two WARNING alerts."""
    tl = Timeline(DAY)
    step = 0
    for _ in range(4):
        tl.add(step * 60, Zone.MAIN_GATE, det("person", "standing", clothing="dark jacket"))
        step += 1
    for _ in range(missed):
        tl.add(step * 60, Zone.MAIN_GATE)
        step += 1
    for _ in range(4):
        tl.add(step * 60, Zone.MAIN_GATE, det("person", "standing", clothing="dark jacket"))
        step += 1
    return tl.observations


def run_flicker() -> ExperimentResult:
    security = scenario_observations("security_events")
    settings = (0, AggregationSettings().max_missed_frames)  # ablated vs production default
    scenario = {f"max_missed_frames={m}": _flicker_summary(security, m) for m in settings}
    # Only the loitering episode at the gate (frames 1-9, with frame 5 undetected).
    gate_episode = [o for o in security if o.frame_id <= 9]
    episode = {f"max_missed_frames={m}": _flicker_summary(gate_episode, m) for m in settings}
    sweep = []
    for missed in range(0, 5):
        for m in settings:
            entry = _flicker_summary(gap_fixture(missed), m)
            sweep.append({"missed_frames_in_gap": missed, **entry})
    base, ablated = episode[f"max_missed_frames={settings[1]}"], episode["max_missed_frames=0"]
    absorbed = max(
        (
            r["missed_frames_in_gap"]
            for r in sweep
            if r["max_missed_frames"] == settings[1] and r["person_events"] == 1
        ),
        default=0,
    )
    rows = [
        {
            "experiment": "flicker",
            "fixture": f"gap={r['missed_frames_in_gap']}",
            "max_missed_frames": r["max_missed_frames"],
            "person_events": r["person_events"],
            "durations_s": str(r["durations_s"]),
            "loitering_alerts": r["loitering_alerts"],
            "loitering_severities": str(r["loitering_severities"]),
        }
        for r in sweep
    ]
    interpretation = [
        f"security_events frames 1-9 (one continuous loitering episode, one undetected frame): "
        f"max_missed_frames={settings[1]} gives {base['person_events']} event(s) "
        f"{base['durations_s']} s and {base['loitering_alerts']} loitering alert(s) "
        f"{base['loitering_severities']}; max_missed_frames=0 gives {ablated['person_events']} "
        f"event(s) {ablated['durations_s']} s and {ablated['loitering_alerts']} loitering "
        f"alert(s) {ablated['loitering_severities']}.",
        "Loitering is decided from an event's duration, so a split changes downstream security "
        "reasoning: fragments that each fall under a threshold lose that alert, and here each "
        "fragment is long enough to warn but too short to reach the critical rung.",
        f"Gap sweep (default setting): a gap of up to {absorbed} consecutive missed frame(s) "
        f"kept one event; a gap of {absorbed + 1} split it (the tolerance is bounded on "
        "purpose, so a person who really left is not merged forever).",
    ]
    return ExperimentResult(
        key="flicker",
        title="Flicker-tolerance ablation (max_missed_frames)",
        research_question="RQ3: does max_missed_frames stop one missed detection from splitting "
        "a continuous event?",
        definition="Same observations and every other setting identical; only "
        f"max_missed_frames varies ({list(settings)}; default {settings[1]}). Fixtures: the "
        "security_events loiterer (frame 5 has no detection), the whole scenario, and a generated "
        "gap sweep (0-4 consecutive missed frames).",
        measured={
            "settings": list(settings),
            "loitering_episode": episode,
            "whole_scenario": scenario,
            "gap_sweep": sweep,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            "Missed frames are inserted by hand into scripted data; real detector misses are "
            "correlated (occlusion, lighting) and come in longer runs.",
            "The result depends on frame spacing (here 60 s): tolerance counts frames, while "
            "max_gap_seconds counts time, and either can close an event.",
            "Tolerance can also bridge a person leaving and a different, similar person "
            "arriving; events never claim identity.",
        ],
    )


# ---- Experiment 4: entity memory ------------------------------------------------------------
def _entity_summary(observations: list[Observation], link: bool) -> dict:
    events, entities, alerts = run_pipeline(observations, link_entities=link)
    vehicle = [e for e in events if e.object_type is ObjectType.VEHICLE]
    pickup_events = [e for e in vehicle if e.attributes.get("body_type") == "pickup"]
    return {
        "entity_linking": link,
        "vehicle_events": len(vehicle),
        "vehicle_entities": len(entities),
        "multi_appearance_entities": sum(1 for e in entities if e.appearance_count > 1),
        "repeat_vehicle_alerts": sum(1 for a in alerts if a.rule_id == "repeat_vehicle"),
        "pickup_events": [
            {
                "event": e.event_id,
                "zone": e.zone.value,
                "color_text": e.attributes.get("color"),
                "entity_id": e.entity_id,
            }
            for e in pickup_events
        ],
        "max_appearances_known_for_a_vehicle": max(
            [e.appearance_count for e in entities] or [1 if vehicle else 0]
        ),
    }


def _event(event_id: str, zone: Zone, start_s: float, end_s: float, **attrs: str) -> Event:
    return Event(
        event_id=event_id,
        object_type=ObjectType.VEHICLE,
        zone=zone,
        first_seen=DAY.replace(minute=0) + _delta(start_s),
        last_seen=DAY.replace(minute=0) + _delta(end_s),
        attributes=attrs,
        activities=["parked"],
        description="controlled fixture",
        observation_ids=[f"o-{event_id}"],
        frame_ids=[int(event_id[-3:])],
        close_reason=CloseReason.SCENARIO_END,
    )


def _delta(seconds: float):
    from datetime import timedelta

    return timedelta(seconds=seconds)


def _two_sightings(first: dict, second: dict, second_zone=Zone.DRIVEWAY) -> list[Observation]:
    tl = Timeline(DAY)
    tl.run(Zone.PARKING_AREA, [0, 60], det("vehicle", "parked", **first))
    tl.run(second_zone, [600, 660], det("vehicle", "parked", **second))
    return tl.observations


def negative_controls() -> list[dict]:
    """Vehicle sightings the documented linker contract must NOT link (or must link).

    `same_vehicle` is the fixture's ground truth about the story (None = not knowable);
    `contract_links` is what the documented rules require (equal normalised colour+body type, or
    an equal explicit plate; missing attributes, conflicting plates and time overlap never link).
    """
    blue = {"color": "blue", "body_type": "pickup"}
    cases = [
        (
            "different colour",
            _two_sightings(blue, {"color": "red", "body_type": "pickup"}),
            False,
            False,
        ),
        (
            "different body type",
            _two_sightings(blue, {"color": "blue", "body_type": "sedan"}),
            False,
            False,
        ),
        ("missing body type", _two_sightings(blue, {"color": "blue"}), None, False),
        (
            "conflicting explicit plates",
            _two_sightings({**blue, "plate": "AB12CDE"}, {**blue, "plate": "ZX98YWV"}),
            False,
            False,
        ),
        (
            "two real pickups, same coarse attributes (false-link limitation)",
            _two_sightings(blue, {"color": "dark blue", "body_type": "pickup truck"}),
            False,
            True,
        ),
        (
            "same plate, different colour (explicit plate evidence)",
            _two_sightings(
                {"color": "blue", "body_type": "pickup", "plate": "AB12CDE"},
                {"color": "red", "body_type": "pickup", "plate": "AB12CDE"},
            ),
            True,
            True,
        ),
    ]
    out = []
    for name, observations, truth, contract in cases:
        events, entities, _ = run_pipeline(observations)
        vehicle = [e for e in events if e.object_type is ObjectType.VEHICLE]
        out.append(_control_row(name, vehicle, entities, truth, contract))
    # overlap in time: built directly, because the aggregator never produces overlapping events
    overlap = [
        _event("EVT-V-001", Zone.PARKING_AREA, 0, 600, color="blue", body_type="pickup"),
        _event("EVT-V-002", Zone.DRIVEWAY, 300, 900, color="blue", body_type="pickup"),
    ]
    linked, entities = EntityLinker().link(overlap)
    out.append(
        _control_row("overlapping in time (cannot be one vehicle)", linked, entities, False, False)
    )
    return out


def _control_row(name, vehicle_events, entities, truth, contract) -> dict:
    linked = len(vehicle_events) == 2 and len({e.entity_id for e in vehicle_events}) == 1
    return {
        "case": name,
        "vehicle_events": len(vehicle_events),
        "actual_linked": linked,
        "match_levels": sorted({e.match_level.value for e in entities if e.appearance_count > 1}),
        "contract_says_link": contract,
        "contract_respected": linked == contract,
        "ground_truth_same_vehicle": truth,
        "false_link": linked and truth is False,
        "missed_link": (not linked) and truth is True,
    }


def run_entity_memory() -> ExperimentResult:
    security = scenario_observations("security_events")
    baseline = _entity_summary(security, True)
    ablated = _entity_summary(security, False)
    controls = negative_controls()
    respected = sum(1 for c in controls if c["contract_respected"])
    false_links = [c["case"] for c in controls if c["false_link"]]
    rows = [
        {
            "experiment": "entity_memory",
            "setting": "linking on" if s["entity_linking"] else "linking off",
            "vehicle_events": s["vehicle_events"],
            "vehicle_entities": s["vehicle_entities"],
            "multi_appearance_entities": s["multi_appearance_entities"],
            "repeat_vehicle_alerts": s["repeat_vehicle_alerts"],
        }
        for s in (baseline, ablated)
    ] + [
        {
            "experiment": "entity_memory",
            "setting": "control: " + c["case"],
            "actual_linked": c["actual_linked"],
            "contract_respected": c["contract_respected"],
            "false_link": c["false_link"],
        }
        for c in controls
    ]
    interpretation = [
        f"security_events: with coarse linking {baseline['vehicle_events']} vehicle events form "
        f"{baseline['vehicle_entities']} entities ({baseline['multi_appearance_entities']} with "
        f"several appearances) and {baseline['repeat_vehicle_alerts']} repeat-vehicle alert(s); "
        f"without linking there are {ablated['vehicle_entities']} entities and "
        f"{ablated['repeat_vehicle_alerts']} repeat-vehicle alert(s).",
        f"The {len(baseline['pickup_events'])} pickup sightings "
        f"({', '.join(e['color_text'] for e in baseline['pickup_events'])}; zones "
        f"{', '.join(e['zone'] for e in baseline['pickup_events'])}) map to entities "
        f"{sorted({str(e['entity_id']) for e in baseline['pickup_events']})} with linking and to "
        f"{sorted({str(e['entity_id'] or 'no entity') for e in ablated['pickup_events']})} without it: the "
        "capability gained is answering 'was a similar vehicle seen more than once' and firing "
        "the repeat rule.",
        f"Negative controls: the documented contract was respected in {respected} of "
        f"{len(controls)} cases. Coarse matching can still merge genuinely different vehicles "
        f"with identical coarse attributes (false links in this fixture set: {false_links or 'none'}); "
        "this is an inherent limit of attribute-only matching, which is why the wording is "
        "'similar vehicle' and never identity.",
    ]
    return ExperimentResult(
        key="entity_memory",
        title="Coarse vehicle entity memory ablation and negative controls",
        research_question="RQ4: what does coarse vehicle linking add compared with treating every "
        "vehicle event independently?",
        definition="production events with EntityLinker (baseline) versus entities=[] (ablation, "
        "done in the harness, production untouched), plus seven vehicle-linking controls checked "
        "against the linker's documented contract and against the fixture's ground truth.",
        measured={
            "baseline": baseline,
            "ablated": ablated,
            "controls": controls,
            "contract_respected": respected,
            "controls_total": len(controls),
            "false_links": false_links,
        },
        rows=rows,
        interpretation=interpretation,
        limitations=[
            "This is coarse attribute grouping of separate events. It is not re-identification "
            "and not tracking; there is no appearance model.",
            "Ground truth 'same vehicle' is authored in the fixture; real colour descriptions from a "
            "vision model vary with lighting, which can cause both false links and missed links.",
            "Only seven controls: they check the documented contract, not the false-link rate.",
        ],
    )
