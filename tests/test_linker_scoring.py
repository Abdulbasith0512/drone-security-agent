"""Entity re-identification: scored links, transparent components, and indexed (linear) linking."""

import random
from datetime import datetime, timedelta

import pytest
from conftest import IST
from test_linker import blue, event, link

from drone_agent.models import CloseReason, Event, MatchLevel, ObjectType, Zone
from drone_agent.pipeline import EntityLinker, EventConfig
from drone_agent.pipeline.config import VehicleMatchSettings
from drone_agent.pipeline.normalize import normalize_attribute, normalize_plate

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)


# ---- scores ---------------------------------------------------------------------------------
def test_an_exact_match_close_in_time_and_place_scores_high_with_visible_components():
    events, (entity,) = link(event(1, 0, blue()), event(2, 10, blue()))
    score = entity.link_scores["EVT-V-002"]
    assert 0.98 < score <= 1.0
    parts = entity.link_components["EVT-V-002"]
    assert parts["color"] == 1.0 and parts["body_type"] == 1.0 and parts["zone"] == 1.0
    assert 0.99 < parts["temporal"] <= 1.0
    assert entity.match_score == score and "EVT-V-001" not in entity.link_scores  # first: no link
    assert entity.match_level is MatchLevel.COARSE  # a high score never means "confirmed"


def test_a_normalised_only_colour_match_earns_partial_credit():
    """'dark blue' is equal to 'blue' only after normalisation, so the link is weaker: 0.82."""
    _, (entity,) = link(event(1, 0, blue("blue")), event(2, 10, blue("dark blue")))
    assert entity.link_components["EVT-V-002"]["color"] == 0.8
    assert 0.85 < entity.link_scores["EVT-V-002"] < 0.92  # (0.8 + 1.0) / 2 x about 0.997


def test_a_different_zone_and_a_long_gap_lower_the_score_but_do_not_by_themselves_unlink():
    near = link(event(1, 0, blue()), event(2, 10, blue()))[1][0].match_score
    moved = link(event(1, 0, blue()), event(2, 10, blue(), zone=Zone.DRIVEWAY))[1][0].match_score
    late = link(event(1, 0, blue()), event(2, 60 * 24 * 3, blue()))[1][0].match_score
    assert moved < near and late < near
    assert moved == pytest.approx(near * 0.92, abs=0.01)
    assert 0.74 < late < 0.8  # three days later: floored, still a (weak) link
    assert len(link(event(1, 0, blue()), event(2, 60 * 24 * 3, blue()))[1]) == 1


def test_the_entity_score_is_its_weakest_link():
    _, (entity,) = link(
        event(1, 0, blue()),
        event(2, 10, blue()),  # strong
        event(3, 400, blue("dark blue"), zone=Zone.DRIVEWAY),  # weaker
    )
    scores = entity.link_scores
    assert set(scores) == {"EVT-V-002", "EVT-V-003"}
    assert entity.match_score == min(scores.values()) == scores["EVT-V-003"]


def test_a_shared_plate_scores_the_configured_plate_score_regardless_of_colour():
    _, (entity,) = link(
        event(1, 0, blue(plate="AB12CD")), event(2, 60, blue("red", plate="ab-12 cd"))
    )
    assert entity.match_level is MatchLevel.CONFIRMED
    assert entity.link_scores["EVT-V-002"] == 0.98
    assert entity.link_components["EVT-V-002"]["plate"] == 1.0


def test_min_link_score_can_refuse_a_weak_but_gate_passing_link():
    strict = EventConfig(vehicle_matching=VehicleMatchSettings(min_link_score=0.95))
    weak = [event(1, 0, blue()), event(2, 10, blue("dark blue"), zone=Zone.DRIVEWAY)]
    _, entities = EntityLinker(strict).link(weak)
    assert len(entities) == 2  # the score, not the gates, refused it
    assert len(EntityLinker().link(weak)[1]) == 1  # default 0.5: gates decide


def test_scores_never_reach_unlinked_or_person_events():
    events, entities = link(event(1, 0, blue()), event(2, 60, {"color": "red", "body_type": "van"}))
    assert all(e.match_score is None and e.link_scores == {} for e in entities)


def test_score_settings_are_validated():
    with pytest.raises(ValueError):
        VehicleMatchSettings(zone_change_factor=1.5)
    with pytest.raises(ValueError):
        VehicleMatchSettings(temporal_tau_seconds=0)


# ---- the index-based linker links exactly who the all-pairs linker linked -----------------------
def naive_assignment(events: list[Event], cfg: EventConfig) -> dict[str, str]:
    """The pre-M9 algorithm (scan every group for every event), as an executable specification."""
    linker = EntityLinker(cfg)
    groups: list[dict] = []
    out: dict[str, str] = {}
    for ev in sorted(events, key=lambda e: (e.first_seen, e.event_id)):
        if ev.object_type is not ObjectType.VEHICLE:
            continue
        coarse, plate = linker._coarse(ev), linker._plate(ev)
        candidates = []
        for g in groups:
            if ev.first_seen < max(e.last_seen for e in g["events"]):
                continue
            if plate and g["plates"]:
                ok = plate in g["plates"]
            else:
                ok = bool(coarse) and coarse == g["coarse"]
            if ok:
                candidates.append(g)
        if candidates:
            group = max(candidates, key=lambda g: max(e.last_seen for e in g["events"]))
        else:
            group = {
                "id": f"ENT-V-{len(groups) + 1:03d}",
                "coarse": coarse,
                "events": [],
                "plates": set(),
            }
            groups.append(group)
        group["events"].append(ev)
        if plate:
            group["plates"].add(plate)
        out[ev.event_id] = group["id"]
    return out


def random_vehicle_events(rng: random.Random, n: int) -> list[Event]:
    colors = ["blue", "dark blue", "red", "white", "grey", "gray", "light blue"]
    bodies = ["pickup", "sedan", "van", "pickup truck", None]
    plates = [None, None, None, "AB12CD", "ab-12 cd", "ZX98YW"]
    events = []
    for i in range(n):
        start = rng.randint(0, 600)
        attrs = {"color": rng.choice(colors)}
        if (body := rng.choice(bodies)) is not None:
            attrs["body_type"] = body
        if (plate := rng.choice(plates)) is not None:
            attrs["plate"] = plate
        events.append(
            Event(
                event_id=f"EVT-V-{i + 1:03d}",
                object_type=ObjectType.VEHICLE,
                zone=rng.choice(list(Zone)),
                first_seen=T0 + timedelta(minutes=start),
                last_seen=T0 + timedelta(minutes=start + rng.randint(0, 20)),
                attributes=attrs,
                observation_ids=[f"o{i}"],
                frame_ids=[i],
                close_reason=CloseReason.GAP,
            )
        )
    return events


@pytest.mark.parametrize("seed", range(12))
def test_indexed_linking_assigns_exactly_like_the_naive_all_pairs_linker(seed):
    rng = random.Random(seed)
    events = random_vehicle_events(rng, rng.randint(20, 120))
    cfg = EventConfig(vehicle_matching=VehicleMatchSettings(min_link_score=0))
    linked, entities = EntityLinker(cfg).link(events)
    got = {e.event_id: e.entity_id for e in linked if e.entity_id}
    assert got == naive_assignment(events, cfg)
    assert sorted(x for e in entities for x in e.event_ids) == sorted(got)


def test_linking_a_large_stream_is_close_to_linear():
    """Doubling the events must not quadruple the work: count candidate comparisons."""
    calls = {"n": 0}
    original = EntityLinker._matches

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    def comparisons(n: int) -> int:
        rng = random.Random(1)
        events = [
            Event(
                event_id=f"EVT-V-{i + 1:06d}",
                object_type=ObjectType.VEHICLE,
                zone=Zone.PARKING_AREA,
                first_seen=T0 + timedelta(minutes=i * 3),
                last_seen=T0 + timedelta(minutes=i * 3 + 1),
                attributes={"color": rng.choice(["blue", "red", "white"]), "body_type": "van"},
                observation_ids=[f"o{i}"],
                frame_ids=[i],
                close_reason=CloseReason.GAP,
            )
            for i in range(n)
        ]
        calls["n"] = 0
        EntityLinker._matches = staticmethod(counting)  # type: ignore[method-assign]
        try:
            EntityLinker().link(events)
        finally:
            EntityLinker._matches = staticmethod(original)  # type: ignore[method-assign]
        return calls["n"]

    small, large = comparisons(2000), comparisons(4000)
    assert large < small * 2.5  # quadratic growth would be about 4x


def test_normalisation_helpers_used_by_the_signature_are_unchanged():
    from drone_agent.pipeline.config import NormalizationSettings

    cfg = NormalizationSettings()
    assert normalize_attribute("color", "Dark Blue", cfg) == "blue"
    assert normalize_plate("ab 12-cd") == "AB12CD"
