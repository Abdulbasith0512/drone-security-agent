from datetime import datetime, timedelta

from conftest import IST

from drone_agent.models import CloseReason, Event, MatchLevel, ObjectType, Zone
from drone_agent.pipeline import EntityLinker

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=IST)


def event(
    n: int,
    minutes: int,
    attrs: dict[str, str] | None,
    kind=ObjectType.VEHICLE,
    zone=Zone.PARKING_AREA,
    length: int = 1,
) -> Event:
    return Event(
        event_id=f"EVT-{'V' if kind is ObjectType.VEHICLE else 'P'}-{n:03d}",
        object_type=kind,
        zone=zone,
        first_seen=T0 + timedelta(minutes=minutes),
        last_seen=T0 + timedelta(minutes=minutes + length),
        attributes=attrs or {},
        observation_ids=[f"obs-{n}"],
        frame_ids=[n],
        close_reason=CloseReason.GAP,
    )


def blue(color="blue", **extra) -> dict[str, str]:
    return {"color": color, "body_type": "pickup", **extra}


def link(*events: Event):
    return EntityLinker().link(list(events))


def test_blue_and_dark_blue_pickups_form_one_coarse_entity():
    events, entities = link(
        event(1, 0, blue("blue")),
        event(2, 60, blue("dark blue")),
        event(3, 300, blue("light blue")),
    )
    (entity,) = entities
    assert entity.entity_id == "ENT-V-001"
    assert entity.event_ids == ["EVT-V-001", "EVT-V-002", "EVT-V-003"]
    assert entity.appearance_count == 3
    assert entity.attributes == {"color": "blue", "body_type": "pickup"}
    assert entity.match_level is MatchLevel.COARSE
    assert entity.label == "similar blue pickup"  # never "the same vehicle"
    assert {e.entity_id for e in events} == {"ENT-V-001"}
    assert (entity.first_seen, entity.last_seen) == (T0, T0 + timedelta(minutes=301))


def test_different_vehicles_get_different_entities():
    _, entities = link(
        event(1, 0, blue()),
        event(2, 60, {"color": "white", "body_type": "sedan"}),
        event(3, 120, {"color": "blue", "body_type": "van"}),
    )
    assert [e.attributes for e in entities] == [
        {"color": "blue", "body_type": "pickup"},
        {"color": "white", "body_type": "sedan"},
        {"color": "blue", "body_type": "van"},
    ]
    assert all(e.appearance_count == 1 for e in entities)


def test_missing_required_attribute_is_never_guessed():
    events, entities = link(event(1, 0, blue()), event(2, 60, {"color": "blue"}))
    assert len(entities) == 2  # unknown body type: not linked to the known pickup
    assert events[1].entity_id == "ENT-V-002"


def test_overlapping_vehicle_events_are_different_vehicles():
    _, entities = link(event(1, 0, blue(), length=30), event(2, 10, blue(), length=5))
    assert len(entities) == 2


def test_people_are_never_identity_linked():
    person_attrs = {"clothing": "dark jacket"}
    events, entities = link(
        event(1, 0, person_attrs, kind=ObjectType.PERSON, zone=Zone.MAIN_GATE),
        event(2, 600, person_attrs, kind=ObjectType.PERSON, zone=Zone.MAIN_GATE),
    )
    assert entities == []
    assert [e.entity_id for e in events] == [None, None]


def test_input_events_are_not_mutated_and_output_keeps_input_order():
    original = [event(2, 60, blue()), event(1, 0, blue())]
    events, entities = link(*original)
    assert [e.event_id for e in events] == ["EVT-V-002", "EVT-V-001"]
    assert all(e.entity_id is None for e in original)
    assert entities[0].event_ids == ["EVT-V-001", "EVT-V-002"]  # chronological


def test_explicit_shared_plate_confirms_the_entity():
    events, entities = link(
        event(1, 0, blue(plate="ab 12-cd")), event(2, 60, blue("dark blue", plate="AB12CD"))
    )
    (entity,) = entities
    assert entity.match_level is MatchLevel.CONFIRMED
    assert entity.attributes["plate"] == "AB12CD"
    assert entity.label.startswith("confirmed")


def test_conflicting_plates_prevent_a_link_even_with_identical_coarse_attributes():
    _, entities = link(event(1, 0, blue(plate="AAA111")), event(2, 60, blue(plate="BBB222")))
    assert len(entities) == 2


def test_plate_on_only_one_event_stays_a_coarse_match():
    _, entities = link(event(1, 0, blue(plate="AAA111")), event(2, 60, blue()))
    (entity,) = entities
    assert entity.match_level is MatchLevel.COARSE
    assert "plate" not in entity.attributes
