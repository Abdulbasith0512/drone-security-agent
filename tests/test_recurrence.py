"""Person recurrence hints: weak, labelled, capped, never an identity link."""

from test_linker import event

from drone_agent.models import ObjectType, Zone
from drone_agent.pipeline.recurrence import MAX_SCORE, similar_earlier_person_events


def person(n: int, minutes: int, clothing: str | None = "dark hoodie", zone=Zone.MAIN_GATE):
    attrs = {"clothing": clothing} if clothing else {}
    return event(n, minutes, attrs, kind=ObjectType.PERSON, zone=zone)


def test_an_earlier_person_in_the_same_zone_with_equal_clothing_is_a_capped_hint():
    earlier, target = person(1, 0), person(2, 30)
    (hint,) = similar_earlier_person_events(target, [earlier, target])
    assert hint.event.event_id == "EVT-P-001"
    assert 0 < hint.score <= MAX_SCORE == 0.6
    assert "similar clothing" in hint.reason and "same person" not in hint.reason


def test_the_score_never_exceeds_the_cap_even_for_back_to_back_events():
    (hint,) = similar_earlier_person_events(person(2, 2), [person(1, 0), person(2, 2)])
    assert hint.score <= MAX_SCORE


def test_a_more_recent_appearance_scores_higher_than_an_old_one():
    target = person(3, 60 * 24)
    old, recent = person(1, 0), person(2, 60 * 23)
    hints = similar_earlier_person_events(target, [old, recent, target])
    assert [h.event.event_id for h in hints] == ["EVT-P-002", "EVT-P-001"]
    assert hints[0].score > hints[1].score


def test_a_different_zone_different_clothing_or_a_later_event_gives_no_hint():
    target = person(2, 30)
    assert similar_earlier_person_events(target, [person(1, 0, zone=Zone.WAREHOUSE), target]) == []
    assert similar_earlier_person_events(target, [person(1, 0, "red jacket"), target]) == []
    assert similar_earlier_person_events(person(1, 0), [person(1, 0), person(2, 30)]) == []


def test_no_recorded_clothing_means_no_hint_at_all():
    assert similar_earlier_person_events(person(2, 30, None), [person(1, 0, None)]) == []
    assert similar_earlier_person_events(person(2, 30), [person(1, 0, None)]) == []


def test_vehicles_are_never_given_a_person_hint():
    vehicle = event(2, 30, {"color": "blue", "body_type": "pickup"})
    assert similar_earlier_person_events(vehicle, [person(1, 0), vehicle]) == []
