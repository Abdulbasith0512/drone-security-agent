from datetime import datetime, timedelta

import pytest
from conftest import IST, ROOT
from pydantic import ValidationError

from drone_agent.models import CloseReason, DetectedObject, ObjectType, Observation, Zone
from drone_agent.pipeline import (
    ConfigError,
    EventAggregator,
    EventConfig,
    aggregate_observations,
    load_config,
)
from drone_agent.pipeline.config import AggregationSettings
from drone_agent.pipeline.normalize import normalize_attribute, normalize_color

START = datetime(2026, 9, 29, 23, 50, tzinfo=IST)


def person(clothing="dark jacket", activity="standing") -> DetectedObject:
    return DetectedObject(
        object_type=ObjectType.PERSON,
        activity=activity,
        attributes={"clothing": clothing} if clothing else {},
        confidence=0.9,
    )


def pickup(color="blue") -> DetectedObject:
    return DetectedObject(
        object_type=ObjectType.VEHICLE,
        activity="parked",
        attributes={"color": color, "body_type": "pickup"},
        confidence=0.9,
    )


def obs(frame: int, seconds: int, zone=Zone.MAIN_GATE, *objects, start=START) -> Observation:
    return Observation(
        observation_id=f"obs-{frame}",
        frame_id=frame,
        timestamp=start + timedelta(seconds=seconds),
        source="scripted",
        zone=zone,
        description=f"frame {frame}",
        detected_objects=list(objects),
    )


def config(**aggregation) -> EventConfig:
    return EventConfig(aggregation=AggregationSettings(**aggregation))


# ---- one event / evidence / duration --------------------------------------------------------
def test_adjacent_frames_form_one_event_with_evidence_and_duration():
    events = aggregate_observations(
        [
            obs(1, 0, Zone.MAIN_GATE, person(activity="standing")),
            obs(2, 60, Zone.MAIN_GATE, person(activity="pacing")),
            obs(3, 120, Zone.MAIN_GATE, person(activity="standing")),
        ]
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.event_id == "EVT-P-001"
    assert ev.observation_ids == ["obs-1", "obs-2", "obs-3"]
    assert ev.frame_ids == [1, 2, 3]
    assert ev.evidence_count == 3
    assert (ev.first_seen, ev.last_seen) == (START, START + timedelta(seconds=120))
    assert ev.duration_seconds == 120
    assert ev.activities == ["standing", "pacing"]
    assert ev.attributes == {"clothing": "dark jacket"}
    assert ev.description == "frame 1"
    assert ev.close_reason is CloseReason.SCENARIO_END


def test_single_sighting_has_zero_duration():
    (ev,) = aggregate_observations([obs(1, 0, Zone.MAIN_GATE, person())])
    assert ev.duration_seconds == 0 and ev.evidence_count == 1


# ---- flicker tolerance and gaps -------------------------------------------------------------
def test_short_missed_detection_is_tolerated():
    events = aggregate_observations(
        [
            obs(1, 0, Zone.MAIN_GATE, person()),
            obs(2, 60, Zone.MAIN_GATE),  # detector missed the person
            obs(3, 120, Zone.MAIN_GATE, person()),
        ]
    )
    assert len(events) == 1
    assert events[0].frame_ids == [1, 3]  # the missed frame is not evidence
    assert events[0].duration_seconds == 120


def test_gap_longer_than_max_gap_splits_the_event():
    events = aggregate_observations(
        [obs(1, 0, Zone.MAIN_GATE, person()), obs(2, 181, Zone.MAIN_GATE, person())]
    )
    assert [e.frame_ids for e in events] == [[1], [2]]
    assert events[0].close_reason is CloseReason.GAP


def test_gap_exactly_at_max_gap_is_still_one_event():
    events = aggregate_observations(
        [obs(1, 0, Zone.MAIN_GATE, person()), obs(2, 180, Zone.MAIN_GATE, person())]
    )
    assert len(events) == 1


def test_too_many_consecutive_missed_frames_split_the_event():
    frames = [obs(1, 0, Zone.MAIN_GATE, person())]
    frames += [obs(i, 10 * i, Zone.MAIN_GATE) for i in (2, 3, 4)]  # 3 misses, all within max gap
    frames.append(obs(5, 50, Zone.MAIN_GATE, person()))
    events = aggregate_observations(frames)  # default max_missed_frames = 2
    assert [e.frame_ids for e in events] == [[1], [5]]
    assert events[0].close_reason is CloseReason.MISSED_FRAMES
    tolerant = aggregate_observations(frames, config(max_missed_frames=3))
    assert [e.frame_ids for e in tolerant] == [[1, 5]]


def test_tolerances_are_configurable():
    frames = [obs(1, 0, Zone.MAIN_GATE, person()), obs(2, 400, Zone.MAIN_GATE, person())]
    assert len(aggregate_observations(frames)) == 2
    assert len(aggregate_observations(frames, config(max_gap_seconds=500))) == 1


# ---- boundaries -----------------------------------------------------------------------------
def test_zone_change_closes_the_event_and_starts_another():
    agg = EventAggregator()
    assert agg.process(obs(1, 0, Zone.MAIN_GATE, person())) == []
    closed = agg.process(obs(2, 60, Zone.DRIVEWAY, person()))
    assert [(e.frame_ids, e.close_reason) for e in closed] == [([1], CloseReason.ZONE_CHANGE)]
    (last,) = agg.flush()
    assert (last.zone, last.frame_ids, last.close_reason) == (
        Zone.DRIVEWAY,
        [2],
        CloseReason.SCENARIO_END,
    )


def test_incompatible_attributes_make_a_separate_event_without_closing_the_first():
    events = aggregate_observations(
        [
            obs(1, 0, Zone.MAIN_GATE, person("dark jacket")),
            obs(2, 60, Zone.MAIN_GATE, person("dark jacket"), person("hi-vis vest")),
            obs(3, 120, Zone.MAIN_GATE, person("dark jacket")),
        ]
    )
    by_clothing = {e.attributes["clothing"]: e for e in events}
    assert by_clothing["dark jacket"].frame_ids == [1, 2, 3]  # never closed by the newcomer
    assert by_clothing["hi-vis vest"].frame_ids == [2]
    assert {e.event_id for e in events} == {"EVT-P-001", "EVT-P-002"}


def test_normalised_variants_stay_in_one_event_but_different_colors_do_not():
    same = aggregate_observations(
        [obs(1, 0, Zone.DRIVEWAY, pickup("blue")), obs(2, 60, Zone.DRIVEWAY, pickup("Dark Blue"))]
    )
    assert len(same) == 1 and same[0].attributes["color"] == "blue"  # modal raw value, first wins
    different = aggregate_observations(
        [obs(1, 0, Zone.DRIVEWAY, pickup("blue")), obs(2, 60, Zone.DRIVEWAY, pickup("white"))]
    )
    assert len(different) == 2


def test_missing_attribute_is_unknown_and_does_not_block_a_match():
    events = aggregate_observations(
        [obs(1, 0, Zone.MAIN_GATE, person("dark jacket")), obs(2, 60, Zone.MAIN_GATE, person(None))]
    )
    assert len(events) == 1


def test_different_object_types_never_merge():
    events = aggregate_observations([obs(1, 0, Zone.MAIN_GATE, person(), pickup())])
    assert sorted(e.event_id for e in events) == ["EVT-P-001", "EVT-V-001"]


def test_event_crossing_midnight_keeps_correct_duration_and_dates():
    late = datetime(2026, 9, 29, 23, 59, tzinfo=IST)
    events = aggregate_observations(
        [
            obs(1, 0, Zone.MAIN_GATE, person(), start=late),
            obs(2, 60, Zone.MAIN_GATE, person(), start=late),
            obs(3, 120, Zone.MAIN_GATE, person(), start=late),
        ]
    )
    (ev,) = events
    assert ev.first_seen.day == 29 and ev.last_seen.day == 30
    assert (ev.first_seen.hour, ev.last_seen.hour, ev.last_seen.minute) == (23, 0, 1)
    assert ev.duration_seconds == 120


def test_flush_closes_open_events_with_scenario_end():
    agg = EventAggregator()
    agg.process(obs(1, 0, Zone.MAIN_GATE, person()))
    agg.process(obs(2, 60, Zone.MAIN_GATE, pickup()))
    closed = agg.flush()
    assert {e.close_reason for e in closed} == {CloseReason.SCENARIO_END}
    assert agg.flush() == []


def test_out_of_order_observations_are_rejected():
    agg = EventAggregator()
    agg.process(obs(2, 60, Zone.MAIN_GATE, person()))
    with pytest.raises(ValueError, match="chronological"):
        agg.process(obs(1, 0, Zone.MAIN_GATE, person()))


def test_aggregation_is_deterministic_and_ids_follow_first_seen_order():
    frames = [
        obs(1, 0, Zone.MAIN_GATE, person()),
        obs(2, 60, Zone.DRIVEWAY, pickup()),
        obs(3, 120, Zone.PARKING_AREA, person("navy shirt")),
    ]
    first, second = aggregate_observations(frames), aggregate_observations(frames)
    assert first == second
    assert [e.event_id for e in first] == ["EVT-P-001", "EVT-V-001", "EVT-P-002"]


# ---- normalisation and config ---------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("blue", "blue"),
        ("dark blue", "blue"),
        ("Light  Blue", "blue"),
        ("dark-blue", "blue"),
        ("grey", "gray"),
        ("dark", "dark"),  # a lone modifier is the colour itself
        ("navy", "navy"),  # no invented synonyms
    ],
)
def test_color_normalisation(raw, expected):
    assert normalize_color(raw, EventConfig().normalization) == expected


def test_body_type_and_free_text_normalisation():
    cfg = EventConfig().normalization
    assert normalize_attribute("body_type", "Pickup Truck", cfg) == "pickup"
    assert normalize_attribute("clothing", "Hi-Vis  Vest", cfg) == "hi vis vest"


def test_shipped_yaml_equals_built_in_defaults():
    assert load_config(ROOT / "config" / "events.yaml") == EventConfig()
    assert load_config() == EventConfig()


def test_invalid_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("aggregation: {max_gap_seconds: -5}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ValidationError):
        AggregationSettings(unknown_key=1)
