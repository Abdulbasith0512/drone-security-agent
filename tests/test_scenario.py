from pathlib import Path

import pytest
from conftest import SCENARIO_DIR

from drone_agent.models import ObjectType, Zone
from drone_agent.scenario import ScenarioError, load_scenario

VALID = """
name: t
start: "2026-09-29T23:55:00+05:30"
telemetry:
  patrol:
    - {zone: main_gate, duration_s: 600}
frames:
  - {frame_id: 1, timestamp_offset: 60, zone: main_gate, description: d,
     objects: [{type: person, confidence: 0.9}]}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_inline_scenario_loads(tmp_path):
    scenario = load_scenario(_write(tmp_path, VALID))
    assert scenario.frames[0].objects[0].object_type is ObjectType.PERSON
    assert scenario.start.utcoffset().total_seconds() == 5.5 * 3600


@pytest.mark.parametrize(
    "old,new,why",
    [
        ("confidence: 0.9", "confidence: 1.5", "confidence out of range"),
        ("zone: main_gate, description", "zone: lobby, description", "unknown zone"),
        ("2026-09-29T23:55:00+05:30", "2026-09-29T23:55:00", "naive start"),
        ("type: person", "type: dragon", "unknown object type"),
        ("description: d,", "description: d, colour: x,", "unknown frame field"),
        ("timestamp_offset: 60", "timestamp_offset: 700", "frame after patrol ends"),
        ("frames:\n", "frames: []\n#", "no frames"),
    ],
)
def test_malformed_scenarios_rejected(tmp_path, old, new, why):
    assert old in VALID, why
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, VALID.replace(old, new)))


def test_frame_zone_must_match_drone_patrol_zone(tmp_path):
    text = VALID.replace("{zone: main_gate, duration_s: 600}", "{zone: warehouse, duration_s: 600}")
    with pytest.raises(ScenarioError, match="patrol zone"):
        load_scenario(_write(tmp_path, text))


def test_duplicate_and_unordered_frames_rejected(tmp_path):
    frame = "  - {frame_id: %d, timestamp_offset: %d, zone: main_gate, description: d}\n"
    with pytest.raises(ScenarioError, match="unique"):
        load_scenario(_write(tmp_path, VALID + frame % (1, 90)))
    with pytest.raises(ScenarioError, match="increasing"):
        load_scenario(_write(tmp_path, VALID + frame % (2, 30)))


def test_yaml_syntax_error_missing_file_and_non_mapping_rejected(tmp_path):
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, "name: [unclosed"))
    with pytest.raises(ScenarioError):
        load_scenario(tmp_path / "missing.yaml")
    with pytest.raises(ScenarioError):
        load_scenario(_write(tmp_path, "- just\n- a list\n"))


def test_normal_day_scenario_is_a_clean_daytime_baseline():
    s = load_scenario(SCENARIO_DIR / "normal_day.yaml")
    assert s.start.hour == 10
    assert all(f.zone is not Zone.RESTRICTED_ZONE for f in s.frames)
    vehicles = [
        (o.attributes["color"], o.attributes["body_type"])
        for f in s.frames
        for o in f.objects
        if o.object_type is ObjectType.VEHICLE
    ]
    assert set(vehicles) == {("white", "van"), ("grey", "hatchback")}
    gate_person = [
        f
        for f in s.frames
        if f.zone is Zone.MAIN_GATE and any(o.object_type is ObjectType.PERSON for o in f.objects)
    ]
    span = gate_person[-1].timestamp_offset - gate_person[0].timestamp_offset
    assert span <= 120  # far below any loitering threshold


def _frames_with(scenario, predicate):
    return [f.frame_id for f in scenario.frames if any(predicate(o) for o in f.objects)]


def test_security_scenario_contains_the_observations_later_rules_need():
    s = load_scenario(SCENARIO_DIR / "security_events.yaml")

    def ts(f):
        return s.clock.at(f.timestamp_offset)

    by_id = {f.frame_id: f for f in s.frames}

    # loitering: person in dark jacket at main_gate for 8 minutes, with a missed-detection gap
    gate_people = [
        by_id[i]
        for i in _frames_with(
            s,
            lambda o: (
                o.object_type is ObjectType.PERSON and o.attributes.get("clothing") == "dark jacket"
            ),
        )
    ]
    assert {f.zone for f in gate_people} == {Zone.MAIN_GATE}
    assert (ts(gate_people[-1]) - ts(gate_people[0])).total_seconds() == 480
    assert 5 not in [f.frame_id for f in gate_people] and by_id[5].objects == []

    # repeat vehicle: blue pickup in three separate sightings; white sedan is a different vehicle
    def blue_pickup(o):
        return o.object_type is ObjectType.VEHICLE and o.attributes["body_type"] == "pickup"

    pickup_frames = _frames_with(s, blue_pickup)
    assert pickup_frames == [10, 11, 13, 14, 16, 17]
    assert all(
        "blue" in o.attributes["color"] for f in s.frames for o in f.objects if blue_pickup(o)
    )
    assert _frames_with(s, lambda o: o.attributes.get("color") == "white") == [26]

    # restricted zone entry by a person
    assert [f.frame_id for f in s.frames if f.zone is Zone.RESTRICTED_ZONE and f.objects] == [
        22,
        23,
        24,
    ]

    # after hours and across midnight
    assert ts(by_id[1]).hour == 23 and ts(by_id[16]).day == 30 and ts(by_id[16]).hour == 0
    assert ts(by_id[22]).hour == 0
