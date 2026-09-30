"""Temporal context engine: the incremental view ends exactly where the batch pipeline ends."""

import pytest
from conftest import SCENARIO_DIR

from drone_agent.agents.context import ContextEngine
from drone_agent.models import CloseReason
from drone_agent.pipeline import EntityLinker, aggregate_observations
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario

SECURITY = SCENARIO_DIR / "security_events.yaml"
NORMAL = SCENARIO_DIR / "normal_day.yaml"


def observations(path):
    return run_scenario(load_scenario(path)).observations


@pytest.mark.parametrize("path", [SECURITY, NORMAL], ids=["security_events", "normal_day"])
def test_flushing_the_engine_gives_exactly_the_batch_events_and_entities(path):
    obs = observations(path)
    engine = ContextEngine()
    for o in obs:
        engine.process(o)
    final = engine.flush()
    batch_events, batch_entities = EntityLinker().link(aggregate_observations(obs))
    assert final.events == batch_events
    assert final.entities == batch_entities
    assert not final.open_events


def test_open_events_are_visible_while_they_grow_and_keep_their_ids():
    obs = observations(SECURITY)
    engine = ContextEngine()
    lengths = []
    for o in obs[:9]:  # the loiterer at the gate, frames 1-9
        update = engine.process(o)
        gate = [e for e in update.open_events if e.event_id == "EVT-P-001"]
        if gate:
            lengths.append(gate[0].evidence_count)
    assert lengths == sorted(lengths) and lengths[0] == 1 and lengths[-1] == 8  # frame 5 missed
    assert all(e.close_reason is CloseReason.OPEN for e in update.open_events)


def test_only_events_that_changed_at_a_step_are_reported_as_changed():
    obs = observations(SECURITY)
    engine = ContextEngine()
    first = engine.process(obs[0])
    assert [e.event_id for e in first.changed] == ["EVT-P-001"]
    second = engine.process(obs[1])
    assert [e.event_id for e in second.changed] == ["EVT-P-001"]  # it grew
    engine.process(obs[2])
    engine.process(obs[3])
    quiet = engine.process(obs[4])  # frame 5: nothing detected, the open event just waits
    assert quiet.changed == []


def test_an_event_closed_by_a_zone_change_is_reported_once_as_closed():
    obs = observations(SECURITY)
    engine = ContextEngine()
    closings = []
    for o in obs[:14]:  # frame 10 leaves the gate, frame 13 leaves the parking area
        update = engine.process(o)
        closings += [(o.frame_id, e.event_id, e.close_reason) for e in update.closed]
    assert closings == [
        (10, "EVT-P-001", CloseReason.ZONE_CHANGE),
        (13, "EVT-V-001", CloseReason.ZONE_CHANGE),
    ]


def test_vehicle_entities_grow_as_the_similar_vehicle_reappears():
    obs = observations(SECURITY)
    engine = ContextEngine()
    counts = []
    for o in obs:
        update = engine.process(o)
        blue = [e for e in update.entities if e.label == "similar blue pickup"]
        counts.append(blue[0].appearance_count if blue else 0)
    assert counts[0] == 0 and max(counts) == 3
    assert counts == sorted(counts)  # memory only ever accumulates
    first_repeat = counts.index(3)
    assert obs[first_repeat].frame_id == 16  # the third sighting opens at frame 16


def test_process_rejects_out_of_order_observations():
    obs = observations(SECURITY)
    engine = ContextEngine()
    engine.process(obs[3])
    with pytest.raises(ValueError, match="chronological"):
        engine.process(obs[1])
