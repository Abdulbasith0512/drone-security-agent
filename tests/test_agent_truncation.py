"""M8a: the template analyst must never silently omit results.

Fixtures are generated (no video, no API): many separate vehicle/person events so that the tool
caps (20 events, 20 entities) and the analyst's own display limits are exceeded.
"""

import json
import re
from datetime import datetime
from pathlib import Path

import pytest
from conftest import IST, SCENARIO_DIR

from drone_agent.agent.citations import validate_answer
from drone_agent.agent.offline import OfflineAnalyst
from drone_agent.agent.session import AnalystSession
from drone_agent.agent.tools import AnalystTools
from drone_agent.eval.base import Timeline, det
from drone_agent.ingest import ingest_observations, ingest_result
from drone_agent.models import Zone
from drone_agent.pipeline import build_events
from drone_agent.rules import analyze
from drone_agent.scenario import load_scenario
from drone_agent.simulate import run_scenario
from drone_agent.store import Store

GOLDEN = json.loads((Path(__file__).parent / "golden_offline_answers.json").read_text())
COLORS = ("blue", "red", "white", "black", "silver", "green")
BODIES = ("pickup", "sedan", "van", "suv")
COMBOS = [(c, b) for b in BODIES for c in COLORS]  # 24 distinct coarse vehicle descriptions


def build_db(path: Path, vehicles: int, people: int, repeats_of_one_vehicle: int = 0) -> Path:
    """`vehicles` events with distinct coarse attributes (so distinct entities), `people` person
    events, and optionally one vehicle that appears `repeats_of_one_vehicle` separate times.
    Events are 10 minutes apart, so none merge."""
    tl = Timeline(datetime(2026, 9, 29, 12, 0, tzinfo=IST))
    t = 0
    for i in range(vehicles):
        color, body = COMBOS[i]
        tl.run(Zone.MAIN_GATE, [t, t + 60], det("vehicle", "parked", color=color, body_type=body))
        t += 600
    for i in range(people):
        tl.run(Zone.MAIN_GATE, [t, t + 60], det("person", "standing", clothing=f"outfit {i}"))
        t += 600
    for _ in range(repeats_of_one_vehicle):
        tl.run(Zone.MAIN_GATE, [t, t + 60], det("vehicle", "parked", color="blue", body_type="van"))
        t += 600
    with Store.open(path) as store:
        ingest_observations(store, "fixture", [], tl.observations)
        build_events(store)
        analyze(store)
    return path


@pytest.fixture(scope="module")
def dbs(tmp_path_factory):
    root = tmp_path_factory.mktemp("trunc")
    return {
        "events23": build_db(root / "events23.db", vehicles=8, people=15),  # 23 events, 8 entities
        "entities11": build_db(root / "entities11.db", vehicles=11, people=0),
        "entities22": build_db(root / "entities22.db", vehicles=22, people=0),
        "repeat25": build_db(root / "repeat25.db", vehicles=0, people=0, repeats_of_one_vehicle=25),
        "small": build_db(root / "small.db", vehicles=2, people=2),  # 4 events: nothing truncated
    }


@pytest.fixture(scope="module")
def security_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("sec") / "security.db"
    with Store.open(path) as store:
        ingest_result(store, run_scenario(load_scenario(SCENARIO_DIR / "security_events.yaml")))
        build_events(store)
        analyze(store)
    return path


def ask(db: Path, question: str):
    """Returns (answer, store-backed session) with the store kept open by the caller."""
    store = Store.open(db)
    session = AnalystSession(store)
    return session.ask(question), session, store


def check_valid(answer, session, store):
    """The answer passed the analyst's own citation check and validates again from outside."""
    assert answer.mode == "template" and answer.notes == []
    report = validate_answer(store, answer.text, retrieved=session.tools.retrieved)
    assert report.ok, report.problems()
    store.close()


def event_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("- [event:"))


# ---- (a) more than 20 events: the answer says how many are shown out of the total ------------
def test_zone_answer_states_shown_of_total_events(dbs):
    answer, session, store = ask(dbs["events23"], "What happened at the main gate?")
    assert "23 event(s)" in answer.text
    assert re.search(r"showing 20 of 23 events", answer.text)
    assert "3 not shown" in answer.text
    assert event_lines(answer.text) == 20  # what is listed is exactly what the note claims
    check_valid(answer, session, store)


def test_objects_answer_states_shown_of_total_events(dbs):
    answer, session, store = ask(dbs["events23"], "What objects were observed?")
    assert re.search(r"showing 20 of 23 events", answer.text)
    assert event_lines(answer.text) == 20
    check_valid(answer, session, store)


def test_summary_and_entity_history_also_state_their_limits(dbs):
    store = Store.open(dbs["entities22"])
    text = AnalystSession(store).summarize().text
    assert "showing 20 of 22 vehicle entities" in text
    store.close()
    answer, session, store = ask(dbs["repeat25"], "Describe ENT-V-001")
    assert re.search(r"showing 20 of 25 appearances", answer.text)
    check_valid(answer, session, store)


def test_vehicle_event_fallback_list_states_its_limit(dbs):
    answer, session, store = ask(dbs["entities22"], "Was an orange car seen?")
    assert "No similar-vehicle entity matches" in answer.text
    assert re.search(r"showing 20 of 22 vehicle events", answer.text)
    check_valid(answer, session, store)


def test_zone_answer_discloses_that_observations_are_a_sample(security_db):
    answer, session, store = ask(security_db, "What happened at the main gate?")
    assert re.search(r"showing 3 of 12 observations", answer.text)
    check_valid(answer, session, store)


# ---- (b) more than 10 entities: how many are shown out of the total --------------------------
def test_vehicle_answer_states_detail_coverage_and_lists_every_entity(dbs):
    answer, session, store = ask(dbs["entities11"], "Which vehicles were observed?")
    assert re.search(r"full history for 3 of 11 similar-vehicle entities", answer.text)
    assert "8 more are listed briefly" in answer.text
    refs = set(re.findall(r"\[entity:(ENT-V-\d+)\]", answer.text))
    assert len(refs) == 11  # nothing is silently dropped
    check_valid(answer, session, store)


def test_entities_beyond_the_tool_cap_are_counted_as_not_shown(dbs):
    answer, session, store = ask(dbs["entities22"], "Which vehicles were observed?")
    assert re.search(r"showing 20 of 22 similar-vehicle entities", answer.text)
    assert "2 not shown" in answer.text
    assert len(set(re.findall(r"\[entity:(ENT-V-\d+)\]", answer.text))) == 20
    check_valid(answer, session, store)


def test_objects_answer_states_entity_limit(dbs):
    answer, session, store = ask(dbs["entities22"], "What objects were observed?")
    assert re.search(r"showing 20 of 22 similar-vehicle entities", answer.text)
    check_valid(answer, session, store)


# ---- (c) compound "people and vehicles" questions report both categories ---------------------
@pytest.mark.parametrize(
    "question",
    [
        "What people and vehicles were observed?",
        "Which vehicles and people were seen?",
        "Were any pedestrians or cars observed?",
    ],
)
def test_compound_question_reports_people_and_vehicles(security_db, question):
    answer, session, store = ask(security_db, question)
    assert "person 13, vehicle 7" in answer.text
    assert "Events by type: person 3, vehicle 4" in answer.text
    assert "[event:EVT-P-001]" in answer.text and "[event:EVT-V-001]" in answer.text
    assert "[entity:ENT-V-001]" in answer.text
    check_valid(answer, session, store)


def test_compound_question_on_a_large_run_reports_both_categories_and_the_limit(dbs):
    answer, session, store = ask(dbs["events23"], "What people and vehicles were observed?")
    assert "Events by type: person 15, vehicle 8" in answer.text
    assert re.search(r"showing 20 of 23 events", answer.text)
    assert "person" in answer.text and "vehicle" in answer.text
    check_valid(answer, session, store)


def test_specific_vehicle_questions_are_not_treated_as_compound(security_db):
    store = Store.open(security_db)
    analyst = OfflineAnalyst(AnalystTools(store))
    assert analyst.answer("Was the blue pickup seen more than once?").matched == "vehicle"
    assert analyst.answer("Did a person see the blue pickup?").matched == "vehicle"
    store.close()


# ---- (d) no truncation: nothing changes and no unnecessary notice ----------------------------
@pytest.mark.parametrize("question", sorted(GOLDEN))
def test_answers_that_are_not_truncated_are_byte_identical_to_the_baseline(security_db, question):
    store = Store.open(security_db)
    text = AnalystSession(store).ask(question).text
    store.close()
    assert text == GOLDEN[question]


def test_small_runs_carry_no_truncation_notice(dbs):
    for question in (
        "What happened at the main gate?",  # 4 events, but 4 observations > 3 sample? checked below
        "Which vehicles were observed?",
        "What people and vehicles were observed?",
        "What objects were observed?",
        "Summarize the run",
    ):
        answer, session, store = ask(dbs["small"], question)
        text = answer.text
        assert "not shown" not in text and "listed briefly" not in text, question
        assert not re.search(r"showing \d+ of \d+ (events|similar-vehicle entities|vehicle)", text)
        check_valid(answer, session, store)


def test_notice_wording_appears_only_when_something_was_omitted(dbs):
    for name in ("events23", "entities22"):
        text = AnalystSession(Store.open(dbs[name])).ask("What objects were observed?").text
        assert "not shown" in text
    text = AnalystSession(Store.open(dbs["entities11"])).ask("What objects were observed?").text
    assert "not shown" not in text  # 11 events and 11 entities are below the caps: listed in full
