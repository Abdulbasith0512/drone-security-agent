"""The dashboard adapter: every payload must equal what the existing backend returns, read-only."""

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import ROOT

from drone_agent.agent.citations import extract_citations
from drone_agent.models import Severity
from drone_agent.rules import load_rules_config
from drone_agent.store import Store, StoreError
from drone_agent.web.api import (
    LIMITS,
    MAX_QUESTION_CHARS,
    MAX_SESSIONS,
    ApiError,
    DashboardApi,
    open_readonly,
)
from drone_agent.web.bundle import BundleError, build_demo_runs
from drone_agent.web.registry import RunRegistry

SCRIPTED, VIDEO = "security_events", "synthetic_video"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    try:
        return build_demo_runs(tmp_path_factory.mktemp("dashapi"))
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise


@pytest.fixture(scope="module")
def api(runs):
    return DashboardApi(RunRegistry(runs))


@pytest.fixture
def scripted_store(runs):
    with Store.open(runs[0].db) as store:
        yield store


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- runs and overview -----------------------------------------------------------------------
def test_runs_list_the_declared_kind_and_headline_counts(api):
    listing = {r["name"]: r for r in api.runs()["runs"]}
    assert listing[SCRIPTED]["label"] == "SCRIPTED" and not listing[SCRIPTED]["has_video"]
    assert listing[VIDEO]["label"] == "SYNTHETIC DEMO" and listing[VIDEO]["has_video"]
    assert listing[VIDEO]["video"]["frame_count"] == 200 and listing[VIDEO]["video"]["fps"] == 10.0
    assert listing[SCRIPTED]["counts"] == {
        "observations": 27, "events": 7, "entities": 2, "alerts": 6
    }  # fmt: skip


def test_the_overview_equals_the_existing_backends_numbers(api, scripted_store):
    ov = api.overview(SCRIPTED)
    s = scripted_store
    assert ov["counts"] == {
        "frames": s.count_frames(),
        "observations": s.count_observations(),
        "detected_objects": s.count_detected_objects(),
        "events": s.count_events(),
        "entities": s.count_entities(),
        "alerts": s.count_alerts(),
    }
    alerts = s.query_alerts()
    for severity, n in ov["severity_breakdown"].items():
        assert n == sum(a.severity.value == severity for a in alerts)
    assert sum(ov["severity_breakdown"].values()) == len(alerts)
    assert ov["rule_breakdown"] == {
        r: sum(a.rule_id == r for a in alerts) for r in {a.rule_id for a in alerts}
    }
    assert ov["object_counts"] == s.detected_object_counts()
    assert {z["zone"]: z["observations"] for z in ov["zones"]} == s.observation_counts_by("zone")
    assert ov["severity_order"] == [sev.value for sev in Severity]
    assert ov["limitations"] and "simulated" in ov["limitations"][0]


def test_the_pipeline_strip_is_the_documented_six_stages_with_backend_counts(api):
    stages = api.overview(SCRIPTED)["pipeline"]
    assert [s["key"] for s in stages] == [
        "observations", "events", "entities", "rules", "alerts", "analyst"
    ]  # fmt: skip
    assert [s["count"] for s in stages] == [27, 7, 2, 4, 6, None]
    assert all(s["panel"].startswith("panel-") for s in stages)


def test_sources_distinguish_carried_forward_from_new_observations(api):
    sources = {s["source"]: s for s in api.overview(VIDEO)["sources"]}
    assert sources["cassette"]["carried_forward"] is False
    assert sources["cassette+carried"]["carried_forward"] is True
    assert sources["cassette"]["count"] == 4 and sources["cassette+carried"]["count"] == 6


def test_suggested_questions_come_from_the_runs_own_data(api):
    scripted = api.overview(SCRIPTED)["suggested_questions"]
    assert "Why was the loitering alert raised?" in scripted
    assert "Was the blue pickup seen more than once?" in scripted
    video = api.overview(VIDEO)["suggested_questions"]
    assert "Why was the loitering alert raised?" not in video


def test_the_rules_are_the_backends_own_configuration_not_a_copy(api):
    rules = {r["id"]: r for r in api.rules()["rules"]}
    config = load_rules_config()
    assert set(rules) == set(config.model_dump())
    assert rules["loitering"]["settings"]["zones"]["main_gate"]["threshold_seconds"] == 300
    assert rules["repeat_vehicle"]["settings"]["minimum_appearances"] == 3


# ---- events, entities, alerts -----------------------------------------------------------------
def test_events_equal_the_store_in_order_with_alert_links(api, scripted_store):
    payload = api.events(SCRIPTED)
    stored = scripted_store.query_events()
    assert [e["event_id"] for e in payload["events"]] == [e.event_id for e in stored]
    for row, ev in zip(payload["events"], stored, strict=True):
        assert row["duration_seconds"] == ev.duration_seconds
        assert (row["zone"], row["type"]) == (ev.zone.value, ev.object_type.value)
        assert row["frame_ids"] == ev.frame_ids and row["observation_ids"] == ev.observation_ids
    first = payload["events"][0]
    assert first["alert_refs"] == [
        f"alert:{a.alert_id}"
        for a in scripted_store.query_alerts()
        if first["event_id"] in a.event_ids
    ]
    assert payload["span"]["first"] == min(e.first_seen for e in stored).isoformat()


def test_event_filters_use_the_stores_filters_and_reject_bad_values(api, scripted_store):
    people = api.events(SCRIPTED, object_type="person")["events"]
    assert [e["event_id"] for e in people] == [
        e.event_id for e in scripted_store.query_events(object_type="person")
    ]
    gate = api.events(SCRIPTED, zone="main_gate")["events"]
    assert gate and all(e["zone"] == "main_gate" for e in gate)
    for kwargs in ({"zone": "moon_base"}, {"object_type": "spaceship"}):
        with pytest.raises(ApiError) as err:
            api.events(SCRIPTED, **kwargs)
        assert err.value.status == 400


def test_event_detail_returns_the_evidence_alerts_and_entity(api, scripted_store):
    detail = api.event_detail(SCRIPTED, "EVT-V-003")
    assert detail["event"]["ref"] == "event:EVT-V-003"
    assert [o["observation_id"] for o in detail["observations"]] == [
        o.observation_id for o in scripted_store.get_event_observations("EVT-V-003")
    ]
    assert detail["entity"]["ref"] == "entity:ENT-V-001"
    assert [a["rule_id"] for a in detail["alerts"]] == ["repeat_vehicle"]
    with pytest.raises(ApiError) as err:
        api.event_detail(SCRIPTED, "EVT-X-999")
    assert err.value.status == 404


def test_people_events_are_never_linked_to_an_entity(api):
    detail = api.event_detail(SCRIPTED, "EVT-P-001")
    assert detail["entity"] is None


def test_entities_are_similar_vehicle_groups_never_confirmed_identity(api):
    payload = api.entities(SCRIPTED)
    assert payload["total"] == 2
    assert all(e["identity_confirmed"] is False for e in payload["entities"])
    assert payload["entities"][0]["label"].startswith("similar ")


def test_alerts_equal_the_store_and_the_detail_is_the_evidence_chain(api, scripted_store):
    listing = api.alerts(SCRIPTED)
    assert [a["alert_id"] for a in listing["alerts"]] == [
        a.alert_id for a in scripted_store.query_alerts()
    ]
    detail = api.alert_detail(SCRIPTED, "ALR-LOITER-EVT-P-001")
    stored = scripted_store.get_alert("ALR-LOITER-EVT-P-001")
    assert detail["alert"]["rule_id"] == "loitering"
    assert detail["alert"]["severity"] == stored.severity.value
    assert detail["alert"]["frame_ids"] == stored.frame_ids
    assert {f["name"]: f["value"] for f in detail["facts"]} == stored.facts
    assert [x["event_ref"] for x in detail["evidence"]] == [f"event:{e}" for e in stored.event_ids]
    assert len(detail["evidence"][0]["observations"]) == 8
    assert "480s" in detail["alert"]["reason"]
    with pytest.raises(ApiError) as err:
        api.alert_detail(SCRIPTED, "ALR-NOPE")
    assert err.value.status == 404


# ---- observations and frames --------------------------------------------------------------------
def test_observations_carry_source_carried_flag_event_links_and_simulated_telemetry(api):
    rows = api.observations(VIDEO)["observations"]
    assert [r["frame_id"] for r in rows] == [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
    for row in rows:
        assert row["carried_forward"] == row["source"].endswith("+carried")
        assert row["image"] is True
        assert row["telemetry_simulated"] is not None
    assert rows[2]["event_refs"] == ["event:EVT-P-001"] and rows[0]["event_refs"] == []


def test_scripted_observations_have_no_image_and_no_carried_flag(api):
    rows = api.observations(SCRIPTED)["observations"]
    assert len(rows) == 27
    assert not any(r["image"] or r["carried_forward"] for r in rows)
    assert {r["source"] for r in rows} == {"scripted"}


def test_observation_filters_and_limits_use_the_store(api, scripted_store):
    gate = api.observations(SCRIPTED, zone="main_gate")
    assert gate["total"] == scripted_store.count_matching_observations(zone="main_gate")
    hits = api.observations(SCRIPTED, query="pickup")
    assert hits["total"] >= 1 and all(
        "pickup" in json.dumps(o).lower() for o in hits["observations"]
    )
    capped = api.observations(SCRIPTED, limit=5)
    assert capped["returned"] == 5 and capped["truncated"] and capped["total"] == 27
    assert api.observations(SCRIPTED, limit=10**6)["returned"] == 27  # clamped, not an error
    with pytest.raises(ApiError):
        api.observations(SCRIPTED, zone="moon_base")


def test_frames_are_served_only_for_registered_video_runs(api):
    assert api.frame_jpeg(VIDEO, 40)[:3] == b"\xff\xd8\xff"
    for name, frame in ((SCRIPTED, 1), (VIDEO, 200), (VIDEO, -1), ("nope", 1)):
        with pytest.raises(ApiError) as err:
            api.frame_jpeg(name, frame)
        assert err.value.status == 404


# ---- the analyst -----------------------------------------------------------------------------
def test_an_answer_is_the_existing_template_analysts_and_is_labelled_offline(api):
    answer = api.ask(SCRIPTED, "Why was the repeat vehicle alert raised?")
    assert answer["mode"] == "template" and answer["refused"] is False and answer["verified"]
    assert answer["label"] == "template answer (deterministic, no LLM)"
    assert "no language model, no network and no Gemini call" in answer["origin"]["analyst"]
    assert "SCRIPTED" in answer["origin"]["observations"]
    assert "[alert:ALR-REPEAT-ENT-V-001]" in answer["text"]


def test_every_citation_in_an_answer_exists_in_the_database(api, scripted_store):
    for question in ("Summarize the run.", "List the alerts.", "What happened at the main gate?"):
        answer = api.ask(SCRIPTED, question)
        assert answer["citations"], question
        for citation in answer["citations"]:
            kind, ident = citation["kind"], citation["id"]
            exists = {
                "event": scripted_store.get_event,
                "alert": scripted_store.get_alert,
                "entity": scripted_store.get_entity,
                "obs": scripted_store.get_observation,
                "frame": lambda i: scripted_store.get_observation_by_frame(int(i)),
            }[kind](ident)
            assert exists is not None, citation
        assert [c["token"] for c in answer["citations"]] == [
            c.token for c in extract_citations(answer["text"])
        ]


def test_follow_up_questions_resolve_from_the_session(api):
    first = api.ask(SCRIPTED, "Why was the repeat vehicle alert raised?")
    assert "Where was it seen?" in first["followups"]
    follow = api.ask(SCRIPTED, "Where was it seen?", first["session_id"])
    assert follow["session_id"] == first["session_id"]
    assert "similar blue pickup" in follow["text"] and "[entity:ENT-V-001]" in follow["text"]
    cold = api.ask(SCRIPTED, "Where was it seen?")  # no session: the pronoun stays unresolved
    assert "similar blue pickup" not in cold["text"]


def test_a_session_is_never_shared_across_runs(api):
    first = api.ask(SCRIPTED, "Why was the repeat vehicle alert raised?")
    other = api.ask(VIDEO, "Where was it seen?", first["session_id"])
    # the video run also has an ENT-V-001: had the context leaked, "it" would have resolved to it
    assert other["text"].startswith("Insufficient evidence")
    assert other["session_id"] != first["session_id"]  # a new session, not the other run's


def test_unsupported_questions_get_an_honest_insufficient_evidence(api):
    answer = api.ask(SCRIPTED, "What is the airspeed velocity of an unladen swallow?")
    assert answer["refused"] is False
    assert answer["text"].startswith("Insufficient evidence")
    assert not answer["text"].lower().count("swallow") > 1  # nothing was invented about it


def test_identity_intent_and_criminality_questions_are_refused_without_asking_the_analyst(
    api, monkeypatch
):
    def fail(*_a, **_k):
        raise AssertionError("the analyst must not be asked")

    monkeypatch.setattr("drone_agent.web.api.AnalystSession", fail)
    for question, category in (
        ("Is the person at the gate a criminal?", "criminality"),
        ("Who is the person at the gate?", "identity"),
        ("What was he planning?", "intent"),
    ):
        answer = api.ask(SCRIPTED, question)
        assert answer["refused"] is True and answer["mode"] == "guard"
        assert answer["guard"]["category"] == category
        assert "not a semantic safety classifier" in answer["guard"]["policy"]
        assert answer["citations"] == [] and answer["suggested_questions"]


def test_a_refused_question_is_not_remembered_by_the_session(api):
    first = api.ask(SCRIPTED, "Why was the repeat vehicle alert raised?")
    api.ask(SCRIPTED, "Is he a criminal?", first["session_id"])
    follow = api.ask(SCRIPTED, "Where was it seen?", first["session_id"])
    assert "similar blue pickup" in follow["text"]


def test_bad_questions_and_unknown_runs_are_client_errors(api):
    for question in ("", "   ", "x" * (MAX_QUESTION_CHARS + 1)):
        with pytest.raises(ApiError) as err:
            api.ask(SCRIPTED, question)
        assert err.value.status == 400
    with pytest.raises(ApiError) as err:
        api.ask("nope", "List the alerts.")
    assert err.value.status == 404


def test_the_number_of_remembered_sessions_is_bounded(runs):
    fresh = DashboardApi(RunRegistry(runs))
    for _ in range(MAX_SESSIONS + 10):
        fresh.ask(SCRIPTED, "List the alerts.")
    assert len(fresh._sessions) == MAX_SESSIONS


def test_the_summary_is_the_deterministic_offline_summary(api):
    summary = api.summary(SCRIPTED)
    assert summary["text"].startswith("Stored run: 27 observation(s)")
    assert summary["label"] == "template answer (deterministic, no LLM)"


# ---- read-only, offline ---------------------------------------------------------------------
def test_databases_are_opened_read_only_so_a_write_cannot_succeed(runs):
    store = open_readonly(runs[0].db)
    try:
        with pytest.raises(Exception, match="readonly"):
            store._conn.execute("DELETE FROM alerts")
        with pytest.raises(Exception, match="readonly"):
            store._conn.execute("INSERT INTO telemetry (ts_utc) VALUES ('x')")
        assert store.count_alerts() == 6
    finally:
        store.close()


def test_using_every_endpoint_leaves_the_database_files_byte_identical(runs):
    fresh = DashboardApi(RunRegistry(runs))
    before = {r.name: fingerprint(r.db) for r in runs}
    for name in (SCRIPTED, VIDEO):
        fresh.runs(), fresh.overview(name), fresh.events(name), fresh.entities(name)
        fresh.alerts(name), fresh.observations(name), fresh.summary(name)
        fresh.ask(name, "Summarize the run."), fresh.ask(name, "Is he a criminal?")
        for alert in fresh.alerts(name)["alerts"]:
            fresh.alert_detail(name, alert["alert_id"])
        for event in fresh.events(name)["events"]:
            fresh.event_detail(name, event["event_id"])
    assert {r.name: fingerprint(r.db) for r in runs} == before


def test_a_missing_or_outdated_database_is_a_store_error(tmp_path):
    import sqlite3

    with pytest.raises(StoreError, match="not found"):
        open_readonly(tmp_path / "missing.db")
    old = tmp_path / "old.db"
    conn = sqlite3.connect(old)
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="schema version 1"):
        open_readonly(old)
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not sqlite" * 50)
    with pytest.raises(StoreError, match="not a readable SQLite"):
        open_readonly(junk)


def test_the_adapter_makes_no_network_connection_and_reads_no_key(runs, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the dashboard adapter must stay offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr("dotenv.load_dotenv", refuse)
    monkeypatch.setenv("GEMINI_API_KEY", "TEST-PLACEHOLDER-NOT-A-REAL-KEY")
    fresh = DashboardApi(RunRegistry(runs))
    payloads = [fresh.overview(SCRIPTED), fresh.ask(SCRIPTED, "List the alerts."), fresh.rules()]
    assert "TEST-PLACEHOLDER" not in json.dumps(payloads)


def test_the_dashboard_never_imports_the_model_or_analyst_llm_layers():
    code = (
        "import sys\n"
        "sys.modules['langchain_core'] = None\n"
        "sys.modules['langchain_google_genai'] = None\n"
        "import drone_agent.web.api, drone_agent.web.server, drone_agent.web.guard\n"
        "import drone_agent.web.frames, drone_agent.web.registry\n"
        "banned = ('drone_agent.agent.llm', 'drone_agent.agent.build',\n"
        "          'drone_agent.vision.gemini')\n"
        "loaded = [m for m in banned if m in sys.modules]\n"
        "assert not loaded, loaded\n"
        "print('ok')\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60
    )
    assert done.returncode == 0 and "ok" in done.stdout, done.stderr


def test_listings_are_not_silently_cut_by_the_backends_small_default_caps(api):
    assert LIMITS.max_events >= 1000 and LIMITS.max_evidence >= 500
    assert api.observations(SCRIPTED)["returned"] == 27
