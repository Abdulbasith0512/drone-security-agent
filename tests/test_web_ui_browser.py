# ruff: noqa: E501  (the one-line JavaScript snippets sent to the browser are long on purpose)
"""The dashboard in a real (headless) browser: what is on screen must equal what the API says.

A local Chrome or Edge is driven over the DevTools protocol (tests/cdp_browser.py); the tests skip
when none is installed. The server is the real dashboard server over the three demo runs, so these
check the page the evaluator will actually see: default run, counts against the API, every
interaction, run switching, the analyst, error states, accessibility and responsiveness.
"""

import json
import re
import threading
import urllib.request

import pytest
from cdp_browser import Browser, find_browser

from drone_agent.web.api import DashboardApi
from drone_agent.web.bundle import BundleError, build_demo_runs, build_real_run
from drone_agent.web.registry import RunRegistry
from drone_agent.web.server import make_server

pytestmark = pytest.mark.skipif(find_browser() is None, reason="no Chrome or Edge is installed")

RUNS = ["real_clip", "security_events", "synthetic_video"]
VIEWPORTS = [(1280, 720), (1440, 900), (1024, 768), (768, 1024), (390, 844)]
READY = (
    "document.querySelector('#kpi-strip .kpi .n') && document.querySelector('#viewer')"
    " && !document.querySelector('#video-body .skel') && !document.querySelector('#investigation-body .skel')"
    " && (() => { const i = document.querySelector('#viewer img'); return !i || (i.complete && i.naturalWidth > 0); })()"
)


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    out = tmp_path_factory.mktemp("uisite")
    try:
        runs = [build_real_run(out), *build_demo_runs(out)]
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise
    httpd = make_server(DashboardApi(RunRegistry(runs)), "127.0.0.1", 0, quiet=True)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def browser():
    b = Browser(find_browser())
    yield b
    b.close()


@pytest.fixture
def page(browser):
    p = browser.new_page()
    p.viewport(1440, 900)
    yield p
    browser.send("Target.closeTarget", {"targetId": p.target})


def api(site, path, body=None):
    request = urllib.request.Request(site.rstrip("/") + path)
    if body is not None:
        request = urllib.request.Request(
            site.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def open_run(page, site, name=None):
    page.goto(site + (f"#run={name}" if name else ""))
    page.wait_js(READY, 30)


def dom(page, selector, expression="true"):
    """Evaluate `expression` (using `el`) on the first element matching `selector`, else None."""
    return page.js(
        f"(() => {{ const el = document.querySelector({json.dumps(selector)}); return el ? ({expression}) : null; }})()"
    )


def click(page, selector):
    assert dom(page, selector, "(el.click(), true)"), selector


def kpis(page):
    return page.js(
        "Object.fromEntries([...document.querySelectorAll('#kpi-strip .kpi')].map(k => [k.dataset.kpi, k.querySelector('.n').textContent]))"
    )


def switch_run(page, name):
    page.js(
        f"(() => {{ const s = document.querySelector('#run-select'); s.value = {json.dumps(name)}; s.dispatchEvent(new Event('change')); }})()"
    )


# ---- the default run is the real video, with real frames --------------------------------------
def test_the_default_run_is_the_real_video_with_real_frames_and_no_errors(page, site):
    runs = api(site, "/api/runs")["runs"]
    first = api(site, f"/api/runs/{runs[0]['name']}/observations")["observations"][0]
    open_run(page, site)
    assert dom(page, "#run-select", "el.value") == runs[0]["name"] == "real_clip"
    assert dom(page, "#kind-banner .kind", "el.textContent") == runs[0]["label"] == "REAL VIDEO"
    chips = page.js(
        "[...document.querySelectorAll('#status-chips li')].map(l => l.textContent.trim().toUpperCase())"
    )
    assert chips == ["SYSTEM ONLINE", "REAL VIDEO", "VLM REPLAY", "REPRODUCIBLE"]
    image = dom(
        page,
        "#viewer img",
        "({w: el.naturalWidth, h: el.naturalHeight, src: el.getAttribute('src')})",
    )
    assert image["w"] > 0 and f"/frames/{first['frame_id']}.jpg" in image["src"]
    assert (
        dom(page, "#viewer", "getComputedStyle(el).aspectRatio") == "1920 / 1080"
    )  # from the API's video size
    assert dom(page, "#run-select optgroup", "el.label") == "Primary demo"
    assert not re.search(r"(?i)\blive\b", page.js("document.body.innerText"))
    assert page.problems() == []


def test_the_overlay_shows_zone_kind_time_and_the_selection_kind(page, site):
    detail = api(site, "/api/runs/real_clip/overview")["run"]
    obs = api(site, "/api/runs/real_clip/observations")["observations"][0]
    open_run(page, site)
    pills = page.js(
        "[...document.querySelectorAll('#viewer .ovpill')].map(p => p.textContent.trim().toLowerCase())"
    )
    assert "main gate" in pills and "real video" in pills and "vlm replay" in pills
    total = detail["video"]["duration_seconds"]
    assert f"0:00.0 / {int(total // 60)}:{total % 60:04.1f}" in pills
    assert f"frame {obs['frame_id']}" in pills and "observation" in pills


# ---- every number on screen equals the API -------------------------------------------------------
@pytest.mark.parametrize("name", RUNS)
def test_every_number_on_screen_equals_the_api(page, site, name):
    ov = api(site, f"/api/runs/{name}/overview")
    obs = api(site, f"/api/runs/{name}/observations")["observations"]
    c = ov["counts"]
    open_run(page, site, name)
    shown = kpis(page)
    assert [shown[k] for k in ("observations", "events", "entities", "alerts")] == [str(c[k]) for k in ("observations", "events", "entities", "alerts")]  # fmt: skip
    fresh = sum(s["count"] for s in ov["sources"] if not s["carried_forward"])
    assert shown["vlm-frames"] == (
        str(fresh) if {o["provenance"] for o in obs} == {"replayed"} else "n/a"
    )
    assert page.js("document.querySelectorAll('#strip .scard').length") == c["observations"]
    assert page.js("document.querySelectorAll('#timeline-body tbody tr').length") == c["events"]
    assert page.js("document.querySelectorAll('#overview-body .entity').length") == c["entities"]
    assert page.js("document.querySelectorAll('#alerts-body .alert-card').length") == c["alerts"]
    stages = page.js("[...document.querySelectorAll('#pipeline .pstage')].map(a => ({name: a.querySelector('.name').textContent, count: (a.querySelector('.count') || {}).textContent ?? null, cap: a.querySelector('.cap').textContent}))")  # fmt: skip
    video = ov["run"]["video"]
    assert stages[0]["count"] == (str(video["frame_count"]) if video else None)
    for stage, expected in zip(stages[1:], ov["pipeline"], strict=True):
        assert stage["count"] == (None if expected["count"] is None else str(expected["count"]))
        assert stage["cap"] == expected["caption"]
    banner = dom(page, "#kind-banner", "el.innerText")
    assert ov["run"]["detail"] in banner
    assert (
        ov["observation_span"]["first"][:19].replace("T", " ") in banner
        and ov["observation_span"]["last"][11:19] in banner
    )
    assert page.problems() == []


@pytest.mark.parametrize("name", RUNS)
def test_the_chips_follow_the_declared_provenance(page, site, name):
    provenance = {
        o["provenance"] for o in api(site, f"/api/runs/{name}/observations")["observations"]
    }
    open_run(page, site, name)
    chips = page.js(
        "[...document.querySelectorAll('#status-chips li')].map(l => l.textContent.trim())"
    )
    wording = {"replayed": "VLM replay", "canned": "Canned replay", "scripted": "Scripted text"}
    assert {wording[p] for p in provenance} <= set(chips)
    assert ("Reproducible" in chips) == (provenance <= {"replayed", "canned"})
    assert ("VLM replay" in chips) == (
        provenance == {"replayed"}
    )  # the synthetic demo is never called a VLM


# ---- alerts: a legitimate zero, and real cards ---------------------------------------------------
def test_the_real_video_shows_a_neutral_zero_alert_state(page, site):
    assert api(site, "/api/runs/real_clip/alerts")["total"] == 0
    open_run(page, site, "real_clip")
    text = dom(page, "#alerts-body .zero", "el.innerText")
    assert "0 ALERTS" in text and "No security rules triggered in this replay." in text
    assert (
        dom(page, "#alerts-body .panel-error") is None
        and dom(page, "#alerts-body .alert-card") is None
    )
    assert dom(page, "#alerts-body .zero .n", "getComputedStyle(el).color") != dom(
        page, ".error", "getComputedStyle(el).color"
    )


def test_the_scripted_run_shows_alert_cards_with_their_fields(page, site):
    alerts = api(site, "/api/runs/security_events/alerts")["alerts"]
    open_run(page, site, "security_events")
    cards = page.js(
        "[...document.querySelectorAll('#alerts-body .alert-card')].map(c => c.innerText)"
    )
    assert len(cards) == len(alerts) > 0
    for card, alert in zip(cards, alerts, strict=True):
        assert alert["title"] in card and alert["severity"] in card and alert["rule_id"] in card
        assert (
            alert["status"].upper() in card.upper()
            and alert["triggered_at"][:19].replace("T", " ") in card
        )
        assert "View evidence" in card
    assert "No security rules triggered" not in dom(page, "#alerts-body", "el.innerText")


def test_view_evidence_opens_the_evidence_and_highlights_the_event(page, site):
    alert = next(
        a for a in api(site, "/api/runs/security_events/alerts")["alerts"] if a["event_refs"]
    )
    detail = api(site, f"/api/runs/security_events/alerts/{alert['alert_id']}")
    event_id = alert["event_refs"][0].split(":")[1]
    open_run(page, site, "security_events")
    click(page, f"[data-alert={alert['alert_id']}] .btn.primary")
    page.wait_js(
        "document.querySelector('.alert-detail') && document.querySelector('.alert-detail').innerText.toLowerCase().includes('why it was raised')"
    )
    assert dom(page, f"[data-alert={alert['alert_id']}]", "el.classList.contains('sel')")
    assert alert["reason"] in dom(page, ".alert-detail", "el.innerText")
    expected_thumbs = sum(len(x["observations"]) for x in detail["evidence"])
    assert page.js("document.querySelectorAll('.alert-detail .evthumb').length") == expected_thumbs
    page.wait_js(
        f"document.querySelector('#investigation-body').innerText.includes({json.dumps(event_id)})"
    )
    assert dom(page, f"svg.timeline rect[data-event={event_id}]", "el.classList.contains('sel')")
    assert dom(page, "#viewer .textframe") is not None  # scripted text frame, never a fake picture


# ---- event, frame and alert interactions ---------------------------------------------------------
def test_an_event_click_selects_it_highlights_evidence_and_moves_the_viewer(page, site):
    obs = {
        o["observation_id"]: o
        for o in api(site, "/api/runs/real_clip/observations")["observations"]
    }
    events = api(site, "/api/runs/real_clip/events")["events"]
    event = next(e for e in events if e["observation_ids"][0] != "obs-0")
    frame = obs[event["observation_ids"][0]]["frame_id"]
    open_run(page, site, "real_clip")
    click(page, f"#timeline-body tr[data-event={event['event_id']}]")
    page.wait_js(
        f"(() => {{ const i = document.querySelector('#viewer img'); return i && i.getAttribute('src').includes('/frames/{frame}.jpg') && i.complete && i.naturalWidth > 0; }})()"
    )
    assert dom(
        page, f"svg.timeline rect[data-event={event['event_id']}]", "el.classList.contains('sel')"
    )
    assert dom(
        page, f"#timeline-body tr[data-event={event['event_id']}]", "el.classList.contains('sel')"
    )
    assert (
        dom(
            page,
            f"#strip [data-obs={event['observation_ids'][0]}]",
            "el.getAttribute('aria-current')",
        )
        == "true"
    )
    investigation = dom(page, "#investigation-body", "el.innerText")
    assert (
        event["event_id"] in investigation
        and "No security rule triggered for this event." in investigation
    )
    assert "Event detected" in dom(page, "#viewer .ovpill.event", "el.textContent") and event[
        "event_id"
    ] in dom(page, "#viewer .ovpill.event", "el.textContent")
    page.wait_js(
        "document.querySelector('#investigation-body .evthumb')"
    )  # the evidence behind the event is loaded
    assert page.js("document.querySelectorAll('#investigation-body .evthumb').length") == len(
        event["observation_ids"]
    )
    assert page.problems() == []


def test_a_frame_click_selects_the_observation_and_updates_the_investigation(page, site):
    obs = api(site, "/api/runs/real_clip/observations")["observations"]
    target = obs[2]
    open_run(page, site, "real_clip")
    click(page, f"#strip [data-obs={target['observation_id']}]")
    page.wait_js(
        f"document.querySelector('#viewer img').getAttribute('src').includes('/frames/{target['frame_id']}.jpg')"
    )
    investigation = dom(page, "#investigation-body", "el.innerText")
    assert (
        target["description"] in investigation
        and target["time"][:19].replace("T", " ") in investigation
    )
    assert (
        dom(
            page,
            f".scrub .mark[data-obs={target['observation_id']}]",
            "el.getAttribute('aria-current')",
        )
        == "true"
    )
    assert f"Observation 3 of {len(obs)}" in dom(page, "#where", "el.textContent")
    assert dom(page, "#viewer .ovpill.event") is None  # a frame was picked, not an event
    page.js("document.querySelector('.transport .btn:last-child').click()")  # Next
    page.wait_js(
        f"document.querySelector('#viewer img').getAttribute('src').includes('/frames/{obs[3]['frame_id']}.jpg')"
    )


def test_carried_forward_frames_are_labelled_as_such(page, site):
    obs = api(site, "/api/runs/real_clip/observations")["observations"]
    carried = next(o for o in obs if o["carried_forward"])
    open_run(page, site, "real_clip")
    assert "CARRIED" in dom(page, f"#strip [data-obs={carried['observation_id']}]", "el.innerText")
    click(page, f"#strip [data-obs={carried['observation_id']}]")
    page.wait_js("document.querySelector('#viewer .ovpill.carried')")
    assert "Carried forward" in dom(page, "#investigation-body", "el.innerText")


# ---- run switching -------------------------------------------------------------------------------
def test_switching_runs_refreshes_the_whole_dashboard_from_the_api(page, site):
    open_run(page, site)
    for name in ("security_events", "synthetic_video", "real_clip"):
        counts = api(site, f"/api/runs/{name}/overview")["counts"]
        label = next(r["label"] for r in api(site, "/api/runs")["runs"] if r["name"] == name)
        switch_run(page, name)
        page.wait_js(
            f"document.querySelector('#kind-banner .kind') && document.querySelector('#kind-banner .kind').textContent === {json.dumps(label)} && {READY}",
            30,
        )
        assert kpis(page)["observations"] == str(counts["observations"]) and kpis(page)[
            "alerts"
        ] == str(counts["alerts"])
        assert page.js("document.location.hash") == f"#run={name}"
        assert (
            page.js("document.querySelectorAll('#timeline-body tbody tr').length")
            == counts["events"]
        )
    assert page.problems() == []


def test_a_fast_switch_never_shows_the_previous_runs_data(page, site):
    real = api(site, "/api/runs/real_clip/overview")["counts"]
    open_run(page, site, "security_events")
    switch_run(page, "real_clip")
    switch_run(page, "synthetic_video")
    switch_run(page, "real_clip")  # the last choice wins, whatever order the answers arrive in
    page.wait_js(
        f"document.querySelector('#kind-banner .kind').textContent === 'REAL VIDEO' && {READY}", 30
    )
    page.js("new Promise(r => setTimeout(r, 800))")
    assert kpis(page)["observations"] == str(real["observations"]) and kpis(page)["events"] == str(
        real["events"]
    )
    assert page.js("document.querySelectorAll('#alerts-body .alert-card').length") == 0
    assert page.problems() == []


def test_a_run_in_the_url_hash_is_opened_and_an_unknown_one_falls_back_to_the_first(page, site):
    open_run(page, site, "synthetic_video")
    assert dom(page, "#run-select", "el.value") == "synthetic_video"
    # editing the address (or back/forward) opens that run without a reload
    page.js("location.hash = '#run=security_events'")
    page.wait_js(
        f"document.querySelector('#kind-banner .kind').textContent === 'SCRIPTED' && {READY}", 30
    )
    assert dom(page, "#run-select", "el.value") == "security_events"
    page.js("location.hash = '#run=does_not_exist'")  # an unknown run is ignored
    page.js("new Promise(r => setTimeout(r, 400))")
    assert dom(page, "#run-select", "el.value") == "security_events"
    page.goto("about:blank")  # a fresh load with an unknown run falls back to the first run
    open_run(page, site, "does_not_exist")
    assert dom(page, "#run-select", "el.value") == "real_clip"


# ---- the analyst ---------------------------------------------------------------------------------
def ask(page, question):
    page.js(
        f"(() => {{ document.querySelector('#chat-q').value = {json.dumps(question)}; document.querySelector('#chat-send').click(); }})()"
    )
    page.wait_js(
        "document.querySelectorAll('#chat-log .msg.bot:not(.busy)').length >= 2 && !document.querySelector('#chat-log .busy')",
        30,
    )


def test_the_analyst_answers_with_the_apis_text_and_clickable_citations(page, site):
    question = "What happened at the main gate?"
    expected = api(site, "/api/runs/real_clip/ask", {"question": question})
    open_run(page, site, "real_clip")
    ask(page, question)
    body = page.js(
        "[...document.querySelectorAll('#chat-log .msg.bot')].pop().querySelector('.body').textContent"
    )
    assert body == expected["text"]  # exactly what the backend said
    tags = page.js(
        "[...document.querySelectorAll('#chat-log .msg.bot:last-child .tag')].map(t => t.textContent)"
    )
    assert tags == ["Answer", "Evidence / citations"]
    tokens = {c["token"] for c in expected["citations"]}
    shown = set(
        page.js(
            "[...document.querySelectorAll('#chat-log .msg.bot:last-child .part:first-child .chip')].map(c => c.textContent)"
        )
    )
    assert shown == tokens
    assert (
        f"{len({c['token'] for c in expected['citations']})} citations checked against the database"
        in dom(page, "#chat-log .msg.bot:last-child", "el.innerText")
    )
    assert expected["label"] in dom(page, "#chat-log .msg.bot:last-child summary", "el.textContent")
    event_id = next(c["id"] for c in expected["citations"] if c["kind"] == "event")
    page.js(
        "[...document.querySelectorAll('#chat-log .msg.bot:last-child .cites .chip')].find(c => c.title.startsWith('Open event:')).click()"
    )
    page.wait_js(
        f"document.querySelector('#investigation-body').innerText.includes({json.dumps(event_id)})"
    )
    assert dom(page, f"svg.timeline rect[data-event={event_id}]", "el.classList.contains('sel')")
    assert dom(page, "#chat-log", "el.getAttribute('role')") == "log"
    # many answers together are taller than the log: the log must scroll and no answer may be squashed
    for more in (
        "What objects were observed?",
        "What happened at the main gate?",
        "Was the black van seen more than once?",
        "Summarize the run.",
    ):
        ask(page, more)
    clipped = page.js(
        "[...document.querySelectorAll('#chat-log .msg')].filter(m => m.scrollHeight > m.clientHeight + 1).length"
    )
    assert clipped == 0, f"{clipped} chat messages are clipped instead of the log scrolling"
    assert page.js(
        "(l => l.scrollHeight > l.clientHeight + 10)(document.querySelector('#chat-log'))"
    ), "the chat log does not scroll"
    page.js("document.querySelector('#chat-log').scrollTop = 0")
    first_answer = page.js(
        "(m => ({shown: m.clientHeight, whole: m.scrollHeight}))(document.querySelectorAll('#chat-log .msg.bot')[1])"
    )
    assert first_answer["shown"] >= first_answer["whole"] - 1 and first_answer["shown"] > 120
    assert page.problems() == []


def test_a_refused_question_is_shown_as_not_answered(page, site):
    open_run(page, site, "real_clip")
    ask(page, "Is the man at the gate a criminal?")
    refused = dom(page, "#chat-log .msg.refused", "el.innerText")
    assert refused and "NOT ANSWERED" in refused.upper() and "Matched wording" in refused


def test_the_question_chips_come_from_the_api(page, site):
    suggested = api(site, "/api/runs/real_clip/overview")["suggested_questions"]
    open_run(page, site, "real_clip")
    assert (
        page.js("[...document.querySelectorAll('#suggestions button')].map(b => b.textContent)")
        == suggested
    )
    assert dom(page, "#chat-q", "el.placeholder") == "Ask the security analyst..."
    assert (
        "Grounded in run evidence" in dom(page, "#panel-analyst", "el.innerText")
        or "GROUNDED IN RUN EVIDENCE" in dom(page, "#panel-analyst", "el.innerText").upper()
    )
    assert "Grounded answers from observed evidence" in dom(page, "#panel-analyst", "el.innerText")


# ---- loading, empty and error states -------------------------------------------------------------
FAIL_FETCH = """(() => {
  const original = window.fetch;
  window.fetch = (url, options) => {
    const u = String(url);
    if (%s) return Promise.resolve(new Response(JSON.stringify({error: 'boom'}), {status: 500, headers: {'Content-Type': 'application/json'}}));
    return original(url, options);
  };
})();"""


def test_one_failing_endpoint_shows_a_local_error_with_retry_and_the_rest_still_renders(page, site):
    page.init_script(FAIL_FETCH % "u.endsWith('/events')")
    page.goto(site)
    page.wait_js("document.querySelector('#timeline-body .panel-error')", 30)
    text = dom(page, "#timeline-body .panel-error", "el.innerText")
    assert "Could not load events" in text and "boom" in text and "Retry" in text
    assert dom(page, "#timeline-body .panel-error", "el.getAttribute('role')") == "alert"
    page.wait_js(READY, 30)
    assert kpis(page)["observations"] and dom(page, "#strip .scard") is not None
    assert dom(page, "#alerts-body .zero") is not None  # other panels are unaffected


def test_a_failing_observations_endpoint_does_not_blank_the_page(page, site):
    page.init_script(FAIL_FETCH % "u.includes('/observations')")
    page.goto(site)
    page.wait_js(
        "document.querySelector('#video-body .panel-error') && document.querySelector('#investigation-body .panel-error')",
        30,
    )
    assert "Could not load observations" in dom(page, "#video-body", "el.innerText")
    assert kpis(page)["events"] and dom(page, "#timeline-body tbody tr") is not None


def test_an_unreachable_api_is_reported_as_offline(page, site):
    page.init_script("window.fetch = () => Promise.reject(new Error('network down'));")
    page.goto(site)
    page.wait_js("document.querySelector('#error') && !document.querySelector('#error').hidden", 30)
    assert "network down" in dom(page, "#error", "el.textContent")
    chips = page.js(
        "[...document.querySelectorAll('#status-chips li')].map(l => l.textContent.trim())"
    )
    assert chips == ["API offline"]


def test_a_frame_that_cannot_load_shows_a_message_not_a_broken_image(page, site):
    page.block_urls(["*/frames/*"])
    page.expected_failures = ["/frames/", "ERR_BLOCKED"]
    page.goto(site)
    page.wait_js("document.querySelector('#viewer .nopic')", 30)
    page.js(
        "document.querySelector('#strip').scrollIntoView({block: 'center'})"
    )  # thumbnails load when seen
    assert "could not be loaded" in dom(page, "#viewer .nopic", "el.textContent")
    # the thumbnails fail too: each is replaced by a note, so no broken-image icon stays on screen
    page.wait_js("document.querySelectorAll('#strip .nopic').length > 0", 30)
    page.wait_js(
        "[...document.images].filter(i => i.hasAttribute('src')).every(i => !(i.complete && i.naturalWidth === 0))",
        30,
    )
    assert kpis(page)["observations"]  # and the rest of the page is unaffected


def test_a_loading_skeleton_is_shown_before_data_arrives(page, site):
    page.init_script(
        "(() => { const o = window.fetch; window.fetch = (u, p) => new Promise(r => setTimeout(() => r(o(u, p)), String(u).includes('/overview') ? 1500 : 0)); })();"
    )
    page.goto(site)
    page.wait_js("document.querySelector('#video-body .skel')", 10)
    assert page.js("document.querySelector('#video-body').getAttribute('aria-busy')") == "true"
    page.wait_js(READY, 30)
    assert dom(page, "#video-body", "el.getAttribute('aria-busy')") is None


# ---- accessibility and keyboard --------------------------------------------------------------------
def test_controls_have_names_images_have_alt_and_the_page_has_landmarks(page, site):
    open_run(page, site, "real_clip")
    page.wait_js("document.querySelectorAll('#strip img').length > 0")
    assert (
        page.js(
            "[...document.querySelectorAll('button')].filter(b => !(b.textContent.trim() || b.getAttribute('aria-label'))).length"
        )
        == 0
    )
    assert (
        page.js("[...document.querySelectorAll('img')].filter(i => !i.hasAttribute('alt')).length")
        == 0
    )
    assert (
        page.js(
            "[...document.querySelectorAll('select, textarea, input')].filter(e => !(e.getAttribute('aria-label') || e.closest('label') || document.querySelector('label[for=' + e.id + ']'))).length"
        )
        == 0
    )
    for landmark in ("header", "main", "footer", "nav#pipeline", "aside"):
        assert dom(page, landmark) is not None, landmark
    assert page.js("document.documentElement.lang") == "en"
    assert page.js("document.querySelectorAll('h1').length") == 1
    assert page.js(
        "[...document.querySelectorAll('[aria-expanded]')].every(e => ['true', 'false'].includes(e.getAttribute('aria-expanded')))"
    )


def test_the_page_is_operable_from_the_keyboard_with_visible_focus(page, site):
    open_run(page, site, "real_clip")
    page.press("Tab")
    assert dom(page, ".skip", "el === document.activeElement")
    for _ in range(60):
        if page.js("document.activeElement.classList.contains('mark')"):
            break
        page.press("Tab")
    assert page.js("document.activeElement.classList.contains('mark')")
    outline = page.js(
        "(() => { const s = getComputedStyle(document.activeElement); return [s.outlineStyle, s.outlineWidth]; })()"
    )
    assert outline == ["solid", "2px"]  # a visible focus ring
    before = page.js("document.activeElement.dataset.obs")
    page.press("ArrowRight")
    page.wait_js(
        f"document.activeElement.classList.contains('mark') && document.activeElement.dataset.obs !== {json.dumps(before)}"
    )
    assert page.js("document.activeElement.getAttribute('aria-current')") == "true"
    events = api(site, "/api/runs/real_clip/events")["events"]
    target = events[-1]["event_id"]
    page.js(f"document.querySelector('svg.timeline rect[data-event={target}]').focus()")
    page.press("Enter")
    page.wait_js(
        f"document.querySelector('#investigation-body').innerText.includes({json.dumps(target)})"
    )
    assert dom(page, f"svg.timeline rect[data-event={target}]", "el.classList.contains('sel')")
    page.js("document.querySelector('#alerts-body details.rules summary').focus()")
    page.press("Enter")
    assert page.js("document.querySelector('#alerts-body details.rules').open")


def test_an_alert_card_expands_and_collapses_from_the_keyboard(page, site):
    alert = api(site, "/api/runs/security_events/alerts")["alerts"][0]
    open_run(page, site, "security_events")
    head = f"[data-alert={alert['alert_id']}] .alert-head"

    def focused_head():
        """True when the focused element is this alert's header, reached by keyboard (so its ring shows)."""
        return page.js(
            f"document.activeElement === document.querySelector({json.dumps(head)})"
            " && document.activeElement.matches(':focus-visible')"
        )

    def settled(expanded: bool):
        page.wait_js(
            f"document.querySelector({json.dumps(head)}).getAttribute('aria-expanded') === '{str(expanded).lower()}'"
        )
        page.js("new Promise(r => setTimeout(r, 300))")  # let any asynchronous re-render finish

    assert dom(page, head, "el.getAttribute('aria-expanded')") == "false"
    page.js(f"document.querySelector({json.dumps(head)}).focus()")
    page.press("Enter")
    settled(True)
    # the detail arrives asynchronously and rebuilds the list: focus must survive that render
    page.wait_js(
        f"document.querySelector('#ad-{alert['alert_id']}') && document.querySelector('#ad-{alert['alert_id']}').innerText.toLowerCase().includes('why it was raised')"
    )
    page.js("new Promise(r => setTimeout(r, 300))")
    assert dom(page, f"#ad-{alert['alert_id']}") is not None
    assert focused_head(), "focus was lost when the alert detail loaded"
    # collapse WITHOUT focusing again: the keypress must reach the same header
    page.press("Enter")
    settled(False)
    assert dom(page, f"#ad-{alert['alert_id']}") is None
    assert focused_head(), "focus was lost when the alert collapsed"
    page.press(" ")  # Space expands it again (the detail is cached now)
    settled(True)
    assert dom(page, f"#ad-{alert['alert_id']}") is not None
    assert focused_head(), "focus was lost when the alert expanded with Space"
    assert (
        page.js("document.activeElement.closest('[data-alert]').dataset.alert") == alert["alert_id"]
    )
    # the "View evidence" button of the same card keeps focus through its own (asynchronous) re-render too
    button = f"[data-alert={alert['alert_id']}] .alert-actions .btn"
    page.js(f"document.querySelector({json.dumps(button)}).focus()")
    page.press("Enter")
    page.js("new Promise(r => setTimeout(r, 400))")
    assert page.js(
        f"document.activeElement === document.querySelector({json.dumps(button)})"
        " && document.activeElement.matches(':focus-visible')"
    ), "focus left the View evidence button after the evidence loaded"


# ---- responsive: five viewports, two kinds of run ------------------------------------------------
OFFENDERS = """(() => {
  const limit = window.innerWidth + 1, bad = [];
  const scrolls = (el) => ['auto', 'scroll', 'hidden'].includes(getComputedStyle(el).overflowX);
  for (const el of document.body.querySelectorAll('*')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.right <= limit) continue;
    let p = el.parentElement, contained = false;
    while (p && p !== document.body) { if (scrolls(p) && p.getBoundingClientRect().right <= limit) { contained = true; break; } p = p.parentElement; }
    if (!contained) bad.push(el.tagName + '.' + el.className + ' right=' + Math.round(r.right));
  }
  return bad.slice(0, 8);
})()"""


@pytest.mark.parametrize("name", ["real_clip", "security_events"])
@pytest.mark.parametrize("width, height", VIEWPORTS)
def test_no_console_errors_no_broken_images_no_horizontal_overflow(page, site, name, width, height):
    page.viewport(width, height)
    open_run(page, site, name)
    page.js(
        "(async () => { for (let y = 0; y < document.body.scrollHeight; y += 400) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 80)); } window.scrollTo(0, 0); const s = document.querySelector('#strip'); s.scrollIntoView({block: 'center'}); await new Promise(r => setTimeout(r, 150)); for (let x = 0; x <= s.scrollWidth; x += 150) { s.scrollLeft = x; await new Promise(r => setTimeout(r, 60)); } s.scrollLeft = 0; window.scrollTo(0, 0); })()"
    )
    # thumbnails load as they scroll into view, two at a time: wait until every one has arrived
    page.wait_js("[...document.images].every(i => i.complete && i.naturalWidth > 0)", 90)
    assert (
        page.js(
            "[...document.images].filter(i => !(i.complete && i.naturalWidth > 0)).map(i => i.src)"
        )
        == []
    )
    assert page.js("document.documentElement.scrollWidth <= window.innerWidth")
    assert page.js(OFFENDERS) == []
    assert dom(page, "#viewer", "el.getBoundingClientRect().height") > 100
    if width > 1180:  # a sticky header must stay compact; below that it scrolls away
        assert dom(page, ".top", "getComputedStyle(el).position") == "sticky"
        assert dom(page, ".top", "el.getBoundingClientRect().height") < 90
    else:
        assert dom(page, ".top", "getComputedStyle(el).position") == "static"
    assert page.problems() == []


def test_the_layout_changes_with_the_viewport(page, site):
    def box(selector):
        return dom(
            page, selector, "(r => ({x: r.x, y: r.y, w: r.width}))(el.getBoundingClientRect())"
        )

    page.viewport(1440, 900)
    open_run(page, site, "real_clip")
    video, side = box("#panel-video"), box("#panel-investigation")
    assert (
        side["x"] > video["x"] + video["w"] - 1 and abs(side["y"] - video["y"]) < 2
    )  # side by side
    page.viewport(768, 1024)
    page.js("window.dispatchEvent(new Event('resize'))")
    page.js("new Promise(r => setTimeout(r, 300))")
    video, side = box("#panel-video"), box("#panel-investigation")
    assert side["y"] > video["y"] and abs(side["x"] - video["x"]) < 2  # stacked
    page.viewport(390, 844)
    page.js("new Promise(r => setTimeout(r, 300))")
    assert box("#panel-video")["w"] <= 390
