"""The dashboard's HTTP server, exercised over a real loopback socket."""

import hashlib
import http.client
import json
import re
import threading

import pytest

from drone_agent.rules import load_rules_config
from drone_agent.web.api import DashboardApi
from drone_agent.web.bundle import BundleError, build_demo_runs
from drone_agent.web.registry import RunRegistry
from drone_agent.web.server import CSP, MAX_BODY_BYTES, STATIC_DIR, STATIC_FILES, make_server

SCRIPTED, VIDEO = "security_events", "synthetic_video"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    try:
        return build_demo_runs(tmp_path_factory.mktemp("dashsrv"))
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise


@pytest.fixture(scope="module")
def server(runs):
    httpd = make_server(DashboardApi(RunRegistry(runs)), "127.0.0.1", 0, quiet=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        data = response.read()
        return response.status, dict(response.getheaders()), data
    finally:
        conn.close()


def get_json(server, path, **kw):
    status, headers, data = request(server, "GET", path, **kw)
    return status, headers, json.loads(data)


def post_json(server, path, payload, headers=None):
    return request(
        server,
        "POST",
        path,
        body=json.dumps(payload),
        headers={"Content-Type": "application/json", **(headers or {})},
    )


# ---- static files ------------------------------------------------------------------------------
def test_the_page_and_its_assets_are_served_with_safe_headers(server):
    for path, (filename, content_type) in STATIC_FILES.items():
        status, headers, body = request(server, "GET", path)
        assert status == 200, path
        assert headers["Content-Type"] == content_type
        assert body == (STATIC_DIR / filename).read_bytes()
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Content-Security-Policy"] == CSP
        assert headers["Cache-Control"] == "no-store"
    assert b"Drone Security Analyst Dashboard" in request(server, "GET", "/")[2]


def test_the_content_security_policy_forbids_anything_outside_localhost():
    assert "default-src 'none'" in CSP and "script-src 'self'" in CSP
    assert "connect-src 'self'" in CSP and "frame-ancestors 'none'" in CSP
    assert "unsafe-inline" not in CSP and "unsafe-eval" not in CSP and "http" not in CSP


@pytest.mark.parametrize(
    "path",
    [
        "/static/../pyproject.toml",
        "/static/%2e%2e/pyproject.toml",
        "/static/..%2fpyproject.toml",
        "/static/",
        "/static",
        "/static/app.js/../../../README.md",
        "/README.md",
        "/index.html",
        "/etc/passwd",
        "//etc/passwd",
        "/static/app.js%00.txt",
    ],
)
def test_only_the_whitelisted_files_are_ever_served(server, path):
    status, _, body = request(server, "GET", path)
    assert status == 404, path
    assert b"[project]" not in body and b"root:" not in body


def test_the_favicon_probe_is_answered_quietly(server):
    assert request(server, "GET", "/favicon.ico")[0] == 204
    assert request(server, "GET", "/static/favicon.svg")[2].startswith(b"<svg")


def test_head_requests_return_headers_without_a_body(server):
    status, headers, body = request(server, "HEAD", "/")
    assert status == 200 and body == b"" and int(headers["Content-Length"]) > 0


# ---- the JSON API over HTTP ----------------------------------------------------------------------
def test_json_endpoints_return_the_adapters_payloads(server):
    status, headers, runs = get_json(server, "/api/runs")
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert [r["name"] for r in runs["runs"]] == [SCRIPTED, VIDEO]
    assert get_json(server, f"/api/runs/{SCRIPTED}/overview")[2]["counts"]["alerts"] == 6
    assert get_json(server, f"/api/runs/{SCRIPTED}/events")[2]["total"] == 7
    assert get_json(server, f"/api/runs/{SCRIPTED}/entities")[2]["total"] == 2
    assert get_json(server, f"/api/runs/{SCRIPTED}/alerts")[2]["total"] == 6
    assert get_json(server, f"/api/runs/{SCRIPTED}/alerts/ALR-LOITER-EVT-P-001")[2]["alert"]
    assert get_json(server, f"/api/runs/{SCRIPTED}/events/EVT-P-001")[2]["observations"]
    assert get_json(server, f"/api/runs/{SCRIPTED}/observations?limit=3")[2]["returned"] == 3
    assert get_json(server, f"/api/runs/{SCRIPTED}/observations?zone=main_gate&q=jacket")[0] == 200
    assert get_json(server, f"/api/runs/{SCRIPTED}/summary")[2]["text"].startswith("Stored run")
    assert len(get_json(server, "/api/rules")[2]["rules"]) == 4


def test_errors_are_json_with_the_right_status(server):
    cases = [
        ("/api/runs/nope/overview", 404),
        ("/api/runs/bad.name/overview", 404),
        (f"/api/runs/{SCRIPTED}/events/EVT-X-999", 404),
        (f"/api/runs/{SCRIPTED}/alerts/ALR-NOPE", 404),
        (f"/api/runs/{SCRIPTED}/events?zone=moon", 400),
        (f"/api/runs/{SCRIPTED}/observations?limit=abc", 400),
        ("/api/nothing", 404),
        (f"/api/runs/{SCRIPTED}/alerts/../../etc", 404),
    ]
    for path, expected in cases:
        status, _, payload = get_json(server, path)
        assert status == expected and "error" in payload, path
        assert "Traceback" not in json.dumps(payload)


def test_a_bad_query_string_is_rejected_not_crashed(server):
    assert request(server, "GET", f"/api/runs/{SCRIPTED}/events?" + "a=1&" * 30)[0] == 400
    assert request(server, "GET", f"/api/runs/{SCRIPTED}/events?q=" + "x" * 600)[0] == 414


def test_frames_are_only_available_for_registered_video_runs(server):
    status, headers, body = request(server, "GET", f"/api/runs/{VIDEO}/frames/40.jpg?w=320")
    assert status == 200 and headers["Content-Type"] == "image/jpeg" and body[:3] == b"\xff\xd8\xff"
    assert request(server, "GET", f"/api/runs/{SCRIPTED}/frames/1.jpg")[0] == 404
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/9999.jpg")[0] == 404
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/abc.jpg")[0] == 404
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/-1.jpg")[0] == 404
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/1234567890123.jpg")[0] == 404
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/40.jpg?w=abc")[0] == 400
    assert request(server, "GET", f"/api/runs/{VIDEO}/frames/../../etc.jpg")[0] == 404


# ---- POST /ask -----------------------------------------------------------------------------------
def test_asking_over_http_returns_a_cited_answer_and_supports_follow_ups(server):
    status, _, data = post_json(
        server,
        f"/api/runs/{SCRIPTED}/ask",
        {"question": "Why was the repeat vehicle alert raised?"},
    )
    first = json.loads(data)
    assert status == 200 and first["mode"] == "template" and first["citations"]
    _, _, data = post_json(
        server,
        f"/api/runs/{SCRIPTED}/ask",
        {"question": "Where was it seen?", "session_id": first["session_id"]},
    )
    assert "similar blue pickup" in json.loads(data)["text"]


def test_a_refused_question_is_answered_with_the_guard_over_http(server):
    _, _, data = post_json(
        server, f"/api/runs/{SCRIPTED}/ask", {"question": "Is the person a criminal?"}
    )
    answer = json.loads(data)
    assert answer["refused"] and answer["guard"]["category"] == "criminality"


@pytest.mark.parametrize(
    ("body", "content_type", "status"),
    [
        (b"not json", "application/json", 400),
        (b"[1, 2]", "application/json", 400),
        (b"{}", "application/json", 400),
        (b'{"question": 5}', "application/json", 400),
        (b'{"question": "x", "session_id": 7}', "application/json", 400),
        (b'{"question": "x", "session_id": "' + b"a" * 100 + b'"}', "application/json", 400),
        (b'{"question": "List the alerts."}', "text/plain", 415),
        (b'{"question": "' + b"x" * (MAX_BODY_BYTES + 10) + b'"}', "application/json", 413),
        (b'{"question": ""}', "application/json", 400),
    ],
)
def test_bad_ask_requests_are_rejected_cleanly(server, body, content_type, status):
    got, _, data = request(
        server,
        "POST",
        f"/api/runs/{SCRIPTED}/ask",
        body=body,
        headers={"Content-Type": content_type},
    )
    assert got == status, data[:80]


def test_post_only_works_on_the_ask_route_and_needs_a_length(server):
    assert post_json(server, "/api/runs", {"question": "x"})[0] == 404
    assert post_json(server, f"/api/runs/{SCRIPTED}/alerts", {"question": "x"})[0] == 404
    assert post_json(server, "/api/runs/nope/ask", {"question": "x"})[0] == 404
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    conn.putrequest("POST", f"/api/runs/{SCRIPTED}/ask")
    conn.putheader("Content-Type", "application/json")
    conn.endheaders()
    assert conn.getresponse().status == 411
    conn.close()


# ---- Host / Origin / methods ---------------------------------------------------------------------
def test_requests_with_a_foreign_host_header_are_refused(server):
    for host in ("evil.example", "127.0.0.1", "127.0.0.1:1", "localhost:9999"):
        assert request(server, "GET", "/api/runs", headers={"Host": host})[0] == 403, host
    ok_host = f"localhost:{server.server_port}"
    assert request(server, "GET", "/api/runs", headers={"Host": ok_host})[0] == 200


def test_cross_origin_requests_are_refused_and_same_origin_is_allowed(server):
    same = f"http://127.0.0.1:{server.server_port}"
    assert request(server, "GET", "/api/runs", headers={"Origin": same})[0] == 200
    for origin in ("http://evil.example", "null", "https://127.0.0.1:1"):
        status, _, _ = post_json(
            server,
            f"/api/runs/{SCRIPTED}/ask",
            {"question": "List the alerts."},
            {"Origin": origin},
        )
        assert status == 403, origin
    assert request(server, "GET", "/api/runs", headers={"Origin": "http://evil.example"})[0] == 403


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_no_write_methods_exist(server, method):
    status, headers, _ = request(server, method, "/api/runs")
    assert status == 405 and "POST" in headers["Allow"] and "PUT" not in headers["Allow"]


def test_the_server_is_bound_to_loopback_and_refuses_other_addresses(server, runs):
    assert server.server_address[0] == "127.0.0.1"
    api = DashboardApi(RunRegistry(runs))
    for host in ("0.0.0.0", "192.168.1.5", ""):
        with pytest.raises(ValueError, match="loopback only"):
            make_server(api, host, 0)


def test_serving_every_route_leaves_the_databases_unchanged(server, runs):
    before = {r.name: hashlib.sha256(r.db.read_bytes()).hexdigest() for r in runs}
    for name in (SCRIPTED, VIDEO):
        for suffix in ("overview", "events", "entities", "alerts", "observations", "summary"):
            assert request(server, "GET", f"/api/runs/{name}/{suffix}")[0] == 200
        post_json(server, f"/api/runs/{name}/ask", {"question": "Summarize the run."})
    after = {r.name: hashlib.sha256(r.db.read_bytes()).hexdigest() for r in runs}
    assert after == before


# ---- the page itself holds no detection logic ------------------------------------------
def page_sources() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in STATIC_DIR.iterdir() if p.is_file()}


def test_javascript_contains_no_rule_ids_thresholds_or_detection_logic():
    rule_ids = list(load_rules_config().model_dump())
    banned = rule_ids + [
        "threshold",
        "loiter",
        "aggregat",
        "linker",
        "max_missed",
        "min_appearances",
    ]
    for name, text in page_sources().items():
        lowered = text.lower()
        for word in banned:
            assert word not in lowered, f"{name} mentions '{word}'"


def test_the_page_makes_no_security_decisions_of_its_own():
    js = page_sources()["app.js"]
    # severity and status are only ever displayed or used to colour; never compared to decide
    assert not re.search(r"severity\s*(===|!==|==|!=|>|<)", js)
    assert not re.search(r"(alert|event)s?\.(filter|some|every)\([^)]*(severity|rule)", js)
    assert "fetch(" in js and "/api/" in js


def test_the_page_loads_nothing_from_outside_localhost():
    for name, text in page_sources().items():
        assert not re.search(
            r"(?i)(https?:)?//(?!www\.w3\.org/2000/svg)[a-z0-9.-]+\.[a-z]{2,}", text
        ), name
    html = page_sources()["index.html"]
    assert 'src="/static/app.js"' in html and 'href="/static/app.css"' in html
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)=\"(https?:)?//", html)
    assert not re.search(r"\son[a-z]+=", html) and "<style" not in html  # no inline handlers/styles


def test_the_page_has_the_five_required_panels_and_the_pipeline():
    html = page_sources()["index.html"]
    for panel in (
        "panel-overview",
        "panel-timeline",
        "panel-alerts",
        "panel-analyst",
        "panel-video",
    ):
        assert f'id="{panel}"' in html
    assert 'id="pipeline"' in html and 'id="kind-banner"' in html
