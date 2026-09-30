"""Opt-in public mode: one configured hostname is accepted, everything else stays refused.

The dashboard is loopback-only with a strict Host and Origin check. Behind a hosting proxy the
browser's Host is the public hostname, so public mode adds exactly that hostname (and its https
origin) and nothing more. Without `--public-host` nothing changes; the existing suite in
tests/test_web_server.py still pins that default and is untouched.
"""

import http.client
import json
import socket
import threading
import time

import pytest

from drone_agent.web import __main__ as web_main
from drone_agent.web.api import DashboardApi
from drone_agent.web.bundle import BundleError, build_demo_runs
from drone_agent.web.registry import RunRegistry
from drone_agent.web.server import PUBLIC_BIND_HOST, make_server, normalise_public_host

SCRIPTED = "security_events"
PUBLIC = "dashboard.example.com"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    try:
        return build_demo_runs(tmp_path_factory.mktemp("dashpub"))
    except BundleError as exc:
        if "MJPG" in str(exc):
            pytest.skip("this OpenCV build cannot write MJPG video")
        raise


def serve(runs, public_host=None, bind="127.0.0.1"):
    httpd = make_server(
        DashboardApi(RunRegistry(runs)), bind, 0, quiet=True, public_host=public_host
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture(scope="module")
def public(runs):
    httpd = serve(runs, PUBLIC)
    yield httpd
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def local(runs):
    httpd = serve(runs)
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def request(server, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read()
    finally:
        conn.close()


def ask(server, headers=None, question="List the alerts."):
    return request(
        server,
        "POST",
        f"/api/runs/{SCRIPTED}/ask",
        body=json.dumps({"question": question}),
        headers={"Content-Type": "application/json", **(headers or {})},
    )


def get(server, path="/api/runs", host=None, origin=None):
    headers = {}
    if host is not None:
        headers["Host"] = host
    if origin is not None:
        headers["Origin"] = origin
    return request(server, "GET", path, headers=headers)[0]


# ---- default local mode is unchanged --------
def test_without_a_public_host_the_server_is_loopback_only_and_strict(local, runs):
    assert local.RequestHandlerClass.public_host is None
    assert local.server_address[0] == "127.0.0.1"
    port = local.server_port
    assert get(local, host=f"127.0.0.1:{port}") == 200
    for host in (PUBLIC, f"{PUBLIC}:{port}", "evil.example"):
        assert get(local, host=host) == 403, host
    for origin in (f"https://{PUBLIC}", f"http://{PUBLIC}"):
        assert ask(local, {"Origin": origin})[0] == 403, origin
    api = DashboardApi(RunRegistry(runs))
    with pytest.raises(ValueError, match="loopback only"):
        make_server(api, PUBLIC_BIND_HOST, 0)  # 0.0.0.0 needs a public hostname


# ---- public mode --------
def test_public_mode_accepts_the_configured_hostname(public):
    port = public.server_port
    for host in (PUBLIC, f"{PUBLIC}:{port}", PUBLIC.upper()):  # as a proxy sends it, or direct
        assert get(public, host=host) == 200, host
        assert get(public, "/", host=host) == 200, host  # the page itself
    assert get(public, host=f"127.0.0.1:{port}") == 200  # local access still works
    assert public.RequestHandlerClass.public_host == PUBLIC


def test_public_mode_accepts_the_configured_https_origin_for_ask(public):
    status, body = ask(public, {"Host": PUBLIC, "Origin": f"https://{PUBLIC}"})
    answer = json.loads(body)
    assert status == 200 and answer["mode"] == "template" and answer["citations"]
    assert get(public, host=PUBLIC, origin=f"https://{PUBLIC}") == 200  # GETs too
    assert ask(public, {"Host": PUBLIC})[0] == 200  # no Origin header (not a browser): as before


@pytest.mark.parametrize(
    "host",
    [
        "evil.example",
        f"{PUBLIC}.evil.example",
        f"evil.{PUBLIC}",
        f"{PUBLIC}:1",  # the wrong port
        "",
        f"{PUBLIC}@evil.example",
    ],
)
def test_an_unrelated_host_is_still_refused_in_public_mode(public, host):
    assert get(public, host=host) == 403, host
    assert ask(public, {"Host": host})[0] == 403, host


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.example",
        "https://evil.example",
        f"http://{PUBLIC}",  # only the https origin of the configured host
        f"https://{PUBLIC}.evil.example",
        f"https://evil.{PUBLIC}",
        f"https://{PUBLIC}:444",
        "null",
    ],
)
def test_an_unrelated_origin_is_still_refused_in_public_mode(public, origin):
    assert ask(public, {"Host": PUBLIC, "Origin": origin})[0] == 403, origin
    assert get(public, host=PUBLIC, origin=origin) == 403, origin


def test_ask_stays_same_origin_protected_in_public_mode(public):
    # the right Host with a foreign Origin, the right Origin with a foreign Host: both refused
    assert ask(public, {"Host": PUBLIC, "Origin": "https://evil.example"})[0] == 403
    assert ask(public, {"Host": "evil.example", "Origin": f"https://{PUBLIC}"})[0] == 403
    # a refused request never reaches the analyst: the same question is answered when allowed
    assert ask(public, {"Host": PUBLIC, "Origin": f"https://{PUBLIC}"})[0] == 200


def test_public_mode_can_bind_all_interfaces_and_only_with_a_hostname(runs):
    httpd = serve(runs, PUBLIC, bind=PUBLIC_BIND_HOST)
    try:
        assert httpd.server_address[0] == PUBLIC_BIND_HOST
        assert get(httpd, host=PUBLIC) == 200 and get(httpd, host="evil.example") == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
    api = DashboardApi(RunRegistry(runs))
    for host in ("192.168.1.5", ""):  # public mode does not open arbitrary binds
        with pytest.raises(ValueError, match="loopback only"):
            make_server(api, host, 0, public_host=PUBLIC)


@pytest.mark.parametrize(
    "bad",
    [
        "", " ", "https://app.onrender.com", "app.onrender.com:8080", "app.onrender.com/x",
        "*.onrender.com", "a b.com", "app..com", "-app.com", "app.com-", "app_one.com",
        " app.com", "app.com\n", "user@app.com",
    ],
)  # fmt: skip
def test_a_public_host_must_be_a_bare_hostname(bad, runs):
    with pytest.raises(ValueError):
        normalise_public_host(bad)
    with pytest.raises(ValueError):
        make_server(DashboardApi(RunRegistry(runs)), "127.0.0.1", 0, public_host=bad)


def test_a_public_host_is_normalised_to_lower_case():
    assert normalise_public_host("My-App.OnRender.com") == "my-app.onrender.com"
    assert normalise_public_host("203.0.113.7") == "203.0.113.7"


def test_public_mode_needs_no_credential_and_reads_no_dotenv(public, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("the dashboard must not load .env")

    monkeypatch.setattr("dotenv.load_dotenv", forbidden)
    status, body = ask(public, {"Host": PUBLIC, "Origin": f"https://{PUBLIC}"})
    assert status == 200 and json.loads(body)["mode"] == "template"  # the offline analyst
    from pathlib import Path

    web = Path(web_main.__file__).parent
    assert not any("GEMINI" in f.read_text(encoding="utf-8") for f in web.glob("*.py"))


# ---- the command line: the exact Render invocation --------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_the_render_command_works_through_the_cli_with_port_from_the_environment(
    tmp_path, monkeypatch
):
    """The Render start command: --demo --host 0.0.0.0 --port $PORT --public-host $HOSTNAME."""
    port = free_port()
    monkeypatch.setenv("PORT", str(port))
    monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "My-Dashboard.onrender.com")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    import os

    argv = [
        "--demo", "--out-dir", str(tmp_path / "out"),
        "--host", "0.0.0.0",
        "--port", os.environ["PORT"],
        "--public-host", os.environ["RENDER_EXTERNAL_HOSTNAME"],
    ]  # fmt: skip
    started = []
    real_make_server = web_main.make_server

    def capture(api, host, port, **kwargs):
        server = real_make_server(api, host, port, quiet=True, **kwargs)
        started.append((server, host, port, kwargs))
        return server

    monkeypatch.setattr(web_main, "make_server", capture)
    from rich.console import Console

    out = Console(record=True, width=160)
    result = []
    thread = threading.Thread(target=lambda: result.append(web_main.main(argv, out)), daemon=True)
    thread.start()
    for _ in range(100):
        if started:
            break
        time.sleep(0.1)
    assert started, out.export_text()
    server, host, bound_port, kwargs = started[0]
    try:
        assert (host, bound_port) == (PUBLIC_BIND_HOST, port)  # $PORT reached the socket as an int
        assert kwargs == {"public_host": "my-dashboard.onrender.com"}  # normalised
        time.sleep(0.3)  # let serve_forever begin
        assert get(server, host="my-dashboard.onrender.com") == 200  # the proxy's Host: no port
        assert get(server, "/", host="my-dashboard.onrender.com") == 200
        assert get(server, host="evil.example") == 403
        status, _ = ask(
            server,
            {"Host": "my-dashboard.onrender.com", "Origin": "https://my-dashboard.onrender.com"},
        )
        assert status == 200
    finally:
        server.shutdown()
        thread.join(timeout=10)
    assert result == [0]
    text = out.export_text()
    assert "https://my-dashboard.onrender.com/" in text and "no authentication" in text


def test_the_local_command_is_unchanged(tmp_path, monkeypatch):
    calls = []

    class FakeServer:
        server_port = 8765

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    def fake(*args, **kwargs):
        calls.append((args[1:], kwargs))
        return FakeServer()

    monkeypatch.setattr(web_main, "make_server", fake)
    from rich.console import Console

    out = Console(record=True, width=160)
    assert web_main.main(["--demo", "--out-dir", str(tmp_path / "out")], out) == 0
    assert calls == [(("127.0.0.1", 8765), {})]  # no public_host, exactly as before
    assert "http://127.0.0.1:8765/" in out.export_text()


def test_bad_public_options_are_refused_before_anything_is_built(tmp_path, capsys, monkeypatch):
    from rich.console import Console

    def never(*args, **kwargs):
        raise AssertionError("a server must not be started for an invalid public host")

    monkeypatch.setattr(web_main, "make_server", never)  # a failure must fail, not serve forever

    out = Console(record=True, width=160)
    missing_dir = str(tmp_path / "never")
    assert (
        web_main.main(["--demo", "--out-dir", missing_dir, "--public-host", "https://x.com"], out)
        == 2
    )
    assert "--public-host" in out.export_text()
    assert not (tmp_path / "never").exists()  # refused before the demo was built
    with pytest.raises(SystemExit):  # 0.0.0.0 alone is still refused by the parser
        web_main.main(["--demo", "--host", "0.0.0.0"])
    assert "--public-host" in capsys.readouterr().err
