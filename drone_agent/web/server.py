"""Localhost HTTP server for the dashboard (stdlib only).

Deliberately small and defensive: it binds to loopback only, serves a FIXED whitelist of static
files (no directory access, no path joining from user input), validates every route with a strict
pattern, checks the Host and Origin headers, caps request bodies and sends a restrictive
Content-Security-Policy. There is no authentication because it is a local, read-only demo.
"""

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from drone_agent.web.api import ApiError, DashboardApi
from drone_agent.web.frames import DEFAULT_WIDTH

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
MAX_BODY_BYTES = 8 * 1024
DRAIN_LIMIT = (
    64 * 1024
)  # most of a rejected request body that is read (and dropped) before replying
REQUEST_TIMEOUT_S = 15  # a client that stalls mid-request is cut off
MAX_QUERY_CHARS = 512
# the UI is a plain HTML/CSS/JS folder at the repository root: <repo>/frontend
STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend"
STATIC_FILES = {  # the only files ever served, by exact URL path
    "/": ("index.html", "text/html; charset=utf-8"),
    "/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/static/app.css": ("app.css", "text/css; charset=utf-8"),
    "/static/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)

_RUN = r"([A-Za-z0-9][A-Za-z0-9_-]{0,39})"
_ID = r"([A-Za-z0-9_.:-]{1,80})"
_GET_ROUTES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), handler)
    for pattern, handler in (
        (r"^/api/runs$", "runs"),
        (r"^/api/rules$", "rules"),
        (rf"^/api/runs/{_RUN}/overview$", "overview"),
        (rf"^/api/runs/{_RUN}/events$", "events"),
        (rf"^/api/runs/{_RUN}/events/{_ID}$", "event_detail"),
        (rf"^/api/runs/{_RUN}/entities$", "entities"),
        (rf"^/api/runs/{_RUN}/alerts$", "alerts"),
        (rf"^/api/runs/{_RUN}/alerts/{_ID}$", "alert_detail"),
        (rf"^/api/runs/{_RUN}/observations$", "observations"),
        (rf"^/api/runs/{_RUN}/summary$", "summary"),
        (rf"^/api/runs/{_RUN}/frames/(\d{{1,9}})\.jpg$", "frame"),
    )
)
_ASK_ROUTE = re.compile(rf"^/api/runs/{_RUN}/ask$")


class DashboardHandler(BaseHTTPRequestHandler):
    api: DashboardApi
    quiet = False
    server_version = "DroneDashboard/1"
    timeout = REQUEST_TIMEOUT_S  # socket timeout for reading the request

    # ---- responses ---------------------------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str, *, no_store: bool = True) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        if no_store:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("ascii")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    # ---- request checks ----------------------------------------------------------------------
    def _allowed_hosts(self) -> set[str]:
        port = self.server.server_port
        return {f"{h}:{port}" for h in ("127.0.0.1", "localhost", "[::1]")}

    def _headers_ok(self) -> bool:
        if self.headers.get("Host", "") not in self._allowed_hosts():
            self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in self._allowed_hosts()}:
            self._error(HTTPStatus.FORBIDDEN, "unexpected Origin")
            return False
        return True

    # ---- GET ---------------------------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._headers_ok():
            return
        parts = urlsplit(self.path)
        if parts.path in STATIC_FILES:
            self._static(parts.path)
            return
        if parts.path == "/favicon.ico":  # browsers probe this; the page links the SVG instead
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if len(parts.query) > MAX_QUERY_CHARS:
            self._error(HTTPStatus.REQUEST_URI_TOO_LONG, "query string too long")
            return
        try:
            query = {k: v[0] for k, v in parse_qs(parts.query, max_num_fields=8).items()}
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "malformed query string")
            return
        for pattern, handler in _GET_ROUTES:
            match = pattern.match(parts.path)
            if match:
                try:
                    self._dispatch(handler, match.groups(), query)
                except ApiError as exc:
                    self._error(exc.status, exc.message)
                except Exception:  # never leak a traceback to the browser
                    self.log_error("unhandled error for %s", parts.path)
                    self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")
                return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    do_HEAD = do_GET

    def _static(self, url_path: str) -> None:
        filename, content_type = STATIC_FILES[url_path]
        try:
            body = (STATIC_DIR / filename).read_bytes()
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "static file missing")
            return
        self._send(HTTPStatus.OK, body, content_type)

    def _dispatch(self, handler: str, groups: tuple[str, ...], query: dict[str, str]) -> None:
        api = self.api
        if handler == "runs":
            self._json(200, api.runs())
        elif handler == "rules":
            self._json(200, api.rules())
        elif handler == "overview":
            self._json(200, api.overview(groups[0]))
        elif handler == "events":
            self._json(200, api.events(groups[0], query.get("zone"), query.get("type")))
        elif handler == "event_detail":
            self._json(200, api.event_detail(groups[0], groups[1]))
        elif handler == "entities":
            self._json(200, api.entities(groups[0]))
        elif handler == "alerts":
            self._json(200, api.alerts(groups[0]))
        elif handler == "alert_detail":
            self._json(200, api.alert_detail(groups[0], groups[1]))
        elif handler == "observations":
            try:
                limit = int(query.get("limit", "200"))
            except ValueError:
                raise ApiError(400, "limit must be an integer") from None
            self._json(200, api.observations(groups[0], query.get("zone"), query.get("q"), limit))
        elif handler == "summary":
            self._json(200, api.summary(groups[0]))
        elif handler == "frame":
            try:
                width = int(query.get("w", str(DEFAULT_WIDTH)))
            except ValueError:
                raise ApiError(400, "w must be an integer") from None
            data = api.frame_jpeg(groups[0], int(groups[1]), width)
            self._send(HTTPStatus.OK, data, "image/jpeg", no_store=False)

    # ---- POST --------------------------------------------------------------------------------
    def _declared_length(self) -> int | None:
        try:
            return int(self.headers.get("Content-Length", ""))
        except ValueError:
            return None

    def _read_body(self) -> bytes:
        """Read what the client declared (at most DRAIN_LIMIT) BEFORE replying, whatever the reply.

        Closing a socket that still holds unread request data makes Windows reset the connection,
        and the client then loses the reply (a 404, 415, 403 or 413 would arrive as an abort).
        """
        length = self._declared_length()
        if length is None or length <= 0:
            return b""
        try:
            return self.rfile.read(min(length, DRAIN_LIMIT))
        except TimeoutError:  # the client promised a body and never sent it
            self.close_connection = True
            return b""

    def do_POST(self) -> None:
        length = self._declared_length()
        raw = self._read_body()
        if not self._headers_ok():
            return
        parts = urlsplit(self.path)
        match = _ASK_ROUTE.match(parts.path)
        if not match:
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return
        if "application/json" not in self.headers.get("Content-Type", ""):
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "send application/json")
            return
        if length is None:
            self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body must be <= {MAX_BODY_BYTES} bytes"
            )
            return
        try:
            body = json.loads(raw or b"null")
        except (ValueError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "body must be valid JSON")
            return
        if not isinstance(body, dict) or not isinstance(body.get("question"), str):
            self._error(
                HTTPStatus.BAD_REQUEST, 'body must be {"question": "...", "session_id"?: "..."}'
            )
            return
        session_id = body.get("session_id")
        if session_id is not None and not (isinstance(session_id, str) and len(session_id) <= 64):
            self._error(HTTPStatus.BAD_REQUEST, "session_id must be a short string")
            return
        try:
            self._json(200, self.api.ask(match.group(1), body["question"], session_id))
        except ApiError as exc:
            self._error(exc.status, exc.message)
        except Exception:
            self.log_error("unhandled error for %s", parts.path)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")

    def _method_not_allowed(self) -> None:
        self._read_body()
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD, POST")
        self.send_header("Content-Length", "0")
        self.end_headers()

    do_PUT = do_DELETE = do_PATCH = do_OPTIONS = _method_not_allowed

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        if not self.quiet:
            super().log_message(format, *args)


def make_server(
    api: DashboardApi, host: str = "127.0.0.1", port: int = 8765, quiet: bool = False
) -> ThreadingHTTPServer:
    """A ready (bound, not yet serving) server. Non-loopback hosts are refused."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"the dashboard binds to loopback only ({', '.join(LOOPBACK_HOSTS)})")
    handler = type("Handler", (DashboardHandler,), {"api": api, "quiet": quiet})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server
