"""A very small Chrome DevTools Protocol driver for the dashboard's browser tests.

It launches a headless Chrome or Edge that is already installed (it never downloads one), talks to
it over the `websockets` package (already installed as a dependency of google-genai) and offers just
what the UI tests need: navigate, evaluate, wait, press keys, screenshot, and collect every console
error, script exception and failed or error-status network request. No Node, no Selenium, no
Playwright. `find_browser()` returns None when no browser is installed, and the tests then skip.
"""

import base64
import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

try:
    from websockets.sync.client import connect
except ImportError:  # pragma: no cover - websockets ships with google-genai
    connect = None

_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
_NAMES = ("google-chrome", "chromium", "chromium-browser", "chrome", "msedge")


def find_browser() -> str | None:
    if connect is None:
        return None
    for path in _CANDIDATES:
        if os.path.isfile(path):
            return path
    for name in _NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


class CdpError(RuntimeError):
    pass


class Browser:
    def __init__(self, exe: str, extra_args: tuple[str, ...] = ()) -> None:
        self._dir = tempfile.mkdtemp(prefix="cdp-profile-")
        self._proc = subprocess.Popen(
            [
                exe,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--hide-scrollbars",
                "--no-default-browser-check",
                "--remote-debugging-port=0",
                f"--user-data-dir={self._dir}",
                *extra_args,
                "about:blank",
            ],  # fmt: skip
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        port_file = Path(self._dir) / "DevToolsActivePort"
        deadline = time.time() + 30
        while not port_file.exists():
            if time.time() > deadline or self._proc.poll() is not None:
                self.close()
                raise CdpError("the browser did not start")
            time.sleep(0.1)
        port, path = port_file.read_text().split("\n")[:2]
        self._ws = connect(f"ws://127.0.0.1:{port}{path}", max_size=None, open_timeout=30)
        self._next = 0
        self.events: list[dict] = []

    def send(self, method: str, params: dict | None = None, session: str | None = None) -> dict:
        self._next += 1
        message = {"id": self._next, "method": method, "params": params or {}}
        if session:
            message["sessionId"] = session
        self._ws.send(json.dumps(message))
        while True:
            data = json.loads(self._ws.recv(timeout=60))
            if data.get("id") == self._next:
                if "error" in data:
                    raise CdpError(f"{method}: {data['error']}")
                return data.get("result", {})
            if "method" in data:
                self.events.append(data)

    def drain(self, seconds: float = 0.05) -> None:
        """Collect pending events without sending anything."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                data = json.loads(self._ws.recv(timeout=max(0.01, deadline - time.time())))
            except TimeoutError:
                return
            if "method" in data:
                self.events.append(data)

    def new_page(self) -> "Page":
        target = self.send("Target.createTarget", {"url": "about:blank"})["targetId"]
        session = self.send("Target.attachToTarget", {"targetId": target, "flatten": True})[
            "sessionId"
        ]
        page = Page(self, session, target)
        page.enable()
        return page

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        shutil.rmtree(self._dir, ignore_errors=True)


class Page:
    def __init__(self, browser: Browser, session: str, target: str) -> None:
        self.b, self.session, self.target = browser, session, target
        self._seen = 0
        self._viewport = (1280, 720)
        self.expected_failures: list[str] = []  # URL fragments whose failure a test provokes

    def cmd(self, method: str, params: dict | None = None) -> dict:
        return self.b.send(method, params, self.session)

    def enable(self) -> None:
        for domain in ("Page", "Runtime", "Log", "Network"):
            self.cmd(f"{domain}.enable")

    def _metrics(self, width: int, height: int) -> None:
        params = {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": width < 600}
        self.cmd("Emulation.setDeviceMetricsOverride", params)

    def viewport(self, width: int, height: int) -> None:
        self._viewport = (width, height)
        self._metrics(width, height)

    def init_script(self, source: str) -> None:
        self.cmd("Page.addScriptToEvaluateOnNewDocument", {"source": source})

    def block_urls(self, patterns: list[str]) -> None:
        self.cmd("Network.setBlockedURLs", {"urls": patterns})

    def goto(self, url: str) -> None:
        self.problems()  # drop anything from the previous page
        if (
            url != "about:blank"
        ):  # a URL that differs only in its #fragment would not reload the page
            self.goto("about:blank")
        self.cmd("Page.navigate", {"url": url})
        base = json.dumps(url.split("#")[0])
        self.wait_js(f"document.readyState === 'complete' && location.href.startsWith({base})", 30)

    def js(self, expression: str):
        params = {"expression": expression, "returnByValue": True, "awaitPromise": True}
        result = self.cmd("Runtime.evaluate", params)
        if "exceptionDetails" in result:
            details = result["exceptionDetails"]
            description = details.get("exception", {}).get("description", "")
            raise CdpError(f"{expression[:80]}: {details.get('text')} {description}")
        return result["result"].get("value")

    def wait_js(self, expression: str, timeout: float = 20):
        deadline = time.time() + timeout
        while True:
            value = self.js(
                f"(() => {{ try {{ return !!({expression}); }} catch (e) {{ return false; }} }})()"
            )
            if value:
                return value
            if time.time() > deadline:
                raise CdpError(f"timed out waiting for: {expression[:160]}")
            time.sleep(0.1)

    def press(self, key: str) -> None:
        codes = {
            "Tab": 9,
            "Enter": 13,
            " ": 32,
            "ArrowLeft": 37,
            "ArrowRight": 39,
            "Home": 36,
            "End": 35,
        }
        text = {"Enter": "\r", " ": " "}.get(key)
        base = {
            "key": key,
            "code": "Space" if key == " " else key,
            "windowsVirtualKeyCode": codes[key],
        }
        self.cmd(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", **base, **({"text": text} if text else {})},
        )
        self.cmd("Input.dispatchKeyEvent", {"type": "keyUp", **base})

    def screenshot(self, path: str | Path, full_page: bool = False) -> None:
        width, height = self._viewport
        if full_page:  # grow the viewport to the whole page: no sticky-header artefacts
            self.js("window.scrollTo(0, 0)")
            height = int(self.cmd("Page.getLayoutMetrics")["cssContentSize"]["height"])
            self._metrics(width, height)
            time.sleep(0.4)
        try:
            Path(path).write_bytes(
                base64.b64decode(self.cmd("Page.captureScreenshot", {"format": "png"})["data"])
            )
        finally:
            if full_page:
                self.viewport(width, self._viewport[1])

    def problems(self) -> list[str]:
        """Console errors, exceptions and failed requests since the last call (then cleared)."""
        self.b.drain()
        found, kept = [], []
        urls: dict[str, str] = {}
        for ev in self.b.events:
            if ev.get("sessionId") != self.session:
                kept.append(ev)
                continue
            method, p = ev["method"], ev.get("params", {})
            if method == "Network.requestWillBeSent":
                urls[p["requestId"]] = p["request"]["url"]
            elif method == "Runtime.exceptionThrown":
                details = p["exceptionDetails"]
                description = details.get("exception", {}).get("description") or details.get("text")
                found.append(f"exception: {description}")
            elif method == "Runtime.consoleAPICalled" and p["type"] in ("error", "assert"):
                said = " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])
                found.append(f"console.{p['type']}: {said}")
            elif method == "Log.entryAdded" and p["entry"]["level"] == "error":
                found.append(f"log: {p['entry']['text']} {p['entry'].get('url', '')}")
            elif method == "Network.loadingFailed":
                url = urls.get(p["requestId"], p["requestId"])
                if not p.get("canceled"):
                    found.append(f"request failed: {url}: {p.get('errorText')}")
            elif method == "Network.responseReceived" and p["response"]["status"] >= 400:
                found.append(f"HTTP {p['response']['status']}: {p['response']['url']}")
        self.b.events = kept
        return [f for f in found if not any(x in f for x in self.expected_failures)]
