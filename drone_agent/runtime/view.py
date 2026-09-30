"""Terminal views of a running patrol: a live dashboard and a plain log.

Both consume `RunEvent`s and label everything honestly: telemetry is SIMULATED, scripted frames are
SCRIPTED, replayed model output is REPLAYED, and a carried observation says CARRIED. Nothing here
computes anything; it only shows what the engine announced.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from drone_agent.runtime.events import RunEvent, RunEventKind

_SEVERITY_STYLE = {
    "low": "cyan",
    "medium": "yellow",
    "high": "bold red",
    "critical": "bold white on red",
}


@dataclass(frozen=True)
class Glyphs:
    """Symbols the views draw with; ASCII where the console cannot encode Unicode."""

    child: str
    up: str
    done: str
    status: dict[str, str]
    full: str
    empty: str
    spark: str
    arrow: str
    sep: str


UNICODE = Glyphs(
    "↳", "▲", "✓", {"open": "●", "escalated": "▲", "acknowledged": "◆", "resolved": "✓"},
    "█", "░", "▁▂▃▄▅▆▇█", "→", "·",
)  # fmt: skip
ASCII = Glyphs(
    "|_", "^", "ok", {"open": "*", "escalated": "^", "acknowledged": "o", "resolved": "v"},
    "#", "-", "_.-~=+*#", "->", "|",
)  # fmt: skip


def glyphs_for(console: Console) -> Glyphs:
    """Unicode symbols on UTF terminals, ASCII on legacy code pages (e.g. cp1252)."""
    return UNICODE if "utf" in (console.encoding or "").lower() else ASCII


def _clock(ts: datetime) -> str:
    return ts.strftime("%H:%M:%S")


def sparkline(values: list[float], g: Glyphs, low: float = 0.0, high: float = 100.0) -> str:
    if not values:
        return ""
    span = max(high - low, 1e-9)
    top = len(g.spark) - 1
    return "".join(g.spark[min(top, max(0, int((v - low) / span * (top + 0.999))))] for v in values)


def bar(value: float | None, g: Glyphs, width: int = 16) -> str:
    if value is None:
        return "n/a".ljust(width)
    filled = round(max(0.0, min(100.0, value)) / 100 * width)
    return g.full * filled + g.empty * (width - filled)


def describe(event: RunEvent, g: Glyphs = UNICODE) -> tuple[str, str] | None:
    """(style, one-line text) for events worth a line in a timeline or log; None = skip."""
    d, t = event.data, _clock(event.at)
    kind = event.kind
    if kind is RunEventKind.STARTED:
        return "bold", f"{t}  PATROL STARTED  {d['scenario']}: {d['description']}"
    if kind is RunEventKind.FRAME:
        mode = "analysed" if d["analyzed"] else "skipped "
        why = f"{d['reason']}" + (f" ({d['detail']})" if d.get("detail") else "")
        note = "  [perception FAILED]" if d.get("dropped") else ""
        style = "green" if d["analyzed"] else "dim"
        return style, f"{t}  FRAME {d['frame_id']:<3} {d['zone']:<15} {mode} {why}{note}"
    if kind is RunEventKind.OBSERVATION:
        objs = ", ".join(
            f"{o['type']}" + (f" ({o['activity']})" if o.get("activity") else "")
            for o in d["objects"]
        )
        seen = f"[{event.provenance}] {d['description']} {g.arrow} {objs or 'nothing'}"
        return "white", f"{t}    {g.child} {seen}"
    if kind is RunEventKind.EVENT:
        who = f"{d['object_type']} in {d['zone']}"
        if d["change"] == "opened":
            return "blue", f"{t}  EVENT {d['event_id']} opened: {who}"
        if d["change"] == "closed":
            return "blue", (
                f"{t}  EVENT {d['event_id']} closed ({d['close_reason']}): {who}, "
                f"{d['duration_s']:.0f}s, {d['observations']} obs"
            )
        return None
    if kind is RunEventKind.ALERT:
        if d["transition"] == "evidence":
            return None
        mark = {
            "raised": "RAISED",
            "escalated": f"ESCALATED {g.up}",
            "resolved": f"RESOLVED {g.done}",
        }.get(d["transition"], d["transition"].upper())
        return _SEVERITY_STYLE.get(d["severity"], "bold"), (
            f"{t}  ALERT {mark} [{d['severity'].upper()}] {d['title']} ({d['alert_id']})"
        )
    if kind is RunEventKind.TELEMETRY and not d["delivered"]:
        return "dim yellow", f"{t}  TELEMETRY  no sample delivered (link?)"
    if kind is RunEventKind.FINISHED:
        return "bold", f"{t}  PATROL FINISHED"
    return None


class LogView:
    """One line per notable event, for terminals without live rendering and for recordings."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.g = glyphs_for(console)

    def __call__(self, event: RunEvent) -> None:
        line = describe(event, self.g)
        if line is not None:
            self.console.print(Text(line[1], style=line[0]))


class LiveView:
    """A dashboard: drone status, timeline, alerts and counters, redrawn as events arrive."""

    def __init__(self, console: Console, speed: float, expected_alerts: int = 0) -> None:
        self.console = console
        self.g = glyphs_for(console)
        self.speed = speed
        self.expected_alerts = expected_alerts
        self.scenario = ""
        self.now: datetime | None = None
        self.offset = 0.0
        self.duration = 0.0
        self.timeline: deque[tuple[str, str]] = deque(maxlen=60)
        self.alerts: dict[str, dict] = {}
        self.telemetry: dict = {}
        self.battery: deque[float] = deque(maxlen=40)
        self.signal: deque[float] = deque(maxlen=40)
        self.frames = self.analyzed = self.carried = self.events = self.observations = 0
        self.missed_telemetry = 0
        self.provenance: dict[str, int] = {}
        self.finished = False

    def __call__(self, event: RunEvent) -> None:
        self.now, self.offset = event.at, event.offset_s
        d = event.data
        if event.kind is RunEventKind.STARTED:
            self.scenario, self.duration = d["scenario"], float(d["duration_s"])
        elif event.kind is RunEventKind.TELEMETRY:
            if d["delivered"]:
                self.telemetry = d
                self.battery.append(d["battery_percent"])
                if d.get("signal_percent") is not None:
                    self.signal.append(d["signal_percent"])
            else:
                self.missed_telemetry += 1
        elif event.kind is RunEventKind.FRAME:
            self.frames += 1
            self.analyzed += 1 if d["analyzed"] else 0
        elif event.kind is RunEventKind.OBSERVATION:
            self.observations += 1
            self.carried += 1 if d["carried"] else 0
            self.provenance[event.provenance] = self.provenance.get(event.provenance, 0) + 1
        elif event.kind is RunEventKind.EVENT:
            if d["change"] == "opened":
                self.events += 1
        elif event.kind is RunEventKind.ALERT:
            self.alerts[d["alert_id"]] = {**d, "at": event.at}
        elif event.kind is RunEventKind.FINISHED:
            self.finished = True
        line = describe(event, self.g)
        if line is not None:
            self.timeline.append(line)

    # ---- rendering -------------------------------------------------------------------------
    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="body"),
            Layout(self._telemetry_panel(), name="telemetry", size=6),
        )
        layout["body"].split_row(
            Layout(self._timeline_panel(), name="timeline", ratio=5),
            Layout(name="right", ratio=4),
        )
        layout["right"].split_column(
            Layout(self._alerts_panel(), name="alerts", ratio=3),
            Layout(self._counters_panel(), name="counters", ratio=2),
        )
        return layout

    def _header(self) -> Panel:
        clock = _clock(self.now) if self.now else "--:--:--"
        pace = f"{self.speed:g}x" if self.speed else "unpaced"
        progress = f"{self.offset:.0f}/{self.duration:.0f}s" if self.duration else ""
        state = "FINISHED" if self.finished else "PATROLLING"
        text = Text.assemble(
            ("DRONE SECURITY PATROL  ", "bold"),
            (f"{self.scenario}  ", "cyan"),
            (f"sim {clock}  {progress}  {pace}  ", ""),
            (state, "bold green" if not self.finished else "bold"),
        )
        text.append("\n")
        text.append(
            f"telemetry SIMULATED {self.g.sep} frames SCRIPTED {self.g.sep} "
            "rules and alerts computed live",
            style="dim",
        )
        return Panel(text, padding=(0, 1))

    def _timeline_panel(self) -> Panel:
        height = max(6, self.console.size.height - 14)
        lines = list(self.timeline)[-height:]
        body = Group(
            *(Text(text, style=style, no_wrap=True, overflow="ellipsis") for style, text in lines)
        )
        return Panel(body, title="Live timeline", border_style="blue")

    def _alerts_panel(self) -> Panel:
        table = Table(show_header=True, header_style="bold", expand=True, box=None, pad_edge=False)
        table.add_column("", width=2)
        table.add_column("Sev")
        table.add_column("Alert", overflow="ellipsis", no_wrap=True)
        rows = sorted(
            self.alerts.values(),
            key=lambda a: (a["status"] == "resolved", -_rank(a["severity"]), a["at"]),
        )
        for a in rows[:12]:
            table.add_row(
                self.g.status.get(a["status"], "?"),
                Text(a["severity"].upper(), style=_SEVERITY_STYLE.get(a["severity"], "")),
                Text(f"{a['title']}", style="dim" if a["status"] == "resolved" else ""),
            )
        if not self.alerts:
            table.add_row("", "", Text("no alerts", style="dim"))
        active = sum(1 for a in self.alerts.values() if a["status"] != "resolved")
        crit = sum(
            1
            for a in self.alerts.values()
            if a["severity"] == "critical" and a["status"] != "resolved"
        )
        return Panel(
            table,
            title=f"Alerts  {active} active" + (f", {crit} critical" if crit else ""),
            border_style="red" if crit else "yellow" if active else "green",
        )

    def _counters_panel(self) -> Panel:
        avoided = 100.0 * (self.frames - self.analyzed) / self.frames if self.frames else 0.0
        prov = "  ".join(f"{k} {v}" for k, v in sorted(self.provenance.items())) or "-"
        lines = [
            f"frames {self.frames}   model calls {self.analyzed}   avoided {avoided:.0f}%",
            f"observations {self.observations} ({self.carried} carried)   events {self.events}",
            f"provenance: {prov}",
        ]
        return Panel("\n".join(lines), title="Perception & sampling", border_style="magenta")

    def _telemetry_panel(self) -> Panel:
        t = self.telemetry
        battery = t.get("battery_percent")
        signal = t.get("signal_percent")
        lines = [
            Text.assemble(
                ("battery ", "bold"),
                f"{bar(battery, self.g)} {battery:.0f}%   " if battery is not None else "n/a   ",
                (sparkline(list(self.battery), self.g), "green"),
            ),
            Text.assemble(
                ("signal  ", "bold"),
                f"{bar(signal, self.g)} {signal:.0f}%   "
                if signal is not None
                else "not reported   ",
                (sparkline(list(self.signal), self.g), "cyan"),
            ),
            Text(
                f"zone {t.get('zone', '-')}   altitude {t.get('altitude_m', float('nan')):.0f} m   "
                f"missed deliveries {self.missed_telemetry}",
            ),
        ]
        return Panel(Group(*lines), title="Drone telemetry (SIMULATED)", border_style="cyan")


def _rank(severity: str) -> int:
    return ("low", "medium", "high", "critical").index(severity)
