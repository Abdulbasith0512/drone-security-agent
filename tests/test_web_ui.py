"""The dashboard UI files: structure, the CSP-safe subset, honest copy and readable colours.

These are static checks over frontend/ (no browser needed); tests/test_web_ui_browser.py drives the
real page. tests/test_web_server.py still pins the security contract of the same files: no rule ids,
thresholds or detection logic, nothing loaded from outside, the required panel ids.
"""

import re

import pytest

from drone_agent.web.server import STATIC_DIR, STATIC_FILES

FILES = {name: (STATIC_DIR / name).read_text(encoding="utf-8") for name, _ in STATIC_FILES.values()}
HTML, JS, CSS = FILES["index.html"], FILES["app.js"], FILES["app.css"]


def test_the_frontend_is_exactly_three_files_plus_the_icon_and_the_readme():
    names = {p.name for p in STATIC_DIR.iterdir()}
    assert names == {"index.html", "app.js", "app.css", "favicon.svg", "README.md"}
    assert set(STATIC_FILES.values()) >= {
        ("index.html", "text/html; charset=utf-8"),
        ("app.js", "text/javascript; charset=utf-8"),
        ("app.css", "text/css; charset=utf-8"),
    }


def test_the_required_structure_and_title_are_present():
    assert "<title>Drone Security Analyst Dashboard</title>" in HTML
    for element_id in (
        "panel-video", "panel-investigation", "panel-timeline", "panel-alerts", "panel-overview",
        "panel-analyst", "pipeline", "kind-banner", "run-select", "status-chips", "kpi-strip",
        "strip",
    ):  # fmt: skip
        assert f'id="{element_id}"' in HTML, element_id
    assert 'lang="en"' in HTML and 'name="viewport"' in HTML
    assert HTML.count("<h1") == 1  # one page heading


def test_nothing_needs_an_inline_style_or_a_data_url_under_the_csp():
    assert "style=" not in HTML and "<style" not in HTML
    assert 'setAttribute("style"' not in JS and "setAttribute('style'" not in JS
    assert "innerHTML" not in JS and "insertAdjacentHTML" not in JS and "eval(" not in JS
    for name, text in FILES.items():
        assert "data:" not in text.replace("Data:", ""), name
        assert "@import" not in text and "url(" not in text, name


def test_the_page_never_says_live():
    visible = JS + HTML + CSS
    visible = visible.replace("aria-live", "")  # the accessibility attribute is not wording
    assert not re.search(r"(?i)\blive\b", visible)


def test_no_result_of_a_particular_run_is_hardcoded():
    """Every number and name on screen must come from the API: none of these may be typed here."""
    for literal in (
        "617", "1920", "1080", "24.68", "d7e3cee", "a8715a8", "gemini", "Gemini", "clip.mp4",
        "real_clip", "security_events", "synthetic_video", "EVT-P-", "EVT-V-", "ENT-V-",
        "main_gate", "2026-09", "14:00", "Drone-Demo",
    ):  # fmt: skip
        assert literal not in JS and literal not in HTML, literal


def test_the_chip_wording_comes_from_the_declared_provenance_not_from_the_run_name():
    assert "replayed" in JS and "canned" in JS and "scripted" in JS  # the three stored values
    assert "VLM replay" in JS and "Canned replay" in JS
    # the synthetic demo is canned text and must never be called a vision model
    assert not re.search(r'canned\s*:\s*"[^"]*VLM', JS)


def test_no_alert_is_invented_and_the_zero_state_is_neutral():
    assert "No security rules triggered in this" in JS
    assert "No security rule triggered for this event." in JS
    zero = re.search(r"\.zero \{[^}]*\}", CSS).group(0)
    assert "crit" not in zero and "danger" not in zero  # not styled as an error


def test_the_ui_has_focus_rings_reduced_motion_and_responsive_breakpoints():
    assert ":focus-visible" in CSS and "outline: 2px solid var(--primary)" in CSS
    assert "prefers-reduced-motion: reduce" in CSS
    assert re.search(r"@media \(max-width: 1180px\)", CSS) and re.search(
        r"@media \(max-width: 720px\)", CSS
    )
    assert "position: sticky" in CSS
    assert (
        "aspectRatio" in JS
    )  # the viewer reserves its size from the API's video size: no layout shift


def test_frames_are_requested_at_one_size_through_a_queue():
    """Each frame is decoded by reading the video up to it, so one size and a queue keep the server
    from decoding everything twice and all at once (it matters most on a small host)."""
    assert re.search(r"const FRAME_W = 640", JS)
    assert set(re.findall(r"frameUrl\([^()]*?,\s*(\w+)\)", JS)) == {"FRAME_W"}
    assert "IntersectionObserver" in JS and "const loader" in JS and "max: 2" in JS
    assert (
        'loading: "lazy"' not in JS
    )  # the queue, not the browser, decides when a picture is fetched
    assert "new Image()" not in JS  # no speculative prefetch burst


# ---- colours: WCAG AA for every text token on every surface it is drawn on -----------------
def tokens() -> dict[str, str]:
    root = re.search(r":root \{(.*?)\n\}", CSS, re.S).group(1)
    return dict(re.findall(r"--([a-z0-9-]+):\s*(#[0-9a-fA-F]{6})\b", root))


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


SURFACES = ("bg", "panel", "panel-2", "panel-3")
TEXT = ("text", "muted", "primary", "ai", "ok", "warn", "crit", "orange")


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("text", TEXT)
def test_text_colours_meet_wcag_aa_on_every_surface(text, surface):
    t = tokens()
    assert contrast(t[text], t[surface]) >= 4.5, (
        f"{text} on {surface}: {contrast(t[text], t[surface]):.2f}"
    )


@pytest.mark.parametrize(
    "fg, bg",
    [("on-primary", "primary"), ("ai", "ai-soft"), ("primary", "primary-soft"),
     ("text", "primary-soft"),
     ("muted", "primary-soft"), ("related", "panel-2"), ("bar", "panel-2")],
)  # fmt: skip
def test_special_pairings_meet_their_contrast(fg, bg):
    t = tokens()
    needed = 3.0 if fg in ("related", "bar") else 4.5  # graphics need 3:1, text 4.5:1
    assert contrast(t[fg], t[bg]) >= needed, f"{fg} on {bg}: {contrast(t[fg], t[bg]):.2f}"
