"""The frontend/ and backend/ folders: real UI files, and a guide whose every path exists."""

import re

import pytest
from conftest import ROOT

from drone_agent.web.server import STATIC_DIR, STATIC_FILES

FRONTEND = ROOT / "frontend"
BACKEND = ROOT / "backend"
LINK = re.compile(r"\]\((?!https?:|#)([^)\s]+)\)")


def test_the_ui_lives_in_the_top_level_frontend_folder_and_the_server_serves_it():
    assert STATIC_DIR == FRONTEND
    for name, _ in STATIC_FILES.values():
        assert (FRONTEND / name).is_file(), name
    assert (FRONTEND / "README.md").is_file()


def test_no_ui_files_are_left_inside_the_python_package():
    package = ROOT / "drone_agent" / "web"
    assert not (package / "static").exists()
    assert not [p for p in package.rglob("*") if p.suffix in {".html", ".css", ".js", ".svg"}]


def test_the_backend_folder_is_a_guide_and_holds_no_code():
    assert (BACKEND / "README.md").is_file()
    assert not [p for p in BACKEND.rglob("*") if p.suffix in {".py", ".js", ".html"}]


@pytest.mark.parametrize("readme", [FRONTEND / "README.md", BACKEND / "README.md"])
def test_every_relative_link_in_the_folder_readmes_points_at_something_that_exists(readme):
    links = LINK.findall(readme.read_text(encoding="utf-8"))
    assert links, readme
    for target in links:
        path = (readme.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{readme.name} links to missing {target}"


def test_every_file_the_backend_guide_names_exists_in_the_folder_it_names():
    checked = 0
    for line in (BACKEND / "README.md").read_text(encoding="utf-8").splitlines():
        folder = re.search(r"\]\(\.\./(drone_agent/[a-z_]+/)\)", line)
        if not folder or "|" not in line:
            continue
        for name in re.findall(r"`([a-z_]+\.py)`", line):
            assert (ROOT / folder.group(1) / name).is_file(), f"{folder.group(1)}{name}"
            checked += 1
    assert checked >= 40  # the guide really does name the files, and they all exist


def test_the_top_level_readme_shows_where_the_frontend_and_backend_are():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for expected in ("frontend/README.md", "backend/README.md", "drone_agent/web/"):
        assert expected in text
