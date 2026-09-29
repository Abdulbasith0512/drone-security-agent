"""M0 smoke test: package imports and the core stays free of LangChain."""

import importlib
import pathlib
import pkgutil

import drone_agent


def test_version():
    assert drone_agent.__version__


def test_core_does_not_import_langchain():
    for mod in pkgutil.walk_packages(drone_agent.__path__, "drone_agent."):
        if mod.name.startswith("drone_agent.agent"):
            continue
        m = importlib.import_module(mod.name)
        src = getattr(m, "__file__", "") or ""
        if src.endswith(".py"):
            assert "langchain" not in pathlib.Path(src).read_text(encoding="utf-8").lower()
