# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the PyPortfolioOpt backend's handling of lazily-imported submodules.

Regression coverage for the `plotting` submodule, which `pypfopt/__init__` does
not import (it pulls in matplotlib). Before `_ensure_lazy_submodules()`, that
made `plotting.*` behave differently depending on what had already run in the
process:

- In a fresh process, `has_symbol("plotting.plot_weights")` returned `False`, so
  `/pyportpf-run plotting.plot_weights …` replied "PyPortfolioOpt has no symbol
  named …" even though the library provides it.
- `run_snippet` raised `AttributeError: module 'pypfopt' has no attribute
  'plotting'`.
- Yet after any `list_entries()` call — whose `__import__` fallback binds the
  submodule as a side effect — both started working.

The order dependence is the bug, so these tests pin the behaviour with no prior
`list_entries()` call, including in a genuinely separate interpreter.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jupyter_ai_fintyche import pyportfolioopt_tools as tools

pypfopt = pytest.importorskip("pypfopt", reason="PyPortfolioOpt is not installed")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def unimported_plotting(monkeypatch: pytest.MonkeyPatch):
    """Simulate a fresh interpreter in which `pypfopt.plotting` is not imported.

    Importing `pypfopt.plotting` anywhere in the session caches it in
    `sys.modules` and binds it onto the package, which would mask the bug. This
    removes both traces so the code under test has to perform the import itself.
    """
    monkeypatch.delitem(sys.modules, "pypfopt.plotting", raising=False)
    if hasattr(pypfopt, "plotting"):
        monkeypatch.delattr(pypfopt, "plotting")


def test_plotting_resolves_without_prior_list(unimported_plotting) -> None:
    """A plotting symbol resolves even though nothing imported the submodule."""
    assert tools.has_symbol("plotting.plot_weights")


def test_plotting_doc_is_available_without_prior_list(unimported_plotting) -> None:
    """`doc` works on a plotting symbol, so the chat can document it."""
    doc = tools.get_doc("plotting.plot_weights")
    assert doc is not None
    assert doc.name == "plotting.plot_weights"
    assert doc.kind == "function"


def test_plotting_executes_in_snippet(unimported_plotting) -> None:
    """A generated snippet can call into `pypfopt.plotting` without AttributeError.

    Resolving the submodule is this module's job; actually rendering needs
    matplotlib, which PyPortfolioOpt treats as an optional extra and imports only
    when a plot function is called — so skip when it is absent.
    """
    matplotlib = pytest.importorskip(
        "matplotlib", reason="matplotlib is required to render a plot"
    )

    # Render headlessly: the server process has no display, and an interactive
    # backend would try to open a window.
    matplotlib.use("Agg", force=True)

    result, _stdout = tools.run_snippet(
        textwrap.dedent(
            """
            weights = {"A": 0.5, "B": 0.3, "C": 0.2}
            result = pypfopt.plotting.plot_weights(weights, showfig=False)
            """
        )
    )
    # `plot_weights` returns the matplotlib Axes it drew on.
    assert result.__class__.__name__.endswith("Axes")


def test_plotting_submodule_is_reachable_in_a_snippet(unimported_plotting) -> None:
    """`pypfopt.plotting` is bound in the snippet namespace even without matplotlib.

    This is the half of the fix that does not depend on the optional extra: the
    attribute lookup must succeed, so a snippet fails (if at all) with
    PyPortfolioOpt's actionable "install matplotlib" message rather than
    `AttributeError: module 'pypfopt' has no attribute 'plotting'`.
    """
    result, _stdout = tools.run_snippet(
        "result = pypfopt.plotting.plot_weights.__name__"
    )
    assert result == "plot_weights"


def test_other_commands_survive_an_unimportable_lazy_submodule(
    monkeypatch: pytest.MonkeyPatch, unimported_plotting
) -> None:
    """A lazy submodule that cannot be imported must not break the whole backend.

    `plotting` depends on matplotlib, an optional extra whose import can also
    fail while initializing a backend. When that happens the rest of
    PyPortfolioOpt must keep working, and the plotting symbol must simply report
    as unavailable.
    """

    def _raise(name: str, *args: object, **kwargs: object):
        if name.startswith("pypfopt."):
            raise ImportError(f"simulated failure importing {name}")
        return original_import_module(name, *args, **kwargs)

    original_import_module = tools.importlib.import_module
    monkeypatch.setattr(tools.importlib, "import_module", _raise)

    # The core API is unaffected...
    assert tools.has_symbol("EfficientFrontier")
    # ...and the unavailable submodule degrades to "no such symbol".
    assert not tools.has_symbol("plotting.plot_weights")


def test_plotting_resolves_in_a_fresh_interpreter() -> None:
    """End-to-end guard against the original order-dependent failure.

    The in-process tests above reconstruct a fresh state with monkeypatching;
    this one uses a real subprocess, so nothing in this test session can mask a
    regression.
    """
    script = textwrap.dedent(
        """
        from jupyter_ai_fintyche import pyportfolioopt_tools as tools

        # Deliberately no `list_entries()` call before this probe.
        assert tools.has_symbol("plotting.plot_weights"), "plotting did not resolve"
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "ok" in proc.stdout
