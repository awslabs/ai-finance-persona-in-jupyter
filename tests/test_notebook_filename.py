# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests that `create_notebook` only ever writes inside the target directory.

The notebook filename reaches these helpers from chat text — FinTyche parses
`... as <filename>` off the end of a `notebook` command — so a value containing
path separators must not escape the workspace directory or overwrite an
unrelated file on disk.

Neither QuantLib nor PyPortfolioOpt is required: both loaders are patched with
small stand-in modules.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from jupyter_ai_fintyche import pyportfolioopt_tools, quantlib_tools

SNIPPET = "result = 1"


@pytest.fixture
def backends(monkeypatch):
    """Patch both libraries' loaders and return the two modules under test."""

    def Date(day, month, year):
        return f"Date({day},{month},{year})"

    fake_ql = types.SimpleNamespace(Date=Date)
    monkeypatch.setattr(quantlib_tools, "_load_quantlib", lambda: fake_ql)

    fake_pypfopt = types.SimpleNamespace(
        expected_returns=types.SimpleNamespace(mean_historical_return=lambda prices: prices)
    )
    monkeypatch.setattr(pyportfolioopt_tools, "_load_pypfopt", lambda: fake_pypfopt)

    return {
        "quantlib": (quantlib_tools, "Date"),
        "pyportfolioopt": (pyportfolioopt_tools, "expected_returns.mean_historical_return"),
    }


@pytest.fixture(params=["quantlib", "pyportfolioopt"])
def backend(request, backends):
    """Run each test against both library backends."""
    return backends[request.param]


@pytest.mark.parametrize(
    "hostile_filename",
    [
        "../escaped.ipynb",
        "../../../escaped.ipynb",
        "sub/dir/escaped.ipynb",
        "/tmp/escaped.ipynb",
    ],
)
def test_filename_cannot_escape_directory(backend, tmp_path, hostile_filename):
    tools, name = backend
    path = Path(tools.create_notebook(name, SNIPPET, str(tmp_path), hostile_filename))

    assert path.parent == tmp_path, f"notebook escaped to {path}"
    assert path.is_file()


def test_plain_filename_is_preserved(backend, tmp_path):
    tools, name = backend
    path = Path(tools.create_notebook(name, SNIPPET, str(tmp_path), "analysis.ipynb"))

    assert path == tmp_path / "analysis.ipynb"
    assert path.is_file()


def test_extension_is_appended(backend, tmp_path):
    tools, name = backend
    path = Path(tools.create_notebook(name, SNIPPET, str(tmp_path), "analysis"))

    assert path == tmp_path / "analysis.ipynb"


@pytest.mark.parametrize("degenerate_filename", ["", ".", "..", "   ", "sub/dir/"])
def test_degenerate_filename_falls_back_to_default(backend, tmp_path, degenerate_filename):
    """A filename that sanitizes down to nothing must still produce a real
    notebook inside the directory, not a dotfile or a write outside it."""
    tools, name = backend
    path = Path(tools.create_notebook(name, SNIPPET, str(tmp_path), degenerate_filename))

    assert path.parent == tmp_path
    assert path.name.endswith(".ipynb")
    assert path.stem not in {"", ".", ".."}
    assert path.is_file()


def test_default_filename_used_when_omitted(backend, tmp_path):
    tools, name = backend
    path = Path(tools.create_notebook(name, SNIPPET, str(tmp_path)))

    assert path.parent == tmp_path
    assert path.name.endswith(".ipynb")
    assert path.is_file()
