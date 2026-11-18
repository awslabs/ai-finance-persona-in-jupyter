# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Introspection, execution, and notebook-generation helpers over the QuantLib
Python bindings (built from the `lballabio/QuantLib` C++ library via SWIG
wrappers generated in `lballabio/QuantLib-SWIG`, and distributed on PyPI as
the `QuantLib` package).

These helpers back the FinTyche chat commands: `list`, `doc <name>`,
`run <name> <args>`, and `notebook <name> <args>`.
"""

from __future__ import annotations

import builtins
import inspect
import io
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Optional

from .market_data import (
    EXCEPTION_BUILTINS,
    YFINANCE_MISSING_MESSAGE,
    load_yfinance,
    uses_yf,
)

# A conservative set of builtins made available to `run`/`notebook` argument
# evaluation. Deliberately excludes `__import__`, `open`, `exec`, `eval`, etc.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "range", "round", "set", "str", "sum",
    "tuple",
)
_SAFE_BUILTINS: dict[str, Any] = {
    **{name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES},
    **EXCEPTION_BUILTINS,
}

# Bound here so tests can stub the loader on this module.
_load_yfinance = load_yfinance


class QuantLibUnavailable(RuntimeError):
    """Raised when the `QuantLib` package is not importable."""


# A library-agnostic alias so the persona can catch the "backend unavailable"
# error uniformly across the QuantLib and PyPortfolioOpt backends.
Unavailable = QuantLibUnavailable


def _load_quantlib():
    try:
        import QuantLib as ql
    except ImportError as e:
        raise QuantLibUnavailable(
            "QuantLib is not installed. Install it with `pip install QuantLib`."
        ) from e
    return ql


@dataclass
class QuantLibEntry:
    name: str
    kind: str  # "class" | "function" | "value"
    summary: str


@dataclass
class QuantLibDoc:
    name: str
    kind: str
    signature: str
    doc: str


def list_entries(query: Optional[str] = None, limit: int = 100) -> tuple[list[QuantLibEntry], int]:
    """List public QuantLib names, optionally filtered by a case-insensitive substring.

    Returns `(entries, total_matched)` — `entries` is capped at `limit` so the
    chat reply stays readable; `total_matched` is the full match count before
    truncation.
    """
    ql = _load_quantlib()
    names = [n for n in dir(ql) if not n.startswith("_")]
    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower()]
    names.sort()
    total = len(names)

    entries = []
    for name in names[:limit]:
        obj = getattr(ql, name)
        if inspect.isclass(obj):
            kind = "class"
        elif callable(obj):
            kind = "function"
        else:
            kind = "value"
        doc_lines = (inspect.getdoc(obj) or "").strip().splitlines()
        summary = doc_lines[0] if doc_lines else ""
        entries.append(QuantLibEntry(name=name, kind=kind, summary=summary))
    return entries, total


def has_symbol(name: str) -> bool:
    """Return whether `name` is a public symbol on the QuantLib module.

    Mirrors ``pyportfolioopt_tools.has_symbol`` so the persona can probe both
    backends through one interface. QuantLib's namespace is flat, so this is a
    simple ``hasattr`` (no dotted-path walk).
    """
    try:
        ql = _load_quantlib()
    except QuantLibUnavailable:
        return False
    return hasattr(ql, name)


def get_doc(name: str) -> Optional[QuantLibDoc]:
    """Return signature + docstring for a QuantLib class/function, or None if unknown."""
    ql = _load_quantlib()
    if not hasattr(ql, name):
        return None
    obj = getattr(ql, name)
    doc = inspect.getdoc(obj) or "No documentation string is available for this symbol."
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"  # many SWIG-wrapped overloads have no introspectable signature
    kind = "class" if inspect.isclass(obj) else "function" if callable(obj) else "value"
    return QuantLibDoc(name=name, kind=kind, signature=sig, doc=doc)


def call_function(name: str, args_str: str = "") -> tuple[Any, str]:
    """Evaluate `QuantLib.<name>(<args_str>)` and return `(result, captured_stdout)`.

    `args_str` is a literal Python argument list as typed by the user in
    chat, e.g. `"15, 6, 2025"` for `ql.Date(15, 6, 2025)`. Evaluated with a
    restricted builtins set — no file, process, or import access — but this
    is still code execution over user-supplied text, so only wire it up to
    trusted chat users.
    """
    ql = _load_quantlib()
    if not hasattr(ql, name):
        raise ValueError(f"QuantLib has no function or class named '{name}'.")

    # A single namespace is passed as `eval`'s globals so that `ql`/`QuantLib`
    # stay reachable from any nested scope the argument expression opens. A
    # generator expression, lambda, or comprehension resolves names through
    # locals -> enclosing function scopes -> globals -> builtins, which never
    # consults a separate `locals` mapping handed to `eval`; binding the module
    # there raises `NameError` instead. `run_snippet` below uses one namespace
    # for the same reason.
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "ql": ql,
        "QuantLib": ql,
    }
    expression = f"QuantLib.{name}({args_str})"

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = eval(expression, namespace)  # noqa: S307 - restricted builtins above
    return result, buffer.getvalue()


def run_snippet(code: str) -> tuple[Any, str]:
    """Execute a multi-statement QuantLib snippet and return `(result, captured_stdout)`.

    Unlike `call_function`, which evaluates a single `QuantLib.<name>(...)`
    expression, this runs a small program so word problems that need more than
    one call can be answered — e.g. pricing an option, which requires building a
    payoff and exercise, wiring a stochastic process into a pricing engine, and
    only then reading `.NPV()`. The snippet MUST assign its answer to a variable
    named `result`.

    `ql` and `QuantLib` are pre-bound in the namespace, plus `yf` (yfinance)
    for fetching spot prices, volatilities, and rates, and `np`/`pd` for
    working with what it returns, so the snippet needs no imports. A snippet
    that uses `yf` while yfinance is absent raises `QuantLibUnavailable` with an
    install hint. It runs with the same restricted builtins as `call_function` (no
    file, process, or import access), but this is still execution of
    model-generated code over user-supplied text, so only wire it up to trusted
    chat users.
    """
    ql = _load_quantlib()

    # A single namespace serves as both globals and locals so that names
    # defined in one statement are visible to later statements (and to any
    # comprehensions/closures QuantLib helpers might use).
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "ql": ql,
        "QuantLib": ql,
    }
    try:  # numpy/pandas come with yfinance, but stay defensive.
        import numpy as np
        import pandas as pd

        namespace["np"] = np
        namespace["pd"] = pd
    except ImportError:
        pass
    yfinance = _load_yfinance()
    if yfinance is not None:
        namespace["yf"] = yfinance
    elif uses_yf(code):
        # Otherwise the snippet dies with a bare `NameError: name 'yf' is not
        # defined`, which reads like a code bug rather than a broken install.
        raise QuantLibUnavailable(YFINANCE_MISSING_MESSAGE)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(code, namespace)  # noqa: S102 - restricted builtins above
    if "result" not in namespace:
        raise ValueError(
            "The generated code ran but never assigned a `result` variable, so "
            "there is no value to report."
        )
    return namespace["result"], buffer.getvalue()


def create_notebook(
    name: str,
    code: str,
    directory: str,
    filename: Optional[str] = None,
    word_problem: Optional[str] = None,
) -> str:
    """Create a Jupyter notebook that runs a QuantLib snippet centered on `name`.

    `code` is a Python snippet (as produced for the `run` command) that uses the
    `ql` namespace and assigns its answer to a `result` variable — anything from
    a single `result = ql.Date(...)` to a full option-pricing pipeline. The
    notebook imports QuantLib, runs the snippet, and displays `result`.
    `word_problem`, when given, is recorded in the header so the notebook
    explains what it computes.

    Writes the notebook into `directory` and returns the absolute path.
    """
    import nbformat as nbf

    doc = get_doc(name)
    if doc is None:
        raise ValueError(f"QuantLib has no function or class named '{name}'.")

    header = (
        f"# QuantLib: `{name}`\n\n"
        f"**Kind:** {doc.kind}  \n"
        f"**Signature:** `{doc.signature}`\n\n"
    )
    if word_problem:
        header += f"**Word problem:** {word_problem}\n\n"
    header += f"{doc.doc}\n\nGenerated by **FinTyche** for Jupyter AI."

    imports = "import QuantLib as ql"
    if uses_yf(code):
        imports += "\nimport numpy as np\nimport pandas as pd\nimport yfinance as yf"

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(header),
        nbf.v4.new_code_cell(imports),
        # Run the snippet, then display `result` as the cell's output.
        nbf.v4.new_code_cell(f"{code.rstrip()}\n\nresult"),
    ]

    default_filename = f"quantlib_{name}.ipynb"
    # The filename reaches us from chat text (`... as <filename>`), so keep only
    # its final path component: a value like `../../notes.ipynb` or
    # `/tmp/notes.ipynb` must not escape `directory` or overwrite a file
    # elsewhere on disk. Anything that sanitizes down to nothing usable falls
    # back to the default name.
    filename = os.path.basename(filename or default_filename).strip()
    if not filename or filename in {".", ".."}:
        filename = default_filename
    if not filename.endswith(".ipynb"):
        filename += ".ipynb"

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    return path
