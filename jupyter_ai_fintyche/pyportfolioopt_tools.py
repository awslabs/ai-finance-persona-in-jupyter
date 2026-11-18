# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Introspection, execution, and notebook-generation helpers over the
PyPortfolioOpt Python library (distributed on PyPI as ``PyPortfolioOpt`` and
imported as ``pypfopt``; sources at
[`PyPortfolio/PyPortfolioOpt`](https://github.com/PyPortfolio/PyPortfolioOpt),
docs at https://pyportfolioopt.readthedocs.io/en/latest/).

These helpers back the FinTyche chat commands for PyPortfolioOpt:
`pyportpf-list`, `pyportpf-doc <name>`, `pyportpf-run <name> <args>`, and
`pyportpf-notebook <name> <args>`.

Unlike QuantLib, whose entire API lives in one flat ``QuantLib`` namespace,
PyPortfolioOpt is a package: top-level optimizer classes (``EfficientFrontier``,
``CLA``, ``HRPOpt``, ``BlackLittermanModel`` …) plus submodules of functions
(``expected_returns.mean_historical_return``, ``risk_models.sample_cov`` …). So
these helpers resolve *dotted* symbol names (e.g. ``expected_returns.mean_historical_return``)
by walking attributes from the ``pypfopt`` package, and `list` surfaces both the
top-level symbols and the public functions inside the submodules.
"""

from __future__ import annotations

import builtins
import importlib
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

# A conservative set of builtins made available to `run`/`notebook` code
# evaluation. Deliberately excludes `__import__`, `open`, `exec`, `eval`, etc.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "len", "list", "max", "min", "range", "round", "set", "sorted", "str",
    "sum", "tuple", "zip",
)
_SAFE_BUILTINS: dict[str, Any] = {
    **{name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES},
    **EXCEPTION_BUILTINS,
}

# Submodules of `pypfopt` whose public functions `list` should surface as dotted
# names. Optimizer classes are exposed at the top level, so we only pull the
# *functional* API out of these (e.g. `risk_models.sample_cov`). `plotting` is
# listed here too; it is made importable by `_ensure_lazy_submodules()`.
_FUNCTION_SUBMODULES = (
    "expected_returns",
    "risk_models",
    "objective_functions",
    "black_litterman",
    "plotting",
)


# Submodules of `pypfopt` that the package's own `__init__` does NOT import, so
# they are absent from the `pypfopt` namespace until something imports them
# explicitly. `plotting` is lazy because it pulls in matplotlib. Without the
# explicit import in `_ensure_lazy_submodules()` below, `plotting.*` symbols
# neither resolve (`has_symbol` -> False) nor execute
# (`AttributeError: module 'pypfopt' has no attribute 'plotting'`).
_LAZY_SUBMODULES = ("plotting",)


class PyPortfolioOptUnavailable(RuntimeError):
    """Raised when the `PyPortfolioOpt` (`pypfopt`) package is not importable."""


# A library-agnostic alias so the persona can catch the "backend unavailable"
# error uniformly across QuantLib and PyPortfolioOpt backends.
Unavailable = PyPortfolioOptUnavailable


def _ensure_lazy_submodules(pypfopt: Any) -> None:
    """Import the `pypfopt` submodules its `__init__` leaves out, best-effort.

    Binds each name in `_LAZY_SUBMODULES` onto the package so dotted lookups
    (`plotting.plot_weights`) and generated snippets (`pypfopt.plotting.…`) both
    work, regardless of what ran earlier in the process.

    Failures are deliberately swallowed: `plotting` requires matplotlib, which is
    an optional extra, and importing it initializes a backend that can raise on a
    headless machine. A missing lazy submodule must degrade to "that symbol is
    unavailable" rather than breaking every other PyPortfolioOpt command.
    """
    for modname in _LAZY_SUBMODULES:
        if hasattr(pypfopt, modname):
            continue
        try:
            importlib.import_module(f"pypfopt.{modname}")
        except Exception:  # noqa: BLE001 - see docstring: optional dependency
            continue


def _patch_scipy_linkage_methods() -> None:
    """Restore the SciPy name `HRPOpt.optimize` validates its argument against.

    PyPortfolioOpt (through at least 1.6.0) checks `linkage_method` against the
    private `scipy.cluster.hierarchy._LINKAGE_METHODS`, which newer SciPy (1.18)
    removed, so every HRP optimization dies with `AttributeError`. Re-adding the
    method names `scipy.cluster.hierarchy.linkage` accepts brings it back.
    """
    try:
        import scipy.cluster.hierarchy as sch
    except ImportError:
        return
    if not hasattr(sch, "_LINKAGE_METHODS"):
        sch._LINKAGE_METHODS = {
            "single", "complete", "average", "weighted", "centroid", "median", "ward",
        }


def _load_pypfopt():
    try:
        import pypfopt
    except ImportError as e:
        raise PyPortfolioOptUnavailable(
            "PyPortfolioOpt is not installed. Install it with "
            "`pip install PyPortfolioOpt`."
        ) from e
    _patch_scipy_linkage_methods()
    # Every entry point into this module (`resolve_symbol`, `list_entries`,
    # `_base_namespace`) funnels through here, so importing the lazy submodules
    # at this one chokepoint makes them uniformly available.
    _ensure_lazy_submodules(pypfopt)
    return pypfopt


@dataclass
class PyPortfolioOptEntry:
    name: str
    kind: str  # "class" | "function" | "module" | "value"
    summary: str


@dataclass
class PyPortfolioOptDoc:
    name: str
    kind: str
    signature: str
    doc: str


def _kind_of(obj: Any) -> str:
    if inspect.isclass(obj):
        return "class"
    if inspect.ismodule(obj):
        return "module"
    if callable(obj):
        return "function"
    return "value"


def _summary_of(obj: Any) -> str:
    doc_lines = (inspect.getdoc(obj) or "").strip().splitlines()
    return doc_lines[0] if doc_lines else ""


def resolve_symbol(name: str) -> Optional[Any]:
    """Resolve a (possibly dotted) PyPortfolioOpt symbol name to its object.

    Walks attributes from the ``pypfopt`` package, so both top-level symbols
    (``EfficientFrontier``) and submodule members
    (``expected_returns.mean_historical_return``) resolve. Returns ``None`` if
    any part of the path is missing.
    """
    pypfopt = _load_pypfopt()
    obj: Any = pypfopt
    for part in name.split("."):
        if not part:
            return None
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def has_symbol(name: str) -> bool:
    """Return whether `name` (dotted allowed) resolves to a PyPortfolioOpt symbol."""
    try:
        return resolve_symbol(name) is not None
    except PyPortfolioOptUnavailable:
        return False


def list_entries(
    query: Optional[str] = None, limit: int = 100
) -> tuple[list[PyPortfolioOptEntry], int]:
    """List public PyPortfolioOpt symbols, optionally filtered by a substring.

    Surfaces the top-level ``pypfopt`` namespace (optimizer classes, helper
    functions, and submodules) plus the public functions inside the main
    submodules as dotted names (e.g. ``risk_models.sample_cov``). Matching is a
    case-insensitive substring test against the (dotted) name.

    Returns `(entries, total_matched)` — `entries` is capped at `limit` so the
    chat reply stays readable; `total_matched` is the full match count before
    truncation.
    """
    pypfopt = _load_pypfopt()

    seen: set[str] = set()
    candidates: list[tuple[str, Any]] = []

    def add(name: str, obj: Any) -> None:
        if name in seen:
            return
        seen.add(name)
        candidates.append((name, obj))

    for name in dir(pypfopt):
        if name.startswith("_"):
            continue
        add(name, getattr(pypfopt, name))

    for modname in _FUNCTION_SUBMODULES:
        try:
            mod = getattr(pypfopt, modname)
        except AttributeError:
            try:  # `plotting` (and friends) may need an explicit import.
                mod = __import__(f"pypfopt.{modname}", fromlist=[modname])
            except Exception:
                continue
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            # Only surface functions/classes actually defined in PyPortfolioOpt,
            # skipping re-exported third-party names (numpy, pandas, cvxpy …).
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if not getattr(obj, "__module__", "").startswith("pypfopt"):
                continue
            add(f"{modname}.{attr}", obj)

    names = sorted(n for n, _ in candidates)
    if query:
        q = query.lower()
        names = [n for n in names if q in n.lower()]
    total = len(names)

    by_name = dict(candidates)
    entries = [
        PyPortfolioOptEntry(
            name=name, kind=_kind_of(by_name[name]), summary=_summary_of(by_name[name])
        )
        for name in names[:limit]
    ]
    return entries, total


def get_doc(name: str) -> Optional[PyPortfolioOptDoc]:
    """Return signature + docstring for a PyPortfolioOpt symbol, or None if unknown.

    Accepts dotted names (e.g. ``EfficientFrontier`` or
    ``expected_returns.mean_historical_return``).
    """
    obj = resolve_symbol(name)
    if obj is None:
        return None
    doc = inspect.getdoc(obj) or "No documentation string is available for this symbol."
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"  # classes/builtins without an introspectable signature
    return PyPortfolioOptDoc(name=name, kind=_kind_of(obj), signature=sig, doc=doc)


# Bound here so tests can stub the loader on this module.
_load_yfinance = load_yfinance


def _base_namespace() -> dict[str, Any]:
    """Build the restricted namespace shared by `call_function` and `run_snippet`.

    Pre-binds ``pypfopt`` plus ``pd``/``np`` (pandas/numpy): portfolio
    optimization is inherently data-driven — expected returns and covariances
    are computed from a price or returns table — so snippets almost always need
    to construct or manipulate a DataFrame. ``yf`` (yfinance) is bound too, so
    snippets fetch real prices and market caps instead of inventing them. No file, process, or
    import access is granted; yfinance does make network requests to Yahoo
    Finance.
    """
    pypfopt = _load_pypfopt()
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "pypfopt": pypfopt,
    }
    try:  # pandas/numpy are declared dependencies, but stay defensive.
        import numpy as np
        import pandas as pd

        namespace["np"] = np
        namespace["pd"] = pd
    except ImportError:
        pass
    yfinance = _load_yfinance()
    if yfinance is not None:
        namespace["yf"] = yfinance
    return namespace


def call_function(name: str, args_str: str = "") -> tuple[Any, str]:
    """Evaluate `pypfopt.<name>(<args_str>)` and return `(result, captured_stdout)`.

    `name` may be dotted (e.g. ``expected_returns.mean_historical_return``).
    `args_str` is a literal Python argument list as typed by the user in chat.
    Evaluated with a restricted builtins set — no file, process, or import
    access — but this is still code execution over user-supplied text, so only
    wire it up to trusted chat users.
    """
    if not has_symbol(name):
        raise ValueError(f"PyPortfolioOpt has no symbol named '{name}'.")

    namespace = _base_namespace()
    expression = f"pypfopt.{name}({args_str})"

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        result = eval(expression, namespace)  # noqa: S307 - restricted builtins above
    return result, buffer.getvalue()


def run_snippet(code: str) -> tuple[Any, str]:
    """Execute a multi-statement PyPortfolioOpt snippet and return `(result, stdout)`.

    Unlike `call_function`, which evaluates a single ``pypfopt.<name>(...)``
    expression, this runs a small program so word problems that need the usual
    optimization pipeline can be answered — build or fetch a price table,
    compute expected returns and a covariance matrix, construct an optimizer such
    as ``EfficientFrontier``, solve for weights, and read
    ``portfolio_performance()``. The snippet MUST assign its answer to a variable
    named ``result``.

    ``pypfopt``, ``pd`` (pandas) and ``np`` (numpy) are pre-bound in the
    namespace, plus ``yf`` (yfinance), so the snippet needs
    no imports. A snippet that uses ``yf`` while yfinance is absent raises
    `PyPortfolioOptUnavailable` with an install hint. It runs with the same restricted
    builtins as `call_function` (no file, process, or import access), but this is
    still execution of model-generated code over user-supplied text, so only wire
    it up to trusted chat users.
    """
    # A single namespace serves as both globals and locals so that names
    # defined in one statement are visible to later statements (and to any
    # comprehensions/closures the library helpers might use).
    namespace = _base_namespace()
    if "yf" not in namespace and uses_yf(code):
        # Without this the snippet dies with a bare `NameError: name 'yf' is
        # not defined`, which reads like a code bug rather than a missing extra.
        raise PyPortfolioOptUnavailable(YFINANCE_MISSING_MESSAGE)

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
    """Create a Jupyter notebook that runs a PyPortfolioOpt snippet on `name`.

    `code` is a Python snippet (as produced for the `pyportpf-run` command) that
    uses the ``pypfopt`` namespace and assigns its answer to a ``result``
    variable — anything from a single ``result = pypfopt.risk_models.sample_cov(prices)``
    to a full efficient-frontier optimization. The notebook imports the library
    (and pandas/numpy), runs the snippet, and displays ``result``.
    `word_problem`, when given, is recorded in the header so the notebook
    explains what it computes.

    Writes the notebook into `directory` and returns the absolute path.
    """
    import nbformat as nbf

    doc = get_doc(name)
    if doc is None:
        raise ValueError(f"PyPortfolioOpt has no symbol named '{name}'.")

    header = (
        f"# PyPortfolioOpt: `{name}`\n\n"
        f"**Kind:** {doc.kind}  \n"
        f"**Signature:** `{doc.signature}`\n\n"
    )
    if word_problem:
        header += f"**Word problem:** {word_problem}\n\n"
    header += f"{doc.doc}\n\nGenerated by **FinTyche** for Jupyter AI."

    imports = "import pypfopt\nimport numpy as np\nimport pandas as pd"
    if uses_yf(code):
        imports += "\nimport yfinance as yf"

    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(header),
        nbf.v4.new_code_cell(imports),
        # Run the snippet, then display `result` as the cell's output.
        nbf.v4.new_code_cell(f"{code.rstrip()}\n\nresult"),
    ]

    default_filename = f"pyportfolioopt_{name.replace('.', '_')}.ipynb"
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
