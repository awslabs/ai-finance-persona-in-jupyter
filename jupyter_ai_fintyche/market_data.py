# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Market-data plumbing shared by the library backends' run sandboxes.

Generated snippets fetch real market inputs with yfinance, pre-bound as `yf`,
instead of typing numbers in from memory, and fall back to simulated data only
when the user asks for it or a fetch fails (see `persona._MARKET_DATA_RULES`).
The helpers here make that possible inside the restricted namespaces that
`quantlib_tools` and `pyportfolioopt_tools` execute snippets in.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Optional

# Builtins a fetch-with-fallback snippet needs on top of each backend's safe
# set: `try`/`except Exception` around a yfinance call, `raise ValueError(...)`
# on empty data, and `isinstance` checks on what came back. Without them those
# snippets die with `NameError` instead of falling back.
EXCEPTION_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "Exception", "ValueError", "KeyError", "IndexError", "TypeError",
        "RuntimeError", "ZeroDivisionError", "isinstance",
    )
}

YFINANCE_MISSING_MESSAGE = (
    "The generated code fetches market data with `yf` (yfinance), which is not "
    "installed even though FinTyche depends on it. Install it with "
    "`pip install yfinance` and restart the Jupyter server."
)


def load_yfinance() -> Optional[Any]:
    """Return the `yfinance` module, or None if it is missing from a broken install."""
    try:
        import yfinance
    except ImportError:
        return None
    return yfinance


def uses_yf(code: str) -> bool:
    """Whether `code` reads the name `yf` (e.g. `yf.download(...)`).

    Parsed rather than regex-matched so a string such as `"yf"` or a name such
    as `xyf` doesn't count. Unparseable code returns False and fails later in
    `exec` with its real SyntaxError.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Name) and node.id == "yf" for node in ast.walk(tree)
    )
