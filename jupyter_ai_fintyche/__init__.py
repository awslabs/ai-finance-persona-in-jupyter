# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
__version__ = "0.1.0"
__all__ = ["FinTychePersona"]


def __getattr__(name: str):
    # Imported on first use, not with the package, so the CLI and the
    # frontend-agnostic core load without Jupyter AI installed.
    if name == "FinTychePersona":
        from .persona import FinTychePersona

        return FinTychePersona
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
