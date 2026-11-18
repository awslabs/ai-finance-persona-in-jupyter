# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for `jupyter_ai_fintyche.quantlib_tools`.

QuantLib itself is not required: these tests patch `_load_quantlib` with a
small stand-in module, so they exercise the argument-evaluation machinery
without depending on the SWIG bindings being installed.
"""

from __future__ import annotations

import types

import pytest

from jupyter_ai_fintyche import quantlib_tools


@pytest.fixture
def fake_quantlib(monkeypatch):
    """Patch `_load_quantlib` to return a minimal stand-in QuantLib module."""

    def Date(day, month, year):
        return f"Date({day},{month},{year})"

    def serial_of(value):
        """Stand-in for any QuantLib helper a user might call inside an argument."""
        return int(value)

    def announce(value):
        print(f"computing {value}")
        return value

    fake = types.SimpleNamespace(Date=Date, serial_of=serial_of, announce=announce)
    monkeypatch.setattr(quantlib_tools, "_load_quantlib", lambda: fake)
    return fake


def test_plain_arguments_evaluate(fake_quantlib):
    result, stdout = quantlib_tools.call_function("Date", "15, 6, 2025")
    assert result == "Date(15,6,2025)"
    assert stdout == ""


def test_no_arguments_evaluate(fake_quantlib):
    result, _ = quantlib_tools.call_function("serial_of", "2025")
    assert result == 2025


def test_stdout_is_captured(fake_quantlib):
    result, stdout = quantlib_tools.call_function("announce", "7")
    assert result == 7
    assert stdout == "computing 7\n"


def test_unknown_symbol_raises_value_error(fake_quantlib):
    with pytest.raises(ValueError, match="no function or class named 'Nope'"):
        quantlib_tools.call_function("Nope", "")


# ---------------------------------------------------------------------------
# Nested-scope access to the injected namespace.
#
# Generator expressions, lambdas, and (before Python 3.12's PEP 709 inlining)
# comprehensions each evaluate in their own scope. A name resolves there via
# locals -> enclosing function scopes -> globals -> builtins, which never
# consults a separate `locals` mapping handed to `eval`. So `ql`/`QuantLib`
# must live in the *globals* mapping, or any argument expression that opens a
# nested scope fails with `NameError: name 'QuantLib' is not defined`.
#
# `run_snippet` already binds a single namespace for this reason; these tests
# hold `call_function` to the same contract.
# ---------------------------------------------------------------------------


def test_generator_expression_can_reference_quantlib(fake_quantlib):
    result, _ = quantlib_tools.call_function(
        "Date", "15, 6, sum(QuantLib.serial_of(d) for d in [1000, 1025])"
    )
    assert result == "Date(15,6,2025)"


def test_lambda_can_reference_quantlib(fake_quantlib):
    result, _ = quantlib_tools.call_function(
        "Date", "15, 6, (lambda: QuantLib.serial_of(2025))()"
    )
    assert result == "Date(15,6,2025)"


def test_list_comprehension_can_reference_quantlib(fake_quantlib):
    """Passes on 3.12+ via PEP 709 inlining; guards the 3.11 floor too."""
    result, _ = quantlib_tools.call_function(
        "Date", "15, 6, max([QuantLib.serial_of(d) for d in [2024, 2025]])"
    )
    assert result == "Date(15,6,2025)"


def test_ql_alias_also_reachable_from_nested_scope(fake_quantlib):
    """The `ql.` prefix is the convention FinTyche's prompts instruct, so it
    must resolve in nested scopes just like `QuantLib.`."""
    result, _ = quantlib_tools.call_function(
        "Date", "15, 6, sum(ql.serial_of(d) for d in [1000, 1025])"
    )
    assert result == "Date(15,6,2025)"


@pytest.mark.parametrize(
    "dangerous_args",
    [
        "open('/etc/passwd').read()",
        "__import__('os').system('true')",
        "eval('1 + 1')",
        "exec('x = 1')",
    ],
)
def test_restricted_builtins_stay_unreachable(fake_quantlib, dangerous_args):
    """Widening argument evaluation must not widen the sandbox: file, import,
    and nested code-execution builtins are deliberately absent."""
    with pytest.raises((NameError, TypeError)):
        quantlib_tools.call_function("serial_of", dangerous_args)
