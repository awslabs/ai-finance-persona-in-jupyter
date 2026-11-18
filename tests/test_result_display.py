# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for how a run's `result` is shown in chat: one bullet per field rather
than a one-line `repr` that ran off the side of the chat box, and no
"yfinance" data source claimed by code that never calls yfinance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jupyter_ai_fintyche import backends, core, persona

pytest.importorskip("QuantLib", reason="QuantLib is not installed")


class _Sink(persona.FinTychePersona):
    """The persona without its chat plumbing; sent messages are collected."""

    def __init__(self):
        self._history = []
        self.messages: list[str] = []

    def send_message(self, body: str) -> None:
        self.messages.append(body)


def _shown_result(code: str) -> str:
    agent = _Sink()
    agent._execute_snippet(persona._BACKENDS["quantlib"], "BlackScholesProcess", code)
    return agent.messages[-1].split("Result:", 1)[1]


def test_dict_result_is_one_bullet_per_field() -> None:
    shown = core._format_result(
        {"weights": {"AAPL": 0.6}, "sharpe": np.float64(0.65), "note": "costs $5"}
    )
    assert shown.splitlines() == [
        "- **weights**:",
        "  - **AAPL**: 0.6",
        "- **sharpe**: 0.65",
        "- **note**: costs \\$5",
    ]


def test_multiline_value_goes_in_a_code_block_under_its_field() -> None:
    shown = core._format_result({"prices": pd.DataFrame({"A": [1.0, 2.0]})})
    assert shown.startswith("- **prices**:\n\n  ````text\n  ")


def test_non_dict_result_stays_a_code_block() -> None:
    assert core._format_result(1.5) == "```text\n1.5\n```"


def test_yfinance_claim_dropped_when_code_never_calls_yf() -> None:
    shown = _shown_result("result = {'npv': 1.0, 'data_source': 'yfinance'}")
    assert "data\\_source" not in shown


def test_yfinance_claim_kept_when_code_calls_yf() -> None:
    code = (
        "try:\n    yf.download\nexcept Exception:\n    pass\n"
        "result = {'npv': 1.0, 'data_source': 'yfinance'}"
    )
    assert "- **data\\_source**: yfinance" in _shown_result(code)


def test_simulated_label_is_kept() -> None:
    shown = _shown_result(
        "result = {'npv': 1.0, 'data_source': 'SIMULATED: yfinance fetch failed'}"
    )
    assert "SIMULATED" in shown


def test_prompt_no_longer_credits_yfinance_for_given_values() -> None:
    assert "fetched or given" not in backends._MARKET_DATA_RULES
