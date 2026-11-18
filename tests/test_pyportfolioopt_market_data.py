# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for how PyPortfolioOpt snippets reach real market data via yfinance.

Generated snippets must fetch prices and market caps with `yf` rather than
synthesizing them. These tests pin the sandbox side of that contract without
touching the network: `yf` is bound when yfinance is importable, a snippet that
needs it fails with an actionable install hint when it isn't, and generated
notebooks import it only when the code uses it.
"""

from __future__ import annotations

import types

import nbformat
import pytest

from jupyter_ai_fintyche import backends, persona
from jupyter_ai_fintyche import pyportfolioopt_tools as tools

pytest.importorskip("pypfopt", reason="PyPortfolioOpt is not installed")


@pytest.fixture
def fake_yfinance(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Stand in for yfinance so tests never hit Yahoo Finance."""
    module = types.ModuleType("yfinance")
    module.marker = "fake-yfinance"  # type: ignore[attr-defined]
    monkeypatch.setattr(tools, "_load_yfinance", lambda: module)
    return module


@pytest.fixture
def no_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_load_yfinance", lambda: None)


def test_yf_is_bound_when_yfinance_is_installed(fake_yfinance) -> None:
    result, _ = tools.run_snippet("result = yf.marker")
    assert result == "fake-yfinance"


def test_snippet_using_yf_without_yfinance_gives_install_hint(no_yfinance) -> None:
    code = 'prices = yf.download(["AAPL"], period="1y")\nresult = prices'
    with pytest.raises(tools.PyPortfolioOptUnavailable, match="pip install yfinance"):
        tools.run_snippet(code)


def test_snippet_without_yf_runs_without_yfinance(no_yfinance) -> None:
    """User-supplied data still works when the optional extra is absent."""
    result, _ = tools.run_snippet(
        "prices = pd.DataFrame({'A': [1.0, 1.1, 1.2]})\nresult = len(prices)"
    )
    assert result == 3


@pytest.mark.parametrize("code", ["xyf = 1\nresult = xyf", "result = 'yf'.upper()"])
def test_yf_detection_ignores_lookalikes(no_yfinance, code: str) -> None:
    """Only a real use of the `yf` name should demand yfinance."""
    tools.run_snippet(code)


@pytest.mark.parametrize(
    ("code", "expects_import"),
    [
        ('prices = yf.download(["AAPL"])\nresult = prices', True),
        ("result = pypfopt.risk_models.sample_cov", False),
    ],
)
def test_notebook_imports_yfinance_only_when_used(
    tmp_path, code: str, expects_import: bool
) -> None:
    path = tools.create_notebook("risk_models.sample_cov", code, str(tmp_path))
    nb = nbformat.read(path, as_version=4)
    imports = nb.cells[1].source
    assert ("import yfinance as yf" in imports) is expects_import


def test_code_prompt_fetches_real_data_and_labels_simulated_fallback() -> None:
    """Guard the prompt rules that produced the reported bad snippets."""
    prompt = backends._PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT
    assert "synthesize a small" not in prompt  # the old instruction to fake prices
    assert "Never type market numbers in from memory" in prompt
    assert 'result["data_source"]' in prompt
    assert '"SIMULATED' in prompt
    assert "market_implied_prior_returns(market_caps, delta, S" in prompt
    assert '"SPY", period="10y"' in prompt
    assert "^IRX" in prompt
    assert "risk_free_rate=RF" in prompt


def test_quantlib_prompt_shares_the_market_data_rules() -> None:
    prompt = backends._QUANTLIB_CODE_EXTRACTION_SYSTEM_PROMPT
    assert backends._MARKET_DATA_RULES in prompt
    assert "`yf` (yfinance)" in prompt


# -- fetch with a labelled fallback -----------------------------------------

_FALLBACK_SNIPPET = """
try:
    raise ValueError("no data for AAPL")
except Exception as e:
    prices = np.random.default_rng(0).normal(size=3)
    source = f"SIMULATED (fake data, not real market prices): yfinance fetch failed: {e}"
result = {"n": len(prices), "data_source": source}
"""


def test_sandbox_allows_try_except_fallback(fake_yfinance) -> None:
    """`Exception`/`ValueError` must exist, or the fallback itself crashes."""
    result, _ = tools.run_snippet(_FALLBACK_SNIPPET)
    assert result["data_source"].startswith("SIMULATED")
    assert "no data for AAPL" in result["data_source"]


def test_quantlib_sandbox_binds_yf_and_numpy(monkeypatch) -> None:
    from jupyter_ai_fintyche import quantlib_tools

    pytest.importorskip("QuantLib")
    module = types.ModuleType("yfinance")
    module.marker = "fake-yfinance"  # type: ignore[attr-defined]
    monkeypatch.setattr(quantlib_tools, "_load_yfinance", lambda: module)
    result, _ = quantlib_tools.run_snippet("result = (yf.marker, np.sqrt(4.0))")
    assert result == ("fake-yfinance", 2.0)

    result, _ = quantlib_tools.run_snippet(_FALLBACK_SNIPPET)
    assert result["data_source"].startswith("SIMULATED")


def test_quantlib_snippet_using_yf_without_yfinance_gives_install_hint(monkeypatch) -> None:
    from jupyter_ai_fintyche import quantlib_tools

    pytest.importorskip("QuantLib")
    monkeypatch.setattr(quantlib_tools, "_load_yfinance", lambda: None)
    with pytest.raises(quantlib_tools.QuantLibUnavailable, match="pip install yfinance"):
        quantlib_tools.run_snippet('result = yf.Ticker("AAPL")')


class _Sink(persona.FinTychePersona):
    def __init__(self):
        self._history = []
        self.messages: list[str] = []

    def send_message(self, body: str) -> None:
        self.messages.append(body)


@pytest.mark.parametrize(
    ("source", "warned"),
    [
        ("SIMULATED (fake data): requested by the user", True),
        ("yfinance", False),
    ],
)
def test_simulated_result_is_shown_with_a_warning(fake_yfinance, source, warned) -> None:
    agent = _Sink()
    backend = persona._BACKENDS["pyportpf"]
    agent._execute_snippet(
        backend, "EfficientFrontier", f"result = {{'data_source': {source!r}}}"
    )
    assert ("simulated (fake) data" in agent.messages[-1]) is warned


# -- the code prompt covers the whole library, not one model ----------------


@pytest.mark.parametrize(
    "form",
    [
        "pypfopt.EfficientFrontier(mu, S",
        "pypfopt.EfficientSemivariance(mu, returns)",
        "pypfopt.EfficientCVaR(mu, returns",
        "pypfopt.EfficientCDaR(mu, returns",
        "pypfopt.HRPOpt(returns)",
        "pypfopt.CLA(mu, S)",
        "pypfopt.BlackLittermanModel(S, pi=pi",
        "pypfopt.DiscreteAllocation(weights",
        "capm_return(prices, market_prices=",
        "add_sector_constraints(",
        "objective_functions.L2_reg",
        "returns_from_prices(prices)",
        "An optimizer instance solves once",
    ],
)
def test_code_prompt_covers_each_part_of_the_library(form: str) -> None:
    assert form in backends._PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT


def test_hrp_optimizes_despite_newer_scipy() -> None:
    """PyPortfolioOpt 1.6.0 reads a SciPy private SciPy 1.18 removed."""
    result, _ = tools.run_snippet(
        "rng = np.random.default_rng(0)\n"
        "returns = pd.DataFrame(rng.normal(0, 0.01, (300, 3)), columns=list('ABC'))\n"
        "hrp = pypfopt.HRPOpt(returns)\n"
        "result = dict(hrp.optimize())"
    )
    assert set(result) == {"A", "B", "C"}
    assert sum(result.values()) == pytest.approx(1.0)
