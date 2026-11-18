# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""The library backends FinTyche exposes, independent of any frontend.

Each backend pairs a library's introspection/execution module with the LLM
prompts FinTyche uses for it. Both the Jupyter AI persona and the CLI read
this module, so it imports nothing from Jupyter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import pyportfolioopt_tools, quantlib_tools

# ---------------------------------------------------------------------------
# Library backends
#
# Each backend bundles everything that differs between the libraries FinTyche
# exposes: the introspection/execution module (all of which share the same API —
# `list_entries`, `get_doc`, `has_symbol`, `call_function`, `run_snippet`,
# `create_notebook`, and an `Unavailable` exception), the `import ... as` alias
# used inside generated code, the slash-command prefix, and the two LLM system
# prompts (one to pick a symbol from a word problem, one to turn a word problem
# into a runnable snippet). Handlers in `core.py` are written once and dispatched per
# backend, so adding another library is a matter of adding a backend here.
# ---------------------------------------------------------------------------


# How generated snippets get market data, shared by both backends. `yf`
# (yfinance) is bound in every run namespace. Simulated data is allowed only when
# the user asks for it or a fetch fails, and the snippet must say so in
# `result["data_source"]`, which FinTyche shows as a warning above the result.
_MARKET_DATA_RULES = (
    "MARKET DATA:\n"
    "- Values the user gives (prices, spot, volatility, rates, market caps) are "
    "used as given.\n"
    "- Any market input the user did not give and that names a real "
    "instrument is fetched with `yf` (yfinance), e.g. "
    '`yf.download(tickers, period="2y", auto_adjust=True, progress=False)["Close"]` '
    "for adjusted prices, "
    '`yf.Ticker(t).history(period="5d")["Close"].iloc[-1]` for the latest '
    'price, `yf.Ticker(t).info["marketCap"]` for market caps, and '
    '`float(yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]) / 100` '
    "for the risk-free rate (13-week T-bill). Never type market numbers in "
    "from memory.\n"
    "- Use simulated data only if the user asks for it, or if a fetch fails or "
    "comes back empty. Wrap each input's fetch in its own "
    "`try`/`except Exception as e:` (never a bare `except:`), and in the "
    "`except` branch replace only that input, keeping every input that was "
    "fetched. Generate simulated values reproducibly with "
    "`np.random.default_rng(0)` (when `np` is available).\n"
    "- When the snippet fetches or simulates market inputs, make `result` a "
    'dict and include `result["data_source"]`: `"yfinance"` when they were '
    "fetched (leave `data_source` out when nothing was fetched), otherwise a "
    'string starting with `"SIMULATED"` that says which inputs are fake and why '
    '(e.g. `f"SIMULATED (fake data, not real market prices): yfinance fetch '
    'failed: {e}"`, or `"SIMULATED (fake data): requested by the user"`).\n\n'
)


_QUANTLIB_CODE_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in the QuantLib quantitative finance library's Python "
    "bindings (imported as `ql`). Given a target QuantLib class or function and "
    "a free-form word problem, write a short Python snippet that uses QuantLib "
    "to compute the answer the user is asking for, centered on `ql.<name>`.\n\n"
    "The snippet runs in a namespace where `ql` (and `QuantLib`) and `yf` "
    "(yfinance) are already imported — do NOT write any `import` statements. "
    "Assign the final answer to a variable named `result`. Write as many statements as you need: many "
    "QuantLib objects are not useful on their own and require follow-up wiring. "
    "In particular, constructing an instrument does NOT price it — e.g. "
    "`ql.VanillaOption(payoff, exercise)` only builds the option; to get a "
    "price you must build a stochastic process from the market data (spot, "
    "risk-free rate, dividend yield, volatility), create a pricing engine such "
    "as `ql.AnalyticEuropeanEngine(process)`, attach it with "
    "`option.setPricingEngine(engine)`, and read `option.NPV()`. Set the "
    "evaluation date with `ql.Settings.instance().evaluationDate` whenever the "
    "problem is time-sensitive.\n\n"
    "Match every constructor's argument list exactly. QuantLib has many "
    "similarly-named classes whose signatures differ — for instance "
    "`ql.BlackScholesProcess` takes three arguments (spot, risk-free, vol) while "
    "`ql.BlackScholesMertonProcess` takes four (spot, dividend, risk-free, vol). "
    "Never carry an argument list over from a class with a similar name.\n\n"
    + _MARKET_DATA_RULES
    + "For an option on a real ticker, fetch the spot as above and estimate "
    "volatility from one year of daily log returns annualized by sqrt(252), "
    "unless the user gives them. When the problem uses no market inputs at all "
    "(e.g. date arithmetic), `result` need not be a dict.\n\n"
    "Respond with ONLY a JSON object and no other text, in one of these two shapes:\n"
    '  {"status": "ok", "code": "<python snippet that ends by assigning `result`>", '
    '"changes": "<only when editing previous code: one sentence on what changed and why>"}\n'
    '  {"status": "missing", "question": "<a single clarifying question asking for the missing value(s)>"}\n\n'
    'Use "missing" only when the word problem genuinely does not provide enough '
    "information to compute the answer and the missing values cannot be "
    "fetched (e.g. contract terms such as strike or maturity). Use "
    "the `ql.` prefix for every QuantLib name (e.g. `ql.Date`, `ql.Actual360()`, "
    "`ql.TARGET()`, `ql.Option.Call`)."
)


_QUANTLIB_DOC_RESOLUTION_SYSTEM_PROMPT = (
    "You are an expert in the QuantLib quantitative finance library's Python "
    "bindings (imported as `ql`). Given a free-form description of a "
    "quant-finance task (a word problem), identify the single most appropriate "
    "QuantLib class or function a user should look at to accomplish it. For a "
    "pricing task, name the primary instrument class the user would price "
    "(e.g. `VanillaOption` for a European or American equity option).\n\n"
    "Respond with ONLY a JSON object and no other text, in one of these two shapes:\n"
    '  {"status": "ok", "name": "<exact QuantLib symbol, no `ql.` prefix>", "note": "<optional one- or two-sentence explanation of how to use it, naming any companion classes needed such as a payoff, exercise, stochastic process, or pricing engine>"}\n'
    '  {"status": "missing", "question": "<a single clarifying question asking for the missing detail(s)>"}\n\n'
    "The `name` must be the exact attribute name on the `QuantLib` module "
    "(e.g. `VanillaOption`, `AnalyticEuropeanEngine`, `BlackScholesMertonProcess`), "
    'not a dotted path. Use "missing" only when the description is too vague to '
    "choose a symbol.\n\n"
    "The `note` describes the symbol and the companion classes it needs. It must "
    "NOT say that you are running, computing, or about to run anything, and must "
    "not promise a result — whether the code is executed is decided after you "
    "reply, not by you."
)


_PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert in the PyPortfolioOpt portfolio-optimization library's "
    "Python API (imported as `pypfopt`). Given a target PyPortfolioOpt class or "
    "function and a free-form word problem, write a short Python snippet that "
    "uses PyPortfolioOpt to compute the answer the user is asking for, centered "
    "on the given symbol reached from the `pypfopt` package (e.g. "
    "`pypfopt.EfficientFrontier` or `pypfopt.expected_returns.mean_historical_return`).\n\n"
    "The snippet runs in a namespace where `pypfopt`, `pd` (pandas), `np` "
    "(numpy) and `yf` (yfinance) are already imported — do NOT write any "
    "`import` statements. Assign the final answer to a variable named `result`. "
    "`print` is not available; put any intermediate values worth showing (such "
    "as `delta` or the prior returns) into `result` as a dict. "
    "Write as many statements as you need.\n\n"
    + _MARKET_DATA_RULES
    + "PRICES: portfolio work starts from historical prices; fetch them unless "
    "the user gave them (the period or dates the user asked for, else 2 years). "
    "Optimizers that model downside or tail risk, and HRP, take daily returns: "
    "`returns = pypfopt.expected_returns.returns_from_prices(prices)`. If the "
    'user gave no tickers, reply with "missing" and ask for them.\n\n'
    "RISK-FREE RATE: define `RF` once near the top. Use the user's value if "
    "they gave one; otherwise fetch the 13-week T-bill yield as above, falling "
    "back to `RF = 0.04` (noted in `data_source`) only if the fetch fails. Pass "
    "`risk_free_rate=RF` to every call that accepts it, so estimates, "
    "objectives, and reported Sharpe ratios agree.\n\n"
    "WORKFLOW: build only the steps the request needs, using the default "
    "(listed first) unless the user asks for something else.\n"
    "1. Expected returns `mu` — `pypfopt.expected_returns.mean_historical_return(prices)`; "
    "`ema_historical_return(prices, span=...)` to weight recent data; "
    "`capm_return(prices, market_prices=<SPY prices>, risk_free_rate=RF)` for "
    "CAPM; or Black-Litterman posterior returns when the user states views "
    "(below).\n"
    "2. Risk model `S` — `pypfopt.risk_models.CovarianceShrinkage(prices).ledoit_wolf()`; "
    "`risk_models.sample_cov(prices)`, `exp_cov(prices)`, or "
    "`semicovariance(prices)` when asked.\n"
    "3. Optimizer, chosen by the risk the user cares about:\n"
    "   - variance (default): `ef = pypfopt.EfficientFrontier(mu, S, "
    "weight_bounds=(0, 1))`, then one of `max_sharpe(risk_free_rate=RF)`, "
    "`min_volatility()`, `max_quadratic_utility(risk_aversion=...)`, "
    "`efficient_risk(target_volatility)`, `efficient_return(target_return)`;\n"
    "   - downside risk: `pypfopt.EfficientSemivariance(mu, returns)` with "
    "`min_semivariance()` or `efficient_return(...)`;\n"
    "   - tail risk: `pypfopt.EfficientCVaR(mu, returns, beta=0.95)` with "
    "`min_cvar()`, or `pypfopt.EfficientCDaR(mu, returns, beta=0.95)` with "
    "`min_cdar()` for drawdowns;\n"
    "   - no return forecasts, or diversification by clustering: "
    "`hrp = pypfopt.HRPOpt(returns)` then `hrp.optimize()`;\n"
    "   - the whole frontier: `pypfopt.CLA(mu, S)`.\n"
    "4. Constraints and extra objectives, added BEFORE the solve call: "
    "`weight_bounds=(-1, 1)` to allow shorting; "
    "`opt.add_sector_constraints(sector_mapper, sector_lower, sector_upper)`; "
    "`opt.add_constraint(lambda w: w[i] <= 0.1)`; "
    "`opt.add_objective(pypfopt.objective_functions.L2_reg, gamma=0.1)` to "
    "avoid many zero weights; `objective_functions.transaction_cost` when "
    "rebalancing from given weights.\n"
    "5. Output — `weights = opt.clean_weights()` and `opt.portfolio_performance(...)`: "
    "`(return, volatility, Sharpe)` with `risk_free_rate=RF` for "
    "EfficientFrontier, EfficientSemivariance, CLA, and HRPOpt; "
    "`(return, CVaR)` or `(return, CDaR)` with no `risk_free_rate` for "
    "EfficientCVaR and EfficientCDaR. For share counts: "
    "`pypfopt.DiscreteAllocation(weights, pypfopt.get_latest_prices(prices), "
    "total_portfolio_value=<amount>).greedy_portfolio()` (or `lp_portfolio()`), "
    "which returns `(allocation, leftover_cash)`.\n\n"
    "RULES THE LIBRARY ENFORCES:\n"
    "- An optimizer instance solves once. To compare objectives (e.g. max "
    "Sharpe vs. min volatility), build a new instance for each.\n"
    "- `max_sharpe` needs at least one asset whose expected return exceeds "
    "`RF`, and `efficient_return`/`efficient_risk` need a reachable target; "
    "otherwise they raise `ValueError`, which should be reported, not hidden.\n\n"
    "BLACK-LITTERMAN, when the user states views on returns, blended with "
    "market equilibrium:\n"
    "- Risk aversion comes from a market index, not the assets, over a long "
    'window: `market_prices = yf.download("SPY", period="10y", '
    'auto_adjust=True, progress=False)["Close"].squeeze()`, then '
    "`delta = pypfopt.black_litterman.market_implied_risk_aversion("
    "market_prices, risk_free_rate=RF)` (one positive number).\n"
    "- `pi = pypfopt.black_litterman.market_implied_prior_returns(market_caps, "
    "delta, S, risk_free_rate=RF)` — all three positional arguments are "
    "required, and the first is market caps, not prices. If market caps can't "
    "be fetched, use equal caps for every ticker (never random ones) and say so "
    "in `data_source`.\n"
    "- `bl = pypfopt.BlackLittermanModel(S, pi=pi, absolute_views=viewdict)` "
    "(add `omega=\"idzorek\", view_confidences=[...]` when the user states "
    "confidence), then `bl.bl_returns()` and `bl.bl_cov()` feed the optimizer.\n\n"
    "SIGNATURES: pass every required argument of every PyPortfolioOpt call you "
    "make, not only the target symbol's. If you are unsure of a function's "
    "signature, use the forms shown above rather than guessing.\n\n"
    "Respond with ONLY a JSON object and no other text, in one of these two shapes:\n"
    '  {"status": "ok", "code": "<python snippet that ends by assigning `result`>", '
    '"changes": "<only when editing previous code: one sentence on what changed and why>"}\n'
    '  {"status": "missing", "question": "<a single clarifying question asking for the missing value(s)>"}\n\n'
    'Use "missing" when the word problem does not say which assets to use or '
    "what to optimize. Use the `pypfopt.` prefix for every PyPortfolioOpt name."
)


_PYPORTFOLIOOPT_DOC_RESOLUTION_SYSTEM_PROMPT = (
    "You are an expert in the PyPortfolioOpt portfolio-optimization library's "
    "Python API (imported as `pypfopt`). Given a free-form description of a "
    "portfolio task (a word problem), identify the single most appropriate "
    "PyPortfolioOpt class or function a user should look at to accomplish it. "
    "For an optimization, name the optimizer for the risk the user cares about: "
    "`EfficientFrontier` (mean-variance: max Sharpe, min volatility, target "
    "return or risk), `EfficientSemivariance` (downside risk), `EfficientCVaR` "
    "or `EfficientCDaR` (tail risk, drawdowns), `HRPOpt` (hierarchical risk "
    "parity, no return forecasts), `CLA` (the full frontier), or "
    "`BlackLittermanModel` (blending the user's views with market "
    "equilibrium). For converting weights into share counts, name "
    "`DiscreteAllocation`. For an estimation task, name the estimator (e.g. "
    "`expected_returns.mean_historical_return` or `risk_models.CovarianceShrinkage`).\n\n"
    "Respond with ONLY a JSON object and no other text, in one of these two shapes:\n"
    '  {"status": "ok", "name": "<PyPortfolioOpt symbol reachable from the `pypfopt` package, e.g. `EfficientFrontier` or `expected_returns.mean_historical_return`; a dotted submodule path is allowed, but no `pypfopt.` prefix>", "note": "<optional one- or two-sentence explanation of how to use it, naming any companion functions needed such as an expected-returns estimator or a covariance model>"}\n'
    '  {"status": "missing", "question": "<a single clarifying question asking for the missing detail(s)>"}\n\n'
    "The `name` must resolve as an attribute path on the `pypfopt` package "
    "(e.g. `EfficientFrontier`, `HRPOpt`, `risk_models.CovarianceShrinkage`) and "
    'must NOT include the `pypfopt.` prefix. Use "missing" only when the '
    "description is too vague to choose a symbol.\n\n"
    "The `note` describes the symbol and the companion functions it needs. It "
    "must NOT say that you are running, computing, or about to run anything, and "
    "must not promise a result — whether the code is executed is decided after "
    "you reply, not by you."
)


@dataclass(frozen=True)
class LibraryBackend:
    """Everything that differs between the libraries FinTyche exposes."""

    key: str  # stable identifier, e.g. "quantlib" | "pyportpf"
    label: str  # human name, e.g. "QuantLib" | "PyPortfolioOpt"
    pip_name: str  # PyPI distribution name, for install hints
    tools: Any  # module exposing the shared introspection/execution API
    module_alias: str  # `import ... as <alias>` used inside generated code
    slash_prefix: str  # slash-command stem, e.g. "quantlib" -> "/quantlib-list"
    doc_resolution_prompt: str
    code_extraction_prompt: str

    def slash(self, verb: str) -> str:
        return f"/{self.slash_prefix}-{verb}"


_BACKENDS: dict[str, LibraryBackend] = {
    "quantlib": LibraryBackend(
        key="quantlib",
        label="QuantLib",
        pip_name="QuantLib",
        tools=quantlib_tools,
        module_alias="ql",
        slash_prefix="quantlib",
        doc_resolution_prompt=_QUANTLIB_DOC_RESOLUTION_SYSTEM_PROMPT,
        code_extraction_prompt=_QUANTLIB_CODE_EXTRACTION_SYSTEM_PROMPT,
    ),
    "pyportpf": LibraryBackend(
        key="pyportpf",
        label="PyPortfolioOpt",
        pip_name="PyPortfolioOpt",
        tools=pyportfolioopt_tools,
        module_alias="pypfopt",
        slash_prefix="pyportpf",
        doc_resolution_prompt=_PYPORTFOLIOOPT_DOC_RESOLUTION_SYSTEM_PROMPT,
        code_extraction_prompt=_PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT,
    ),
}

_VERBS = ("list", "doc", "run", "notebook", "help")

_SLASH_DESCRIPTIONS = {
    "quantlib": {
        "list": "List QuantLib functions/classes, optionally filtered by keyword.",
        "doc": "Find the right QuantLib symbol for a task and show its docs.",
        "run": "Run QuantLib.<name>(...), describing the arguments as a free-form word problem.",
        "notebook": "Create a notebook that solves a QuantLib word problem.",
        "help": "Show FinTyche's QuantLib command reference.",
    },
    "pyportpf": {
        "list": "List PyPortfolioOpt classes/functions, optionally filtered by keyword.",
        "doc": "Find the right PyPortfolioOpt symbol for a task and show its docs.",
        "run": "Run a PyPortfolioOpt symbol from a free-form word problem and show the result.",
        "notebook": "Create a notebook that solves a PyPortfolioOpt word problem.",
        "help": "Show FinTyche's PyPortfolioOpt command reference.",
    },
}


def _build_command_aliases() -> dict[str, tuple[str, str]]:
    """Map every command spelling onto a `(backend_key, verb)` pair.

    - QuantLib keeps the bare verbs (`list`, `doc`, `run`, `notebook`, `help`)
      as well as its `/quantlib-*` slash forms — this is backward compatible.
    - PyPortfolioOpt is reached through its `/pyportpf-*` slash forms, and the
      matching bare spellings (`pyportpf-list`, …). The bare verbs cannot be
      shared between two libraries, so PyPortfolioOpt keeps its prefix.
    """
    aliases: dict[str, tuple[str, str]] = {}
    for verb in _VERBS:
        # QuantLib owns the bare verbs (backward compatible).
        aliases[verb] = ("quantlib", verb)
        for backend in _BACKENDS.values():
            aliases[backend.slash(verb)] = (backend.key, verb)
            # Bare spelling of the slash form, e.g. "pyportpf-list".
            aliases[f"{backend.slash_prefix}-{verb}"] = (backend.key, verb)
    return aliases



_COMMAND_ALIASES = _build_command_aliases()
