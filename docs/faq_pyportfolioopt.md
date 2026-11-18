# PyPortfolioOpt Commands FAQ

Answers to common questions about how FinTyche's `/pyportpf-*` chat commands
work under the hood. File and line references point at the extension source in
[`jupyter_ai_fintyche/`](../jupyter_ai_fintyche/). This mirrors
[`faq_quantlib.md`](faq_quantlib.md): both libraries share one implementation
(the generic command handlers in [`core.py`](../jupyter_ai_fintyche/core.py)),
differing only in a per-library **backend** (`_BACKENDS`) that supplies the
introspection module, the code alias, the slash-command prefix, and two LLM
system prompts.

## Q: Where does `/pyportpf-list` get the list of classes/functions from? Is it stored locally or fetched from a web search?

**It's entirely local — no web search is involved.** The list is generated at
runtime by Python introspection of the installed `PyPortfolioOpt` package
(imported as `pypfopt`).

The flow:

1. [`core.py`](../jupyter_ai_fintyche/core.py) calls
   `pyportfolioopt_tools.list_entries(query=...)` through the PyPortfolioOpt
   backend.
2. `list_entries` imports the locally installed package (`import pypfopt`),
   raising `PyPortfolioOptUnavailable` if it isn't `pip install`ed.
3. Unlike QuantLib — whose entire API is a flat `dir(ql)` namespace —
   PyPortfolioOpt is a **package**: top-level optimizer classes
   (`EfficientFrontier`, `CLA`, `HRPOpt`, `BlackLittermanModel`, …) plus
   submodules of functions. So `list_entries` catalogs the top-level public
   names **and** walks the main submodules (`expected_returns`, `risk_models`,
   `objective_functions`, `black_litterman`, `plotting`), surfacing their public
   functions as **dotted names** (e.g. `risk_models.sample_cov`,
   `expected_returns.mean_historical_return`). It only includes functions/classes
   actually defined in `pypfopt` (by checking `__module__`), so re-exported
   third-party names (numpy, pandas, cvxpy) are skipped.
4. It optionally filters by your case-insensitive substring query, sorts, caps
   the result at `limit` (default 100), and uses `inspect` to classify each name
   as class/function/module/value and grab the first docstring line as a summary.

Because the catalog comes from introspecting whatever PyPortfolioOpt is installed
in your environment, it is **always in sync with your installed version** and is
**not** stored as a data file in this repo. It is not cached — every
`/pyportpf-list` invocation re-runs the walk fresh.

## Q: Where does `/pyportpf-doc` get the exact symbol it needs? Does it use a skill file?

When you give `/pyportpf-doc` a plain-text word problem (rather than an exact
symbol name), the decision is made by a **live LLM call guided by a hardcoded
Python system prompt** — **not** a skill file.

The flow (`_handle_doc` → `_resolve_doc_symbol`, both generic over the backend):

1. If the query already resolves to an exact PyPortfolioOpt symbol
   (`pyportfolioopt_tools.has_symbol`, which walks dotted attribute paths from
   the `pypfopt` package), it skips the LLM and documents it directly.
2. Otherwise it calls the LLM via `litellm.acompletion` with:
   - **system message** = the backend's `doc_resolution_prompt`
     (`_PYPORTFOLIOOPT_DOC_RESOLUTION_SYSTEM_PROMPT`)
   - **user message** = `"Word problem: {your text}"`
3. The model returns a JSON object naming the single best PyPortfolioOpt symbol
   (a dotted submodule path such as `expected_returns.mean_historical_return` is
   allowed) plus an optional `note` on how to use it.

The symbol-selection guidance lives in the
`_PYPORTFOLIOOPT_DOC_RESOLUTION_SYSTEM_PROMPT` constant in
[`backends.py`](../jupyter_ai_fintyche/backends.py). It tells the model to name
the **primary optimizer class** for a mean-variance task (e.g.
`EfficientFrontier`) or the relevant **estimator** for an estimation task (e.g.
`expected_returns.mean_historical_return`, `risk_models.CovarianceShrinkage`),
and to name companion functions in the `note`. The final doc text (signature +
docstring) then comes from local `inspect` introspection via
`pyportfolioopt_tools.get_doc`.

**Does it use a skill file? No — not for `/pyportpf-doc`.** As with QuantLib, the
skill file [`skills/system_prompt.md`](../jupyter_ai_fintyche/skills/system_prompt.md)
is only loaded as the persona's **general system prompt** for free-form
quant-finance chat. The `/pyportpf-doc` symbol-resolution path uses its own
separate, hardcoded prompt.

## Q: How does `/pyportpf-run` work, and how does it determine the sequence of PyPortfolioOpt calls needed to produce an allocation?

The command is `pyportpf-run <name> <word problem>`. `_handle_run` splits it into
the symbol `name` and free-text `prompt_text`, validates the name against the
real library (`pyportfolioopt_tools.has_symbol`, reached through the
PyPortfolioOpt backend), then takes one of two paths.

### Path A — no word problem (direct literal call)

If you type only a name and literal args (or no args), it skips the LLM and calls
`_execute_run` → `pyportfolioopt_tools.call_function`, which `eval`s the single
expression `pypfopt.<name>(<args_str>)` in a restricted namespace. **No
sequencing happens here** — it's one call. (For most optimizers this needs input
data, so the word-problem path below is the usual one.)

### Path B — word problem (multi-step optimization)

This is where a *sequence* of PyPortfolioOpt calls is determined. The flow is
`_handle_run` → `_extract_and_dispatch` → `_extract_snippet` →
`_execute_snippet`.

**The sequence is decided entirely by the LLM, not by any hardcoded recipe in the
code:**

1. **Grounding context.** `_extract_snippet` fetches the target symbol's real
   signature and docstring via `pyportfolioopt_tools.get_doc(name)`, and
   `_build_extraction_prompt` packages `{symbol, signature, docstring, word
   problem}` as the user message.
2. **The system prompt teaches the optimization pattern.** The
   `_PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT` constant is where the "how to
   get to an allocation" knowledge lives — as prose, not code. It tells the model
   that `pypfopt`, `pd` (pandas) and `np` (numpy) are pre-imported, that portfolio
   optimization is **data-driven**, and walks it through a general workflow,
   each step with a default: expected returns (historical mean, EMA, CAPM, or
   Black-Litterman), a risk model (Ledoit-Wolf shrinkage by default), an
   optimizer chosen by the risk the user cares about (`EfficientFrontier` for
   variance, `EfficientSemivariance` for downside risk, `EfficientCVaR` /
   `EfficientCDaR` for tail risk and drawdowns, `HRPOpt`, `CLA`), constraints
   and extra objectives added before solving, and the output
   (`clean_weights()`, `portfolio_performance()`, `DiscreteAllocation`). It also
   states the rules the library enforces, such as one objective per optimizer
   instance. When the user provides **no** price data, the
   prompt instructs the model to **fetch real adjusted prices with yfinance**
   (`yf`, always pre-bound; yfinance is a core dependency) and forbids
   market numbers recalled from memory. For Black-Litterman it
   spells out the three inputs to the market prior: risk aversion from a market
   index such as SPY, market caps from `yf.Ticker(t).info`, and the covariance
   matrix, all with one shared risk-free rate (the 13-week T-bill yield, `^IRX`,
   unless the user gives one). Simulated data is used only when the user asks
   for it or a fetch fails; the snippet then reports it in
   `result["data_source"]`, and FinTyche shows the result under a warning that
   it uses fake data. The same rules (`_MARKET_DATA_RULES`) apply to QuantLib
   snippets, whose sandbox also binds `yf`. It assigns the final answer to
   `result` and
   replies with a JSON object (`{"status":"ok","code":...}` or
   `{"status":"missing","question":...}`).
3. **Missing-info handshake.** If the model returns `status: "missing"`,
   `_extract_and_dispatch` stores `self._pending_run` (including which backend it
   belongs to) and asks a clarifying question; your next message is merged in and
   re-extracted (`_handle_pending_run`) rather than requiring a full retype.
4. **Execution.** On `status: "ok"`, `_execute_snippet` runs the whole
   multi-statement snippet via `run_snippet`, which uses a single `exec`
   namespace with `pypfopt`, `pd`, `np`, and `yf` pre-bound, restricted builtins (no direct
   import/file/process builtins, and not a security sandbox), captured stdout, and a required `result`
   variable. It replies with both the generated code and the computed value.

### Summary

- **Path A** (`call_function`): one literal `eval` — no sequence logic.
- **Path B** (`run_snippet`): the LLM authors the ordered sequence of
  PyPortfolioOpt calls. The domain knowledge of *what* sequence produces an
  allocation (estimate returns → estimate covariance → build optimizer → solve
  objective → clean weights → performance) is encoded as **instructions in
  `_PYPORTFOLIOOPT_CODE_EXTRACTION_SYSTEM_PROMPT`**, combined with the live
  docstring of the target symbol. The Python code only orchestrates, validates
  the symbol, handles missing-info follow-ups, and runs it with restricted builtins (not a sandbox).

One consequence, shared with QuantLib: because the sequence is model-generated
and run with `exec`, correctness depends on the LLM getting the wiring right, and
— for PyPortfolioOpt specifically — on the market data the snippet fetches.
Fetched data reflects whatever Yahoo Finance returns at run time, so results
change from run to run; pass your own price or returns table in the word problem
when you need a fixed, reproducible input.

## Q: Why does PyPortfolioOpt keep the `pyportpf-` prefix on its bare commands, while QuantLib gets the bare verbs `list`/`doc`/`run`/`notebook`?

Because the two libraries share the same five verbs, and a bare `list` (or `run`,
etc.) can only route to one of them. For backward compatibility QuantLib keeps
the bare verbs; PyPortfolioOpt is reached through its `/pyportpf-*` slash forms
or the equivalent bare spellings (`pyportpf-list`, `pyportpf-doc`, …). See
`_build_command_aliases` in [`backends.py`](../jupyter_ai_fintyche/backends.py),
which builds the `{spelling → (backend, verb)}` dispatch table. The bare-symbol
documentation shortcut (typing just `EfficientFrontier`) checks QuantLib first,
then PyPortfolioOpt (`_symbol_backend`).
