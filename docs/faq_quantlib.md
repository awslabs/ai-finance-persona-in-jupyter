# QuantLib Commands FAQ

Answers to common questions about how FinTyche's `/quantlib-*` chat commands
work under the hood. File and line references point at the extension source in
[`jupyter_ai_fintyche/`](../jupyter_ai_fintyche/). The parallel
[`faq_pyportfolioopt.md`](faq_pyportfolioopt.md) covers the `/pyportpf-*`
commands, which share the same generic handlers via a per-library backend.

## Q: Where does `/quantlib-list` get the list of functions from? Is it stored locally or fetched from a web search?

**It's entirely local — no web search is involved.** The list is generated at
runtime by Python introspection of the installed `QuantLib` package.

The flow:

1. [`core.py`](../jupyter_ai_fintyche/core.py) calls
   `quantlib_tools.list_entries(query=...)`.
2. `list_entries` imports the locally installed QuantLib module
   (`import QuantLib as ql`), raising `QuantLibUnavailable` if it isn't
   `pip install`ed.
3. It builds the catalog with Python's built-in `dir(ql)` — the public
   attributes of the module (names starting with `_` are filtered out).
4. It optionally filters by your case-insensitive substring query, sorts,
   caps the result at `limit` (default 100), and uses `inspect` to classify
   each name as class/function/value and grab the first docstring line as a
   summary.

Because the catalog comes from `dir()` on whatever QuantLib is installed in
your environment, it is **always in sync with your installed version** and is
**not** stored as a data file in this repo. Note that it is not cached — every
`/quantlib-list` invocation re-runs `dir()` and `inspect.getdoc()` fresh.

## Q: Where does `/quantlib-doc` get the exact function and approach it needs to decide on pricing a security? Does it use a skill file?

When you give `/quantlib-doc` a plain-text word problem (rather than an exact
symbol name), the decision is made by a **live LLM call guided by a hardcoded
Python system prompt** — **not** a skill file.

The flow (`_handle_doc` → `_resolve_doc_symbol`):

1. If the query is already an exact QuantLib symbol, it skips the LLM and
   documents it directly.
2. Otherwise it calls the LLM via `litellm.acompletion` with:
   - **system message** = the backend's `doc_resolution_prompt`
     (`_QUANTLIB_DOC_RESOLUTION_SYSTEM_PROMPT`)
   - **user message** = `"Word problem: {your text}"`
3. The model returns a JSON object naming the single best QuantLib symbol plus
   an optional `note` on how to use it.

The "function + approach" instructions live in the
`_QUANTLIB_DOC_RESOLUTION_SYSTEM_PROMPT` constant in
[`backends.py`](../jupyter_ai_fintyche/backends.py). That prompt tells
the model to, for a pricing task, name the **primary instrument class** the user
would price (e.g. `VanillaOption` for a European/American equity option), and in
the `note` to name the **companion classes** needed — a payoff, exercise,
stochastic process, or pricing engine. So the pricing "approach" pattern is baked
into that string constant, while the actual reasoning happens in the LLM at
runtime. The final doc text (signature + docstring) then comes from local
`inspect` introspection via `quantlib_tools.get_doc`.

**Does it use a skill file? No — not for `/quantlib-doc`.** There *is* a skill
file, [`skills/system_prompt.md`](../jupyter_ai_fintyche/skills/system_prompt.md),
but it is only loaded as the persona's **general system prompt** (via
`_load_system_prompt`) for free-form quant-finance chat. The `/quantlib-doc`
symbol-resolution path uses its own separate, hardcoded prompt and never reads
that skill file.

## Q: What role does `skills/quant_skills.py` play?

It's a **standalone helper library** of quant-analytics functions — not part of
the command-handling logic. It defines three classes:

- **`RiskAnalytics`** — `sharpe_ratio`, `max_drawdown`, `value_at_risk`
- **`AlphaFactors`** — `rsi`, `macd`
- **`VectorizedBacktest`** — `run_sma_crossover`

**No Python code in the extension imports or calls `quant_skills`.** It is
referenced only in documentation and prompt text:

1. `README.md` and `INSTRUCTIONS.md` — docs showing users
   `from quant_skills import RiskAnalytics, AlphaFactors, VectorizedBacktest`.
2. [`skills/system_prompt.md`](../jupyter_ai_fintyche/skills/system_prompt.md)
   — the persona's general system prompt *tells the LLM these helpers exist*
   (alpha factors, risk analytics, backtesting).

So it's a library meant to be **imported inside notebook/chat code cells** —
either by the user directly, or in code the LLM writes when answering a
free-form quant question. It plays **no role** in `/quantlib-list`,
`/quantlib-doc`, `/quantlib-run`, or `/quantlib-notebook`, which all go through
`quantlib_tools.py` and introspect the real QuantLib package.

Caveat: for `import quant_skills` to work in a running kernel, the file's
directory must be on the kernel's Python path. It lives inside the installed
package at `jupyter_ai_fintyche/skills/`, while `INSTRUCTIONS.md` refers to it
at a different location (`~/skills/quant-finance/quant_skills.py`), so whether a
bare `from quant_skills import ...` succeeds depends on the user's environment.

## Q: How does `/quantlib-run` work, and how does it determine the sequence of QuantLib functions needed to compute a final price?

The command is `run <QuantLib-name> <word problem>`. `_handle_run` splits it into
the symbol `name` and free-text `prompt_text`, validates the name against the
real library (`quantlib_tools.has_symbol`, reached through the QuantLib backend),
then takes one of two paths.

### Path A — no word problem (direct literal call)

If you type only a name and literal args (or no args), it skips the LLM and calls
`_execute_run` → `quantlib_tools.call_function`, which simply `eval`s the single
expression `QuantLib.<name>(<args_str>)` in a restricted namespace. **No
sequencing happens here** — it's one call.

### Path B — word problem (multi-step pricing)

This is where a *sequence* of QuantLib calls is determined. The flow is
`_handle_run` → `_extract_and_dispatch` → `_extract_snippet` →
`_execute_snippet`.

**The sequence is decided entirely by the LLM, not by any hardcoded pricing
recipe in the code:**

1. **Grounding context.** `_extract_snippet` fetches the target symbol's real
   signature and docstring via `quantlib_tools.get_doc(name)`, and
   `_build_extraction_prompt` packages `{symbol, signature, docstring, word
   problem}` as the user message.
2. **The system prompt teaches the pricing pattern.** The
   `_QUANTLIB_CODE_EXTRACTION_SYSTEM_PROMPT` constant is where the "how to get to a final
   price" knowledge lives — as prose, not code. It instructs the model that it
   may write **as many statements as needed**, that **constructing an instrument
   does not price it** (e.g. `ql.VanillaOption(payoff, exercise)` only builds the
   option), and that to get a price it must build a **stochastic process** from
   market data (spot, rate, dividend, vol) → create a **pricing engine** (e.g.
   `ql.AnalyticEuropeanEngine(process)`) → attach it with
   `option.setPricingEngine(engine)` → read `option.NPV()`. It also says to set
   `ql.Settings.instance().evaluationDate` for time-sensitive problems, never
   invent missing market data, assign the final answer to a variable named
   `result`, and reply with a JSON object (`{"status":"ok","code":...}` or
   `{"status":"missing","question":...}`). So the model composes the ordered
   pipeline itself, guided by this prompt plus the docstring.
3. **Missing-info handshake.** If the model returns `status: "missing"`,
   `_extract_and_dispatch` stores `self._pending_run` and asks a clarifying
   question; your next message is merged in and re-extracted
   (`_handle_pending_run`) rather than requiring a full retype.
4. **Execution.** On `status: "ok"`, `_execute_snippet` runs the whole
   multi-statement snippet via `run_snippet`, which uses a single `exec`
   namespace (so names defined in one statement are visible to later ones), with
   `ql`/`QuantLib` pre-bound, restricted builtins (no direct import/file/process
   builtins, and not a security sandbox), captured stdout, and a required `result` variable. It replies with
   both the generated code and the computed value.

### Summary

- **Path A** (`call_function`): one literal `eval` — no sequence logic.
- **Path B** (`run_snippet`): the LLM authors the ordered sequence of QuantLib
  calls. The domain knowledge of *what* sequence prices a security (build
  instrument → process → engine → `setPricingEngine` → `NPV`) is encoded as
  **instructions in `_QUANTLIB_CODE_EXTRACTION_SYSTEM_PROMPT`**, combined with the live
  docstring of the target symbol. The Python code only orchestrates, validates
  the symbol, handles missing-info follow-ups, and runs it with restricted builtins (not a sandbox) — it never
  hardcodes the pricing pipeline itself.

One consequence: because the sequence is model-generated and run with `exec`,
correctness depends on the LLM getting the wiring right for the specific
instrument. The docstring grounding helps, but there is no validation that the
produced pipeline is the canonically correct one for a given security — it runs
whatever the model wrote (with restricted builtins, which are not a security sandbox, and not verified for financial correctness).
