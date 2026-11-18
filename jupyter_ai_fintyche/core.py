# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FinTyche's conversation logic, independent of any frontend.

`FinTycheCore` holds everything FinTyche does with a message: route it to a
command, resolve a word problem to a library symbol, write and run code for it,
keep the conversation history, and answer free-form questions. It never touches
Jupyter. A frontend subclasses it and supplies the I/O it calls; see the class
docstring. The Jupyter AI persona (`persona.py`) is one; a terminal CLI and
the eval harness are meant to be others.

Two library backends are exposed through the same set of verbs (`list`, `doc`,
`run`, `notebook`, `help`):

- **QuantLib** — derivatives/fixed-income analytics, via `/quantlib-*` slash
  commands (or the bare verbs `list`/`doc`/`run`/`notebook`/`help`).
- **PyPortfolioOpt** — portfolio optimization, via `/pyportpf-*` slash commands
  (or the bare `pyportpf-list`/`pyportpf-doc`/… spellings).

The two share one implementation: each backend (see `backends._BACKENDS`)
supplies its own introspection/execution module and its own LLM prompts, and
every command handler takes a `backend` argument.

Once a word problem has been resolved to a symbol, who runs the resulting code
is the user's choice, via the execution mode: ask first (default), run
automatically, or leave it to the user.
The resolved symbol is remembered in `_last_doc`, so a message that just asks to
run it ("sure", "rerun that") runs it without the whole command being retyped;
the model judges whether a message means that, so no particular phrase is
required. The code generated for it is pinned in `_pinned`: agreeing to run it,
and repeating the command, run exactly that code, and any later request for the
same symbol (a fix after an error, a changed detail, a notebook) edits it rather
than starting over. Free-form chat has no
execution channel of its own — `_run_llm_chat` makes a plain completion call with
no tools — so the system prompt forbids it from claiming it will run anything.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .backends import _BACKENDS, _COMMAND_ALIASES, LibraryBackend
from .market_data import uses_yf

FINTYCHE_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "skills" / "system_prompt.md"
)

_DEFAULT_SYSTEM_PROMPT = "You are FinTyche, a professional quantitative financial analyst."

# History is trimmed to fit the selected model's context window, not a fixed turn
# count. Budget = window - reply_reserve; `_fit_history` keeps the most recent
# whole pairs that fit, but never fewer than _MIN_HISTORY_PAIRS, so no model ends
# up with less memory than before this change.
_MODEL_CONTEXT_FALLBACK = 32000  # assumed input window when litellm can't say
_RESPONSE_TOKEN_RESERVE = 2048  # reply room when the model's output size is unknown
_RESPONSE_RESERVE_CAP = 8192  # cap the reply reserve so a huge output window (e.g.
#                               64k) doesn't swallow most of the budget
_MIN_HISTORY_PAIRS = 6  # always keep at least this many recent pairs, budget aside
_HISTORY_HARD_CAP_PAIRS = 200  # absolute ceiling, and the in-memory store cap
# Upper bound on code/result/traceback text stored in one execution record, so a
# single large run can't crowd out the conversation before budget trimming runs.
_MAX_RECORDED_CODE_CHARS = 2000
_NOTEBOOK_FILENAME_RE = re.compile(r"\s+as\s+(\S+)\s*$", re.IGNORECASE)

_CANCEL_WORDS = {"cancel", "stop", "nevermind", "never mind"}

# Cycling "is working ..." progress line shown while a command runs (see `FinTycheCore._working_status`).
_STATUS_TICK_SECONDS = 0.5
_STATUS_DOT_FRAMES = ("", ".", "..", "...")

# One `handle_message` call (a command, a run, a notebook, a free-form chat
# reply — whatever one incoming message triggers) is "a task". Past this many
# seconds it is cancelled rather than left to hang the chat indefinitely on a
# stuck LLM call or a slow/looping generated snippet.
_TASK_TIMEOUT_SECONDS = 600  # 10 minutes

# `eval`/`exec`'d code (`_execute_run`/`_execute_snippet` below) runs
# synchronously, outside any `await`, so `asyncio.wait_for` around the
# surrounding task can't preempt it — the timeout is only checked once the
# call returns. Running it on this thread pool instead lets the caller give
# up after `_TASK_TIMEOUT_SECONDS` even though the thread itself can't be
# killed and keeps running until the call it's stuck in returns.
_SNIPPET_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    thread_name_prefix="fintyche-snippet"
)


def _run_with_timeout(fn, *args, timeout: float = _TASK_TIMEOUT_SECONDS):
    """Run `fn(*args)` on `_SNIPPET_EXECUTOR`, giving up after `timeout` seconds.

    Raises the original exception if `fn` raised one, or `TimeoutError` if it
    didn't finish in time.
    """
    future = _SNIPPET_EXECUTOR.submit(fn, *args)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(
            f"This took longer than {int(timeout)} seconds, so FinTyche "
            "stopped waiting for it."
        ) from None

# ---------------------------------------------------------------------------
# Execution mode — who runs the generated code
#
# Once FinTyche has resolved a word problem to a library symbol, the code that
# answers it can be run by FinTyche or left for the user to run with an
# explicit `run` command. `run_snippet` `exec`s model-generated code (under
# restricted builtins), so the default asks for confirmation first rather than
# executing unprompted.
# ---------------------------------------------------------------------------

_EXECUTION_MODE_SETTING_ID = "execution_mode"

_MODE_ASK = "ask"
_MODE_AUTO = "auto"
_MODE_MANUAL = "manual"

_DEFAULT_EXECUTION_MODE = _MODE_ASK


# Appended to the chat system prompt so a reply that names a command describes
# what will actually happen next. In ask mode FinTyche asks whether to run it
# itself, after showing the code, so the reply must not ask too; in auto mode
# nothing is asked at all.
_CHAT_MODE_INSTRUCTIONS = {
    _MODE_ASK: (
        "Running code is set to *Ask first*: after your reply FinTyche writes "
        "the code for the command you name, shows it, and asks the user whether "
        "to run it. Do not ask that yourself, and do not tell the user what to "
        "reply."
    ),
    _MODE_AUTO: (
        "Running code is set to *Run automatically*: FinTyche runs the "
        "command you name as soon as your reply ends, and posts the code and "
        "result below it. Say that it will run right after this message. Do "
        "not ask whether to run it, offer a choice, or tell the user what to "
        "reply."
    ),
    _MODE_MANUAL: (
        "Running code is set to *I'll run it myself*: FinTyche will not run "
        "the command you name. Tell the user to send it when they are ready."
    ),
}


# Classifies a message once FinTyche has something it could run, so agreeing
# ("sure", "rerun that with 5% vol") or declining ("nah, not now") is understood
# however it's phrased, rather than matched against a list of phrases.
_RUN_INTENT_SYSTEM_PROMPT = (
    "You read one chat message sent to FinTyche, a quant-finance assistant "
    "that writes and runs code. You are told what FinTyche last offered to "
    "run or ran. Decide what the message means:\n"
    '- "run": the user wants that run, or run again, now — agreement to the '
    "offer, or a request to (re)run it, possibly with a change to its inputs.\n"
    '- "decline": the user does not want it run.\n'
    '- "other": anything else — a question, a comment, or a new or different '
    "task, including one that names another function or class.\n\n"
    "Respond with ONLY a JSON object and no other text:\n"
    '  {"intent": "run" | "decline" | "other", "changes": "<for run only: any '
    'change to the inputs the user asked for, as a short phrase; otherwise empty>"}\n'
    'When unsure, answer "other".'
)

# A `…-run <Symbol> <word problem>` command suggested inside a free-form chat
# reply. Free-form chat cannot execute anything itself, so when it tells the user
# which command to send, that symbol is recorded as the last resolution — which
# is what lets a following "go ahead" run it instead of dead-ending.
_SUGGESTED_RUN_RE = re.compile(
    r"/?(?:(?P<prefix>quantlib|pyportpf)-)?run\s+"
    r"(?P<name>[A-Za-z_][\w.]*)\s+"
    r"(?P<prompt>[^\n`]{4,})"
)

# The messages FinTyche posts for generated code, read back by the persona's
# `_restore_from_chat` to find the last run after a restart. They mirror the
# formats in `_execute_snippet` and `_send_proposal`.
_POSTED_CODE_RES = (
    re.compile(
        r"\*\*Error running the generated code for `(?P<name>[\w.]+)`:\*\* "
        r"(?P<error>[^\n]*)\n\n```python\n(?P<code>.*?)\n```",
        re.DOTALL,
    ),
    re.compile(r"\*\*`(?P<name>[\w.]+)`\*\* →\n\n```python\n(?P<code>.*?)\n```", re.DOTALL),
    re.compile(r"\*\*Code for `(?P<name>[\w.]+)`:\*\*\n\n```python\n(?P<code>.*?)\n```", re.DOTALL),
)


def _build_extraction_prompt(label: str, doc: Any, prompt_text: str) -> str:
    return (
        f"Target {label} symbol: {doc.name} ({doc.kind})\n"
        f"Signature: {doc.name}{doc.signature}\n"
        f"Docstring: {doc.doc}\n\n"
        f"Word problem: {prompt_text}"
    )


# Names of functions an error message blames, e.g. the `market_implied_prior_returns`
# in "market_implied_prior_returns() missing 1 required positional argument".
_ERROR_FUNCTION_RE = re.compile(r"\b([A-Za-z_]\w*)\(\)")


def _error_signatures(backend: "LibraryBackend", error: str) -> list[str]:
    """Real signatures of the library functions an error message names.

    The failing call is often a companion function rather than the target
    symbol, and its signature is exactly what the model got wrong, so it is
    looked up from the library instead of left to the model's memory.
    """
    lines: list[str] = []
    for fname in dict.fromkeys(_ERROR_FUNCTION_RE.findall(error)):
        try:
            entries, _ = backend.tools.list_entries(query=fname, limit=200)
        except Exception:
            continue
        for entry in entries:
            if entry.name.split(".")[-1] != fname:
                continue
            doc = backend.tools.get_doc(entry.name)
            if doc is not None:
                lines.append(f"{backend.module_alias}.{doc.name}{doc.signature}")
    return lines


def _build_edit_prompt(backend: "LibraryBackend", pinned: dict[str, Any]) -> str:
    """Instructions to edit pinned code rather than write a new snippet.

    Without this, a fix or a changed detail regenerates from the word problem
    alone, and the model is free to swap data, estimators, and parameters that
    had nothing to do with the request.
    """
    text = (
        "\n\nPrevious code for this symbol:\n"
        f"```python\n{pinned['code']}\n```\n"
        f"It was written for: {pinned['prompt']}\n\n"
    )
    error = pinned.get("error")
    if error:
        text += f"Running it failed with:\n{error}\n\n"
        signatures = _error_signatures(backend, error)
        if signatures:
            text += "Actual signatures:\n" + "\n".join(signatures) + "\n\n"
        text += "Fix that error. "
    else:
        text += "Update it for the word problem above. "
    return text + (
        "Edit the previous code as little as possible: keep its tickers, data "
        "source, estimators, parameters, and views unless the error or the word "
        "problem requires a change. Describe the edit in `changes`."
    )


def _truncate_for_history(text: str) -> str:
    """Cap text stored in one history entry so a single large run (code, a big
    result, stdout, or a traceback) can't crowd out the rest of the transcript."""
    if len(text) > _MAX_RECORDED_CODE_CHARS:
        return text[:_MAX_RECORDED_CODE_CHARS] + "\n… (truncated)"
    return text


# Characters a result value could otherwise use as markdown (`$` starts LaTeX).
_MARKDOWN_SPECIAL_RE = re.compile(r"([\\`*_\[\]<>#|~$&])")


def _escape_markdown(text: str) -> str:
    return _MARKDOWN_SPECIAL_RE.sub(r"\\\1", text)


def _result_lines(fields: dict, indent: str = "") -> list[str]:
    """One bullet per field, nesting for a nested dict. A value whose text
    spans lines (e.g. a DataFrame) goes in a code block under its field."""
    lines = []
    for key, value in fields.items():
        label = f"{indent}- **{_escape_markdown(str(key))}**:"
        if isinstance(value, dict) and value:
            lines.append(label)
            lines.extend(_result_lines(value, indent + "  "))
            continue
        if getattr(value, "shape", None) == () and hasattr(value, "item"):
            value = value.item()  # `0.12`, not `np.float64(0.12)`
        text = value if isinstance(value, str) else repr(value)
        if "\n" in text:
            block = "\n".join(f"{indent}  {line}" for line in text.splitlines())
            lines.append(f"{label}\n\n{indent}  ````text\n{block}\n{indent}  ````")
        else:
            lines.append(f"{label} {_escape_markdown(text)}")
    return lines


def _format_result(result: Any) -> str:
    """`result` as shown in chat. A dict is one bullet per field, which wraps to
    the chat panel's width; in a code block a long line would scroll instead."""
    if isinstance(result, dict) and result:
        return "\n".join(_result_lines(result))
    return f"```text\n{result!r}\n```"


def _snippet_request(name: str, code: str, prompt_text: str = "") -> str:
    """The 'user' side of a recorded snippet run: the word problem plus the code
    generated for it, so a later turn sees both what was asked and what ran."""
    ask = f"{prompt_text}\n\n" if prompt_text else ""
    return (
        f"{ask}Ran this generated code for `{name}`:\n"
        f"```python\n{_truncate_for_history(code)}\n```"
    )


def _loads_json_object(raw: Optional[str]) -> Optional[dict]:
    """Parse a model reply that should be a single JSON object, tolerating a
    ```json fence. Returns the dict, or None if it isn't valid JSON object."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_extraction_response(raw: Optional[str]) -> Optional[dict[str, str]]:
    data = _loads_json_object(raw)
    if data is None:
        return None
    status = data.get("status")
    if status == "ok" and isinstance(data.get("code"), str):
        parsed = {"status": "ok", "code": data["code"]}
        if isinstance(data.get("changes"), str) and data["changes"].strip():
            parsed["changes"] = data["changes"].strip()
        return parsed
    if status == "missing" and isinstance(data.get("question"), str):
        return {"status": "missing", "question": data["question"]}
    return None


def _parse_doc_resolution(raw: Optional[str]) -> Optional[dict[str, str]]:
    data = _loads_json_object(raw)
    if data is None:
        return None
    status = data.get("status")
    if status == "ok" and isinstance(data.get("name"), str):
        result = {"status": "ok", "name": data["name"]}
        if isinstance(data.get("note"), str) and data["note"].strip():
            result["note"] = data["note"].strip()
        return result
    if status == "missing" and isinstance(data.get("question"), str):
        return {"status": "missing", "question": data["question"]}
    return None


def _parse_run_intent(raw: Optional[str]) -> Optional[dict[str, str]]:
    data = _loads_json_object(raw)
    if data is None or data.get("intent") not in ("run", "decline", "other"):
        return None
    changes = data.get("changes")
    return {
        "intent": data["intent"],
        "changes": changes.strip() if isinstance(changes, str) else "",
    }


@dataclass
class RunOutcome:
    """What one execution produced, for frontends that need more than the
    markdown message (the CLI's `--json` output, an eval harness)."""

    backend: str  # backend key, e.g. "quantlib"
    name: str  # the library symbol the code was written around
    code: str  # the code that ran (`<name>(<args>)` for a direct call)
    prompt: str  # the word problem, or "" for a direct call
    result: Any = None  # the value of `result`; None if the code raised
    stdout: str = ""
    error: Optional[str] = None  # "<Type>: <message>" if the code raised


# What `_on_notice` reports, in place of a result:
# - "failed": the request couldn't be carried out (a failed or unreadable model
#   call, an unknown symbol, no model, litellm missing, a notebook not written);
# - "unavailable": the library itself isn't installed or can't be loaded;
# - "usage": the command was incomplete, and the reply says how to use it;
# - "question": a clarifying question that is not tracked as a pending command.
_NOTICE_KINDS = ("failed", "unavailable", "usage", "question")


def load_system_prompt() -> str:
    """FinTyche's chat system prompt, from `skills/system_prompt.md`."""
    try:
        return FINTYCHE_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    except OSError:
        return _DEFAULT_SYSTEM_PROMPT


class FinTycheCore:
    """FinTyche's frontend-agnostic logic, mixed into each frontend.

    A frontend subclasses this and implements the I/O methods stubbed out under
    "Frontend contract" below, plus `system_prompt` and `log` attributes, and
    calls `_init_state()` from its `__init__`. The persona lists `BasePersona`
    before this class so `BasePersona`'s real I/O methods take precedence over
    the stubs here; it defines `_resolve_model` itself.

    The class attributes under "Frontend wording" say where things live in a
    given frontend (where to configure a model, how to open a notebook), so
    help and error text never points at UI the user doesn't have.
    """

    # ---- frontend wording ----

    # Shown when `_resolve_model` finds no model.
    no_model_message = (
        "**No chat model is configured.** Configure a model to enable word "
        "problems and free-form questions."
    )
    # Help's headline, after "**FinTyche** — ".
    product_tagline = "quantitative finance assistant"
    # Help's model line when no model is configured, after "none configured yet — ".
    model_settings_hint = "configure one to enable word problems and chat"
    # Where the execution mode is set, as help refers to it.
    execution_mode_location = "the execution mode"
    # Which model answers free-form questions, as help refers to it.
    chat_model_location = "the configured model"
    # Follows "**Notebook created:** `<path>`".
    notebook_open_hint = "Open it in Jupyter to run the demo."

    # ---- frontend contract ----

    def send_message(self, body: str) -> None:
        """Post one markdown message to the user."""
        raise NotImplementedError

    async def stream_message(self, chunks: Any) -> None:
        """Post a reply streamed from `chunks`, an async iterator of LiteLLM
        streaming chunks. Must consume the iterator to the end: the reply
        recorded in history is collected as the chunks are read."""
        raise NotImplementedError

    def set_status(self, status: str = "") -> None:
        """Show `status` as a transient progress line."""
        raise NotImplementedError

    def clear_status(self) -> None:
        """Remove the progress line."""
        raise NotImplementedError

    def get_workspace_dir(self) -> str:
        """The directory notebooks are written into."""
        raise NotImplementedError

    def process_attachments(self, message: Any) -> Optional[str]:
        """Text of any files attached to `message`, or None. Only called when
        `handle_message` was given a message object."""
        raise NotImplementedError

    def _resolve_model(self) -> tuple[Optional[str], dict[str, Any]]:
        """`(litellm_model_id, kwargs)` for the model to call, or `(None, {})`
        when none is configured."""
        raise NotImplementedError

    def _on_run(self, outcome: RunOutcome) -> None:
        """Called after every execution, whether it succeeded or not. A no-op
        unless a frontend wants the structured outcome."""

    def _on_notice(self, kind: str, text: str) -> None:
        """Called when FinTyche answers with something other than a result,
        so a frontend can tell what happened without parsing `text`. `kind` is
        one of `_NOTICE_KINDS`. A no-op by default."""

    def _on_notebook(self, backend: str, name: str, code: str, path: str) -> None:
        """Called after a notebook is written to `path`. A no-op by default."""

    def _send_notice(self, kind: str, text: str) -> None:
        """Send `text` to the user, reporting it as a `kind` notice first."""
        self._on_notice(kind, text)
        self.send_message(text)

    # ---- conversation state ----

    def _init_state(self, execution_mode: str = _DEFAULT_EXECUTION_MODE) -> None:
        """Set up the conversation state; call once from the frontend's __init__."""
        self._history: list[dict[str, str]] = []
        # A word-problem command (`run` or `notebook`) awaiting a clarifying
        # answer. Keys: "backend" (key), "kind" ("run"|"notebook"), "name",
        # "prompt", and "filename" (notebook only, may be None).
        self._pending_run: Optional[dict[str, Any]] = None

        # The most recent symbol FinTyche resolved for the user, and the word
        # problem it came from. Keys: "backend" (key), "name", "prompt". This is
        # what "run it" refers to — without it there is no referent for a
        # conversational run request, so the symbol would have to be retyped.
        self._last_doc: Optional[dict[str, str]] = None

        # The code generated for the last resolved symbol, pinned so that running
        # it again runs exactly what was shown. Keys: "backend", "name",
        # "prompt" (the word problem it was written for), "code", and "error"
        # (the last run's error, or None). A later request for the same symbol
        # edits this code instead of regenerating it from scratch.
        self._pinned: Optional[dict[str, Any]] = None

        # True while an "shall I run it?" offer is outstanding, so the next
        # message is read as a yes/no answer to it.
        self._pending_confirm = False

        # Who runs generated code: `_MODE_ASK`, `_MODE_AUTO`, or `_MODE_MANUAL`.
        self._execution_mode = execution_mode

    @contextlib.asynccontextmanager
    async def _working_status(self):
        """Show a cycling "is working ..." indicator with elapsed seconds
        as the frontend's progress line for the duration of the `async with`
        block, via `set_status()`/`clear_status()`.

        Ticks independently of whatever runs inside the block (an LLM call, a
        library introspection call, code execution), so slash commands show
        visible progress even though they don't stream tokens the way
        free-form chat replies do.
        """
        start = time.monotonic()

        async def _tick() -> None:
            frame = 0
            while True:
                dots = _STATUS_DOT_FRAMES[frame % len(_STATUS_DOT_FRAMES)]
                elapsed = int(time.monotonic() - start)
                self.set_status(f"is working{dots} {elapsed}s")
                frame += 1
                await asyncio.sleep(_STATUS_TICK_SECONDS)

        task = asyncio.ensure_future(_tick())
        try:
            yield
        finally:
            task.cancel()
            # `asyncio.wait` waits for the ticker without raising its
            # cancellation, so a cancellation of *this* task (in the CLI, Ctrl-C
            # during synchronous code execution) still propagates instead of
            # being swallowed here.
            try:
                await asyncio.wait({task})
            finally:
                self.clear_status()
            if not task.cancelled() and task.exception() is not None:
                self.log.warning("The progress line failed.", exc_info=task.exception())

    async def handle_message(self, body: str, message: Any = None) -> None:
        """Route one already-stripped message body to a command, a pending
        flow, a run request, the bare-symbol doc shortcut, or free-form chat.

        `message` is the frontend's original message object, if any; it is
        only handed back to `process_attachments` for free-form chat.

        The whole thing is bounded by `_TASK_TIMEOUT_SECONDS`: if routing or
        an LLM call runs past it, the task is cancelled; generated code is
        run on a worker thread (see `_run_with_timeout`) so a looping or
        slow/hanging snippet is given up on at the same deadline instead of
        blocking this cancellation from ever being checked. Either way, the
        chat is never left hanging indefinitely."""
        try:
            await asyncio.wait_for(
                self._route_message(body, message), timeout=_TASK_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            # `_working_status`'s own `finally` clears the status line as the
            # cancellation propagates through it; `_pending_confirm`/
            # `_pending_run` are reset so a half-finished flow doesn't leave
            # the next message answering a question that no longer applies.
            self._pending_confirm = False
            self._pending_run = None
            minutes = _TASK_TIMEOUT_SECONDS // 60
            self.send_message(
                f"**This is taking longer than {minutes} minutes, so FinTyche "
                "stopped it.** Try again, perhaps with a narrower request."
            )

    async def _route_message(self, body: str, message: Any = None) -> None:
        """The routing logic `handle_message` runs under its timeout."""
        if not body:
            self._handle_help()
            return

        tokens = body.split(maxsplit=1)
        first_word = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""

        resolved = _COMMAND_ALIASES.get(first_word)

        # An explicit command always wins, even mid-flow: a pending clarifying
        # question or run offer must never swallow a deliberate command (typing
        # `/quantlib-run …` while a question was outstanding used to be fed back
        # as "additional detail" instead of being dispatched).
        if resolved is not None:
            self._pending_confirm = False
            self._pending_run = None
            async with self._working_status():
                await self._dispatch_command(resolved, body, rest)
            return

        # A pending `run` awaiting a missing argument takes priority over
        # free-form chat, mirroring QuickAgent's interactive-setup priority —
        # the user's next message is the answer, not a new conversation.
        if self._pending_run is not None:
            async with self._working_status():
                await self._handle_pending_run(body)
            return

        # Once there is something to run, the message may accept or decline it
        # — an answer to the offer, or "rerun that with 5% vol" later on. The
        # model judges that; anything else carries on to normal routing, so an
        # offer never swallows a new question.
        if self._last_doc is not None:
            intent = await self._classify_run_intent(body)
            if intent is not None and intent["intent"] == "run":
                self._pending_confirm = False
                async with self._working_status():
                    await self._run_last_doc(extra=intent["changes"])
                return
            if intent is not None and intent["intent"] == "decline" and self._pending_confirm:
                self._pending_confirm = False
                self._send_decline()
                return

        # Bare `<symbol-name>` shortcut for documentation, e.g. "Date" alone
        # (QuantLib) or "EfficientFrontier" (PyPortfolioOpt).
        if len(tokens) == 1:
            backend = self._symbol_backend(body)
            if backend is not None:
                async with self._working_status():
                    await self._handle_doc(backend, body)
                return

        await self._run_llm_chat(body, message)

    def _restore_last_run(self, turns: list[dict[str, str]]) -> None:
        """Pin the last code FinTyche posted in `turns` (the saved messages,
        one per entry), found by the formats
        it posts code in. The word problem it was written for isn't saved, so it
        is recovered from the turns before it: the `…-run <name> <problem>`
        command a reply suggested or the user sent, else the longest request
        since the previous code (an answer such as "yes" is never the problem).
        Reruns use the pinned code verbatim either way."""
        for index in range(len(turns) - 1, -1, -1):
            if turns[index]["role"] != "assistant":
                continue
            found = None
            for pattern in _POSTED_CODE_RES:
                for match in pattern.finditer(turns[index]["content"]):
                    if found is None or match.start() > found.start():
                        found = match
            if found is None:
                continue
            backend = self._symbol_backend(found["name"])
            if backend is None:
                return
            prompt = self._recover_word_problem(turns[:index], found["name"], found["code"])
            error = found.groupdict().get("error")
            self._last_doc = {"backend": backend.key, "name": found["name"], "prompt": prompt}
            self._pinned = {
                "backend": backend.key,
                "name": found["name"],
                "prompt": prompt,
                "code": found["code"],
                "error": error,
            }
            return

    @staticmethod
    def _recover_word_problem(turns: list[dict[str, str]], name: str, code: str) -> str:
        """The word problem behind `code` for `name`, from the turns before it.
        The search stops at different code posted earlier, which belongs to a
        previous run; the same code (its proposal before the run) does not."""
        since_last_code: list[str] = []
        for turn in reversed(turns):
            for match in _SUGGESTED_RUN_RE.finditer(turn["content"]):
                if match["name"] == name:
                    return match["prompt"].strip().rstrip("`").strip()
            if turn["role"] == "assistant" and any(
                match["code"] != code
                for pattern in _POSTED_CODE_RES
                for match in pattern.finditer(turn["content"])
            ):
                break
            if turn["role"] == "user":
                since_last_code.append(turn["content"])
        if since_last_code:
            return max(since_last_code, key=len)
        return f"the code for `{name}` shown earlier in this chat"

    async def _dispatch_command(
        self, resolved: tuple[str, str], body: str, rest: str
    ) -> None:
        """Run the handler for an already-resolved `(backend_key, verb)` pair."""
        backend_key, verb = resolved
        backend = _BACKENDS[backend_key]
        if verb == "list":
            self._handle_list(backend, rest)
        elif verb == "doc":
            await self._handle_doc(backend, rest)
        elif verb == "run":
            await self._handle_run(backend, rest)
        elif verb == "notebook":
            await self._handle_notebook(backend, rest)
        elif verb == "help":
            self._handle_help()

    def _send_decline(self) -> None:
        """Acknowledge a declined run offer and say how to run it later."""
        last = self._last_doc
        hint = ""
        if last is not None:
            backend = _BACKENDS[last["backend"]]
            hint = (
                f" Say `{backend.slash_prefix}-run {last['name']} "
                "<word problem>` whenever you want to run it."
            )
        self.send_message(f"Okay, not running it.{hint}")

    async def _classify_run_intent(self, body: str) -> Optional[dict[str, str]]:
        """Ask the model whether `body` accepts or declines running `_last_doc`.

        Returns `{"intent": "run" | "decline" | "other", "changes": ...}`, where
        `changes` is any change to the inputs the user asked for. Returns None
        when that can't be decided (no model, a failed call, an unreadable
        reply); the message is then handled as ordinary chat.
        """
        last = self._last_doc
        if last is None:
            return None
        try:
            import litellm
        except ImportError:
            return None
        model_id, model_kwargs = self._resolve_model()
        if model_id is None:
            return None

        offered = (
            "FinTyche has just offered to run"
            if self._pending_confirm
            else "The last thing FinTyche worked on is"
        )
        problem = last.get("prompt") or "(no word problem yet)"
        request = f"{offered} `{last['name']}` for: {problem}\n\nMessage: {body}"
        try:
            response = await litellm.acompletion(
                model=model_id,
                messages=[
                    {"role": "system", "content": _RUN_INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
                **model_kwargs,
            )
            raw = response.choices[0].message.content
        except Exception:
            self.log.exception("Error classifying a run request via '%s'.", model_id)
            return None
        return _parse_run_intent(raw)

    async def _run_last_doc(self, extra: str = "") -> None:
        """Run the last resolved symbol against its word problem.

        `extra` is appended to that word problem, so "run the model with 5% vol"
        refines the original problem rather than discarding the addition.

        Always answers the user: it either runs something or explains why it
        can't (no symbol resolved yet, or no word problem to build code from).
        """
        last = self._last_doc
        if last is None:
            self._send_notice(
                "usage",
                "I don't have a symbol to run yet — tell me what you want to "
                "compute (or say `help` for the command reference) and I'll pick "
                "one first."
            )
            return

        backend = _BACKENDS[last["backend"]]
        name = last["name"]
        prompt_text = last.get("prompt", "")
        if not prompt_text:
            self._send_notice(
                "usage",
                f"I have `{name}` but no word problem to build code from. "
                f"Say `{backend.slash_prefix}-run {name} <word problem>` and "
                "describe the inputs in plain text."
            )
            return

        if extra:
            prompt_text = f"{prompt_text}\n\nAdditional detail: {extra}"

        await self._extract_and_dispatch(backend, "run", name, prompt_text)

    def _symbol_backend(self, name: str) -> Optional[LibraryBackend]:
        """Return the backend for which `name` is a known symbol, or None.

        QuantLib is checked first so its flat namespace keeps priority for the
        bare-symbol documentation shortcut; PyPortfolioOpt is the fallback.
        """
        for backend in _BACKENDS.values():
            try:
                if backend.tools.has_symbol(name):
                    return backend
            except backend.tools.Unavailable:
                continue
        return None

    # ---- library commands (generic over a backend) ----

    def _handle_list(self, backend: LibraryBackend, query: str) -> None:
        try:
            entries, total = backend.tools.list_entries(query=query or None)
        except backend.tools.Unavailable as e:
            self._send_notice("unavailable", str(e))
            return

        if not entries:
            filter_note = f" matching `{query}`" if query else ""
            self.send_message(f"No {backend.label} symbols found{filter_note}.")
            return

        lines = [
            f"**{backend.label} symbols**{f' matching `{query}`' if query else ''} "
            f"({len(entries)} of {total} shown):\n"
        ]
        for entry in entries:
            summary = f" — {entry.summary}" if entry.summary else ""
            lines.append(f"- `{entry.name}` (*{entry.kind}*){summary}")
        if total > len(entries):
            lines.append(
                f"\n_{total - len(entries)} more not shown — narrow with "
                f"`{backend.slash_prefix}-list <keyword>`._"
            )
        lines.append(
            f"\nSay `{backend.slash_prefix}-doc <name>` for details, "
            f"`{backend.slash_prefix}-run <name> <word problem>` to call it, or "
            f"`{backend.slash_prefix}-notebook <name> <word problem>` to generate "
            "a demo notebook."
        )
        self.send_message("\n".join(lines))

    async def _handle_doc(self, backend: LibraryBackend, query: str) -> None:
        if not query:
            self._send_notice(
                "usage",
                f"Usage: `{backend.slash_prefix}-doc <word problem>` — describe "
                f"what you want to do and FinTyche finds the right "
                f"{backend.label} symbol and shows its docs. An exact symbol name "
                "works too."
            )
            return

        # An exact symbol name is shown directly, with no LLM call — this also
        # backs the bare-`<symbol>` documentation shortcut. There is no word
        # problem in this case, so there is nothing to build code from and no
        # run to offer.
        if backend.tools.has_symbol(query):
            self._last_doc = {"backend": backend.key, "name": query, "prompt": ""}
            self._send_doc(backend, query)
            return

        # Otherwise treat the text as a word problem and let the LLM name the
        # most appropriate symbol to document.
        resolved = await self._resolve_doc_symbol(backend, query)
        if resolved is None:
            return
        if resolved["status"] == "missing":
            self._send_notice("question", resolved["question"])
            return

        name = resolved["name"]
        if not backend.tools.has_symbol(name):
            self._send_notice(
                "failed",
                f"FinTyche suggested `{name}` for that, but {backend.label} has "
                f"no such symbol. Try rephrasing, or "
                f"`{backend.slash_prefix}-list <keyword>` to search by name."
            )
            return

        # Remember the resolution so "run it" has a referent, then either run it,
        # offer to, or leave it to the user, per the execution-mode setting.
        self._last_doc = {"backend": backend.key, "name": name, "prompt": query}

        if self._execution_mode == _MODE_AUTO:
            self._send_doc(backend, name, note=resolved.get("note"), offer=False)
            await self._extract_and_dispatch(backend, "run", name, query)
            return

        if self._execution_mode == _MODE_ASK:
            # Write the code now and show it with the offer, so a "yes" runs
            # what the user saw rather than code generated after they agreed.
            self._send_doc(backend, name, note=resolved.get("note"), offer=False)
            await self._extract_and_dispatch(backend, "propose", name, query)
            return

        self._send_doc(backend, name, note=resolved.get("note"))

    def _send_doc(
        self,
        backend: LibraryBackend,
        name: str,
        note: Optional[str] = None,
        offer: bool = True,
    ) -> None:
        """Send the signature and docstring for an exact library symbol.

        `offer` appends the "say `…-run …`" hint. Callers that immediately run
        the symbol or ask to run it pass False, so the message does not tell the
        user to do something FinTyche is already doing.
        """
        try:
            doc = backend.tools.get_doc(name)
        except backend.tools.Unavailable as e:
            self._send_notice("unavailable", str(e))
            return

        if doc is None:
            self._send_notice(
                "failed",
                f"{backend.label} has no symbol named `{name}`. "
                f"Try `{backend.slash_prefix}-list {name}` to search for similar names."
            )
            return

        prefix = f"{note}\n\n" if note else ""
        body = (
            f"{prefix}"
            f"### `{doc.name}` (*{doc.kind}*)\n\n"
            f"**Signature:** `{doc.name}{doc.signature}`\n\n"
            f"```text\n{doc.doc}\n```"
        )
        if offer:
            body += (
                f"\n\nSay `{backend.slash_prefix}-run {doc.name} <word problem>` "
                f"to call it, or `{backend.slash_prefix}-notebook {doc.name} "
                "<word problem>` to generate a demo notebook."
            )
        self.send_message(body)

    async def _resolve_doc_symbol(
        self, backend: LibraryBackend, prompt_text: str
    ) -> Optional[dict[str, str]]:
        """Ask the LLM which library symbol best fits a word problem.

        Returns the parsed response — `{"status": "ok", "name": ..., "note"?: ...}`
        or `{"status": "missing", "question": ...}` — or `None` if something went
        wrong, in which case an explanatory message has already been sent.
        """
        try:
            import litellm
        except ImportError:
            self._send_notice(
                "failed",
                "**`litellm` is not installed**, so `doc` can only look up exact "
                f"{backend.label} symbol names. Install it with "
                "`pip install litellm` to describe what you need in plain text."
            )
            return None

        model_id, model_kwargs = self._resolve_model()
        if model_id is None:
            self._send_notice("failed", self.no_model_message)
            return None

        try:
            response = await litellm.acompletion(
                model=model_id,
                messages=[
                    {"role": "system", "content": backend.doc_resolution_prompt},
                    {"role": "user", "content": f"Word problem: {prompt_text}"},
                ],
                **model_kwargs,
            )
            raw = response.choices[0].message.content
        except Exception as e:
            self.log.exception("Error resolving a doc symbol via '%s'.", model_id)
            self._send_notice(
                "failed",
                f"**Error finding a {backend.label} symbol via `{model_id}`:** {e}"
            )
            return None

        parsed = _parse_doc_resolution(raw)
        if parsed is None:
            self._send_notice(
                "failed",
                f"Couldn't work out which {backend.label} symbol you need. Try "
                f"rephrasing, or `{backend.slash_prefix}-list <keyword>` to search "
                "by name."
            )
            return None
        return parsed

    async def _handle_run(self, backend: LibraryBackend, rest: str) -> None:
        # Bare `run` — run the last resolved symbol if there is one, rather than
        # only printing usage.
        if not rest:
            if self._last_doc is not None and self._last_doc.get("prompt"):
                await self._run_last_doc()
                return
            self._send_notice(
                "usage",
                f"Usage: `{backend.slash_prefix}-run <{backend.label}-name> "
                "<word problem>` — describe the arguments in plain text."
            )
            return
        parts = rest.split(maxsplit=1)
        name = parts[0]
        prompt_text = parts[1] if len(parts) > 1 else ""

        # `run /pyportpf-notebook …` — a command wrapped in "run". Only the
        # prefixed spellings count; a bare `list` could be a symbol name.
        nested = _COMMAND_ALIASES.get(name.lower())
        if nested is not None and "-" in name:
            await self._dispatch_command(nested, rest, prompt_text)
            return

        if not backend.tools.has_symbol(name):
            # `run it`, `run the model with 5% vol` — the leading token refers
            # to the last resolved symbol rather than naming one, and any change
            # it asks for refines the word problem. A token that names something
            # else (a typo) still errors below, rather than silently running
            # something the user didn't name.
            intent = await self._classify_run_intent(f"run {rest}")
            if intent is not None and intent["intent"] == "run":
                await self._run_last_doc(extra=intent["changes"])
                return
            hint = (
                ""
                if self._last_doc is not None
                else " (Nothing has been run yet, so there's nothing to run again.)"
            )
            self._send_notice(
                "failed",
                f"{backend.label} has no symbol named `{name}`. "
                f"Try `{backend.slash_prefix}-list {name}` to search for similar names."
                + hint
            )
            return

        # No argument text at all — nothing to extract, so call it directly.
        if not prompt_text:
            self._execute_run(backend, name, "")
            return

        await self._extract_and_dispatch(backend, "run", name, prompt_text)

    def _execute_run(self, backend: LibraryBackend, name: str, args_str: str) -> None:
        try:
            result, stdout = _run_with_timeout(backend.tools.call_function, name, args_str)
        except backend.tools.Unavailable as e:
            call = f"{name}({args_str})"
            error = f"{type(e).__name__}: {e}"
            self._on_run(RunOutcome(backend.key, name, call, "", error=error))
            self._send_notice("unavailable", str(e))
            return
        except Exception as e:
            call = f"{name}({args_str})"
            error = f"{type(e).__name__}: {e}"
            self._on_run(RunOutcome(backend.key, name, call, "", error=error))
            self.send_message(
                f"**Error running `{name}({args_str})`:** {type(e).__name__}: {e}"
            )
            self._record_execution(
                f"Ran `{name}({args_str})`.",
                _truncate_for_history(f"Error: {type(e).__name__}: {e}"),
            )
            return

        self._on_run(
            RunOutcome(backend.key, name, f"{name}({args_str})", "", result, stdout)
        )
        body = f"**`{name}({args_str})`** →\n\n{_format_result(result)}"
        if stdout:
            body += f"\n\nCaptured output:\n```text\n{stdout}```"
        self.send_message(body)
        outcome = f"Result: {result!r}"
        if stdout:
            outcome += f"\nCaptured output:\n{stdout}"
        self._record_execution(
            f"Ran `{name}({args_str})`.", _truncate_for_history(outcome)
        )

    def _execute_snippet(
        self,
        backend: LibraryBackend,
        name: str,
        code: str,
        prompt_text: str = "",
        note: str = "",
    ) -> Optional[str]:
        """Run a model-generated snippet and report its `result`.

        Used by the word-problem path: the model may need several statements
        (build the inputs, wire up the pipeline, compute the answer) rather than
        a single `<alias>.<name>(...)` call, so we execute the whole snippet and
        show both the code and the value it produced. `note` (what changed, or
        which data came from the user) is shown above the code.

        Returns the error as `"<Type>: <message>"` if the code raised, else None.
        """
        prefix = f"{note}\n\n" if note else ""
        try:
            result, stdout = _run_with_timeout(backend.tools.run_snippet, code)
        except backend.tools.Unavailable as e:
            error = f"{type(e).__name__}: {e}"
            self._on_run(RunOutcome(backend.key, name, code, prompt_text, error=error))
            self._send_notice("unavailable", str(e))
            return error
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            self._on_run(RunOutcome(backend.key, name, code, prompt_text, error=error))
            self.send_message(
                f"{prefix}**Error running the generated code for `{name}`:** "
                f"{error}\n\n```python\n{code}\n```"
            )
            self._record_execution(
                _snippet_request(name, code, prompt_text),
                _truncate_for_history(f"Error: {error}"),
            )
            return error

        self._on_run(RunOutcome(backend.key, name, code, prompt_text, result, stdout))
        data_source = result.get("data_source") if isinstance(result, dict) else None
        if (
            isinstance(data_source, str)
            and "yfinance" in data_source.lower()
            and not data_source.upper().startswith("SIMULATED")
            and not uses_yf(code)
        ):
            # The model's label, not a fact: code that never calls `yf` fetched
            # nothing, so don't claim a source.
            del result["data_source"]
        if isinstance(data_source, str) and data_source.upper().startswith("SIMULATED"):
            prefix += (
                "⚠️ **This result uses simulated (fake) data, not real market "
                f"data.** {data_source}\n\n"
            )
        body = (
            f"{prefix}**`{name}`** →\n\n"
            f"```python\n{code}\n```\n\n"
            f"Result:\n\n{_format_result(result)}"
        )
        if stdout:
            body += f"\n\nCaptured output:\n```text\n{stdout}```"
        self.send_message(body)
        outcome = f"Result: {result!r}"
        if stdout:
            outcome += f"\nCaptured output:\n{stdout}"
        self._record_execution(
            _snippet_request(name, code, prompt_text), _truncate_for_history(outcome)
        )
        return None

    def _record_execution(self, request: str, outcome: str) -> None:
        """Record a run and its outcome as a balanced user->assistant pair, so a
        later chat turn can debug a run that command paths execute outside
        ``_run_llm_chat``. See ``_append_pair`` for why pairs stay balanced."""
        self._append_pair(request, outcome)

    def _append_pair(self, user_content: str, assistant_content: str) -> None:
        """Append a user->assistant pair and trim the in-memory store to the hard
        cap. Every write goes through here, so history stays strictly alternating
        (which Bedrock Converse requires) and can't grow without bound over a long
        session."""
        self._history.append({"role": "user", "content": user_content})
        self._history.append({"role": "assistant", "content": assistant_content})
        excess = len(self._history) - _HISTORY_HARD_CAP_PAIRS * 2
        if excess > 0:
            del self._history[:excess]

    async def _extract_snippet(
        self,
        backend: LibraryBackend,
        name: str,
        prompt_text: str,
        base: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, str]]:
        """Ask the LLM to turn a word problem into a snippet for `name`.

        With `base` (pinned code for the same symbol), the model edits that code
        — fixing its recorded error, if any — instead of writing a new snippet.

        Returns the parsed response — `{"status": "ok", "code": ...}` (plus an
        optional `changes` note) or
        `{"status": "missing", "question": ...}` — or `None` if something went
        wrong, in which case an explanatory message has already been sent.
        Shared by the `run` and `notebook` commands.
        """
        try:
            import litellm
        except ImportError:
            self._send_notice(
                "failed",
                "**`litellm` is not installed**, so word problems can't be "
                "parsed. Install it with `pip install litellm` to enable "
                f"`{backend.slash_prefix}-run`/`{backend.slash_prefix}-notebook "
                f"{name} <word problem>`."
            )
            return None

        doc = backend.tools.get_doc(name)
        if doc is None:
            self._send_notice("failed", f"{backend.label} has no symbol named `{name}`.")
            return None

        model_id, model_kwargs = self._resolve_model()
        if model_id is None:
            self._send_notice("failed", self.no_model_message)
            return None

        request = _build_extraction_prompt(backend.label, doc, prompt_text)
        if base is not None:
            request += _build_edit_prompt(backend, base)
        messages = [
            {"role": "system", "content": backend.code_extraction_prompt},
            {"role": "user", "content": request},
        ]
        self.set_status("is writing code...")
        try:
            response = await litellm.acompletion(
                model=model_id, messages=messages, **model_kwargs
            )
            raw = response.choices[0].message.content
        except Exception as e:
            self.log.exception("Error extracting code for '%s' via '%s'.", name, model_id)
            self._send_notice("failed", f"**Error extracting code via `{model_id}`:** {e}")
            return None
        finally:
            self.clear_status()

        parsed = _parse_extraction_response(raw)
        if parsed is None:
            self._send_notice(
                "failed",
                f"Couldn't understand the model's response for `{name}`. "
                "Try rephrasing the word problem."
            )
            return None
        return parsed

    async def _extract_and_dispatch(
        self,
        backend: LibraryBackend,
        kind: str,
        name: str,
        prompt_text: str,
        filename: Optional[str] = None,
    ) -> None:
        """Get the code for a word-problem command and act on it.

        `kind` is "run" (execute it), "notebook" (write it to a notebook), or
        "propose" (show it and offer to run it). The code is pinned: the same
        request again reuses it verbatim, so what was shown is what runs, and
        a different request for the same symbol, or one whose last run failed,
        edits it instead of regenerating it. When information is missing, asks
        a clarifying question and stores `self._pending_run` so the user's next
        message supplies it (see `_handle_pending_run`) rather than requiring
        the whole command to be retyped.
        """
        # Whatever happens next, this is the symbol and word problem in play, so
        # a later bare "run it" resolves to it.
        self._last_doc = {
            "backend": backend.key,
            "name": name,
            "prompt": prompt_text,
        }

        pinned = self._pinned
        if pinned is not None and (pinned["backend"], pinned["name"]) != (
            backend.key,
            name,
        ):
            pinned = None
        if pinned is not None and pinned["prompt"] == prompt_text and not pinned["error"]:
            parsed: Optional[dict[str, str]] = {"status": "ok", "code": pinned["code"]}
        else:
            parsed = await self._extract_snippet(backend, name, prompt_text, base=pinned)
        if parsed is None:
            return

        if parsed["status"] == "missing":
            # A clarifying question replaces any outstanding run offer: the next
            # message answers the question, not the offer.
            self._pending_confirm = False
            self._pending_run = {
                "backend": backend.key,
                "kind": kind,
                "name": name,
                "prompt": prompt_text,
                "filename": filename,
            }
            self.send_message(parsed["question"])
            return

        self._pending_run = None
        code = parsed["code"]
        self._pinned = {
            "backend": backend.key,
            "name": name,
            "prompt": prompt_text,
            "code": code,
            "error": None,
        }
        note = f"**Changed:** {parsed['changes']}" if parsed.get("changes") else ""

        if kind == "notebook":
            self._create_notebook_from_code(backend, name, code, filename, prompt_text)
        elif kind == "propose":
            self._send_proposal(backend, name, code, note)
        else:
            self._pinned["error"] = self._execute_snippet(
                backend, name, code, prompt_text, note=note
            )

    def _send_proposal(
        self, backend: LibraryBackend, name: str, code: str, note: str = ""
    ) -> None:
        """Show pinned code and say how to run it, per the execution mode."""
        prefix = f"{note}\n\n" if note else ""
        if self._execution_mode == _MODE_ASK:
            self._pending_confirm = True
            ask = "Shall I run exactly this code?"
        else:
            ask = (
                f"Ask me to run it (or send `{backend.slash_prefix}-run {name} …` "
                "with the same word problem) to run exactly this code."
            )
        self.send_message(
            f"{prefix}**Code for `{name}`:**\n\n```python\n{code}\n```\n\n{ask}"
        )

    async def _handle_pending_run(self, body: str) -> None:
        pending = self._pending_run
        assert pending is not None

        if body.strip().lower() in _CANCEL_WORDS:
            self._pending_run = None
            self.send_message(f"Okay, cancelled the pending `{pending['kind']}` request.")
            return

        backend = _BACKENDS[pending["backend"]]
        merged_prompt = f"{pending['prompt']}\n\nAdditional detail: {body.strip()}"
        await self._extract_and_dispatch(
            backend,
            pending["kind"],
            pending["name"],
            merged_prompt,
            pending.get("filename"),
        )

    async def _handle_notebook(self, backend: LibraryBackend, rest: str) -> None:
        if not rest:
            self._send_notice(
                "usage",
                f"Usage: `{backend.slash_prefix}-notebook <{backend.label}-name> "
                "<word problem> [as <filename>]` — describe the problem in plain "
                "text and FinTyche writes a notebook that solves it."
            )
            return

        filename = None
        match = _NOTEBOOK_FILENAME_RE.search(rest)
        if match:
            filename = match.group(1)
            rest = rest[: match.start()]

        parts = rest.strip().split(maxsplit=1)
        name = parts[0]
        prompt_text = parts[1].strip() if len(parts) > 1 else ""

        if not backend.tools.has_symbol(name):
            self._send_notice(
                "failed",
                f"{backend.label} has no symbol named `{name}`. "
                f"Try `{backend.slash_prefix}-list {name}` to search for similar names."
            )
            return

        # No word problem: use the code already pinned for this symbol, so the
        # notebook matches what ran in chat.
        pinned = self._pinned
        if (
            not prompt_text
            and pinned is not None
            and (pinned["backend"], pinned["name"]) == (backend.key, name)
        ):
            self._create_notebook_from_code(
                backend, name, pinned["code"], filename, pinned["prompt"]
            )
            return

        # No word problem and nothing pinned — nothing to extract, so build a
        # notebook that calls it with no arguments (mirrors `run <name>`).
        if not prompt_text:
            self._create_notebook_from_code(
                backend, name, f"result = {backend.module_alias}.{name}()", filename
            )
            return

        await self._extract_and_dispatch(backend, "notebook", name, prompt_text, filename)

    def _create_notebook_from_code(
        self,
        backend: LibraryBackend,
        name: str,
        code: str,
        filename: Optional[str],
        word_problem: Optional[str] = None,
    ) -> None:
        """Write `code` into a notebook and report the path (or any error)."""
        try:
            directory = self.get_workspace_dir()
            path = backend.tools.create_notebook(
                name, code, directory, filename, word_problem
            )
        except backend.tools.Unavailable as e:
            self._send_notice("unavailable", str(e))
            return
        except Exception as e:
            self._send_notice(
                "failed",
                f"**Error creating notebook for `{name}`:** {type(e).__name__}: {e}"
            )
            return

        self._on_notebook(backend.key, name, code, path)
        rel_path = os.path.relpath(path, directory)
        self.send_message(
            f"**Notebook created:** `{rel_path}`\n\n"
            f"{self.notebook_open_hint}"
        )

    def _handle_help(self) -> None:
        model_id, _ = self._resolve_model()
        model_note = (
            f"currently `{model_id}`"
            if model_id
            else f"none configured yet — {self.model_settings_hint}"
        )
        self.send_message(
            f"**FinTyche** — {self.product_tagline}\n\n"
            "FinTyche exposes two libraries through the same set of verbs "
            "(`list`, `doc`, `run`, `notebook`, `help`):\n\n"
            "**QuantLib** — derivatives & fixed-income analytics. Send the bare "
            "verbs (`list`, `doc`, `run`, `notebook`, `help`) or the slash forms "
            "`/quantlib-list`, `/quantlib-doc`, `/quantlib-run`, "
            "`/quantlib-notebook`, `/quantlib-help`.\n\n"
            "**PyPortfolioOpt** — portfolio optimization (mean-variance, "
            "Black-Litterman, HRP, …). Send `/pyportpf-list`, `/pyportpf-doc`, "
            "`/pyportpf-run`, `/pyportpf-notebook`, `/pyportpf-help` (or the bare "
            "spellings `pyportpf-list`, `pyportpf-doc`, …).\n\n"
            "**For each library:**\n"
            "- `…-list [filter]` — list functions/classes, optionally filtered by keyword\n"
            "- `…-doc <word problem>` — describe what you want and FinTyche finds "
            "the right symbol and shows its docs (an exact symbol name works too)\n"
            "- `<name>` — bare symbol name is a shortcut for its documentation\n"
            "- `…-run <name> <word problem>` — describe the problem in plain text; "
            "FinTyche writes the code around `<name>` (including any setup), runs "
            "it, and shows the result; it asks a follow-up if something's missing "
            "(reply `cancel` to abandon it)\n"
            "- `…-notebook <name> <word problem> [as <filename>]` — same as `run`, "
            "but writes the generated code into a Jupyter notebook in your workspace\n"
            "- `…-help` — show this message\n\n"
            "**Running the code.** When FinTyche picks a symbol for your word "
            f"problem, {self.execution_mode_location} "
            "decides what happens next: *Ask first* (default) shows the code and "
            "asks whether to run it, *Run automatically* runs it straight away, "
            "and *I'll run it myself* only shows the command. Once a symbol is on "
            "the table you can just ask to run it (or run it again, with a "
            "change) in your own words instead of retyping the whole command.\n\n"
            "Anything else is answered as a quant-finance chat question using "
            f"{self.chat_model_location} ({model_note}).\n\n"
            "**Examples:** `run Date the fifteenth of June 2025`, then "
            "`/pyportpf-run EfficientFrontier maximize the Sharpe ratio of a "
            "portfolio of AAPL, MSFT, GOOG, META`."
        )

    def _model_token_budget(self, model_id: str, model_kwargs: dict) -> int:
        """Input-token budget for history: the model's context window minus room
        reserved for the reply. One margin only (no extra fraction). Falls back to
        a generous constant when litellm can't report the window."""
        info: dict = {}
        try:
            import litellm

            info = litellm.get_model_info(model_id) or {}
        except Exception:  # unknown model, missing litellm, lookup failure
            info = {}
        window = info.get("max_input_tokens") or _MODEL_CONTEXT_FALLBACK
        # Reserve room for the reply: the user's own max_tokens if set, else the
        # model's output size, capped so a 64k output window doesn't dominate.
        reserve = model_kwargs.get("max_tokens") or min(
            info.get("max_output_tokens") or _RESPONSE_TOKEN_RESERVE,
            _RESPONSE_RESERVE_CAP,
        )
        return max(window - reserve, _RESPONSE_TOKEN_RESERVE)

    def _count_tokens(self, model_id: str, messages: list[dict]) -> int:
        """Token count for `messages`, falling back to a length-based estimate if
        litellm can't count for this model. Uses chars/3 (not /4) plus per-message
        overhead, which over-estimates rather than risks overflowing the window."""
        try:
            import litellm

            return litellm.token_counter(model=model_id, messages=messages)
        except Exception:
            return sum(len(m["content"]) for m in messages) // 3 + 4 * len(messages)

    def _fit_history(
        self,
        model_id: str,
        fixed_messages: list[dict],
        history: list[dict],
        model_kwargs: Optional[dict] = None,
    ) -> list[dict]:
        """Most recent whole user/assistant pairs from `history` that fit the
        token budget alongside `fixed_messages` (system prompt, context, current
        turn), but never fewer than `_MIN_HISTORY_PAIRS`. Pairs are kept intact so
        replayed history stays strictly alternating, which Bedrock Converse
        requires."""
        budget = self._model_token_budget(model_id, model_kwargs or {})
        used = self._count_tokens(model_id, fixed_messages)
        kept: list[dict] = []
        pairs = 0
        # Walk newest -> oldest, one pair at a time, counting each pair once.
        i = len(history)
        while i >= 2 and pairs < _HISTORY_HARD_CAP_PAIRS:
            pair = history[i - 2 : i]
            cost = self._count_tokens(model_id, pair)
            over_budget = used + cost > budget
            # Always keep the most recent `_MIN_HISTORY_PAIRS` even if over
            # budget, so a run and its error are never dropped — that is the very
            # context a "why did it fail?" follow-up needs. Recorded entries are
            # truncated (`_MAX_RECORDED_CODE_CHARS`), so this stays bounded.
            if over_budget and pairs >= _MIN_HISTORY_PAIRS:
                break
            kept = pair + kept
            used += cost
            i -= 2
            pairs += 1
        return kept

    def _assemble_messages(
        self, model_id: str, model_kwargs: dict, user_message: str, context: str
    ) -> list[dict]:
        """Build the messages sent to the model: system prompt, optional attached
        context, budget-fitted history, then the current turn."""
        mode_note = _CHAT_MODE_INSTRUCTIONS[self._execution_mode]
        messages: list[dict] = [
            {"role": "system", "content": f"{self.system_prompt}\n\n{mode_note}"}
        ]
        if context:
            messages.append(
                {"role": "system", "content": f"Attached context:\n{context}"}
            )
        current = {"role": "user", "content": user_message}
        messages.extend(
            self._fit_history(model_id, messages + [current], self._history, model_kwargs)
        )
        messages.append(current)
        return messages

    async def _run_llm_chat(self, user_message: str, message: Any = None) -> None:
        try:
            import litellm
        except ImportError:
            self._send_notice(
                "failed",
                "**`litellm` is not installed**, so free-form chat is unavailable. "
                "Install it with `pip install litellm` to enable it — `list`, "
                "`doc`, and `notebook` work without it (as does `run <name>` "
                "with no arguments).\n\n"
                "Say `help` to see the command reference."
            )
            return

        model_id, model_kwargs = self._resolve_model()
        if model_id is None:
            self._send_notice("failed", self.no_model_message)
            return

        context = (self.process_attachments(message) or "") if message else ""
        chat_messages = self._assemble_messages(
            model_id, model_kwargs, user_message, context
        )

        collected: list[str] = []

        async def token_stream():
            response = await litellm.acompletion(
                model=model_id, messages=chat_messages, stream=True, **model_kwargs
            )
            async for chunk in response:
                try:
                    delta = chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    delta = None
                if delta:
                    collected.append(delta)
                yield chunk

        try:
            await self.stream_message(token_stream())
        except Exception as e:
            self.log.exception("Error calling model '%s'.", model_id)
            self._send_notice("failed", f"**Error calling model `{model_id}`:** {e}")
            return

        reply = "".join(collected)
        self._append_pair(user_message, reply)
        if self._remember_suggested_run(reply) and self._execution_mode != _MODE_MANUAL:
            # Write the suggested command's code now: ask mode shows it with the
            # offer, so "run it" runs what was shown; auto mode runs it.
            last = self._last_doc
            assert last is not None
            kind = "propose" if self._execution_mode == _MODE_ASK else "run"
            async with self._working_status():
                await self._extract_and_dispatch(
                    _BACKENDS[last["backend"]], kind, last["name"], last["prompt"]
                )

    def _remember_suggested_run(self, reply: str) -> bool:
        """Record a `…-run <Symbol> <problem>` command the chat reply suggested.

        Free-form chat has no execution channel, so it can only tell the user
        which command to send. Recording that symbol lets the user answer "run
        it" (or, in ask mode, "yes") instead of retyping the command — which is
        the difference between the reply's suggestion being actionable and being
        a dead end. Returns whether a command was found.
        """
        for match in _SUGGESTED_RUN_RE.finditer(reply):
            prefix = match.group("prefix")
            name = match.group("name")
            prompt_text = match.group("prompt").strip().rstrip("`").strip()
            if not prompt_text:
                continue

            # Trust the named backend when given; otherwise find the one that
            # actually has the symbol, so a bare `run Foo …` lands correctly.
            candidates = (
                [_BACKENDS[prefix]] if prefix in _BACKENDS else list(_BACKENDS.values())
            )
            for backend in candidates:
                try:
                    if not backend.tools.has_symbol(name):
                        continue
                except backend.tools.Unavailable:
                    continue
                self._last_doc = {
                    "backend": backend.key,
                    "name": name,
                    "prompt": prompt_text,
                }
                # In ask mode the reply has already offered to run it, so let a
                # bare "yes" answer that offer.
                if self._execution_mode == _MODE_ASK:
                    self._pending_confirm = True
                return True
        return False
