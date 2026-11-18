# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FinTyche on the command line.

The same agent as the Jupyter AI persona: `TerminalAgent` mixes in
`core.FinTycheCore`, so a message typed here is routed, answered, and run
exactly as it would be in a chat. Three ways in:

- `fintyche` — an interactive session; conversation state (the last symbol,
  pinned code, history) carries from one line to the next, so "run it" and
  "rerun that with 5% vol" work as they do in chat. Ctrl-C abandons the line
  or the request in progress (generated code that is already running finishes
  first; press Ctrl-C again to stop it); Ctrl-D or `exit` quits.
- `fintyche <message>` — handle one message and exit. On a terminal, a
  clarifying question or a "Shall I run exactly this code?" offer is answered
  at a `>` prompt; an empty answer ends the session.
- `fintyche --json <message>` — handle one message without prompting and
  print one JSON object describing what happened (see `summarize`), for
  scripts and the eval harness.

The model is a LiteLLM model ID from `--model` or `$FINTYCHE_MODEL`;
credentials come from the environment the way LiteLLM reads them (provider API
keys, or AWS configuration for `bedrock/...` models). Generated code runs in
this process under the same restricted builtins as in Jupyter.

`--mode` decides what happens once FinTyche has picked a symbol for a request
that did not name one (`…-doc <word problem>`, or a chat question): `ask`
(default) shows the code and waits for a yes, `auto` runs it, `manual` only
shows the command. An explicit `…-run <name> <word problem>` is itself the
go-ahead and runs in every mode, as it does in Jupyter.

Each conversation runs on one event loop (`asyncio.Runner`), and terminal input
is read between messages rather than inside the loop, so Ctrl-C reaches the
prompt or the request in progress directly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from .core import (
    _MODE_ASK,
    _MODE_AUTO,
    _MODE_MANUAL,
    FinTycheCore,
    RunOutcome,
    load_system_prompt,
)

MODEL_ENV_VAR = "FINTYCHE_MODEL"
_EXIT_WORDS = {"exit", "quit", ":q"}
_HISTORY_FILE = Path.home() / ".fintyche_history"
_HISTORY_LENGTH = 1000
# A leading `@FinTyche`, as typed in chat, which the persona also ignores.
_MENTION_RE = re.compile(r"^@\S+\s*")

# Exit code per `summarize` status; anything unlisted exits 0.
_EXIT_CODES = {"error": 1, "failed": 1, "usage": 2, "interrupted": 130}


class TerminalAgent(FinTycheCore):
    """FinTyche for a terminal: messages are printed (as rendered markdown
    when `rich` is installed and stdout is a terminal), progress goes to
    stderr, and every message, execution, notice, and notebook is also kept
    for `summarize`."""

    no_model_message = (
        "**No model is configured.** Pass `--model <litellm-model-id>` or set "
        f"`${MODEL_ENV_VAR}`, e.g. `--model bedrock/us.anthropic.claude-sonnet-5-5`."
    )
    product_tagline = "quantitative finance assistant for the command line"
    model_settings_hint = f"pass `--model` or set `${MODEL_ENV_VAR}`"
    execution_mode_location = "the `--mode` option (`ask`, `auto`, `manual`; `--yes` is `auto`)"
    chat_model_location = f"the model from `--model` or `${MODEL_ENV_VAR}`"
    notebook_open_hint = "Open it with `jupyter lab` to run it."

    def __init__(
        self,
        model: Optional[str],
        mode: str = _MODE_ASK,
        workspace: Optional[str] = None,
        quiet: bool = False,
    ):
        self._init_state(mode)
        self.system_prompt = load_system_prompt()
        self.log = logging.getLogger("fintyche")
        self.model = model
        self.workspace = workspace or os.getcwd()
        # Quiet: print nothing; only collect (the `--json` mode).
        self.quiet = quiet
        self.messages: list[str] = []
        self.replies: list[str] = []  # streamed free-form chat replies
        self.runs: list[RunOutcome] = []
        self.notices: list[tuple[str, str]] = []  # (kind, text)
        self.notebooks: list[dict[str, str]] = []
        self._console = _rich_console() if not quiet and sys.stdout.isatty() else None
        self._status_shown = False
        # The pending command and pinned code as the current message found
        # them; see `begin_turn`.
        self._turn_pending_run: Optional[dict[str, Any]] = None
        self._turn_pinned: Optional[dict[str, Any]] = None

    def begin_turn(self) -> None:
        """Forget what earlier messages produced, so `summarize` describes only
        the next one. Conversation state is kept, and noted as it stands, so a
        question or proposal left over from an earlier message is not taken
        for one this message made (the core always stores a new dict when it
        asks or proposes)."""
        self._turn_pending_run = self._pending_run
        self._turn_pinned = self._pinned
        self.messages.clear()
        self.replies.clear()
        self.runs.clear()
        self.notices.clear()
        self.notebooks.clear()

    # ---- frontend contract ----

    def send_message(self, body: str) -> None:
        self.clear_status()
        self.messages.append(body)
        if self.quiet:
            return
        if self._console is not None:
            from rich.markdown import Markdown

            self._console.print(Markdown(body))
            self._console.print()
        else:
            print(body, end="\n\n", flush=True)

    async def stream_message(self, chunks: Any) -> None:
        parts: list[str] = []
        async for chunk in chunks:
            try:
                delta = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                delta = None
            if not delta:
                continue
            if not parts:
                self.clear_status()
            parts.append(delta)
            if not self.quiet:
                print(delta, end="", flush=True)
        if not self.quiet:
            print("\n", flush=True)
        self.replies.append("".join(parts))

    def set_status(self, status: str = "") -> None:
        if self.quiet or not sys.stderr.isatty():
            return
        sys.stderr.write(f"\r\033[K{status}")
        sys.stderr.flush()
        self._status_shown = True

    def clear_status(self) -> None:
        if self._status_shown:
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
            self._status_shown = False

    def get_workspace_dir(self) -> str:
        return self.workspace

    def process_attachments(self, message: Any) -> Optional[str]:
        return None

    def _resolve_model(self) -> tuple[Optional[str], dict[str, Any]]:
        return (self.model, {}) if self.model else (None, {})

    def _on_run(self, outcome: RunOutcome) -> None:
        self.runs.append(outcome)

    def _on_notice(self, kind: str, text: str) -> None:
        self.notices.append((kind, text))

    def _on_notebook(self, backend: str, name: str, code: str, path: str) -> None:
        self.notebooks.append({"backend": backend, "symbol": name, "code": code, "path": path})

    # ---- state a caller can ask about ----

    @property
    def asked_this_turn(self) -> bool:
        """Whether the current message left a clarifying question pending."""
        return self._pending_run is not None and self._pending_run is not self._turn_pending_run

    @property
    def proposed_this_turn(self) -> bool:
        """Whether the current message wrote code and is waiting for a yes."""
        return (
            self._pending_confirm
            and self._pinned is not None
            and self._pinned is not self._turn_pinned
        )

    @property
    def awaiting_answer(self) -> bool:
        """Whether FinTyche has asked something the next message answers:
        a clarifying question or an offer to run the code it showed."""
        return self._pending_run is not None or self._pending_confirm


def _rich_console() -> Any:
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console()


def _jsonable(value: Any) -> Any:
    """`value` as strict-JSON data: containers recursively; numpy scalars via
    `.item()`; pandas objects and arrays via `.to_dict()`/`.tolist()`, which
    keep their shape even at size 1; NaN and infinities as None; anything else
    as its repr."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and isinstance(value, np.generic):
        return _jsonable(value.item())
    for method in ("to_dict", "tolist"):
        convert = getattr(value, method, None)
        if callable(convert):
            try:
                return _jsonable(convert())
            except (TypeError, ValueError):
                continue
    return repr(value)


def summarize(agent: TerminalAgent, message: str) -> dict[str, Any]:
    """Describe what handling `message` did, from the agent's hooks.

    `status` is one of:
    - "ok" / "error" — code ran, and succeeded or raised (`error` says how);
    - "missing" — FinTyche asked a clarifying question (`question`);
    - "proposed" — code was written and shown, awaiting a yes (ask mode);
    - "notebook" — code was written into a notebook (`path`), not run;
    - "failed" — the request couldn't be carried out: a failed or unreadable
      model call, an unknown symbol, no model, or a library not installed
      (when that stops a run, the run's "error" is reported instead);
    - "usage" — the command was incomplete; the reply says how to use it;
    - "answered" — anything else: a list, doc, help, or chat reply.
    """
    run = agent.runs[-1] if agent.runs else None
    notice_kinds = {kind for kind, _ in agent.notices}
    pinned = agent._pinned
    if agent.asked_this_turn or "question" in notice_kinds:
        status = "missing"
    elif run is not None:
        status = "error" if run.error else "ok"
    elif agent.notebooks:
        status = "notebook"
    elif agent.proposed_this_turn:
        status = "proposed"
    elif notice_kinds & {"failed", "unavailable"}:
        status = "failed"
    elif "usage" in notice_kinds:
        status = "usage"
    else:
        status = "answered"

    data: dict[str, Any] = {
        "input": message,
        "status": status,
        "model": agent.model,
        "mode": agent._execution_mode,
    }
    if run is not None:
        data.update(
            backend=run.backend,
            symbol=run.name,
            prompt=run.prompt,
            code=run.code,
            result=_jsonable(run.result),
            result_repr=None if run.error else repr(run.result),
            stdout=run.stdout,
            error=run.error,
        )
    elif status == "notebook":
        data.update(agent.notebooks[-1])
    elif status == "proposed":
        data.update(
            backend=pinned["backend"],
            symbol=pinned["name"],
            prompt=pinned["prompt"],
            code=pinned["code"],
        )
    elif status == "missing":
        data["question"] = agent.messages[-1] if agent.messages else None
        if agent.asked_this_turn:
            data.update(backend=agent._pending_run["backend"], symbol=agent._pending_run["name"])
    elif status in ("failed", "usage"):
        data["error"] = next(
            text for kind, text in reversed(agent.notices) if kind != "question"
        )
    data["messages"] = list(agent.messages)
    if agent.replies:
        data["reply"] = agent.replies[-1]
    return data


def _enable_line_editing(history: bool) -> Optional[Any]:
    """Turn on readline editing for `input()`, loading the saved history when
    `history`. Returns the readline module to save history with, or None."""
    try:
        import readline
    except ImportError:
        return None
    readline.set_history_length(_HISTORY_LENGTH)
    if history:
        with contextlib.suppress(OSError):
            readline.read_history_file(str(_HISTORY_FILE))
    return readline


def _handle(runner: asyncio.Runner, agent: TerminalAgent, message: str) -> None:
    agent.begin_turn()
    runner.run(agent.handle_message(_MENTION_RE.sub("", message.strip())))


def run_once(runner: asyncio.Runner, agent: TerminalAgent, message: str, follow_up: bool) -> None:
    """Handle `message`, then — when `follow_up` — keep answering FinTyche's
    clarifying questions or run offers from the terminal until it has none or
    the user enters an empty line (or EOF)."""
    _handle(runner, agent, message)
    while follow_up and agent.awaiting_answer:
        try:
            answer = input("> ").strip()
        except EOFError:
            break
        if not answer:
            break
        _handle(runner, agent, answer)


def repl(runner: asyncio.Runner, agent: TerminalAgent, read=input) -> None:
    """Read and handle messages until EOF or an exit word. Ctrl-C clears the
    line at the prompt, and abandons the request while one is in progress
    (once any generated code already running returns, or at a second Ctrl-C)."""
    readline = _enable_line_editing(history=True)
    model_note = agent.model or f"none — pass --model or set ${MODEL_ENV_VAR}"
    print(
        f"FinTyche (model: {model_note}; mode: {agent._execution_mode}). "
        "Ask a question, or type /quantlib-help or /pyportpf-help. "
        "exit or Ctrl-D quits.\n"
    )
    try:
        while True:
            try:
                line = read("fintyche> ").strip()
            except KeyboardInterrupt:
                print()
                continue
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.lower() in _EXIT_WORDS:
                break
            try:
                _handle(runner, agent, line)
            except KeyboardInterrupt:
                agent.clear_status()
                print("\n(interrupted)\n")
    finally:
        if readline is not None:
            with contextlib.suppress(OSError):
                readline.write_history_file(str(_HISTORY_FILE))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fintyche",
        description=(
            "FinTyche: price instruments with QuantLib and optimize portfolios "
            "with PyPortfolioOpt from plain-English requests. With no message, "
            "starts an interactive session."
        ),
        epilog=(
            "examples:\n"
            '  fintyche -m bedrock/us.anthropic.claude-sonnet-5-5\n'
            "  fintyche /quantlib-list Lookback\n"
            '  fintyche "/quantlib-run Date the fifteenth of June 2025"\n'
            '  fintyche --json "/pyportpf-run EfficientFrontier max Sharpe for AAPL, MSFT"\n'
            "\n"
            "With --json, the exit code is 1 for status error/failed, 2 for usage,\n"
            "130 if interrupted, else 0; plain one-shot runs use the same codes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="what to do, as you would type it in chat (a command or a question)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=os.environ.get(MODEL_ENV_VAR),
        help=f"LiteLLM model ID (default: ${MODEL_ENV_VAR})",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--mode",
        choices=(_MODE_ASK, _MODE_AUTO, _MODE_MANUAL),
        default=_MODE_ASK,
        help="what happens once FinTyche picks a symbol for a `…-doc` word problem "
        "or a chat question: ask first (default), run automatically, or only show "
        "the command; an explicit `…-run` always runs",
    )
    mode.add_argument(
        "-y",
        "--yes",
        dest="mode",
        action="store_const",
        const=_MODE_AUTO,
        help="same as --mode auto",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="handle one message without prompting and print a JSON summary",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="directory notebooks are written into (default: the current directory)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

    message = " ".join(args.message).strip()
    if not message and not sys.stdin.isatty():
        message = sys.stdin.read().strip()
    if args.json and not message:
        parser.error("--json needs a message (as arguments or on stdin)")
    if args.out is not None and not os.path.isdir(args.out):
        parser.error(f"--out {args.out!r} is not a directory")

    agent = TerminalAgent(args.model, args.mode, args.out, quiet=args.json)
    if args.json:
        return _main_json(agent, message)

    with asyncio.Runner() as runner:
        if not message:
            repl(runner, agent)
            return 0
        _enable_line_editing(history=False)
        # Only prompt when someone can see the prompt and answer it.
        follow_up = sys.stdin.isatty() and sys.stdout.isatty()
        try:
            run_once(runner, agent, message, follow_up)
        except KeyboardInterrupt:
            agent.clear_status()
            print(file=sys.stderr)
            return _EXIT_CODES["interrupted"]
    return _EXIT_CODES.get(summarize(agent, message)["status"], 0)


def _main_json(agent: TerminalAgent, message: str) -> int:
    """Handle one message and print exactly one JSON object, whatever happens."""
    # stdout carries only the JSON: anything a library prints (litellm prints
    # a help banner when a call fails) goes to stderr instead.
    with contextlib.redirect_stdout(sys.stderr):
        try:
            with asyncio.Runner() as runner:
                _handle(runner, agent, message)
            summary = summarize(agent, message)
        except KeyboardInterrupt:
            summary = {"input": message, "status": "interrupted", "messages": agent.messages}
        except Exception as e:
            agent.log.exception("FinTyche failed handling %r.", message)
            summary = {
                "input": message,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "messages": agent.messages,
            }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return _EXIT_CODES.get(summary["status"], 0)


if __name__ == "__main__":
    sys.exit(main())
