# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the `fintyche` command line.

`main()` is driven in-process with the LLM stubbed, and these assert on what a
script sees: the JSON summary on stdout and the exit code.
"""

from __future__ import annotations

import io
import json
import signal
import sys
import types

import pytest

from jupyter_ai_fintyche import cli, core

MODEL = "stub-model"


def _stub_llm(monkeypatch, *replies: str) -> None:
    """Stub `litellm` so successive calls return `replies` in order."""
    queue = list(replies)

    async def acompletion(model, messages, **kwargs):
        message = types.SimpleNamespace(content=queue.pop(0))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=acompletion))


def _code(code: str) -> str:
    return json.dumps({"status": "ok", "code": code})


def _run_json(capsys, *argv: str) -> tuple[int, dict]:
    code = cli.main(["--json", "-m", MODEL, *argv])
    return code, json.loads(capsys.readouterr().out)


def test_json_run_reports_code_and_result(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, _code("result = {'price': 1.5, 'n': 2}"))

    code, out = _run_json(capsys, "--yes", "/quantlib-run Date price something")

    assert code == 0
    assert out["status"] == "ok"
    assert (out["backend"], out["symbol"]) == ("quantlib", "Date")
    assert out["code"] == "result = {'price': 1.5, 'n': 2}"
    assert out["result"] == {"price": 1.5, "n": 2}
    assert out["error"] is None


def test_json_run_error_exits_nonzero(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, _code("result = 1 / 0"))

    code, out = _run_json(capsys, "--yes", "/quantlib-run Date divide by zero")

    assert code == 1
    assert out["status"] == "error"
    assert out["error"].startswith("ZeroDivisionError")


def test_json_in_ask_mode_proposes_without_running(monkeypatch, capsys) -> None:
    """Ask mode governs code FinTyche proposes after picking a symbol (an
    explicit `…-run` command is itself the go-ahead, in every mode)."""
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, json.dumps({"status": "ok", "name": "Date"}), _code("result = 2"))

    code, out = _run_json(capsys, "/quantlib-doc the date two days from now")

    assert code == 0
    assert out["status"] == "proposed"
    assert out["code"] == "result = 2"
    assert "result" not in out


def test_json_reports_a_clarifying_question(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, json.dumps({"status": "missing", "question": "What strike?"}))

    code, out = _run_json(capsys, "--yes", "/quantlib-run VanillaOption price a call")

    assert code == 0
    assert out["status"] == "missing"
    assert out["question"] == "What strike?"
    assert out["symbol"] == "VanillaOption"


def test_json_without_a_model_fails_cleanly(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    monkeypatch.delenv(cli.MODEL_ENV_VAR, raising=False)

    code = cli.main(["--json", "--yes", "/quantlib-run Date the first of May"])
    out = json.loads(capsys.readouterr().out)

    assert code == 1
    assert out["status"] == "failed"
    assert out["messages"] == [cli.TerminalAgent.no_model_message]


def test_json_stdout_carries_only_json_when_a_library_prints(monkeypatch, capsys) -> None:
    """litellm prints a help banner to stdout when a call fails."""
    pytest.importorskip("QuantLib")

    async def acompletion(model, messages, **kwargs):
        print("Give Feedback / Get Help: ...")
        raise RuntimeError("token expired")

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=acompletion))

    code = cli.main(["--json", "-m", MODEL, "--yes", "/quantlib-run Date x"])
    captured = capsys.readouterr()

    assert code == 1
    assert json.loads(captured.out)["status"] == "failed"
    assert "Give Feedback" in captured.err


def test_plain_one_shot_prints_markdown(capsys) -> None:
    pytest.importorskip("QuantLib")

    assert cli.main(["/quantlib-list", "Lookback"]) == 0
    assert "ContinuousFixedLookbackOption" in capsys.readouterr().out


def test_json_requires_a_message(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--json"])
    assert exc.value.code == 2


def test_yes_is_auto_mode() -> None:
    args = cli.build_parser().parse_args(["--yes", "hi"])
    assert args.mode == core._MODE_AUTO


def test_jsonable_handles_library_values() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    value = {
        "weights": pd.Series({"AAPL": 0.6, "MSFT": 0.4}),
        "sharpe": np.float64(1.25),
        "perf": (0.1, 0.2),
        "date": object(),
    }
    out = cli._jsonable(value)

    assert out["weights"] == {"AAPL": 0.6, "MSFT": 0.4}
    assert out["sharpe"] == 1.25
    assert out["perf"] == [0.1, 0.2]
    assert out["date"].startswith("<object object")
    json.dumps(out)


def test_jsonable_keeps_size_one_containers() -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    assert cli._jsonable(pd.Series({"AAPL": 1.0})) == {"AAPL": 1.0}
    assert cli._jsonable(np.array([5.0])) == [5.0]
    assert cli._jsonable(np.float64("inf")) is None


def test_repl_keeps_state_between_lines(monkeypatch, capsys) -> None:
    """A later line can run what an earlier one resolved, as in chat."""
    pytest.importorskip("QuantLib")
    _stub_llm(
        monkeypatch,
        json.dumps({"status": "ok", "name": "Date"}),  # doc: pick the symbol
        _code("result = 7"),  # write the code it proposes
        json.dumps({"intent": "run", "changes": ""}),  # "yes" means run it
    )
    _no_history_file(monkeypatch)
    agent = cli.TerminalAgent(MODEL)

    with cli.asyncio.Runner() as runner:
        cli.repl(runner, agent, read=_scripted("/quantlib-doc a date", "yes", "exit"))

    assert [run.result for run in agent.runs] == [7]
    assert "Shall I run exactly this code?" in capsys.readouterr().out


def test_repl_survives_ctrl_c_at_the_prompt_and_mid_request(monkeypatch, capsys) -> None:
    """Ctrl-C clears the line at the prompt, abandons a request in progress,
    and leaves the session running either way."""
    pytest.importorskip("QuantLib")

    async def interrupted(model, messages, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=interrupted))
    _no_history_file(monkeypatch)
    agent = cli.TerminalAgent(MODEL)

    with cli.asyncio.Runner() as runner:
        cli.repl(
            runner,
            agent,
            read=_scripted(KeyboardInterrupt, "/quantlib-run Date x", "/quantlib-list Lookback"),
        )

    out = capsys.readouterr().out
    assert "(interrupted)" in out
    assert "ContinuousFixedLookbackOption" in out


def _scripted(*lines):
    """A stand-in for `input` that returns `lines` in order (raising any that
    are exceptions), then EOF."""
    queue = list(lines)

    def read(prompt: str = "") -> str:
        if not queue:
            raise EOFError
        line = queue.pop(0)
        if isinstance(line, type) and issubclass(line, BaseException):
            raise line
        return line

    return read


def _no_history_file(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_HISTORY_FILE", cli.Path("/nonexistent/fintyche_history"))


# -- status labels the eval harness relies on ---------------------------------


def test_json_unknown_suggested_symbol_is_failed(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, json.dumps({"status": "ok", "name": "NoSuchWidget"}))

    code, out = _run_json(capsys, "/quantlib-doc price a widget")

    assert (code, out["status"]) == (1, "failed")
    assert "NoSuchWidget" in out["error"]


def test_json_incomplete_command_is_usage(capsys) -> None:
    pytest.importorskip("QuantLib")

    code, out = _run_json(capsys, "/quantlib-run")

    assert (code, out["status"]) == (2, "usage")
    assert out["error"].startswith("Usage:")


def test_json_doc_clarifying_question_is_missing(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, json.dumps({"status": "missing", "question": "Which instrument?"}))

    code, out = _run_json(capsys, "/quantlib-doc price it")

    assert (code, out["status"], out["question"]) == (0, "missing", "Which instrument?")


def test_json_unavailable_library_is_failed(monkeypatch, capsys) -> None:
    from jupyter_ai_fintyche import quantlib_tools

    def unavailable(*args, **kwargs):
        raise quantlib_tools.QuantLibUnavailable("QuantLib is not installed.")

    monkeypatch.setattr(quantlib_tools, "list_entries", unavailable)

    code, out = _run_json(capsys, "/quantlib-list Lookback")

    assert (code, out["status"], out["error"]) == (1, "failed", "QuantLib is not installed.")


def test_json_notebook_reports_its_path_and_code(monkeypatch, capsys, tmp_path) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, _code("result = ql.Date(1, 5, 2025)"))

    code, out = _run_json(
        capsys, "--out", str(tmp_path), "/quantlib-notebook Date the first of May as may"
    )

    assert (code, out["status"]) == (0, "notebook")
    assert out["path"] == str(tmp_path / "may.ipynb")
    assert out["code"] == "result = ql.Date(1, 5, 2025)"


def test_json_nan_result_is_strict_json(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, _code("result = {'price': float('nan'), 'n': 1}"))

    code = cli.main(["--json", "-m", MODEL, "/quantlib-run Date nan"])
    raw = capsys.readouterr().out

    assert code == 0
    assert "NaN" not in raw
    assert json.loads(raw)["result"] == {"price": None, "n": 1}


def test_json_unexpected_exception_still_prints_json(monkeypatch, capsys) -> None:
    async def boom(self, body, message=None):
        raise RuntimeError("kaput")

    monkeypatch.setattr(cli.TerminalAgent, "handle_message", boom)

    code, out = _run_json(capsys, "anything")

    assert (code, out["status"], out["error"]) == (1, "failed", "RuntimeError: kaput")


def test_json_interrupt_still_prints_json(monkeypatch, capsys) -> None:
    async def interrupted(self, body, message=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.TerminalAgent, "handle_message", interrupted)

    code, out = _run_json(capsys, "anything")

    assert (code, out["status"]) == (130, "interrupted")


def test_plain_one_shot_exit_code_matches_json(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _stub_llm(monkeypatch, json.dumps({"status": "ok", "name": "NoSuchWidget"}))

    assert cli.main(["-m", MODEL, "/quantlib-doc price a widget"]) == 1


def test_message_on_stdin(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    monkeypatch.setattr(sys, "stdin", io.StringIO("/quantlib-list Lookback\n"))

    code, out = _run_json(capsys)

    assert (code, out["status"]) == (0, "answered")
    assert "ContinuousFixedLookbackOption" in out["messages"][0]


def test_out_must_be_a_directory(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--out", str(tmp_path / "missing"), "/quantlib-help"])
    assert exc.value.code == 2


def test_a_leading_mention_is_ignored(capsys) -> None:
    pytest.importorskip("QuantLib")

    code, out = _run_json(capsys, "@FinTyche /quantlib-list Lookback")

    assert out["status"] == "answered"
    assert "ContinuousFixedLookbackOption" in out["messages"][0]


def test_a_message_clears_the_progress_line(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stderr", types.SimpleNamespace(
        isatty=lambda: True, write=lambda s: written.append(s), flush=lambda: None
    ))
    written: list[str] = []
    agent = cli.TerminalAgent(MODEL)

    agent.set_status("is working.. 2s")
    agent.send_message("done")

    assert written[-1] == "\r\033[K"


# -- a real Ctrl-C (SIGINT) while generated code is running --------------------


def _code_that_gets_ctrl_c(monkeypatch) -> None:
    """Run generated code that receives a real SIGINT partway through, the way
    a user's Ctrl-C arrives during a slow yfinance download."""
    from jupyter_ai_fintyche import quantlib_tools

    def run_snippet(code):
        signal.raise_signal(signal.SIGINT)
        return {"price": 1.0}, ""

    monkeypatch.setattr(quantlib_tools, "run_snippet", run_snippet)
    _stub_llm(monkeypatch, _code("result = 1"))


def test_ctrl_c_during_code_execution_abandons_the_request(monkeypatch) -> None:
    pytest.importorskip("QuantLib")
    _code_that_gets_ctrl_c(monkeypatch)
    agent = cli.TerminalAgent(MODEL, quiet=True)

    with cli.asyncio.Runner() as runner, pytest.raises(KeyboardInterrupt):
        cli._handle(runner, agent, "/quantlib-run Date x")


def test_json_reports_ctrl_c_during_code_execution(monkeypatch, capsys) -> None:
    pytest.importorskip("QuantLib")
    _code_that_gets_ctrl_c(monkeypatch)

    code, out = _run_json(capsys, "/quantlib-run Date x")

    assert (code, out["status"]) == (130, "interrupted")


# -- status describes the latest message, not state left by earlier ones -------


def _turns(monkeypatch, replies: list[str], messages: list[str], mode=core._MODE_ASK) -> dict:
    """Send `messages` to one agent in turn and summarize the last one."""
    _stub_llm(monkeypatch, *replies)
    agent = cli.TerminalAgent(MODEL, mode, quiet=True)
    with cli.asyncio.Runner() as runner:
        for message in messages:
            cli._handle(runner, agent, message)
    return cli.summarize(agent, messages[-1])


def test_a_failure_after_a_question_is_not_reported_as_missing(monkeypatch) -> None:
    pytest.importorskip("QuantLib")
    replies = [json.dumps({"status": "missing", "question": "What strike?"}), "not json"]

    out = _turns(monkeypatch, replies, ["/quantlib-run VanillaOption price a call", "strike 100"])

    assert out["status"] == "failed"
    assert "question" not in out


def test_chat_after_a_proposal_is_not_reported_as_proposed(monkeypatch) -> None:
    pytest.importorskip("QuantLib")
    replies = [
        json.dumps({"status": "ok", "name": "Date"}),  # doc picks a symbol
        _code("result = 2"),  # and proposes code
        json.dumps({"intent": "other", "changes": ""}),  # the next message is new
    ]

    async def no_chat(self, user_message, message=None):
        self.send_message("an answer")

    monkeypatch.setattr(cli.TerminalAgent, "_run_llm_chat", no_chat)

    out = _turns(monkeypatch, replies, ["/quantlib-doc a date", "what is a lookback?"])

    assert out["status"] == "answered"
    assert "code" not in out


def test_a_failed_chat_after_a_proposal_is_failed(monkeypatch) -> None:
    pytest.importorskip("QuantLib")
    replies = [
        json.dumps({"status": "ok", "name": "Date"}),  # doc picks a symbol
        _code("result = 2"),  # and proposes code
        json.dumps({"intent": "other", "changes": ""}),  # the next message is new
        "not a stream",  # the chat call returns something unusable
    ]

    out = _turns(monkeypatch, replies, ["/quantlib-doc a date", "what is a lookback?"])

    assert out["status"] == "failed"
    assert out["error"].startswith("**Error calling model")
