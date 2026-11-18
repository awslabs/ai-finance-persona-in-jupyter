# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for pinned code: what FinTyche shows is what runs.

Drives the real `_extract_and_dispatch` with the model call and snippet
execution stubbed, so these pin which code runs and what the code writer is
told — not what a model would write. The persona is built without
`BasePersona.__init__`, which needs a live chat stack.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types

import pytest

from jupyter_ai_fintyche import persona as p

pytest.importorskip("pypfopt", reason="PyPortfolioOpt is not installed")

PROBLEM = "Black-Litterman for AAPL, MSFT with views AAPL 8%"


class _Agent(p.FinTychePersona):
    def __init__(self, replies: list[str], mode: str = p._MODE_ASK):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []
        self.messages: list[str] = []
        self.ran: list[str] = []
        self.errors: list[str] = []  # queued run errors, consumed per run
        self._init_state(mode)
        # The stubbed run-intent judgement: these messages ask to run it.
        self.run_requests = {"yes": "", "run it": "", "run it with 3 years of data": "use 3 years of data"}

    def send_message(self, body: str) -> None:
        self.messages.append(body)

    def set_status(self, status: str = "") -> None:
        pass

    def clear_status(self) -> None:
        pass

    def _report_model_config(self) -> None:
        pass

    def _resolve_model(self):
        return "fake-model", {}

    async def _classify_run_intent(self, body):
        if body in self.run_requests:
            return {"intent": "run", "changes": self.run_requests[body]}
        return {"intent": "other", "changes": ""}

    def _execute_snippet(self, backend, name, code, prompt_text="", note=""):
        self.ran.append(code)
        return self.errors.pop(0) if self.errors else None


@pytest.fixture
def agent_with(monkeypatch: pytest.MonkeyPatch):
    def make(replies: list[str], mode: str = p._MODE_ASK) -> _Agent:
        agent = _Agent(replies, mode)

        async def acompletion(model, messages, **kwargs):
            agent.requests.append(list(messages))
            message = types.SimpleNamespace(content=agent.replies.pop(0))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        monkeypatch.setitem(
            sys.modules, "litellm", types.SimpleNamespace(acompletion=acompletion)
        )
        return agent

    return make


def _ok(code: str, **extra: str) -> str:
    return json.dumps({"status": "ok", "code": code, **extra})


def _dispatch(agent: _Agent, kind: str, prompt: str = PROBLEM) -> None:
    backend = p._BACKENDS["pyportpf"]
    asyncio.run(agent._extract_and_dispatch(backend, kind, "BlackLittermanModel", prompt))


class _Msg:
    def __init__(self, body: str):
        self.body = body


def _send(agent: _Agent, body: str) -> None:
    asyncio.run(agent.process_message(_Msg(body)))


def test_yes_runs_exactly_the_proposed_code(agent_with) -> None:
    agent = agent_with([_ok("result = 1")])
    _dispatch(agent, "propose")
    assert "result = 1" in agent.messages[-1]
    assert agent._pending_confirm is True

    _send(agent, "yes")

    assert agent.ran == ["result = 1"]
    assert len(agent.requests) == 1  # no second model call


def test_run_it_again_reruns_without_regenerating(agent_with) -> None:
    agent = agent_with([_ok("result = 1")])
    _dispatch(agent, "run")
    _send(agent, "run it")
    assert agent.ran == ["result = 1", "result = 1"]
    assert len(agent.requests) == 1


def test_fix_after_error_edits_the_failed_code(agent_with) -> None:
    failed = "pi = pypfopt.black_litterman.market_implied_prior_returns(prices)"
    agent = agent_with([_ok(failed), _ok("result = 2", changes="Passed S.")])
    agent.errors.append(
        "TypeError: market_implied_prior_returns() missing 1 required "
        "positional argument: 'cov_matrix'"
    )
    _dispatch(agent, "run")

    # The chat suggests the command again, reworded; it is still a repair.
    _dispatch(agent, "propose", prompt="BL for AAPL, MSFT, pass the covariance")

    edit_request = agent.requests[1][-1]["content"]
    assert failed in edit_request
    assert "missing 1 required positional argument" in edit_request
    # The real signature of the function the error names, not the model's memory.
    assert "market_implied_prior_returns(market_caps, risk_aversion, cov_matrix" in edit_request
    assert "**Changed:** Passed S." in agent.messages[-1]
    assert agent._pinned["code"] == "result = 2"


def test_changed_request_edits_rather_than_regenerates(agent_with) -> None:
    agent = agent_with([_ok("result = 1"), _ok("result = 3")])
    _dispatch(agent, "run")
    _send(agent, "run it with 3 years of data")
    assert "Previous code for this symbol" in agent.requests[1][-1]["content"]
    assert agent.ran == ["result = 1", "result = 3"]


def test_notebook_without_a_word_problem_uses_pinned_code(agent_with, tmp_path) -> None:
    agent = agent_with([_ok("result = 1")])
    agent.get_workspace_dir = lambda: str(tmp_path)  # type: ignore[method-assign]
    _dispatch(agent, "run")

    _send(agent, "/pyportpf-notebook BlackLittermanModel as bl.ipynb")

    import nbformat

    nb = nbformat.read(tmp_path / "bl.ipynb", as_version=4)
    assert "result = 1" in nb.cells[2].source
    assert len(agent.requests) == 1


def test_run_wrapped_around_a_command_dispatches_it(agent_with, tmp_path) -> None:
    """`run /pyportpf-notebook …` used to report "no symbol named /pyportpf-notebook"."""
    agent = agent_with([])
    agent.get_workspace_dir = lambda: str(tmp_path)  # type: ignore[method-assign]
    _send(agent, "run /pyportpf-notebook EfficientFrontier as ef.ipynb")
    assert (tmp_path / "ef.ipynb").exists()
    assert "Notebook created" in agent.messages[-1]
