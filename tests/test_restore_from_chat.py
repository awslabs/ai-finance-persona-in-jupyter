# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for rebuilding FinTyche's memory from the chat's saved messages.

FinTyche's memory lives in the server process, so after a restart a reopened
chat showed its whole history while FinTyche saw none of it ("rerun" got "I
don't see a previous model"). `_restore_from_chat` replays the saved messages
on the first message handled, as chat apps rebuild a conversation from its
transcript. These use an in-memory chat of real `Message` objects.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from jupyterlab_chat.models import Message

from jupyter_ai_fintyche import persona as p

pytest.importorskip("pypfopt", reason="PyPortfolioOpt is not installed")

ME = "jupyter-ai-personas::jupyter_ai_fintyche::FinTychePersona"
USER = "912d5e168e46414cb62d04bec81fe364"
PROBLEM = "Black-Litterman for AAPL, MSFT with views AAPL 8%"
CODE = "tickers = ['AAPL', 'MSFT']\nresult = {'n': len(tickers)}"


class _Chat:
    def __init__(self, messages: list[Message], fail: bool = False):
        self.messages = messages
        self.fail = fail
        self.reads = 0

    def get_messages(self) -> list[Message]:
        self.reads += 1
        if self.fail:
            raise RuntimeError("chat unavailable")
        return list(self.messages)


class _Agent(p.FinTychePersona):
    id = ME
    log = logging.getLogger("test")

    def __init__(self, chat: _Chat):
        self.chat = chat
        self._init_state(p._MODE_AUTO)
        self._restored = False
        self.messages: list[str] = []
        self.ran: list[str] = []
        self.chat_calls: list[str] = []

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
        intent = "run" if body == "rerun" else "other"
        return {"intent": intent, "changes": ""}

    async def _extract_snippet(self, *args, **kwargs):
        raise AssertionError("a rerun must not ask the model for new code")

    def _execute_snippet(self, backend, name, code, prompt_text="", note="") -> None:
        self.ran.append(code)

    async def _run_llm_chat(self, user_message, message=None) -> None:
        self.chat_calls.append(user_message)


def _msg(n: int, sender: str, body: str, **extra) -> Message:
    return Message(id=f"m{n}", time=float(n), sender=sender, body=body, **extra)


def _transcript() -> list[Message]:
    """A day-old chat: a question, a reply suggesting a command, "yes", the
    proposed code, then the result posted in a second message."""
    return [
        _msg(0, ME, "Hi! I'm FinTyche."),  # a greeting before any request
        _msg(1, USER, "Give me the best risk-adjusted mix of AAPL and MSFT"),
        _msg(2, ME, f"Here's the command:\n\n/pyportpf-run BlackLittermanModel {PROBLEM}\n"),
        _msg(3, USER, "yes"),
        _msg(4, ME, f"**Code for `BlackLittermanModel`:**\n\n```python\n{CODE}\n```\n\nShall I run it?"),
        _msg(5, ME, f"**`BlackLittermanModel`** →\n\n```python\n{CODE}\n```\n\nResult:\n\n- **n**: 2"),
        _msg(6, USER, "a message that was deleted", deleted=True),
        _msg(7, USER, "thanks"),
        _msg(8, ME, "You're welcome."),
    ]


def test_history_is_rebuilt_alternating_from_the_transcript() -> None:
    agent = _Agent(_Chat(_transcript()))
    agent._restore_from_chat()

    roles = [m["role"] for m in agent._history]
    assert roles == ["user", "assistant"] * 3
    # The greeting before any request is dropped; FinTyche's code and result
    # messages are one turn; the deleted message is gone.
    assert agent._history[0]["content"].startswith("Give me the best")
    assert "Shall I run it?" in agent._history[3]["content"]
    assert "Result:" in agent._history[3]["content"]
    assert all("deleted" not in m["content"] for m in agent._history)


def test_last_run_is_pinned_with_its_word_problem() -> None:
    agent = _Agent(_Chat(_transcript()))
    agent._restore_from_chat()

    assert agent._last_doc == {
        "backend": "pyportpf",
        "name": "BlackLittermanModel",
        "prompt": PROBLEM,  # from the suggested command, not "yes"
    }
    assert agent._pinned["code"] == CODE
    assert agent._pinned["error"] is None


def test_a_failed_last_run_is_pinned_with_its_error() -> None:
    messages = _transcript()[:4] + [
        _msg(9, ME, "**Error running the generated code for `BlackLittermanModel`:** "
                    f"TypeError: bad call\n\n```python\n{CODE}\n```"),
    ]
    agent = _Agent(_Chat(messages))
    agent._restore_from_chat()

    assert agent._pinned["error"] == "TypeError: bad call"


def test_without_a_suggested_command_the_longest_request_is_used() -> None:
    messages = [
        _msg(1, USER, "Optimize AAPL and MSFT with Black-Litterman, AAPL 8%"),
        _msg(2, USER, "ok"),
        _msg(3, ME, f"**`BlackLittermanModel`** →\n\n```python\n{CODE}\n```"),
    ]
    agent = _Agent(_Chat(messages))
    agent._restore_from_chat()
    assert agent._last_doc["prompt"] == "Optimize AAPL and MSFT with Black-Litterman, AAPL 8%"


def test_rerun_in_a_reopened_chat_runs_the_saved_code() -> None:
    chat = _Chat(_transcript())
    agent = _Agent(chat)

    asyncio.run(agent.process_message(_msg(10, USER, "rerun")))

    assert agent.ran == [CODE]
    assert agent.chat_calls == []


def test_the_message_being_handled_is_not_replayed_as_history() -> None:
    current = _msg(10, USER, "what did we do last time?")
    agent = _Agent(_Chat([*_transcript(), current]))

    asyncio.run(agent.process_message(current))

    assert all(m["content"] != current.body for m in agent._history)
    assert agent.chat_calls == [current.body]


def test_restore_reads_the_chat_once() -> None:
    chat = _Chat(_transcript())
    agent = _Agent(chat)
    agent._restore_from_chat()
    agent._restore_from_chat()
    assert chat.reads == 1


def test_restore_keeps_memory_the_session_already_has() -> None:
    agent = _Agent(_Chat(_transcript()))
    agent._history = [{"role": "user", "content": "now"}, {"role": "assistant", "content": "ok"}]
    agent._last_doc = {"backend": "quantlib", "name": "Date", "prompt": "today"}

    agent._restore_from_chat()

    assert len(agent._history) == 2
    assert agent._last_doc["name"] == "Date"


def test_unreadable_chat_leaves_memory_empty() -> None:
    agent = _Agent(_Chat([], fail=True))
    agent._restore_from_chat()
    assert agent._history == [] and agent._last_doc is None
