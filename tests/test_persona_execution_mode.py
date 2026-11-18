# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for who runs the generated code, and for acting on a request to run it.

These drive `FinTychePersona.process_message` directly with the LLM calls and
snippet execution stubbed out, so they assert on routing decisions — which
handler a message reaches and what gets executed — rather than on model output.
Whether a message accepts or declines a run is the model's judgement
(`_classify_run_intent`); here it is stubbed per message via `intents`.

The persona is built with `object.__new__` to skip `BasePersona.__init__`, which
needs a live chat/awareness stack; every attribute the routed paths touch is set
explicitly below.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types

import pytest

from jupyter_ai_fintyche import core
from jupyter_ai_fintyche import persona as p


class _Recorder(p.FinTychePersona):
    """A persona that records outgoing messages and executions instead of
    talking to a chat, an LLM, or the libraries."""

    def __init__(self, mode: str = core._DEFAULT_EXECUTION_MODE):
        # Deliberately no super().__init__() — see module docstring.
        self.messages: list[str] = []
        self.executed: list[tuple[str, str, str]] = []  # (backend, name, prompt)
        self.chat_calls: list[str] = []
        self._init_state(mode)
        # The symbol the stubbed doc-resolution step returns.
        self.resolved_name = "VanillaOption"
        # What the stubbed run-intent judgement says per message; anything
        # unlisted is "other".
        self.intents: dict[str, dict[str, str]] = {}
        self.classified: list[str] = []

    # -- framework seams -------------------------------------------------
    def send_message(self, body: str) -> None:
        self.messages.append(body)

    def _report_model_config(self) -> None:
        pass

    def set_status(self, status: str = "is typing...") -> None:
        pass

    def clear_status(self) -> None:
        pass

    def _send_doc(self, backend, name, note=None, offer=True) -> None:
        self.messages.append(f"[doc {name} offer={offer}]")

    async def _run_llm_chat(self, user_message, message=None) -> None:
        self.chat_calls.append(user_message)

    async def _classify_run_intent(self, body):
        self.classified.append(body)
        return self.intents.get(body, {"intent": "other", "changes": ""})

    async def _resolve_doc_symbol(self, backend, prompt_text):
        return {"status": "ok", "name": self.resolved_name}

    async def _extract_and_dispatch(
        self, backend, kind, name, prompt_text, filename=None
    ) -> None:
        # Mirror the real method's bookkeeping, then record instead of running.
        # A "propose" shows code and (in ask mode) arms the offer, as
        # `_send_proposal` does; only "run" counts as an execution.
        self._last_doc = {
            "backend": backend.key,
            "name": name,
            "prompt": prompt_text,
        }
        if kind == "propose":
            if self._execution_mode == p._MODE_ASK:
                self._pending_confirm = True
            self.messages.append(f"[proposal {name}]")
            return
        self.executed.append((backend.key, name, prompt_text))

    # -- helpers ---------------------------------------------------------
    @property
    def last(self) -> str:
        return self.messages[-1] if self.messages else ""


class _Msg:
    def __init__(self, body: str):
        self.body = body


def send(persona: _Recorder, body: str) -> None:
    asyncio.run(persona.process_message(_Msg(body)))


WORD_PROBLEM = "price a European call, spot 100, strike 105, 20% vol"


RUN = {"intent": "run", "changes": ""}
DECLINE = {"intent": "decline", "changes": ""}


def _resolve(persona: _Recorder, problem: str = WORD_PROBLEM) -> None:
    """Drive a doc resolution so a symbol is on the table."""
    send(persona, f"doc {problem}")


# ---------------------------------------------------------------------------
# Asking for a run in your own words runs it; no particular phrase is needed.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrasing", ["rerun", "sure, go for it", "let's see the numbers", "do it again"]
)
def test_message_the_model_reads_as_run_executes(phrasing):
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)
    persona.intents[phrasing] = RUN

    send(persona, phrasing)

    assert persona.executed == [("quantlib", "VanillaOption", WORD_PROBLEM)]
    assert persona.chat_calls == []


def test_nothing_to_run_goes_straight_to_chat():
    persona = _Recorder()
    send(persona, "could you rerun that")
    assert persona.classified == []  # no model call to judge it
    assert persona.chat_calls == ["could you rerun that"]


def test_run_command_with_nothing_to_run_says_so():
    persona = _Recorder()
    send(persona, "run it")
    assert persona.executed == []
    assert "nothing to run again" in persona.last


def test_run_request_with_a_change_refines_the_word_problem():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)
    persona.intents["run that again with 5% vol"] = {"intent": "run", "changes": "use 5% vol"}

    send(persona, "run that again with 5% vol")

    _, name, prompt = persona.executed[0]
    assert name == "VanillaOption"
    assert WORD_PROBLEM in prompt
    assert "Additional detail: use 5% vol" in prompt


def test_run_command_with_a_referent_is_judged_too():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)
    persona.intents["run the model with 5% vol"] = {"intent": "run", "changes": "5% vol"}

    send(persona, "run the model with 5% vol")

    assert persona.executed[0][1] == "VanillaOption"
    assert "Additional detail: 5% vol" in persona.executed[0][2]


def test_mistyped_symbol_still_errors_rather_than_running_last():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)

    send(persona, "run VanilaOption price a call")

    assert persona.executed == []
    assert "has no symbol named `VanilaOption`" in persona.last


def test_bare_run_uses_last_resolved_symbol():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)

    send(persona, "run")

    assert persona.executed == [("quantlib", "VanillaOption", WORD_PROBLEM)]


def test_bare_run_without_context_shows_usage():
    persona = _Recorder()
    send(persona, "run")
    assert persona.executed == []
    assert "Usage:" in persona.last


# ---------------------------------------------------------------------------
# Execution mode: the user's choice of who runs the code.
# ---------------------------------------------------------------------------


def test_auto_mode_runs_without_being_asked():
    persona = _Recorder(mode=p._MODE_AUTO)
    _resolve(persona)
    assert persona.executed == [("quantlib", "VanillaOption", WORD_PROBLEM)]
    assert persona._pending_confirm is False


def test_manual_mode_never_runs_on_its_own():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)
    assert persona.executed == []
    assert persona._pending_confirm is False
    # Manual mode keeps the "here's the command" hint.
    assert "offer=True" in persona.messages[-1]


def test_ask_mode_offers_then_runs_on_agreement():
    persona = _Recorder(mode=p._MODE_ASK)
    _resolve(persona)

    assert persona.executed == []
    assert persona._pending_confirm is True
    # The offer shows the code that agreeing will run.
    assert persona.last == "[proposal VanillaOption]"

    persona.intents["yeah why not"] = RUN
    send(persona, "yeah why not")

    assert persona.executed == [("quantlib", "VanillaOption", WORD_PROBLEM)]
    assert persona._pending_confirm is False


def test_ask_mode_declines_when_the_model_reads_a_no():
    persona = _Recorder(mode=p._MODE_ASK)
    _resolve(persona)
    persona.intents["nah, not now"] = DECLINE

    send(persona, "nah, not now")

    assert persona.executed == []
    assert persona._pending_confirm is False
    assert "not running it" in persona.last
    assert "quantlib-run VanillaOption" in persona.last


def test_ask_mode_unrelated_reply_falls_through_to_chat():
    """An offer must not hijack the next message when it isn't an answer."""
    persona = _Recorder(mode=p._MODE_ASK)
    _resolve(persona)

    send(persona, "what is implied volatility?")

    assert persona.executed == []
    assert persona.chat_calls == ["what is implied volatility?"]
    assert persona._pending_confirm is True  # still open


def test_decline_with_no_offer_open_is_just_chat():
    persona = _Recorder(mode=p._MODE_MANUAL)
    _resolve(persona)
    persona.intents["no thanks"] = DECLINE

    send(persona, "no thanks")

    assert persona.chat_calls == ["no thanks"]


def test_exact_symbol_lookup_makes_no_run_offer():
    """A bare symbol name carries no word problem, so there is nothing to run."""
    persona = _Recorder(mode=p._MODE_ASK)
    send(persona, "Date")
    assert persona.executed == []
    assert persona._pending_confirm is False


def test_run_it_after_exact_symbol_lookup_asks_for_a_word_problem():
    persona = _Recorder(mode=p._MODE_ASK)
    send(persona, "Date")
    persona.intents["run it"] = RUN

    send(persona, "run it")

    assert persona.executed == []
    assert "no word problem" in persona.last


# ---------------------------------------------------------------------------
# Mode setting plumbing.
# ---------------------------------------------------------------------------


def test_update_model_settings_switches_mode():
    persona = _Recorder(mode=p._MODE_ASK)
    asyncio.run(
        persona.update_model_settings({p._EXECUTION_MODE_SETTING_ID: p._MODE_AUTO})
    )
    assert persona._execution_mode == p._MODE_AUTO


def test_update_model_settings_ignores_unknown_value():
    persona = _Recorder(mode=p._MODE_ASK)
    asyncio.run(
        persona.update_model_settings({p._EXECUTION_MODE_SETTING_ID: "bogus"})
    )
    assert persona._execution_mode == p._MODE_ASK


def test_mode_change_clears_a_pending_offer():
    persona = _Recorder(mode=p._MODE_ASK)
    _resolve(persona)
    assert persona._pending_confirm is True

    asyncio.run(
        persona.update_model_settings({p._EXECUTION_MODE_SETTING_ID: p._MODE_MANUAL})
    )

    assert persona._pending_confirm is False


def test_setting_reports_current_mode():
    setting = p._execution_mode_setting(p._MODE_AUTO)
    assert setting.id == p._EXECUTION_MODE_SETTING_ID
    assert setting.current == p._MODE_AUTO
    assert [o.id for o in setting.options] == [
        p._MODE_ASK,
        p._MODE_AUTO,
        p._MODE_MANUAL,
    ]


# ---------------------------------------------------------------------------
# An explicit command must always win over an outstanding prompt.
# ---------------------------------------------------------------------------


def test_command_escapes_a_pending_clarifying_question():
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._pending_run = {
        "backend": "quantlib",
        "kind": "run",
        "name": "VanillaOption",
        "prompt": "an earlier problem",
        "filename": None,
    }

    send(persona, "/quantlib-run EuropeanExercise expire in a year")

    assert persona._pending_run is None
    assert persona.executed == [
        ("quantlib", "EuropeanExercise", "expire in a year")
    ]


def test_command_escapes_a_pending_offer():
    persona = _Recorder(mode=p._MODE_ASK)
    _resolve(persona)
    assert persona._pending_confirm is True

    send(persona, "list swap")

    assert persona._pending_confirm is False
    assert persona.executed == []


def test_clarifying_answer_still_reaches_the_extractor():
    """A plain-language answer to a clarifying question is not a command."""
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._pending_run = {
        "backend": "quantlib",
        "kind": "run",
        "name": "VanillaOption",
        "prompt": "price a call",
        "filename": None,
    }

    send(persona, "the volatility is 20%")

    assert len(persona.executed) == 1
    _, name, prompt = persona.executed[0]
    assert name == "VanillaOption"
    assert "the volatility is 20%" in prompt


def test_no_is_a_valid_clarifying_answer_not_a_cancel():
    """Regression guard: "no" must answer a question, not cancel it."""
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._pending_run = {
        "backend": "quantlib",
        "kind": "run",
        "name": "VanillaOption",
        "prompt": "price an option",
        "filename": None,
    }

    send(persona, "no")

    assert len(persona.executed) == 1
    assert "no" in persona.executed[0][2]


def test_cancel_word_abandons_a_pending_run():
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._pending_run = {
        "backend": "quantlib",
        "kind": "run",
        "name": "VanillaOption",
        "prompt": "price an option",
        "filename": None,
    }

    send(persona, "cancel")

    assert persona.executed == []
    assert persona._pending_run is None
    assert "cancelled" in persona.last.lower()


# ---------------------------------------------------------------------------
# Free-form chat can only suggest a command; that suggestion must be actionable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "Send this:\n\n/quantlib-run VanillaOption price a call, spot 100\n",
        "Try `quantlib-run VanillaOption price a call, spot 100`",
        "Use: run VanillaOption price a call, spot 100",
    ],
)
def test_chat_suggested_command_is_remembered(reply):
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._remember_suggested_run(reply)

    assert persona._last_doc is not None
    assert persona._last_doc["name"] == "VanillaOption"
    assert persona._last_doc["backend"] == "quantlib"
    assert "spot 100" in persona._last_doc["prompt"]


def test_run_request_works_after_a_chat_suggestion():
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._remember_suggested_run(
        "/quantlib-run VanillaOption price a call, spot 100"
    )
    persona.intents["go ahead"] = RUN

    send(persona, "go ahead")

    assert len(persona.executed) == 1
    assert persona.executed[0][1] == "VanillaOption"


def test_ask_mode_lets_agreement_answer_a_chat_offer():
    persona = _Recorder(mode=p._MODE_ASK)
    persona._remember_suggested_run(
        "/quantlib-run VanillaOption price a call, spot 100"
    )
    assert persona._pending_confirm is True
    persona.intents["yes"] = RUN

    send(persona, "yes")

    assert len(persona.executed) == 1
    assert persona.executed[0][1] == "VanillaOption"


def test_manual_mode_does_not_arm_a_confirm_from_chat():
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._remember_suggested_run(
        "/quantlib-run VanillaOption price a call, spot 100"
    )
    assert persona._pending_confirm is False


def test_chat_suggestion_resolves_the_pyportfolioopt_backend():
    persona = _Recorder(mode=p._MODE_MANUAL)
    persona._remember_suggested_run(
        "/pyportpf-run EfficientFrontier maximize the Sharpe ratio of AAPL, MSFT"
    )
    assert persona._last_doc["backend"] == "pyportpf"
    assert persona._last_doc["name"] == "EfficientFrontier"


def test_chat_prose_without_a_command_is_not_remembered():
    persona = _Recorder(mode=p._MODE_ASK)
    persona._remember_suggested_run(
        "Implied volatility is the volatility implied by an option's market "
        "price. You can run scenarios to explore it."
    )
    assert persona._last_doc is None
    assert persona._pending_confirm is False


def test_chat_suggestion_with_an_unknown_symbol_is_ignored():
    persona = _Recorder(mode=p._MODE_ASK)
    persona._remember_suggested_run("/quantlib-run NotARealSymbol do something")
    assert persona._last_doc is None
    assert persona._pending_confirm is False



# ---------------------------------------------------------------------------
# The real judgement: what the model is asked, and what happens if it fails.
# ---------------------------------------------------------------------------


class _Judged(_Recorder):
    """Uses the real `_classify_run_intent`, with litellm stubbed."""

    log = logging.getLogger("test")
    _classify_run_intent = p.FinTychePersona._classify_run_intent

    def _resolve_model(self):
        return "fake-model", {}


def _stub_litellm(monkeypatch, reply=None, error=None) -> list:
    requests: list = []

    async def acompletion(model, messages, **kwargs):
        requests.append(messages)
        if error is not None:
            raise error
        message = types.SimpleNamespace(content=reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(acompletion=acompletion))
    return requests


def test_judgement_is_told_the_offer_and_the_message(monkeypatch):
    persona = _Judged(mode=p._MODE_ASK)
    _resolve(persona)
    requests = _stub_litellm(monkeypatch, '{"intent": "run", "changes": ""}')

    send(persona, "sounds good to me")

    asked = requests[0][-1]["content"]
    assert "just offered to run `VanillaOption`" in asked
    assert WORD_PROBLEM in asked and "Message: sounds good to me" in asked
    assert persona.executed == [("quantlib", "VanillaOption", WORD_PROBLEM)]


@pytest.mark.parametrize("reply", ["not json", '{"intent": "maybe"}'])
def test_unreadable_judgement_falls_back_to_chat(monkeypatch, reply):
    persona = _Judged(mode=p._MODE_ASK)
    _resolve(persona)
    _stub_litellm(monkeypatch, reply)

    send(persona, "hmm")

    assert persona.executed == []
    assert persona.chat_calls == ["hmm"]


def test_failed_judgement_falls_back_to_chat(monkeypatch):
    persona = _Judged(mode=p._MODE_ASK)
    _resolve(persona)
    _stub_litellm(monkeypatch, error=RuntimeError("token expired"))

    send(persona, "yes")

    assert persona.executed == []
    assert persona.chat_calls == ["yes"]


# ---------------------------------------------------------------------------
# Nothing tells the user to type a particular phrase.
# ---------------------------------------------------------------------------


def test_auto_mode_chat_is_told_not_to_ask():
    note = core._CHAT_MODE_INSTRUCTIONS[p._MODE_AUTO]
    assert "Do not ask whether to run it" in note


@pytest.mark.parametrize("mode", [p._MODE_ASK, p._MODE_AUTO])
def test_chat_is_never_told_to_suggest_a_reply_phrase(mode):
    assert "run it**" not in core._CHAT_MODE_INSTRUCTIONS[mode]
    assert "tell the user what to reply" in core._CHAT_MODE_INSTRUCTIONS[mode]


def test_system_prompt_names_no_magic_phrase():
    text = core.FINTYCHE_SYSTEM_PROMPT_PATH.read_text()
    assert '"run it"' not in text
