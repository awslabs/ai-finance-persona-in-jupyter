# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests for chat-history recording, context-window-aware trimming, and message
assembly.

Personas are built with ``__new__`` (no ``super().__init__``, which needs a live
chat/awareness stack); only the attributes the exercised paths touch are set.
``_count_tokens`` / ``_model_token_budget`` are stubbed where a test needs
deterministic, offline arithmetic instead of litellm's live model tables.
"""

from __future__ import annotations

import time

import litellm

import jupyter_ai_fintyche.persona as p
from jupyter_ai_fintyche import core


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #
class _Tools:
    class Unavailable(Exception):
        pass

    def __init__(self, result=None, stdout="", raise_exc=None):
        self._result = result
        self._stdout = stdout
        self._raise = raise_exc

    def call_function(self, name, args_str):
        if self._raise is not None:
            raise self._raise
        return self._result, self._stdout

    def run_snippet(self, code):
        if self._raise is not None:
            raise self._raise
        return self._result, self._stdout


class _Backend:
    def __init__(self, tools):
        self.tools = tools
        self.key = "quantlib"
        self.label = "QuantLib"
        self.slash_prefix = "quantlib"


def _make_persona():
    q = p.FinTychePersona.__new__(p.FinTychePersona)
    q._init_state()
    q.sent = []
    q.send_message = lambda body: q.sent.append(body)
    return q


def _pairs(n):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"q{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out


def _alternates(history):
    expected = ("user", "assistant")
    return all(m["role"] == expected[i % 2] for i, m in enumerate(history))


# --------------------------------------------------------------------------- #
# Recording command executions
# --------------------------------------------------------------------------- #
def test_record_execution_appends_balanced_pair():
    q = _make_persona()
    q._record_execution("did a thing", "it worked")
    assert q._history == [
        {"role": "user", "content": "did a thing"},
        {"role": "assistant", "content": "it worked"},
    ]


def test_execute_run_success_is_recorded():
    q = _make_persona()
    q._execute_run(_Backend(_Tools(result=7.2)), "VanillaOption", "a=1")
    assert q._history[0] == {"role": "user", "content": "Ran `VanillaOption(a=1)`."}
    assert q._history[1]["role"] == "assistant"
    assert "7.2" in q._history[1]["content"]


def test_execute_run_error_is_recorded():
    q = _make_persona()
    q._execute_run(_Backend(_Tools(raise_exc=TypeError("bad arg"))), "VanillaOption", "a=1")
    assert "Error: TypeError: bad arg" in q._history[1]["content"]


def test_execute_snippet_records_code_result_and_word_problem():
    q = _make_persona()
    q._execute_snippet(
        _Backend(_Tools(result=42)),
        "VanillaOption",
        "x = 6 * 7\nresult = x",
        prompt_text="what is six times seven",
    )
    # request side carries both the word problem and the code
    assert "what is six times seven" in q._history[0]["content"]
    assert "6 * 7" in q._history[0]["content"]
    assert "42" in q._history[1]["content"]


def test_execute_snippet_error_records_traceback():
    q = _make_persona()
    q._execute_snippet(
        _Backend(_Tools(raise_exc=ValueError("pi must be Nx1"))),
        "BlackLitterman",
        "bad = code()",
        prompt_text="optimize",
    )
    assert "Error: ValueError: pi must be Nx1" in q._history[1]["content"]
    assert "bad = code()" in q._history[0]["content"]


def test_unavailable_is_not_recorded():
    q = _make_persona()
    q._execute_run(_Backend(_Tools(raise_exc=_Tools.Unavailable("no quantlib"))), "X", "a=1")
    assert q._history == []
    assert q.sent  # user still told


def test_repeated_runs_keep_history_alternating():
    q = _make_persona()
    backend = _Backend(_Tools(result=1))
    for _ in range(3):
        q._execute_run(backend, "VanillaOption", "a=1")
    assert len(q._history) == 6
    assert _alternates(q._history)


def test_large_result_is_truncated_in_history():
    q = _make_persona()
    huge = list(range(100000))  # repr is far over the cap
    q._execute_run(_Backend(_Tools(result=huge)), "X", "a=1")
    assert "(truncated)" in q._history[1]["content"]
    assert len(q._history[1]["content"]) <= core._MAX_RECORDED_CODE_CHARS + 40


def test_snippet_request_truncates_long_code():
    out = core._snippet_request("Foo", "y = 1\n" * 1000)
    assert "(truncated)" in out


# --------------------------------------------------------------------------- #
# Code-execution timeout
#
# `eval`/`exec`'d code runs synchronously, so it can't be cancelled by
# `asyncio.wait_for` around the surrounding task (that only preempts at an
# `await`); `_run_with_timeout` is what actually bounds it, by running the
# call on a worker thread and giving up on `future.result()` after the
# deadline, even though the thread itself keeps running in the background.
# --------------------------------------------------------------------------- #
def test_run_with_timeout_gives_up_on_a_hanging_call():
    start = time.monotonic()
    try:
        core._run_with_timeout(time.sleep, 2, timeout=0.05)
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected a TimeoutError")
    # Bounded by the timeout, not by the 2-second sleep it gave up on.
    assert time.monotonic() - start < 1


def test_execute_run_reports_a_hanging_call_as_a_timeout(monkeypatch):
    original = core._run_with_timeout
    monkeypatch.setattr(
        core,
        "_run_with_timeout",
        lambda fn, *args, timeout=None: original(fn, *args, timeout=0.05),
    )
    q = _make_persona()
    backend = _Backend(_Tools())
    backend.tools.call_function = lambda name, args_str: time.sleep(2)
    q._execute_run(backend, "Slow", "")
    assert "TimeoutError" in q._history[1]["content"]


def test_append_pair_trims_to_hard_cap():
    q = _make_persona()
    for i in range(p._HISTORY_HARD_CAP_PAIRS + 20):
        q._append_pair(f"u{i}", f"a{i}")
    assert len(q._history) == p._HISTORY_HARD_CAP_PAIRS * 2
    assert _alternates(q._history)
    # oldest dropped, newest kept
    assert q._history[-1] == {"role": "assistant", "content": f"a{p._HISTORY_HARD_CAP_PAIRS + 19}"}


# --------------------------------------------------------------------------- #
# Token budget
# --------------------------------------------------------------------------- #
_SONNET = {"max_input_tokens": 200000, "max_output_tokens": 64000}


def test_budget_reserves_output_capped(monkeypatch):
    monkeypatch.setattr(litellm, "get_model_info", lambda m: dict(_SONNET))
    q = _make_persona()
    # single margin: window minus the reply reserve, and the 64k output is capped
    assert q._model_token_budget("any", {}) == 200000 - core._RESPONSE_RESERVE_CAP


def test_budget_honors_user_max_tokens(monkeypatch):
    monkeypatch.setattr(litellm, "get_model_info", lambda m: dict(_SONNET))
    q = _make_persona()
    assert q._model_token_budget("any", {"max_tokens": 1000}) == 200000 - 1000


def test_budget_falls_back_for_unknown_model(monkeypatch):
    def boom(_m):
        raise RuntimeError("unknown")

    monkeypatch.setattr(litellm, "get_model_info", boom)
    q = _make_persona()
    assert q._model_token_budget("x", {}) == core._MODEL_CONTEXT_FALLBACK - core._RESPONSE_TOKEN_RESERVE


# --------------------------------------------------------------------------- #
# Fitting history to the budget
# --------------------------------------------------------------------------- #
def _fixed():
    return [{"role": "system", "content": "sys"}]


def test_large_window_keeps_all_history(monkeypatch):
    monkeypatch.setattr(litellm, "get_model_info", lambda m: dict(_SONNET))
    q = _make_persona()
    hist = _pairs(30)
    assert q._fit_history("any", _fixed(), hist) == hist


def test_trims_to_budget_above_floor():
    q = _make_persona()
    q._count_tokens = lambda mid, msgs: 100 * len(msgs)  # 100/msg, 200/pair
    q._model_token_budget = lambda mid, kw: 3200  # fixed 100 + 15 pairs of 200
    hist = _pairs(50)
    kept = q._fit_history("any", _fixed(), hist)
    assert len(kept) % 2 == 0
    assert 0 < len(kept) < len(hist)
    assert kept == hist[len(hist) - len(kept):]  # most-recent tail


def test_floor_keeps_min_pairs_when_over_budget():
    q = _make_persona()
    q._count_tokens = lambda mid, msgs: 100 * len(msgs)
    q._model_token_budget = lambda mid, kw: 0  # nothing "fits"
    kept = q._fit_history("any", _fixed(), _pairs(20))
    # the floor still keeps the most recent pairs so a run is never dropped
    assert len(kept) == core._MIN_HISTORY_PAIRS * 2


def test_oversized_newest_pair_is_kept():
    q = _make_persona()
    hist = _pairs(3) + [
        {"role": "user", "content": "x" * 50000},
        {"role": "assistant", "content": "y" * 50000},
    ]
    q._model_token_budget = lambda mid, kw: 1  # tiny budget
    kept = q._fit_history("any", _fixed(), hist)
    assert kept[-2:] == hist[-2:]  # the big newest pair survives


def test_hard_cap_bounds_pairs():
    q = _make_persona()
    q._count_tokens = lambda mid, msgs: 1
    q._model_token_budget = lambda mid, kw: 10**9
    kept = q._fit_history("any", _fixed(), _pairs(p._HISTORY_HARD_CAP_PAIRS + 50))
    assert len(kept) == p._HISTORY_HARD_CAP_PAIRS * 2


def test_fit_passes_model_kwargs_to_budget():
    q = _make_persona()
    seen = {}
    q._count_tokens = lambda mid, msgs: 1
    q._model_token_budget = lambda mid, kw: seen.update(kw) or 10**9
    q._fit_history("any", _fixed(), _pairs(2), {"max_tokens": 1234})
    assert seen.get("max_tokens") == 1234


# --------------------------------------------------------------------------- #
# Message assembly order
# --------------------------------------------------------------------------- #
def test_assemble_messages_order(monkeypatch):
    monkeypatch.setattr(p.FinTychePersona, "system_prompt", "SYS")
    q = _make_persona()
    q._history = _pairs(2)
    q._fit_history = lambda mid, fixed, hist, kw: hist  # keep all
    msgs = q._assemble_messages("any", {}, "NOW", "CTX")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith("SYS\n\n")
    assert msgs[1] == {"role": "system", "content": "Attached context:\nCTX"}
    assert msgs[2:-1] == _pairs(2)
    assert msgs[-1] == {"role": "user", "content": "NOW"}


def test_assemble_messages_without_context_has_no_context_block(monkeypatch):
    monkeypatch.setattr(p.FinTychePersona, "system_prompt", "SYS")
    q = _make_persona()
    q._history = []
    q._fit_history = lambda mid, fixed, hist, kw: hist
    msgs = q._assemble_messages("any", {}, "NOW", "")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[-1] == {"role": "user", "content": "NOW"}


def test_chat_is_told_the_execution_mode(monkeypatch):
    """In auto mode the reply must not ask whether to run it: the run starts as
    soon as the reply ends."""
    monkeypatch.setattr(p.FinTychePersona, "system_prompt", "SYS")
    q = _make_persona()
    q._fit_history = lambda mid, fixed, hist, kw: hist
    for mode in (p._MODE_ASK, p._MODE_AUTO, p._MODE_MANUAL):
        q._execution_mode = mode
        system = q._assemble_messages("any", {}, "NOW", "")[0]["content"]
        assert system.endswith(core._CHAT_MODE_INSTRUCTIONS[mode])
    assert "Do not ask whether to run it" in core._CHAT_MODE_INSTRUCTIONS[p._MODE_AUTO]
