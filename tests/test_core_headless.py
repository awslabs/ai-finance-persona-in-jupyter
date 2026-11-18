# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Tests that FinTyche's logic runs without Jupyter.

`core.FinTycheCore` is shared by the Jupyter AI persona and the CLI, so it
must import and handle messages with no Jupyter package present. These drive a
bare subclass that supplies the frontend I/O itself, with the LLM stubbed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from jupyter_ai_fintyche import core

REPO_ROOT = Path(__file__).resolve().parent.parent


class _Terminal(core.FinTycheCore):
    """The smallest frontend: implements the whole contract and collects what
    FinTyche would show."""

    system_prompt = "SYS"
    log = logging.getLogger("test")

    def __init__(self, workspace: Path | None = None, mode: str = core._MODE_AUTO):
        self._init_state(mode)
        self.messages: list[str] = []
        self.streamed: list[str] = []
        self.workspace = workspace

    def send_message(self, body: str) -> None:
        self.messages.append(body)

    async def stream_message(self, chunks) -> None:
        async for chunk in chunks:
            self.streamed.append(chunk.choices[0].delta.content)

    def set_status(self, status: str = "") -> None:
        pass

    def clear_status(self) -> None:
        pass

    def get_workspace_dir(self) -> str:
        return str(self.workspace)

    def process_attachments(self, message) -> None:
        return None

    def _resolve_model(self):
        return "stub-model", {}


def _stub_completion(monkeypatch, reply: str, stream_tokens: tuple[str, ...] = ()) -> list:
    """Stub `litellm` so each call returns `reply`, or streams `stream_tokens`
    when called with `stream=True`. Returns the list of messages sent."""
    calls: list = []

    async def acompletion(model, messages, stream=False, **kwargs):
        calls.append(messages)
        if stream:
            async def chunks():
                for token in stream_tokens:
                    delta = types.SimpleNamespace(content=token)
                    yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)])
            return chunks()
        message = types.SimpleNamespace(content=reply)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(
            acompletion=acompletion,
            get_model_info=lambda m: {},
            token_counter=lambda model, messages: 1,
        ),
    )
    return calls


def test_core_imports_without_jupyter() -> None:
    """In a fresh interpreter with every Jupyter package blocked, the core and
    backends still import; only the persona needs Jupyter."""
    script = textwrap.dedent(
        """
        import builtins

        real_import = builtins.__import__
        BLOCKED = {"jupyter_ai_persona_manager", "jupyterlab_chat", "jupyter_server"}

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in BLOCKED:
                raise ImportError(f"blocked {name}")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = guarded
        from jupyter_ai_fintyche import backends, core

        assert core.FinTycheCore and backends._BACKENDS
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"


def test_run_command_writes_and_runs_code_without_a_persona(monkeypatch) -> None:
    pytest.importorskip("QuantLib")
    _stub_completion(
        monkeypatch, json.dumps({"status": "ok", "code": "result = ql.Date(15, 6, 2025)"})
    )
    agent = _Terminal()

    asyncio.run(agent.handle_message("/quantlib-run Date the fifteenth of June 2025"))

    assert "Date(15,6,2025)" in agent.messages[-1]
    assert agent._pinned["code"] == "result = ql.Date(15, 6, 2025)"
    assert agent._pinned["error"] is None


def test_notebook_command_writes_into_the_frontends_workspace(monkeypatch, tmp_path) -> None:
    pytest.importorskip("QuantLib")
    _stub_completion(
        monkeypatch, json.dumps({"status": "ok", "code": "result = ql.Date(15, 6, 2025)"})
    )
    agent = _Terminal(workspace=tmp_path)

    asyncio.run(
        agent.handle_message("/quantlib-notebook Date the fifteenth of June 2025 as june")
    )

    assert (tmp_path / "june.ipynb").exists()
    assert agent.messages[-1].endswith(core.FinTycheCore.notebook_open_hint)


def test_free_form_chat_streams_and_records_the_reply(monkeypatch) -> None:
    calls = _stub_completion(monkeypatch, "", stream_tokens=("Put-call ", "parity."))
    agent = _Terminal()

    asyncio.run(agent.handle_message("what links calls and puts?"))

    assert agent.streamed == ["Put-call ", "parity."]
    assert calls[-1][0] == {
        "role": "system",
        "content": f"SYS\n\n{core._CHAT_MODE_INSTRUCTIONS[core._MODE_AUTO]}",
    }
    assert agent._history[-2:] == [
        {"role": "user", "content": "what links calls and puts?"},
        {"role": "assistant", "content": "Put-call parity."},
    ]


def test_help_uses_the_frontends_wording() -> None:
    class _NoModel(_Terminal):
        model_settings_hint = "pass --model"
        execution_mode_location = "the --mode flag"

        def _resolve_model(self):
            return None, {}

    agent = _NoModel()
    asyncio.run(agent.handle_message("/quantlib-help"))

    assert "none configured yet — pass --model" in agent.messages[-1]
    assert "the --mode flag decides what happens next" in agent.messages[-1]
    assert "Jupyter AI" not in agent.messages[-1]


def test_a_missing_contract_method_fails_loudly() -> None:
    class _Incomplete(core.FinTycheCore):
        pass

    with pytest.raises(NotImplementedError):
        _Incomplete().get_workspace_dir()


def test_package_exposes_the_persona_lazily() -> None:
    import jupyter_ai_fintyche
    from jupyter_ai_fintyche import persona

    assert jupyter_ai_fintyche.FinTychePersona is persona.FinTychePersona
    assert issubclass(persona.FinTychePersona, core.FinTycheCore)
