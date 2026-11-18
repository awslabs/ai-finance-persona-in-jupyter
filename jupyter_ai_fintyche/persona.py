# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""FinTyche persona for Jupyter AI.

A single persona (no interactive agent-builder flow, unlike QuickAgent) that
gives chat users direct access to quantitative-finance Python libraries — list
available functions/classes, read their documentation, call them by describing
the arguments as a free-form word problem, or generate a demo notebook — plus a
per-chat model picker for free-form quant-finance questions, whose options come
from Jupyternaut's chat-model settings so FinTyche runs the same model(s) you
configured in Jupyter AI's settings.

What FinTyche does with a message lives in `core.FinTycheCore`, which this
persona mixes in; this module adds only what is specific to Jupyter AI: the
model picker and "Running code" setting, slash-command autocomplete, model
resolution from Jupyternaut's AI Settings, and rebuilding memory from the
chat's saved messages after a restart.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any, Optional

from jupyter_ai_persona_manager import (
    BasePersona,
    CommandOption,
    ModelConfiguration,
    PersonaDefaults,
    SettingConfiguration,
    SettingOption,
)
from jupyterlab_chat.models import Message

from .backends import _BACKENDS, _SLASH_DESCRIPTIONS, _VERBS
from .core import (
    _EXECUTION_MODE_SETTING_ID,
    _HISTORY_HARD_CAP_PAIRS,
    _MODE_ASK,
    _MODE_AUTO,
    _MODE_MANUAL,
    FinTycheCore,
    load_system_prompt,
)
from .models import DEFAULT_MODEL_ID, build_model_options

FINTYCHE_AVATAR_PATH = str(
    Path(__file__).resolve().parent / "static" / "fintyche.svg"
)


_EXECUTION_MODE_OPTIONS = (
    SettingOption(
        id=_MODE_ASK,
        name="Ask first",
        description=(
            "FinTyche offers to run the code and waits for you to confirm."
        ),
    ),
    SettingOption(
        id=_MODE_AUTO,
        name="Run automatically",
        description=(
            "FinTyche writes and runs the code as soon as it has picked a "
            "symbol, with no confirmation step."
        ),
    ),
    SettingOption(
        id=_MODE_MANUAL,
        name="I'll run it myself",
        description=(
            "FinTyche only shows the documentation and the `run` command to "
            "use; it never executes anything on its own."
        ),
    ),
)


def _execution_mode_setting(current: str) -> SettingConfiguration:
    """The execution-mode picker, reporting `current` as the live selection."""
    return SettingConfiguration(
        id=_EXECUTION_MODE_SETTING_ID,
        name="Running code",
        current=current,
        description=(
            "Who runs the code once FinTyche has picked a library symbol for "
            "your word problem."
        ),
        options=[opt.model_copy(deep=True) for opt in _EXECUTION_MODE_OPTIONS],
    )


def _build_slash_commands() -> list[CommandOption]:
    commands: list[CommandOption] = []
    for backend in _BACKENDS.values():
        for verb in _VERBS:
            commands.append(
                CommandOption(
                    name=backend.slash(verb),
                    description=_SLASH_DESCRIPTIONS[backend.key][verb],
                )
            )
    return commands


_SLASH_COMMANDS = _build_slash_commands()


class FinTychePersona(BasePersona, FinTycheCore):
    """A Jupyter AI persona giving chat access to quant-finance libraries."""

    # Where things live in Jupyter AI; see "frontend wording" in FinTycheCore.
    no_model_message = (
        "**No chat model is configured.** FinTyche uses the same models as "
        "Jupyternaut — configure a chat model in **Settings ▸ AI Settings**, then "
        "select it from the model picker next to FinTyche's name."
    )
    product_tagline = "quantitative finance assistant for Jupyter AI"
    model_settings_hint = "set one in Settings ▸ AI Settings"
    execution_mode_location = "the **Running code** setting next to the model picker"
    chat_model_location = "the model selected in the picker next to FinTyche's name"
    notebook_open_hint = "Open it from the file browser to run the demo."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_state()
        # Whether `_history` and the last run have been rebuilt from the chat's
        # saved messages; see `_restore_from_chat`.
        self._restored = False
        # `_execution_mode` (set by `_init_state`) is the source of truth for
        # the "Running code" setting: `_report_model_config` republishes it, so
        # the user's choice survives the per-message picker refresh. Updated via
        # `update_model_settings`.

        # Advertise the model picker and slash commands here (not in
        # `prepare()`) so they appear in the UI immediately, the same way
        # Jupyternaut's model settings are always available. `prepare()` only
        # runs on the first routed message, which would leave the picker
        # missing until then.
        self._report_model_config()
        self.report_slash_commands(_SLASH_COMMANDS)

    @property
    def defaults(self) -> PersonaDefaults:
        return PersonaDefaults(
            name="FinTyche",
            avatar_path=FINTYCHE_AVATAR_PATH,
            description=(
                "AI quantitative analyst for Jupyter AI: list, document, and run "
                "any QuantLib or PyPortfolioOpt function or class, generate demo "
                "notebooks, and answer quant-finance questions."
            ),
            system_prompt=self._load_system_prompt(),
        )

    def _load_system_prompt(self) -> str:
        return load_system_prompt()

    async def update_model(self, model_id: str) -> None:
        """Apply a user's model selection from the per-chat picker.

        No-op by design, mirroring Jupyternaut: FinTyche's backend is
        stateless (a fresh LiteLLM call per message), so there is nothing to
        switch eagerly. `BasePersona.apply_model_spec` records the new selection
        on awareness, and each message resolves it fresh via `_resolve_model`.
        A stale/unknown ID is tolerated and falls back to the default there.
        """

    def _config_manager(self) -> Any:
        """Return Jupyternaut's ConfigManager (holds the AI Settings), or None."""
        try:
            settings = self.parent.parent.serverapp.web_app.settings
        except AttributeError:
            return None
        return settings.get("jupyternaut.config_manager")

    def _report_model_config(self) -> None:
        """Publish the model picker, sourced from Jupyternaut's saved models.

        Options are the user's custom models; the picker adds a built-in
        "Default" row on its own (a `None` selection) that resolves to
        Jupyternaut's configured default model. Re-called on each message so the
        picker tracks custom models added/edited in Settings ▸ AI Settings while
        the chat is open, preserving the user's own selection when it survives.
        """
        options = build_model_options(self._config_manager())
        option_ids = {opt.id for opt in options}
        current = self.get_model()
        # Drop a stale selection (e.g. a deleted custom model) so the picker
        # falls back to the built-in Default row; `_resolve_model` does the same.
        if (
            current is not None
            and current != DEFAULT_MODEL_ID
            and current not in option_ids
        ):
            current = None
        self.report_model_configuration(
            ModelConfiguration(
                current=current,
                options=options,
                settings=[_execution_mode_setting(self._execution_mode)],
            )
        )

    async def update_model_settings(self, settings: dict[str, str | None]) -> None:
        """Apply a user's selection from the settings shown with the picker.

        Only `execution_mode` is offered. An unknown ID or value is ignored so a
        stale UI selection cannot put the persona into an undefined mode.
        """
        mode = settings.get(_EXECUTION_MODE_SETTING_ID)
        if mode in {_MODE_ASK, _MODE_AUTO, _MODE_MANUAL}:
            self._execution_mode = mode
            # A mode change invalidates any offer made under the previous mode.
            self._pending_confirm = False

    def sync_model_configuration(self) -> None:
        """Re-publish the model picker after Jupyternaut's config changes.

        Jupyternaut's Config REST API (`/api/jupyternaut/config`) calls this on
        every live persona after a config write, discovering it by duck-typing
        (`getattr(persona, "sync_model_configuration", None)`) rather than by
        type — so exposing this exact method name is all it takes for FinTyche
        to refresh its picker the moment a model is added/edited in
        Settings ▸ AI Settings, without a JupyterLab restart. Delegates to
        `_report_model_config`, the same refresh used on init and per message.
        """
        self._report_model_config()

    # ---- message dispatch ----

    def _strip_mention(self, body: str) -> str:
        return re.sub(r"@\S+\s*", "", body, count=1).strip()

    async def process_message(self, message: Message) -> None:
        # Refresh the picker from AI Settings so it reflects any model the user
        # has (re)configured for Jupyternaut since the chat opened.
        self._report_model_config()
        self._restore_from_chat(exclude_id=getattr(message, "id", None))

        body = self._strip_mention(message.body.strip())

        await self.handle_message(body, message)

    def _restore_from_chat(self, exclude_id: Optional[str] = None) -> None:
        """Rebuild memory from the chat's saved messages, once per instance.

        FinTyche's memory (`_history`, `_last_doc`, `_pinned`) lives in the
        server process, which starts empty after a restart, while the chat file
        keeps every message. As chat apps do, the saved conversation is treated
        as the source of truth: it is replayed into `_history` (which
        `_fit_history` then trims to the model's budget, newest first), and the
        last code FinTyche posted is pinned again, so "rerun that" works in a
        chat reopened days later. `exclude_id` is the message being handled,
        which is sent as the current turn rather than as history.
        """
        if getattr(self, "_restored", True):
            return
        self._restored = True
        try:
            saved = self.chat.get_messages()
        except Exception:
            self.log.exception("Could not read this chat's saved messages.")
            return

        entries = [
            {
                "role": "assistant" if msg.sender == self.id else "user",
                "content": (msg.body or "").strip(),
            }
            for msg in saved
            if not msg.deleted and msg.id != exclude_id and (msg.body or "").strip()
        ]
        turns: list[dict[str, str]] = []
        for entry in entries:
            role, text = entry["role"], entry["content"]
            if turns and turns[-1]["role"] == role:
                # FinTyche often answers in several messages (docs, then code,
                # then the result); one turn keeps history alternating, which
                # Bedrock Converse requires.
                turns[-1]["content"] += "\n\n" + text
            else:
                turns.append({"role": role, "content": text})

        if not self._history:
            pairs: list[dict[str, str]] = []
            for user, reply in itertools.pairwise(turns):
                if user["role"] == "user" and reply["role"] == "assistant":
                    pairs += [user, reply]
            self._history = pairs[-_HISTORY_HARD_CAP_PAIRS * 2 :]

        if self._last_doc is None:
            self._restore_last_run(entries)

    # ---- model resolution ----

    def _resolve_model(self) -> tuple[Optional[str], dict[str, Any]]:
        """Resolve the picker's current selection to a LiteLLM model + kwargs.

        Mirrors Jupyternaut's own resolution so FinTyche runs exactly the
        model the picker shows:

        - A custom-model ID resolves to that custom model's LiteLLM model ID and
          saved parameters (looked up from config by ID).
        - The built-in "Default" row (`None` selection, or `DEFAULT_MODEL_ID`)
          resolves to Jupyternaut's configured default model (`chat_model`) and
          its saved `fields` parameters.
        - A stale custom-model ID (deleted model) falls back to the default.

        Returns `(None, {})` when no model can be resolved (no ConfigManager, or
        the default was selected but none is configured), so callers prompt the
        user to configure one.
        """
        cm = self._config_manager()
        if cm is None:
            return None, {}

        selected = self.get_model()
        if selected and selected != DEFAULT_MODEL_ID:
            custom_model = cm.get_custom_model(selected)
            if custom_model is not None:
                return custom_model.model_id, dict(custom_model.params)
            # A stale custom-model ID falls back to the default; any other ID is
            # a LiteLLM model used as-is, with any parameters saved under config
            # `fields` (matching how the default's params are looked up).
            if not selected.startswith("custom-"):
                try:
                    params = cm.get_config().fields.get(selected, {})
                except Exception:
                    params = {}
                return selected, dict(params)

        return cm.chat_model, dict(cm.chat_model_args or {})
