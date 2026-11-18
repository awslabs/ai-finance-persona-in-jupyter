# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Sources FinTyche's model picker from Jupyternaut's saved models.

FinTyche does not maintain its own model list. Its per-chat model picker
offers exactly the models you saved for Jupyternaut in Jupyter AI's settings
(**Settings ▸ AI Settings**): your custom models, plus the built-in "Default"
row the picker renders on its own (a ``None`` selection), which resolves to
Jupyternaut's configured default model. Options are read from Jupyternaut's
``ConfigManager`` so FinTyche stays in lockstep with those settings.

The persona hands us the ``ConfigManager`` (it owns the handle to the server
extension); see ``FinTychePersona._config_manager``, ``_report_model_config``,
and ``_resolve_model`` in ``persona.py``.
"""

from __future__ import annotations

from typing import Any

from jupyter_ai_persona_manager import ModelOption

# Sentinel selection meaning "use Jupyternaut's configured default model",
# matching Jupyternaut's own persona. The picker's built-in "Default" row sends
# a `None` selection, which resolves to the same place; this ID is recognized
# too so an explicit default selection also resolves to the configured default
# rather than being treated as a literal LiteLLM model ID.
DEFAULT_MODEL_ID = "jupyternaut/default"


def build_model_options(config_manager: Any) -> list[ModelOption]:
    """Return the picker options: the user's saved custom models, in order.

    The picker renders a built-in "Default" row on its own (a ``None``
    selection), so Jupyternaut's configured default model is offered without
    adding it here. Returns an empty list when no ``ConfigManager`` is available
    or no custom models are saved.
    """
    if config_manager is None:
        return []
    return [
        ModelOption(id=model.id, name=model.name, description=model.description)
        for model in config_manager.custom_models
    ]
