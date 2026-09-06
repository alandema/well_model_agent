"""The one place that reads configuration files from disk."""

import json
import os


def load_config(json_config_path: str) -> dict:
    """Load the LLM prompts config and resolve system-prompt paths.

    Every other module receives configs as plain dicts; only this function
    touches the filesystem for configuration.
    """
    with open(json_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(json_config_path))
    for model_config in config.values():
        prompt_path = model_config.get("system_prompt_path")
        if prompt_path:
            model_config["system_prompt_path"] = os.path.normpath(
                os.path.join(config_dir, prompt_path)
            )

    return config


def read_system_prompt(model_config: dict) -> str:
    """Read the system-prompt file referenced by a model config.

    Returns an empty string when the config has no prompt file, so nodes
    can simply skip prepending a system message.
    """
    prompt_path = model_config.get("system_prompt_path")
    if not prompt_path:
        return ""
    with open(os.fspath(prompt_path), "r", encoding="utf-8") as f:
        return f.read()
