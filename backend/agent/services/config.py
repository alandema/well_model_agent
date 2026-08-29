import json
import os


def load_config(json_config_path: str) -> dict:
    """Load JSON configuration and resolve Markdown system-prompt paths."""
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
