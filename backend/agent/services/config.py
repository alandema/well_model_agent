import json


def load_config(json_config_path: str) -> dict:
    """Load the configuration from a JSON file."""
    with open(json_config_path, "r") as f:
        config = json.load(f)
    return config
