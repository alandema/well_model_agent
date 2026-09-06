"""Builds raw chat models. All binding (tools / structured output) and the
system prompt live in the node classes that own the model."""

from langchain_openrouter import ChatOpenRouter


def create_chat_model(model_config: dict) -> ChatOpenRouter:
    """Create a raw chat model from its config entry (no tools, no prompt)."""
    return ChatOpenRouter(
        model=model_config["model_id"],
        **model_config.get("model_config", {}),
    )
