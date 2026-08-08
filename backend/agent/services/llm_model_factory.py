import json
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate


def create_llm_model(llm_model_config: dict, tools: list = None):
    """Create a chat model with the system prompt and tools attached.

    Returns a Runnable that, when invoked with a list of messages, produces
    an AIMessage. The system prompt is prepended via a ChatPromptTemplate
    (the documented LangChain pattern), and tools are bound with bind_tools.
    """
    llm_model_id = llm_model_config["model_id"]
    llm_model_kwargs = llm_model_config.get("model_config", {})
    model = ChatOpenRouter(
        model=llm_model_id,
        **llm_model_kwargs
    )

    # Bind tools first (bind_tools returns a RunnableBinding that still
    # supports being the target of a prompt | model chain).
    if tools:
        model = model.bind_tools(tools)

    # Attach the system prompt using the documented prompt | model pattern.
    system_prompt = llm_model_config.get("system_prompt", "")
    if system_prompt:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )
        model = prompt | model
    return model
