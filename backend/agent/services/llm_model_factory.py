import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate


def create_llm_model(llm_model_config: dict, tools: list = None,
                     output_schema=None):
    """Create a chat model with the system prompt and tools attached.

    Returns a chat-model runnable. When tools and an output schema are both
    provided, the model is configured to choose between ordinary tool calls
    and provider-native structured output in the same invocation.
    """
    llm_model_id = llm_model_config["model_id"]
    llm_model_kwargs = llm_model_config.get("model_config", {})

    model = ChatOpenRouter(
        model=llm_model_id,
        **llm_model_kwargs
    )
    system_prompt = ""
    system_prompt_path = llm_model_config.get("system_prompt_path")
    if system_prompt_path:
        with open(os.fspath(system_prompt_path), "r", encoding="utf-8") as f:
            system_prompt = f.read()

    if tools is not None and output_schema is not None:
        model = model.bind_tools(
            tools,
            response_format=output_schema,
            strict=True,
        )
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("placeholder", "{messages}")]
            )
            model = prompt | model
    elif tools is not None:
        model = model.bind_tools(tools)
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("placeholder", "{messages}")]
            )
            model = prompt | model
    elif output_schema is not None:
        model = model.with_structured_output(
            output_schema
        )
        if system_prompt:
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("placeholder", "{messages}")]
            )
            model = prompt | model
    elif system_prompt:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )
        model = prompt | model

    return model
