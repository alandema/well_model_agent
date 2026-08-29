import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent


def create_llm_model(llm_model_config: dict, tools: list = None,
                     output_schema=None):
    """Create a chat model with the system prompt and tools attached.

    Returns a Runnable that, when invoked with a list of messages, produces
    an AIMessage. The system prompt is prepended via a ChatPromptTemplate
    (the documented LangChain pattern), and tools are bound with bind_tools.
    """
    llm_model_id = llm_model_config["model_id"]
    llm_model_kwargs = llm_model_config.get("model_config", {})

    # Bind tools first (bind_tools returns a RunnableBinding that still
    # supports being the target of a prompt | model chain).

    model = ChatOpenRouter(
        model=llm_model_id,
        **llm_model_kwargs
    )

    if output_schema is not None and tools is None:
        model = model.with_structured_output(
            output_schema, method="json_schema"
        )
    if output_schema is None and tools is not None:
        model = model.bind_tools(tools)

    # Attach the system prompt using the documented prompt | model pattern.
    system_prompt = ""
    system_prompt_path = llm_model_config.get("system_prompt_path")
    if system_prompt_path:
        with open(os.fspath(system_prompt_path), "r", encoding="utf-8") as f:
            system_prompt = f.read()

    if output_schema is not None and tools is not None:
        model = create_agent(
            model=model,
            tools=tools,
            response_format=output_schema
        )
    elif system_prompt:
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("placeholder", "{messages}")]
        )
        model = prompt | model

    return model
