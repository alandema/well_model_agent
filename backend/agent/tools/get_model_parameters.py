"""Tool that returns the parameters required to run a given well model."""

import importlib
import pkgutil

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent import well_models as _pkg


def _find_model(model_id: str):
    """Return the model class whose ``id`` matches ``model_id``."""
    for module_info in pkgutil.iter_modules(_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{_pkg.__name__}.{module_info.name}")
        for obj in vars(module).values():
            if isinstance(obj, type) and getattr(obj, "id", None) == model_id:
                return obj
    return None


class GetModelParametersInput(BaseModel):
    """Input schema for the get_model_parameters tool."""

    model_id: str = Field(
        description="The id of the well model (as returned by well_model_picker).",
    )


@tool(
    args_schema=GetModelParametersInput,
    description=(
        "Given a well model id, return the list of parameters required to "
        "run that model, including name, description, unit and default "
        "value for each parameter."
    ),
)
def get_model_parameters(model_id: str) -> dict:
    """Return the parameters needed to run the specified well model."""
    model = _find_model(model_id)
    if model is None:
        return {"error": f"Unknown model id '{model_id}'."}
    return {
        "model_id": model_id,
        "parameters": getattr(model, "parameters", []),
    }
