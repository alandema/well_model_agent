"""Generic LangChain tool that runs any registered well model."""

import os
import time
from typing import Dict, Optional

import importlib
import pkgutil

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent import well_models as _pkg


# Directory where simulation CSV outputs are stored.
_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", ".outputs"
)


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


class RunModelInput(BaseModel):
    """Input schema for the run_model tool."""

    model_id: str = Field(
        description="The id of the well model to run (as returned by well_model_picker).",
    )
    parameters: Optional[Dict] = Field(
        default=None,
        description=(
            "Optional dict of model parameters (as returned by "
            "get_model_parameters). When omitted or partially provided, "
            "model defaults are used for the missing values."
        ),
    )


@tool(
    args_schema=RunModelInput,
    description=(
        "Run a well model identified by its model_id. The full simulation "
        "result is saved to a CSV file. Returns the file name, absolute "
        "path and a head sample (first rows) of the CSV. Use "
        "well_model_picker to discover available models and "
        "get_model_parameters to learn the required parameters."
    ),
)
def run_model(model_id: str, parameters: Optional[Dict] = None) -> dict:
    """Execute the requested well model and save results to CSV."""
    model = _find_model(model_id)
    if model is None:
        return {"error": f"Unknown model id '{model_id}'."}

    params = parameters or {}
    cfg = model.build_config(params)
    result = model.run(cfg)
    csv_content = model.to_csv(result)

    # Persist the CSV to disk.
    out_dir = os.path.join(_OUTPUT_DIR, model_id)
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{model_id}_{timestamp}.csv"
    file_path = os.path.abspath(os.path.join(out_dir, filename))
    with open(file_path, "w", newline="") as f:
        f.write(csv_content)

    # Build a head sample (header + first 5 data rows) for the model.
    lines = csv_content.splitlines()
    head_sample = "\n".join(lines[:6])

    return {
        "name": filename,
        "path": file_path,
        "rows": len(lines) - 1,
        "head_sample": head_sample,
    }
