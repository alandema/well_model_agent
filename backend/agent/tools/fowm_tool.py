"""LangChain tool exposing the FOWM (Fast Offshore Wells Model) to the agent."""

import os
import time
from typing import Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.well_models.fowm import FOWM, FOWMConfig


# Directory where simulation CSV outputs are stored (mirrors the
# ``.checkpoints`` folder used by the graph checkpointer).
_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", ".outputs", "fowm"
)


class FOWMInput(BaseModel):
    """Input schema for the FOWM tool.

    All fields are optional; when omitted the notebook defaults are used.
    """

    gas_lift_rate: Optional[float] = Field(
        default=None,
        description="Gas-lift rate [Sm3/day]. Default 165000.",
    )
    choke_opening: Optional[float] = Field(
        default=None,
        description="Choke opening [% 0-100]. Default 16.",
    )
    separator_pressure: Optional[float] = Field(
        default=None,
        description="Separator pressure [Pa]. Default 1013250.",
    )
    reservoir_pressure: Optional[float] = Field(
        default=None,
        description="Reservoir pressure [Pa]. Default 2.25e7.",
    )
    t_end: Optional[float] = Field(
        default=None,
        description="End time of the simulation [s]. Default 100000.",
    )
    n_points: Optional[int] = Field(
        default=None,
        description="Number of time-grid points. Default 1001.",
    )


def _build_config(user_input: FOWMInput) -> FOWMConfig:
    """Translate the tool input into a FOWMConfig, applying defaults."""
    cfg = FOWMConfig()

    u = list(cfg.u)
    if user_input.gas_lift_rate is not None:
        u[0] = user_input.gas_lift_rate
    if user_input.choke_opening is not None:
        u[1] = user_input.choke_opening
    if user_input.separator_pressure is not None:
        u[2] = user_input.separator_pressure
    if user_input.reservoir_pressure is not None:
        u[3] = user_input.reservoir_pressure
    cfg.u = u

    if user_input.t_end is not None or user_input.n_points is not None:
        import numpy as np

        t_end = user_input.t_end if user_input.t_end is not None else 100000.0
        n_points = user_input.n_points if user_input.n_points is not None else 1001
        cfg.t = list(np.linspace(0, t_end, n_points))

    return cfg


@tool(
    args_schema=FOWMInput,
    description=(
        "Run the Fast Offshore Wells Model (FOWM), a dynamic multiphase oil "
        "production model. The full simulation result (time, six mass-balance "
        "states and four pressures: Ppdg, Ptt, Prt, Prb at every time step) is "
        "saved to a CSV file. Returns the file name, absolute path and a head "
        "sample (first rows) of the CSV. All inputs are optional; defaults "
        "are used when omitted."
    ),
)
def run_fowm(
    gas_lift_rate: Optional[float] = None,
    choke_opening: Optional[float] = None,
    separator_pressure: Optional[float] = None,
    reservoir_pressure: Optional[float] = None,
    t_end: Optional[float] = None,
    n_points: Optional[int] = None,
) -> dict:
    """Execute the FOWM simulation, save all points to CSV, return metadata."""
    user_input = FOWMInput(
        gas_lift_rate=gas_lift_rate,
        choke_opening=choke_opening,
        separator_pressure=separator_pressure,
        reservoir_pressure=reservoir_pressure,
        t_end=t_end,
        n_points=n_points,
    )
    cfg = _build_config(user_input)
    result = FOWM.run(cfg)
    csv_content = FOWM.to_csv(result)

    # Persist the CSV to disk (like the .checkpoints folder pattern).
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"fowm_{timestamp}.csv"
    file_path = os.path.abspath(os.path.join(_OUTPUT_DIR, filename))
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
