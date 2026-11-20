from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import Optional
import json

JSON_PATH = "C:\\Users\\aland\\Desktop\\well_model_agent\\control_room\\app\\configs\\model_config.json"


class SetpointInput(BaseModel):
    choke_opening: Optional[float] = Field(
        None,
        description="Choke opening value. In percentage. If not provided, the choke opening will not be changed.",
        minimum=0.0,
        maximum=100.0,
    )
    gas_injection_rate: Optional[float] = Field(
        None,
        description="Gas injection rate value. In m³/day. If not provided, the gas injection rate will not be changed.",
        examples=[165000.0, 1.65e5]
    )


@tool(
    args_schema=SetpointInput,
    description=(
        "Sets the choke opening and gas injection rate for the well. Use this tool to adjust the well's operating conditions."
    ),
)
def setpoint(input: SetpointInput) -> str:
    """Sets the choke opening and gas injection rate for the well."""
    try:
        with open(JSON_PATH, "r") as f:
            config = json.load(f)

        if input.choke_opening is not None:
            config["choke_opening"] = input.choke_opening
        if input.gas_injection_rate is not None:
            config["gas_injection_rate"] = input.gas_injection_rate

        with open(JSON_PATH, "w") as f:
            json.dump(config, f, indent=4)

        return "Setpoint updated successfully."
    except Exception as e:
        return f"Error updating setpoint: {str(e)}"
