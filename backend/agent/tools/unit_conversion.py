import pint
from pydantic import BaseModel, Field
from typing import Optional
from langchain.tools import tool


class InputUnit(BaseModel):
    """Input schema for the unit conversion tool."""
    value: float = Field(..., description="The numerical value to convert.")
    from_unit: str = Field(..., description="The unit of the input value.")
    to_unit: Optional[str] = Field(
        None, description="The unit to convert the value to. If omitted, the tool will return the value in the international system of units.")


@tool(
    description="Uses Pint python library to convert a numerical value from one unit to another.",
    return_direct=True,
)
def convert_units(input_unit: InputUnit) -> float:
    ureg = pint.UnitRegistry()
    quantity = input_unit.value * ureg(input_unit.from_unit)

    if input_unit.to_unit:
        converted_quantity = quantity.to(input_unit.to_unit)
    else:
        converted_quantity = quantity.to_base_units()

    return converted_quantity.magnitude
