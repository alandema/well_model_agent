from langchain.tools import tool
from pydantic import BaseModel, Field

class AddInput(BaseModel):
    a: float = Field(..., description="The first number to add.")
    b: float = Field(..., description="The second number to add.")

@tool(args_schema=AddInput, description="Add two numbers together.")
def add(a: float, b: float) -> float:
    return a + b