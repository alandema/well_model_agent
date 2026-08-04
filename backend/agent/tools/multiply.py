from langchain.tools import tool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: float = Field(..., description="The first number to multiply.")
    b: float = Field(..., description="The second number to multiply.")

@tool(args_schema=MultiplyInput, description="Multiply two numbers together.")
def multiply(a: float, b: float) -> float:
    return a * b