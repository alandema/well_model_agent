import subprocess
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class TerminalCommandInput(BaseModel):
    command: str = Field(..., description="The shell command to execute.")


@tool(
    args_schema=TerminalCommandInput,
    description="Run a shell command on the terminal and return the output.",
)
def run_terminal_command(command: str) -> str:
    """Run a shell command on the terminal and return the output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Execution failed: {str(e)}"
