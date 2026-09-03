import os
import subprocess
import sys
import tempfile

from langchain_core.tools import tool
from pydantic import BaseModel, Field

MAX_OUTPUT_CHARS = 8000
TIMEOUT_SECONDS = 60

# Packages guaranteed to be available (mirrors backend/requirements.txt).
# Surfed to the model on ImportError so it can self-correct instead of
# retrying with another missing package.
AVAILABLE_PACKAGES = (
    "numpy, scipy, pandas, matplotlib, scikit-learn, pint, statistics, math"
)


class PythonCodeInput(BaseModel):
    code: str = Field(
        ...,
        description=(
            "Complete, self-contained Python code to execute. "
            "Import all needed packages and print() results to see them."
        ),
    )


@tool(
    args_schema=PythonCodeInput,
    description=(
        "Execute Python code and return its stdout (and any error traceback). "
        f"Pre-installed packages: {AVAILABLE_PACKAGES} — restrict imports to "
        "these, they cover math, statistics, optimization and machine-learning "
        "problems. Code must be self-contained: import everything you need and "
        "use print() to output results. State does NOT persist between calls. "
        "For plots, save figures to a file path and report that path. "
        "Timeouts after 60 seconds. Do not attempt to build complex "
        "machine-learning pipelines or train large models, as this tool is not "
        "designed for that."
    ),
)
def python_repl(code: str) -> str:
    """Run self-contained Python code in a subprocess and return stdout/errors."""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script_path = f.name

    try:
        # MPLBACKEND=Agg keeps matplotlib headless inside the container.
        env = {**os.environ, "MPLBACKEND": "Agg"}
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=tempfile.gettempdir(),
            env=env,
        )

        output = result.stdout
        if result.returncode != 0:
            stderr = result.stderr
            if "ModuleNotFoundError" in stderr or "ImportError" in stderr:
                output += (
                    f"\nError (exit code {result.returncode}):\n{stderr}"
                    f"\nHint: the packages available in this environment are: "
                    f"{AVAILABLE_PACKAGES}. Rewrite the code to use only these."
                )
            else:
                output += f"\nError (exit code {result.returncode}):\n{stderr}"

        output = output.strip()
        if not output:
            output = "(no output — use print() to display results)"
        if len(output) > MAX_OUTPUT_CHARS:
            output = (
                output[:MAX_OUTPUT_CHARS]
                + f"\n... [output truncated, {len(output)} chars total]"
            )
        return output
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as e:
        return f"Execution failed: {e}"
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
