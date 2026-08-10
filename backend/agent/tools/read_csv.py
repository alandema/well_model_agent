from langchain.tools import tool
import csv
import os
import posixpath
from pydantic import BaseModel, Field
from typing import Optional


EXPECTED_OUTPUT_DIR = "/app/agent/.outputs"


def _escape_markdown_cell(value: object) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")


def _adjust_file_path(file_path: str) -> str:
    """Validate and map the container path to the local output directory."""
    normalized_path = posixpath.normpath(file_path.replace("\\", "/"))
    expected_prefix = EXPECTED_OUTPUT_DIR + "/"

    if not normalized_path.startswith(expected_prefix):
        raise ValueError(
            f"CSV path must be inside {EXPECTED_OUTPUT_DIR}: {file_path}")

    relative_path = normalized_path[len(expected_prefix):]
    if not relative_path or relative_path == ".":
        raise ValueError(f"Invalid CSV path: {file_path}")

    output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", ".outputs"))
    return os.path.join(output_dir, *relative_path.split("/"))


class CSVReaderInput(BaseModel):
    file_path: str = Field(..., description="Path to the CSV file to read.")
    columns: list[str] | None = Field(
        None, description="Optional list of columns to read from the CSV file. If not provided, all columns will be read.")
    rows: Optional[tuple[int, int]] = Field(
        None, description="Optional tuple specifying the range of rows to read (start, end). If not provided, all rows will be read.")


@tool(
    args_schema=CSVReaderInput,
    description="Reads a CSV file and returns its contents as a string. Prefer to use the summarize_csv tool. Only use this if you need to read the values directly.",
)
def read_csv(
    file_path: str,
    columns: list[str] | None = None,
    rows: Optional[tuple[int, int]] = None,
) -> str:
    try:
        adjusted_file_path = _adjust_file_path(file_path)
        with open(adjusted_file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                return "The CSV file is empty or does not have a header."
            csv_rows = list(reader)
            if rows is not None:
                start, end = rows
                csv_rows = csv_rows[start:end]
            if columns:
                # Check if all specified columns exist in the CSV
                missing_columns = [
                    col for col in columns if col not in reader.fieldnames]
                if missing_columns:
                    return f"The following columns are missing in the CSV file: {', '.join(missing_columns)}"
                # Filter rows to include only specified columns
                rows = [{col: row[col] for col in columns} for row in csv_rows]
            else:
                # Read all columns
                rows = csv_rows

            # Convert rows to a Markdown table
            headers = columns if columns else reader.fieldnames
            header_row = "| " + " | ".join(
                _escape_markdown_cell(header) for header in headers) + " |"
            separator_row = "| " + " | ".join("---" for _ in headers) + " |"
            data_rows = [
                "| " + " | ".join(
                    _escape_markdown_cell(row.get(header)) for header in headers
                ) + " |"
                for row in rows
            ]
            output = "\n".join([header_row, separator_row, *data_rows])
            if len(output) > 10000:  # Limit output to 10,000 characters
                output = output[:10000] + \
                    "\n... (output is truncated because it exceeds 10,000 characters)"
            return output if output else "The CSV file is empty."
    except ValueError as e:
        return str(e)
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred while reading the CSV file: {str(e)}"
