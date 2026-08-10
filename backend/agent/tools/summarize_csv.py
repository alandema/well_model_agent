import csv
import os
import posixpath
from numbers import Real

from langchain.tools import tool
from pydantic import BaseModel, Field

from agent.tools.read_csv import _adjust_file_path


class CSVSummaryInput(BaseModel):
    file_path: str = Field(...,
                           description="Path to the CSV file to summarize.")
    columns: list[str] | None = Field(
        None,
        description="Numeric columns to summarize, for example ['t', 'Ppdg'].",
    )


@tool(
    args_schema=CSVSummaryInput,
    description=(
        "Summarizes numeric CSV columns with count, first, last, minimum, "
        "maximum, mean, and the time of extrema. Use this after fowm_model "
        "for concise simulation analysis instead of reading the full CSV."
    ),
)
def summarize_csv(file_path: str, columns: list[str] | None = None) -> str:
    """Return concise statistics for selected numeric CSV columns."""
    try:
        adjusted_file_path = _adjust_file_path(file_path)
        with open(adjusted_file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                return "The CSV file is empty or does not have a header."
            rows = list(reader)

        selected = columns or [
            column for column in reader.fieldnames if column != "t"
        ]
        missing = [
            column for column in selected if column not in reader.fieldnames]
        if missing:
            return f"Missing columns: {', '.join(missing)}"

        summaries = []
        for column in selected:
            values = []
            times = []
            for row in rows:
                try:
                    value = float(row[column])
                except (TypeError, ValueError):
                    continue
                values.append(value)
                if "t" in row:
                    try:
                        times.append(float(row["t"]))
                    except (TypeError, ValueError):
                        times.append(None)

            if not values:
                continue

            min_index = min(range(len(values)), key=values.__getitem__)
            max_index = max(range(len(values)), key=values.__getitem__)
            line = (
                f"{column}: count={len(values)}, first={values[0]:.6g}, "
                f"last={values[-1]:.6g}, min={values[min_index]:.6g}, "
                f"max={values[max_index]:.6g}, mean={sum(values) / len(values):.6g}"
            )
            if times and len(times) == len(values):
                line += (
                    f", min_t={times[min_index]}, max_t={times[max_index]}"
                )
            summaries.append(line)

        return "\n".join(summaries) if summaries else "No numeric data found."
    except ValueError as error:
        return str(error)
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as error:
        return f"An error occurred while summarizing the CSV file: {error}"
