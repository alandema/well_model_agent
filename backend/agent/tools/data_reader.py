"""LangGraph tool for reading a SQLite database."""

import sqlite3
from typing import Any
from pydantic import BaseModel, Field
from langchain.tools import ToolRuntime, tool


class ReadSQLiteInput(BaseModel):
    query: str = Field(
        description="The SQL query to execute against the SQLite database.")


@tool(
    args_schema=ReadSQLiteInput,
    description="Run a read-only SQL query against a pre-configured SQLite database."
)
def read_sqlite(query: str, runtime: ToolRuntime[None, Any]) -> list[dict[str, Any]]:
    """Run a read-only SQL query against a pre-configured SQLite database."""
    db_path = runtime.config.get("configurable", {}).get("db_path")
    if not db_path:
        raise ValueError("Missing database path: configurable.db_path")

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()
