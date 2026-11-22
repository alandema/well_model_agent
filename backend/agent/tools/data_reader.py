"""LangGraph tool for reading sensor data from the control-room SQLite DB."""

import os
import sqlite3
from typing import Any, Literal
from langchain.tools import tool
from pydantic import Field, BaseModel
from datetime import datetime


DB_PATH = os.getenv(
    "CONTROL_ROOM_DB_PATH",
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..",
            "control_room", "data", "control_room.db",
        )
    ),
)

sensor_name_channel_mapping = {
    "PDG_pressure": "channel_0",
    "top_riser_pressure": "channel_1",
    "flowline_pressure": "channel_2",
    "oil_flow_rate": "channel_3",
    "choke_opening": "channel_4",
    "gas_injection_rate": "channel_5",
    "reservoir_pressure": "channel_6",
}

SensorName = Literal[*sensor_name_channel_mapping.keys()]


class ReadSQLiteInput(BaseModel):
    """Choose a date range and the number of evenly spaced readings to return."""

    sensor_names: list[SensorName] | None = Field(
        default=None,
        description=(
            "Exact sensor names to return. Omit or pass an empty list for all."
        ),
    )
    sample_size: int = Field(
        default=1000,
        gt=0,
        description=(
            "Number of readings to return, selected at even intervals across "
            "the date range. Defaults to 1,000. If the range contains fewer "
            "readings, all available readings are returned."
        ),
        ge=1
    )
    initial_time: datetime | None = Field(
        default=None,
        description=(
            "Inclusive ISO 8601 lower bound of the range to sample. If omitted, "
            "the range starts at the earliest available reading."
        ),
    )
    end_time: datetime | None = Field(
        default=None,
        description=(
            "Inclusive ISO 8601 upper bound of the range to sample. If omitted, "
            "the range ends at the latest available reading."
        ),
    )


def _quote_identifier(identifier: str) -> str:
    """Quote a SQLite identifier, escaping embedded double quotes."""
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


@tool(
    args_schema=ReadSQLiteInput,
    description=(
        "Read sensor values from the control-room SQLite database. The inclusive "
        "initial_time and end_time bounds define the range; omitted bounds use "
        "the earliest or latest available reading. Return up to sample_size "
        "readings evenly spaced across that range, ordered by timestamp."
    ),
)
def data_reader(
    sample_size: int = 1000,
    sensor_names: list[str] | None = None,
    initial_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return evenly spaced timestamped readings, keyed by sensor name."""
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise ValueError("sample_size must be an integer")
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if initial_time is not None and end_time is not None and initial_time > end_time:
        raise ValueError(
            "initial_time must be earlier than or equal to end_time")

    selected_sensors = list(dict.fromkeys(sensor_names or ()))
    invalid_names = [
        name for name in selected_sensors
        if name not in sensor_name_channel_mapping
    ]
    if invalid_names:
        available_names = ", ".join(sensor_name_channel_mapping)
        raise ValueError(
            f"Unknown sensor names: {', '.join(invalid_names)}. "
            f"Available sensor names: {available_names}"
        )

    if not selected_sensors:
        selected_sensors = list(sensor_name_channel_mapping)

    selected_columns = [
        f"{_quote_identifier(sensor_name_channel_mapping[name])} "
        f"AS {_quote_identifier(name)}"
        for name in selected_sensors
    ]
    conditions = []
    parameters: list[str | int] = []

    if initial_time is not None:
        conditions.append(f"{_quote_identifier('timestamp')} >= ?")
        parameters.append(
            initial_time.isoformat() if hasattr(
                initial_time, "isoformat") else str(initial_time)
        )
    if end_time is not None:
        conditions.append(f"{_quote_identifier('timestamp')} <= ?")
        parameters.append(
            end_time.isoformat() if hasattr(end_time, "isoformat") else str(end_time)
        )

    query = (
        f"SELECT {_quote_identifier('timestamp')}, "
        f"{', '.join(selected_columns)} "
        f"FROM {_quote_identifier('sensor_readings')}"
    )
    if conditions:
        query += f" WHERE {' AND '.join(conditions)}"

    query += f" ORDER BY {_quote_identifier('timestamp')} ASC"

    connection = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()
        if len(rows) <= sample_size:
            return [dict(row) for row in rows]

        # Include both ends of the available range and distribute sample
        # positions evenly across the ordered readings.
        last_index = len(rows) - 1
        intervals = sample_size - 1
        if intervals == 0:
            indices = [0]
        else:
            indices = [
                (position * last_index + intervals // 2) // intervals
                for position in range(sample_size)
            ]
        return [dict(rows[index]) for index in indices]
    finally:
        connection.close()
