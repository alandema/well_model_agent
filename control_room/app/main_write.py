"""Internal service that runs the FOWM model and appends sensor readings to SQLite.

Reads ``app/configs/simulation_config.json`` and appends one row every
``data_write_freq`` seconds of real time. Each row advances the simulation by
``model_simulated_time`` seconds, so the stored timestamp is the current UTC
time plus ``model_simulated_time``. The channels and units written are defined
in ``app/configs/sensor_mappings.json``; each cell is stored as
``"<value> <unit>"``.
"""

import datetime
import json
import os
import sqlite3
import time

from well_models.fowm_model import load_initial_state, load_params, run_fowm_step

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "simulation_config.json"
)
SENSOR_MAPPINGS_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "sensor_mappings.json"
)
# Control-room root; relative db_path values are resolved against it.
CONTROL_ROOM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(SENSOR_MAPPINGS_PATH, "r", encoding="utf-8") as f:
    _SENSOR_MAPPINGS = json.load(f)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_reading(
    conn: sqlite3.Connection, timestamp: str, values: dict, input_values: dict
) -> None:
    """Append a single row to the sensor_readings table.

    ``values`` holds the model outputs and ``input_values`` the model inputs.
    Each mapped channel is stored as ``"<value> <unit>"``, using the channel
    and unit from configs/sensor_mappings.json.
    """
    merged = {**values, **input_values}
    columns = []
    row = []
    for name, entry in _SENSOR_MAPPINGS.items():
        columns.append(entry["channel"])
        value = merged.get(name)
        row.append(None if value is None else f"{value} {entry['unit']}")

    placeholders = ", ".join("?" for _ in range(len(columns) + 1))
    conn.execute(
        f"INSERT INTO sensor_readings (timestamp, {', '.join(columns)}) "
        f"VALUES ({placeholders})",
        [timestamp] + row,
    )
    conn.commit()


def main() -> None:
    config = load_config()
    db_path = config["db_path"]
    if not os.path.isabs(db_path):
        db_path = os.path.join(CONTROL_ROOM_DIR, db_path)
    write_freq = config["data_write_freq"]
    sim_time = config["model_simulated_time"]

    if not os.path.exists(db_path):
        raise SystemExit(
            f"Database not found at {db_path}. Run scripts/db_prepare.py first."
        )

    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA busy_timeout = 5000")

    print(
        f"[main_write] Writing to {db_path} every {write_freq}s, "
        f"simulating {sim_time}s ahead",
        flush=True,
    )

    # x0 is only used at service start; each step simulates a single next
    # state, model_simulated_time seconds ahead, and feeds it back for
    # continuity.

    state = load_initial_state()
    while True:
        params = load_params()
        state, values, input_values = run_fowm_step(state, sim_time, params)
        ts = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=sim_time)
        ).isoformat()
        append_reading(conn, ts, values, input_values)
        time.sleep(write_freq)


if __name__ == "__main__":
    main()
