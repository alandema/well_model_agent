"""Internal service that runs the FOWM model and appends sensor readings to SQLite.

Reads ``app/configs/simulation_config.json`` and appends one row every
``data_write_freq`` seconds of real time. Each row advances the simulation by
``model_simulated_time`` seconds, so the stored timestamp is the current UTC
time plus ``model_simulated_time``. Only ``channel_0`` is used, carrying the
"PDG Pressure" sensor reading.
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
# Control-room root; relative db_path values are resolved against it.
CONTROL_ROOM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The only sensor of interest for now: PDG Pressure goes into channel_0.
SENSOR_CHANNEL = 0
NUM_CHANNELS = 10


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_reading(conn: sqlite3.Connection, timestamp: str, value: float) -> None:
    channels = [None] * NUM_CHANNELS
    channels[SENSOR_CHANNEL] = value
    columns = ", ".join(f"channel_{i}" for i in range(NUM_CHANNELS))
    placeholders = ", ".join("?" for _ in range(NUM_CHANNELS + 1))
    conn.execute(
        f"INSERT INTO sensor_readings (timestamp, {columns}) VALUES ({placeholders})",
        [timestamp] + channels,
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
        state, value = run_fowm_step(state, sim_time, params)
        ts = (
            datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(seconds=sim_time)
        ).isoformat()
        append_reading(conn, ts, float(value))
        time.sleep(write_freq)


if __name__ == "__main__":
    main()
