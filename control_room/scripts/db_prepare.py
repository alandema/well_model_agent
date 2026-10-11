"""Prepare the control-room SQLite database and simulation configuration.

Run manually before starting the services:

    python scripts/db_prepare.py

Creates ``data/control_room.db`` with a single ``sensor_readings`` table
(``timestamp`` + ``channel_0`` .. ``channel_9``). Channel cells are TEXT
holding ``"<value> <unit>"``, so the unit travels with each reading.
"""

import json
import os
import sqlite3

CONTROL_ROOM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(
    CONTROL_ROOM_DIR, "app", "configs", "simulation_config.json"
)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

DB_PATH = os.path.join(CONTROL_ROOM_DIR, config["db_path"])
DB_DIR = os.path.dirname(DB_PATH)

NUM_CHANNELS = 10

# Stored relative to the control_room root so the same config works on the
# host and inside the container. Forward slashes work on both platforms.


def main() -> None:
    os.makedirs(DB_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    try:
        # Default rollback journal: WAL needs shared memory (mmap), which is
        # unavailable on Windows Docker bind mounts.
        conn.execute("PRAGMA journal_mode = DELETE")
        channels = ", ".join(f"channel_{i} TEXT" for i in range(NUM_CHANNELS))
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                timestamp TEXT PRIMARY KEY,
                {channels}
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    main()
