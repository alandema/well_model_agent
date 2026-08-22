import sqlite3
import os
from pathlib import Path

# The Compose bind mount exposes this same path to every container and to the
# host as ./data/telemetry.db. The environment variable also makes the path
# explicit for future readers such as the backend.
DEFAULT_DB_PATH = Path(
    os.getenv(
        "TELEMETRY_DB_PATH",
        Path(__file__).resolve().parents[2] / "data" / "telemetry.db",
    )
)


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured for short concurrent operations."""
    target_path = DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(target_path),
        timeout=5,
        check_same_thread=False,
    )
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init_db() -> None:
    """Initializes the SQLite schema if it does not already exist."""
    conn = get_db_connection()
    with conn:
        # WAL lets the Streamlit reader and other readers run while the
        # producer performs its short insert transaction.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                well_id TEXT NOT NULL,
                var1 REAL,
                var2 REAL,
                var3 REAL,
                var4 REAL,
                var5 REAL,
                var6 REAL,
                var7 REAL,
                var8 REAL,
                var9 REAL,
                var10 REAL
            );
            """
        )
    conn.close()
