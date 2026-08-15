import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional
# Default DB Path in the control_room storage area
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "telemetry.db"


def get_db_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Returns a sqlite3 connection with WAL mode enabled for concurrent reads/writes."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: Optional[Path | str] = None) -> None:
    """Initializes the SQLite schema if it does not already exist."""
    conn = get_db_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
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


def insert_readings(reading: Dict[str, Any], well_id: Optional[str] = None) -> None:
    """Inserts a single reading dictionary into the SQLite database."""
    conn = get_db_connection()
    with conn:
        timestamp = reading.get("timestamp")
        well_id = reading.get("well_id", well_id)
        conn.execute(
            """
            INSERT INTO sensor_readings (
                timestamp, well_id, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                timestamp,
                well_id,
                reading.get("var1"),
                reading.get("var2"),
                reading.get("var3"),
                reading.get("var4"),
                reading.get("var5"),
                reading.get("var6"),
                reading.get("var7"),
                reading.get("var8"),
                reading.get("var9"),
                reading.get("var10"),
            ),
        )
    conn.close()

def get_latest_telemetry_df(
    limit: int = 60
) -> list:
    """Fetches the latest telemetry readings from the SQLite database as a DataFrame."""
    conn = get_db_connection(DEFAULT_DB_PATH)
    query = """
        SELECT * FROM sensor_readings
        ORDER BY timestamp DESC
        LIMIT ?;
    """
    sqlite_cursor = conn.cursor()
    sqlite_cursor.execute(query, (limit,))
    rows = sqlite_cursor.fetchall()
    conn.close()
    return rows




