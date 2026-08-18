import random
import sqlite3
import time
from app.storage import get_db_connection


def generate_random_reading(well_id: str) -> dict:
    """Generates a random telemetry reading for a given well."""
    return {
        "timestamp": time.time(),
        "well_id": well_id,
        "var1": random.uniform(0, 100),
        "var2": random.uniform(0, 100),
        "var3": random.uniform(0, 100),
        "var4": random.uniform(0, 100),
        "var5": random.uniform(0, 100),
        "var6": random.uniform(0, 100),
        "var7": random.uniform(0, 100),
        "var8": random.uniform(0, 100),
        "var9": random.uniform(0, 100),
        "var10": random.uniform(0, 100),
    }


def main():
    well_ids = [f"Well {i+1}" for i in range(4)]  # Simulate 4 wells
    while True:
        cycle_started = time.monotonic()
        conn = None
        try:
            readings = [generate_random_reading(
                well_id) for well_id in well_ids]
            conn = get_db_connection()
            with conn:
                conn.executemany(
                    """
                    INSERT INTO sensor_readings (
                        timestamp, well_id, var1, var2, var3, var4, var5, var6, var7, var8, var9, var10
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        tuple(reading[field] for field in (
                            "timestamp", "well_id", "var1", "var2", "var3", "var4",
                            "var5", "var6", "var7", "var8", "var9", "var10",
                        ))
                        for reading in readings
                    ],
                )
            print(f"[telemetry] wrote {len(readings)} readings", flush=True)
        except sqlite3.Error as exc:
            print(
                f"[telemetry] database write failed; retrying: {exc}", flush=True)
        finally:
            if conn is not None:
                conn.close()

        # Keep the start of each write cycle approximately one second apart.
        time.sleep(max(0, 1 - (time.monotonic() - cycle_started)))


if __name__ == "__main__":
    main()
