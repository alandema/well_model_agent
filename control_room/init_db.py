import argparse
import sys
from pathlib import Path

# Ensure control_room is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app.storage import init_db, DEFAULT_DB_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the SQLite database schema for industrial control room telemetry."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="If set, deletes existing database file before recreating the schema",
    )
    args = parser.parse_args()

    db_file = Path(DEFAULT_DB_PATH).resolve()

    if args.reset and db_file.exists():
        print(f"[Init DB] Deleting existing database at {db_file}...")
        db_file.unlink(missing_ok=True)
        db_file.with_suffix(".db-wal").unlink(missing_ok=True)
        db_file.with_suffix(".db-shm").unlink(missing_ok=True)

    print(f"[Init DB] Initializing SQLite schema at: {db_file}")
    init_db(db_file)
    print("[Init DB] Database and schema successfully initialized!")


if __name__ == "__main__":
    main()
