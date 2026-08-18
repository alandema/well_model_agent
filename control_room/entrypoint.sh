#!/bin/sh

set -eu

echo "[control-room] Ensuring telemetry database exists at ${TELEMETRY_DB_PATH:-/data/telemetry.db}"
python -u init_db.py

python -u -m app.well_model.multi_well_model &
WRITER_PID=$!

cleanup() {
	kill "$WRITER_PID" 2>/dev/null || true
	wait "$WRITER_PID" 2>/dev/null || true
}

trap cleanup INT TERM EXIT

streamlit run app/main.py --server.address 0.0.0.0 --server.port 8601


python -m app.well_model.multi_well_model &
streamlit run app/main.py --server.address 0.0.0.0 --server.port 8601