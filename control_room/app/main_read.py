"""Minimal control-room Streamlit app.

Reads the "PDG Pressure" sensor (``channel_0``) from the SQLite database
and displays it as a scatter chart, refreshing every ``data_read_freq``
seconds (from ``app/configs/simulation_config.json``).
"""

import json
import os
import sqlite3

import pandas as pd
import streamlit as st

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "simulation_config.json"
)
# Control-room root; relative db_path values are resolved against it.
CONTROL_ROOM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENSOR_CHANNEL = "channel_0"
SENSOR_NAME = "PDG Pressure"
HISTORY_ROWS = 300


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_readings(db_path: str, limit: int) -> pd.DataFrame:
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        rows = conn.execute(
            f"SELECT timestamp, {SENSOR_CHANNEL} FROM sensor_readings "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    rows.reverse()
    return pd.DataFrame(rows, columns=["timestamp", SENSOR_CHANNEL])


st.set_page_config(page_title="Control room", layout="wide")
st.title("Control room")

config = load_config()

db_path = config["db_path"]
if not os.path.isabs(db_path):
    db_path = os.path.join(CONTROL_ROOM_DIR, db_path)

if not os.path.exists(db_path):
    st.error("Database not found. Run scripts/db_prepare.py first.")
    st.stop()


@st.fragment(run_every=config["data_read_freq"])
def show_readings() -> None:
    df = read_readings(db_path, HISTORY_ROWS)
    if df.empty:
        st.write("No data available.")
        return
    st.scatter_chart(
        df,
        x="timestamp",
        y=SENSOR_CHANNEL,
        x_label="Timestamp",
        y_label=f"{SENSOR_NAME} (Pa)",
    )


show_readings()
