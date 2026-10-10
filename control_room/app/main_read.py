"""Minimal control-room Streamlit app.

Reads mapped sensor channels from SQLite and displays one scatter chart per
sensor, refreshing every ``data_read_freq`` seconds.
"""

import json
import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "configs", "simulation_config.json"
)
# Control-room root; relative db_path values are resolved against it.
CONTROL_ROOM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENSOR_CHANNEL_MAPS = {
    "channel_0": "PDG Pressure",
    "channel_1": "Top Riser Pressure",
    "channel_2": "Flowline Pressure",
    "channel_3": "Oil Flow Rate",
}

HISTORY_ROWS = 300


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_readings(db_path: str, limit: int) -> pd.DataFrame:
    channels = list(SENSOR_CHANNEL_MAPS)
    channel_columns = ", ".join(channels)
    conn = sqlite3.connect(db_path, timeout=5)
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        rows = conn.execute(
            f"SELECT timestamp, {channel_columns} FROM sensor_readings "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    rows.reverse()
    return pd.DataFrame(rows, columns=["timestamp", *channels])


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
    post_script_template = """
            const gd = document.getElementById('{plot_id}');
            const storageKey = __STORAGE_KEY__;
            const readView = () => {
                try { return JSON.parse(localStorage.getItem(storageKey) || '{}'); }
                catch (_) { return {}; }
            };
            const savedView = readView();
            if (Object.keys(savedView).length) Plotly.relayout(gd, savedView);
            gd.on('plotly_relayout', (event) => {
                const view = readView();
                for (const axis of ['xaxis', 'yaxis']) {
                    if (event[`${axis}.autorange`]) {
                        delete view[`${axis}.range[0]`];
                        delete view[`${axis}.range[1]`];
                    } else {
                        for (const bound of ['range[0]', 'range[1]']) {
                            const key = `${axis}.${bound}`;
                            if (event[key] !== undefined) view[key] = event[key];
                        }
                    }
                }
                try {
                    if (Object.keys(view).length) localStorage.setItem(storageKey, JSON.stringify(view));
                    else localStorage.removeItem(storageKey);
                } catch (_) {}
            });
        """
    for channel, sensor_name in SENSOR_CHANNEL_MAPS.items():
        fig = px.line(
            df,
            x="timestamp",
            y=channel,
            labels={"timestamp": "Timestamp", channel: sensor_name},
        )
        chart_html = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            post_script=post_script_template.replace(
                "__STORAGE_KEY__", json.dumps(f"control-room-{channel}-view")
            ),
        )
        st.subheader(sensor_name)
        st.iframe(chart_html, height=500)


show_readings()
