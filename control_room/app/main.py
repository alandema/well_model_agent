
import streamlit as st
import datetime
import altair as alt

from app.storage import get_db_connection

st.set_page_config(page_title="Well telemetry", layout="wide")
st.title("Well telemetry")


history = st.sidebar.slider(
    "Readings per well", min_value=10, max_value=300, value=60)
well_options = [f"Well {i}" for i in range(1, 5)]
wells = st.sidebar.multiselect("Wells", well_options, default=well_options)


@st.fragment(run_every="2s")
def show_live_telemetry() -> None:
    """Displays live telemetry data for selected wells."""
    query = f"""
        SELECT * FROM sensor_readings
        WHERE well_id IN ({','.join(['?']*len(wells))})
        ORDER BY timestamp DESC
        LIMIT ?
    """
    params = wells + [history]
    conn = get_db_connection()
    try:
        data = conn.execute(query, params).fetchall()
    finally:
        conn.close()
    if not data:
        st.write("No data available.")
        return

    # display line chart using altair and no pandas

    chart_data = {
        # SQLite returns values as strings because the existing schema declares
        # timestamp as TEXT. Convert the stored epoch value before using it.
        "timestamp": [datetime.datetime.fromtimestamp(float(row[1])) for row in data],
        "well_id": [row[2] for row in data],
        "var1": [row[3] for row in data],
        "var2": [row[4] for row in data],
        "var3": [row[5] for row in data],
        "var4": [row[6] for row in data],
        "var5": [row[7] for row in data],
        "var6": [row[8] for row in data],
        "var7": [row[9] for row in data],
        "var8": [row[10] for row in data],
        "var9": [row[11] for row in data],
        "var10": [row[12] for row in data],
    }

    chart = alt.Chart(alt.Data(values=[{
        "timestamp": chart_data["timestamp"][i],
        "well_id": chart_data["well_id"][i],
        "var1": chart_data["var1"][i],
        "var2": chart_data["var2"][i],
        "var3": chart_data["var3"][i],
        "var4": chart_data["var4"][i],
        "var5": chart_data["var5"][i],
        "var6": chart_data["var6"][i],
        "var7": chart_data["var7"][i],
        "var8": chart_data["var8"][i],
        "var9": chart_data["var9"][i],
        "var10": chart_data["var10"][i],
    } for i in range(len(chart_data["timestamp"]))])).mark_line().encode(
        x="timestamp:T",
        y="var1:Q",
        color="well_id:N"
    )

    st.altair_chart(chart, width='stretch')


show_live_telemetry()
