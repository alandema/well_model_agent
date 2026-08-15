
import streamlit as st
import altair as alt
import datetime

from app.storage import DEFAULT_DB_PATH, get_latest_telemetry_df


st.set_page_config(page_title="Well telemetry", layout="wide")
st.title("Well telemetry")


history = st.sidebar.slider("Readings per well", min_value=10, max_value=300, value=60)
well_options = [f"Well {i}" for i in range(1, 5)]
wells = st.sidebar.multiselect("Wells", well_options, default=well_options)


@st.fragment(run_every="1s")
def show_live_telemetry() -> None:
    data = get_latest_telemetry_df(limit=history)

    if not data:
        st.warning("No telemetry data available.")
        return

    now = datetime.datetime.now()
    st.caption(f"Updated {now.strftime('%H:%M:%S')} · {len(data)} readings")


    variables = [f"var{i}" for i in range(1, 11)]


    if not chart_data.empty and chart_data["value"].notna().any():
        st.subheader(variable)
    chart = (
    alt.Chart(chart_data)
    .mark_line(point=True)
    .encode(
    x=alt.X("timestamp:T", title="Time"),
    y=alt.Y("value:Q", title=variable),
    color=alt.Color("well_id:N", title="Well"),
    tooltip=["timestamp:T", "well_id:N", "value:Q"],
    )
    .properties(height=180)
    .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

show_live_telemetry()
