import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from sklearn.cluster import DBSCAN
from datetime import date

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Citibike Station Balance Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Month mappings ─────────────────────────────────────────────────────────────
months = {
    "202409": "September 2024",
    "202412": "December 2024",
    "202503": "March 2025",
    "202506": "June 2025"
}

# ── Load processed data ────────────────────────────────────────────────────────
@st.cache_data
def load_processed_data():
    data = {}
    for code in months:
        fp = f"processed_{code}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df = df[df["date"] != date(2024, 8, 31)]
            data[code] = df
    return data

# ── Fetch weather via Open-Meteo ───────────────────────────────────────────────
@st.cache_data
def load_weather(start_date: date, end_date: date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "daily": ["temperature_2m_max", "relativehumidity_2m_max"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "America/New_York"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({
        "date": pd.to_datetime(d["time"]).dt.date,
        "temperature": d["temperature_2m_max"],
        "humidity": d["relativehumidity_2m_max"]
    })

# ── Create clustered map ────────────────────────────────────────────────────────
def create_map_visualization(df_day, radius_m, categories):
    df_day["diff"] = df_day["departures"] - df_day["arrivals"]
    df_day = df_day.dropna(subset=["lat", "lng"])
    coords = np.radians(df_day[["lat", "lng"]].to_numpy())
    eps = (radius_m / 1000) / 6371.0088
    df_day["cluster"] = DBSCAN(eps=eps, min_samples=1,
                               algorithm="ball_tree", metric="haversine")\
        .fit_predict(coords)

    agg = (
        df_day.groupby("cluster")
              .agg({
                  "lat": "mean",
                  "lng": "mean",
                  "arrivals": "sum",
                  "departures": "sum",
                  "station_name": lambda names: ", ".join(names[:3]) + ("..." if len(names) > 3 else "")
              })
              .reset_index()
    )
    agg["diff"] = agg["departures"] - agg["arrivals"]
    agg["hover"] = agg["station_name"]

    fig = go.Figure()
    for name, mask, color in [
        ("More departures", agg["diff"] > 0, "green"),
        ("More arrivals",   agg["diff"] < 0, "red"),
        ("Balanced",        agg["diff"] == 0, "yellow")
    ]:
        if name in categories:
            sub = agg[mask]
            if not sub.empty:
                fig.add_trace(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lng"], mode="markers",
                    marker=dict(size=12, color=color, opacity=0.8),
                    text=sub["hover"], hovertemplate="%{text}<extra></extra>",
                    name=name
                ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=11,
            bounds=dict(north=40.9176, south=40.4774,
                        east=-73.7004, west=-74.2591)
        ),
        uirevision="station_map",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=14),
            bordercolor="black", borderwidth=1,
            itemclick="toggleothers", itemdoubleclick="toggle"
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    return fig

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("NYC Citibike Station Balance Visualization")

    data = load_processed_data()
    if not data:
        st.error("No data loaded.")
        return

    # combine all dates
    combined = pd.concat(data.values(), ignore_index=True)
    dates = sorted(combined["date"].unique())

    # ── Calendar in sidebar ──────────────────────────────────────────────────────
    st.sidebar.header("Select Date")
    selected_date = st.sidebar.date_input(
        "", value=dates[0], min_value=dates[0], max_value=dates[-1]
    )

    # ── Sidebar controls ─────────────────────────────────────────────────────────
    st.sidebar.header("Map Options")
    radius_m = st.sidebar.slider("Clustering radius (m):", 100, 200, 100, 10)
    show_dep = st.sidebar.checkbox("More departures", True)
    show_arr = st.sidebar.checkbox("More arrivals", True)
    show_bal = st.sidebar.checkbox("Balanced", True)
    categories = [
        name for name, chk in zip(
            ["More departures", "More arrivals", "Balanced"],
            [show_dep, show_arr, show_bal]
        ) if chk
    ]

    # ── Map ─────────────────────────────────────────────────────────────────────
    st.subheader(f"Station Map — {selected_date.strftime('%d/%m/%y')}")
    df_day = combined[combined["date"] == selected_date]
    if df_day.empty:
        st.warning("No Citibike data for this date.")
    else:
        fig_map = create_map_visualization(df_day, radius_m, categories)
        st.plotly_chart(fig_map, use_container_width=True, key="station_map")

    # ── Monthly Trend ───────────────────────────────────────────────────────────
    st.subheader("Monthly Trend")
    month_code = selected_date.strftime("%Y%m")
    if month_code in data:
        df_month = data[month_code]
        daily = df_month.groupby("date").agg({
            "departures": "sum", "arrivals": "sum"
        }).reset_index()
        daily["net"] = daily["departures"] - daily["arrivals"]
        fig_trend = go.Figure(go.Scatter(
            x=daily["date"], y=daily["net"], mode="lines+markers"
        ))
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Net Change (Departures − Arrivals)"
        )
        fig_trend.update_xaxes(tickformat="%d/%m/%y")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No monthly data for selected date.")

    # ── Total Rides per Month ────────────────────────────────────────────────────
    st.subheader("Total Rides per Month")
    rides = [int(df["departures"].sum()) for df in data.values()]
    fig_rides = go.Figure(go.Bar(x=list(months.values()), y=rides))
    fig_rides.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Departures (Rides)"
    )
    st.plotly_chart(fig_rides, use_container_width=True)

    # ── Daily Temperature & Humidity ────────────────────────────────────────────
    st.subheader("Daily Temperature & Humidity")
    weather = load_weather(dates[0], dates[-1])
    fig_weather = go.Figure()
    fig_weather.add_trace(go.Scatter(
        x=weather["date"], y=weather["temperature"],
        mode="lines+markers", name="Temp (°C)"
    ))
    fig_weather.add_trace(go.Bar(
        x=weather["date"], y=weather["humidity"],
        name="Humidity (%)", opacity=0.5
    ))
    fig_weather.update_layout(
        xaxis_title="Date",
        yaxis_title="Temp / Humidity",
        legend=dict(orientation="h", y=1.02, x=0)
    )
    fig_weather.update_xaxes(tickformat="%d/%m/%y")
    st.plotly_chart(fig_weather, use_container_width=True)

if __name__ == "__main__":
    main()
