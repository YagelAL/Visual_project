import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from sklearn.cluster import DBSCAN
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from datetime import datetime, date

# Set page config
st.set_page_config(
    page_title="NYC Citibike Station Balance Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directory path
base_dir = os.path.dirname(os.path.abspath(__file__))

# Month mappings
months = {
    "202409": "September 2024",
    "202412": "December 2024",
    "202503": "March 2025",
    "202506": "June 2025"
}

@st.cache_data
def load_processed_data():
    """Load all processed data files"""
    data = {}
    for month_code, month_name in months.items():
        file_path = os.path.join(base_dir, f"processed_{month_code}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date']).dt.date
            data[month_code] = df
        else:
            st.warning(f"Data file not found for {month_name}: {file_path}")
    return data

def create_map_visualization(df, selected_date):
    """Create map visualization showing aggregated station gains/losses"""
    date_data = df[df['date'] == selected_date].copy()
    if date_data.empty:
        st.error(f"No data available for {selected_date}")
        return None

    # Compute difference for coloring
    date_data['diff'] = date_data['departures'] - date_data['arrivals']

    # Normalize diff values for colormap
    max_abs = date_data['diff'].abs().max()
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    colormap = cm.get_cmap("RdYlGn")

    # Apply colors
    date_data['color'] = date_data['diff'].apply(lambda x: to_hex(colormap(norm(x))))

    # Cluster stations within 100m using DBSCAN (haversine metric requires radians)
    coords = date_data[['lat', 'lng']].to_numpy()
    kms_per_radian = 6371.0088
    epsilon = 0.1 / kms_per_radian  # 0.1 km = 100 m
    db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine')
    date_data['cluster'] = db.fit_predict(np.radians(coords))

    # Aggregate per cluster
    agg = date_data.groupby('cluster').agg({
        'lat': 'mean',
        'lng': 'mean',
        'arrivals': 'sum',
        'departures': 'sum',
        'station_name': lambda names: ', '.join(names[:3])
    }).reset_index()
    agg['diff'] = agg['departures'] - agg['arrivals']

    # Size scaling (min 4, max 20)
    abs_diff = agg['diff'].abs()
    pct95 = np.percentile(abs_diff, 95) if not abs_diff.empty else 1
    agg['size'] = 4 + (abs_diff / pct95 * 16)
    agg['size'] = np.clip(agg['size'], 4, 20)

    # Colors for aggregated clusters
    agg['color'] = agg['diff'].apply(lambda x: to_hex(colormap(norm(x))))

    # Hover text
    agg['hover_text'] = agg.apply(
        lambda row: (
            f"<b>{row['station_name']}</b><br>"
            f"Departures: {int(row['departures'])}<br>"
            f"Arrivals: {int(row['arrivals'])}<br>"
            f"Diff: {int(row['diff'])}"
        ), axis=1
    )

    # Build Mapbox figure
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=agg['lat'],
        lon=agg['lng'],
        mode='markers',
        marker=dict(
            size=agg['size'],
            color=agg['color'],
            opacity=0.7,
            sizemode='diameter'
        ),
        text=agg['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        name='Clusters',
        showlegend=False
    ))
    fig.update_layout(
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvb20iLCJhIjoiY21jd2FiMWMyMDBlaDJsc2VxaW50Z2ttcCJ9.eYdMHhxN1no9KMtxEUZDGg',
            style='mapbox://styles/mapbox/streets-v11',
            center=dict(lat=40.7589, lon=-73.9851),
            zoom=11
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def main():
    st.title("🚲 NYC Citibike Station Balance Visualization")
    st.markdown("Visualize which areas gained or lost bikes at the end of each day (clusters within 100m)")

    # Load data
    with st.spinner("Loading data..."):
        processed_data = load_processed_data()
    if not processed_data:
        st.error("No processed data found. Please run the preprocessing script first.")
        return

    # Sidebar controls
    st.sidebar.header("Controls")
    months_available = list(processed_data.keys())
    selected_month = st.sidebar.selectbox(
        "Select Month:",
        options=months_available,
        format_func=lambda x: months[x],
        index=0
    )

    month_data = processed_data[selected_month]
    dates = sorted(month_data['date'].unique())
    selected_date = st.sidebar.selectbox(
        "Select Date:",
        options=dates,
        format_func=lambda d: d.strftime("%Y-%m-%d (%A)"),
        index=len(dates)//2
    )

    # Summary metrics
    summary = month_data[month_data['date'] == selected_date]
    if not summary.empty:
        total = len(summary)
        st.sidebar.subheader("Summary Statistics")
        st.sidebar.metric("Total Active Stations", total)
        net_total = summary['departures'].sum() - summary['arrivals'].sum()
        st.sidebar.metric("Total Net Diff", f"{int(net_total):+d} bikes")

    # Main layout
    col1, col2 = st.columns([3,1])
    with col1:
        st.subheader(f"Station Balance Map - {months[selected_month]}")
        st.caption(f"Date: {selected_date.strftime('%Y-%m-%d (%A)')}")
        fig = create_map_visualization(month_data, selected_date)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Legend & Info")
        st.markdown("""
        🟢 **Green**: More departures than arrivals  
        🔴 **Red**: More arrivals than departures  
        **Darker colors** indicate larger imbalance  
        Stations are grouped if they are within 100 m.
        """)

    # Monthly trend chart
    st.subheader("Monthly Trends")
    daily = month_data.groupby('date').agg({
        'departures': 'sum',
        'arrivals': 'sum'
    }).reset_index()
    daily['net_diff'] = daily['departures'] - daily['arrivals']
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['net_diff'],
        mode='lines+markers',
        name='Daily Net Diff'
    ))
    fig_trend.update_layout(
        title=f"Daily Net Difference Trend - {months[selected_month]}",
        xaxis_title="Date",
        yaxis_title="Bikes (Departures - Arrivals)",
        height=400
    )
    st.plotly_chart(fig_trend, use_container_width=True)


    # Raw data table
    with st.expander("View Raw Data"):
        st.dataframe(
            summary[['station_name','departures','arrivals']].assign(
                diff=lambda df: df['departures'] - df['arrivals']
            ).sort_values('diff', ascending=False),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
