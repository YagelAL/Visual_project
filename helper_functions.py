import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import random
import math
import warnings
from libpysal.weights import W
import time
from libpysal.weights import KNN
import geopandas as gpd
from spopt.region import RegionKMeansHeuristic
from pyproj import Transformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.express as px

# ── Configuration ──────────────────────────────────────────────────────────────
# Performance settings - adjust these values based on your system capabilities
MAX_STATIONS_DEFAULT = 200  # Default maximum stations for performance
MAX_STATIONS_VISUALIZATION = 100  # Maximum stations for complex visualizations

# ── Month mappings ─────────────────────────────────────────────────────────────
months = {
    "202409": "September 2024",
    "202412": "December 2024",
    "202503": "March 2025",
    "202506": "June 2025"
}


# ── DATA LOADING AND PREPARATION ───────────────────────────────────────────────

def generate_test_data(num_stations=50, num_days=30):
    """
    Generate lightweight synthetic test data for presentations/demos
    
    Args:
        num_stations: Number of stations to generate (default: 50)
        num_days: Number of days of data (default: 30)
    
    Returns:
        Dictionary with test data in the same format as load_processed_data()
    """
    import random
    from datetime import datetime, timedelta
    
    # NYC-like station locations (Manhattan area)
    station_locations = []
    for i in range(num_stations):
        lat = 40.7 + random.uniform(0, 0.08)  # Manhattan latitude range
        lng = -74.0 + random.uniform(0, 0.03)  # Manhattan longitude range
        station_locations.append({
            'station_name': f"Test Station {i+1:02d}",
            'lat': lat,
            'lng': lng
        })
    
    # Generate data for each month code
    data = {}
    base_date = datetime(2024, 9, 1)  # Start from September 2024
    
    for month_code in months.keys():
        month_data = []
        
        # Generate data for each day
        for day in range(num_days):
            current_date = base_date + timedelta(days=day)
            
            for station in station_locations:
                # Simulate realistic bike share patterns
                base_activity = random.randint(10, 100)
                
                # Add some patterns based on location and day
                weekday_factor = 1.2 if current_date.weekday() < 5 else 0.8
                location_factor = 1.5 if abs(station['lat'] - 40.75) < 0.02 else 1.0  # Central locations busier
                
                departures = max(0, int(base_activity * weekday_factor * location_factor * random.uniform(0.7, 1.3)))
                arrivals = max(0, int(base_activity * weekday_factor * location_factor * random.uniform(0.7, 1.3)))
                
                month_data.append({
                    'date': current_date.date(),
                    'station_name': station['station_name'],
                    'lat': station['lat'],
                    'lng': station['lng'],
                    'departures': departures,
                    'arrivals': arrivals
                })
        
        data[month_code] = pd.DataFrame(month_data)
        
        # Move to next month for variety
        if month_code == "202409":
            base_date = datetime(2024, 12, 1)
        elif month_code == "202412":
            base_date = datetime(2025, 3, 1)
        elif month_code == "202503":
            base_date = datetime(2025, 6, 1)
    
    return data

def limit_stations_for_performance(df, max_stations=None, station_column='station_name'):
    """
    Utility function to limit stations for performance with configurable maximum
    
    Args:
        df: DataFrame containing station data
        max_stations: Maximum number of stations to keep (uses MAX_STATIONS_DEFAULT if None)
        station_column: Name of the column containing station names
    
    Returns:
        DataFrame with limited stations
    """
    if max_stations is None:
        max_stations = MAX_STATIONS_DEFAULT
    
    unique_stations = df[station_column].unique()
    
    if len(unique_stations) > max_stations:
        selected_stations = pd.Series(unique_stations).sample(n=max_stations, random_state=42).tolist()
        return df[df[station_column].isin(selected_stations)]
    
    return df

@st.cache_data
def load_processed_data(max_stations=None, use_test_data=False):
    """Load and combine all processed monthly data files - configurable station limit for performance"""
    if max_stations is None:
        max_stations = MAX_STATIONS_DEFAULT
    
    # Use test data for presentations/demos
    if use_test_data:
        st.info("🧪 Using synthetic test data for demonstration")
        return generate_test_data(num_stations=min(50, max_stations), num_days=30)
        
    data = {}
    all_stations = set()
    
    # First pass: collect all unique stations
    for code in months:
        fp = f"processed_{code}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            all_stations.update(df['station_name'].unique())
    
    # If no CSV files found, fall back to test data
    if not all_stations:
        st.warning("📁 No CSV files found. Using synthetic test data instead.")
        return generate_test_data(num_stations=min(50, max_stations), num_days=30)
    
    # Limit to specified number of random stations for performance
    if len(all_stations) > max_stations:
        selected_stations = pd.Series(list(all_stations)).sample(n=max_stations, random_state=42).tolist()
        st.info(f"📊 Limited to {max_stations} stations for optimal performance (out of {len(all_stations)} available)")
    else:
        selected_stations = list(all_stations)
    
    # Second pass: load data only for selected stations
    for code in months:
        fp = f"processed_{code}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            # drop the bogus 2024-08-31 row
            df = df[df["date"] != date(2024, 8, 31)]
            # Filter to only selected stations
            df = df[df['station_name'].isin(selected_stations)]
            data[code] = df
    
    return data


@st.cache_data
def prepare_geodata_and_weights(full_df, max_stations=None):
    """
    Prepare geodata optimized for performance with configurable station limit
    """
    if max_stations is None:
        max_stations = MAX_STATIONS_DEFAULT
        
    # Ensure we don't exceed the specified maximum for performance
    if len(full_df) > max_stations:
        full_df = full_df.sample(n=max_stations, random_state=42)
        st.info(f"📊 Limited to {max_stations} stations for optimal performance")

    g = gpd.GeoDataFrame(
        full_df[['station_name', 'lat', 'lng']],
        geometry=gpd.points_from_xy(full_df.lng, full_df.lat),
        crs="EPSG:4326"
    ).to_crs("EPSG:3857")

    # Use distance-based weights with reasonable threshold for better connectivity
    try:
        from libpysal.weights import DistanceBand
        # Use 400m threshold for NYC (good balance of connectivity vs locality)
        threshold = 400  # meters in projected CRS
        w = DistanceBand.from_dataframe(g, threshold=threshold, silence_warnings=True)

        # If too disconnected, increase threshold
        if w.n_components > len(g) * 0.2:  # Too many components
            threshold = 600
            w = DistanceBand.from_dataframe(g, threshold=threshold, silence_warnings=True)

        if w.n_components > len(g) * 0.2:  # Still too disconnected
            # Fallback to KNN with more neighbors
            k = min(8, len(g) - 1)
            w = KNN.from_dataframe(g, k=k)

    except:
        # Final fallback
        k = min(6, len(g) - 1)
        w = KNN.from_dataframe(g, k=k)

    g['X'] = g.geometry.x
    g['Y'] = g.geometry.y
    return g, w


@st.cache_data
def prepare_daily_time_series_data(combined, selected_month, max_stations=None):
    """
    Prepare daily time series data for a specific month
    """
    df2 = combined.copy()
    df2['date'] = pd.to_datetime(df2['date'])

    # Filter by selected month
    month_data = df2[df2['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
    # Limit stations for performance using utility function
    month_data = limit_stations_for_performance(month_data, max_stations)

    # Get all days in the month
    days_in_month = sorted(month_data['date'].dt.date.unique())

    series_list = []
    day_info = []

    for day in days_in_month:
        day_data = month_data[month_data['date'].dt.date == day]

        if not day_data.empty:
            # Aggregate by station for this day
            agg = (
                day_data
                .groupby(['station_name'], as_index=False)
                .agg({'departures': 'sum', 'arrivals': 'sum'})
            )
            agg['net_balance'] = agg['departures'] - agg['arrivals']

            # Create series for this day
            ser = agg.set_index('station_name')['net_balance']
            ser.name = day.strftime('%d/%m')
            series_list.append(ser)

            # Store day information
            day_info.append({
                'date': day,
                'day_label': day.strftime('%d/%m'),
                'stations_count': len(agg)
            })

    if not series_list:
        return pd.DataFrame(), pd.DataFrame(), []

    # Create pivot table
    pivot = pd.concat(series_list, axis=1).fillna(0)

    # Get station coordinates
    station_coords = (
        combined
        .drop_duplicates('station_name')[['station_name', 'lat', 'lng']]
        .reset_index(drop=True)
    )

    return pivot, station_coords, day_info


# ── CLUSTERING FUNCTIONS ───────────────────────────────────────────────────────

@st.cache_data
def perform_time_series_clustering(pivot_net_filtered, n_clusters, station_coords):
    """Perform K-means clustering on time series data"""
    ts_data = pivot_net_filtered.values

    if ts_data.shape[0] < n_clusters:
        st.warning(
            f"Not enough data points ({ts_data.shape[0]}) for {n_clusters} clusters. Returning empty clustering results.")
        return pd.DataFrame(), None

    scaler = StandardScaler()
    ts_data_scaled = scaler.fit_transform(ts_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ts_data_scaled) + 1

    results = pd.DataFrame({
        'station_name': pivot_net_filtered.index,
        'cluster': cluster_labels
    })

    results = results.merge(station_coords, on='station_name', how='left')
    results = results.dropna(subset=['lat', 'lng'])

    return results, kmeans


# ── VISUALIZATION FUNCTIONS ────────────────────────────────────────────────────

def create_map_visualization(df_day, radius_m, categories, max_stations=None):
    """Create static map visualization with DBSCAN clustering"""
    df_day = df_day.copy()
    
    # Limit stations for performance using the utility function
    df_day = limit_stations_for_performance(df_day, max_stations)
    
    df_day["diff"] = df_day["departures"] - df_day["arrivals"]
    df_day = df_day.dropna(subset=["lat", "lng"])
    coords = np.radians(df_day[["lat", "lng"]].to_numpy())
    eps = (radius_m / 1000) / 6371.0088
    df_day["cluster"] = DBSCAN(
        eps=eps, min_samples=1,
        algorithm="ball_tree", metric="haversine"
    ).fit_predict(coords)

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
        ("More arrivals", agg["diff"] < 0, "red"),
        ("Balanced", agg["diff"] == 0, "yellow")
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


def create_time_series_cluster_map(clustered_df):
    """Create map visualization for time series clustering results"""
    clustered_df = clustered_df.dropna(subset=['lat', 'lng'])
    if clustered_df.empty:
        return go.Figure()

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]

    fig = go.Figure()
    unique_clusters = sorted(clustered_df['cluster'].unique())

    for i, cluster_id in enumerate(unique_clusters):
        cluster_df = clustered_df[clustered_df['cluster'] == cluster_id]

        if not cluster_df.empty:
            fig.add_trace(go.Scattermapbox(
                lat=cluster_df["lat"],
                lon=cluster_df["lng"],
                mode="markers",
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    opacity=0.9
                ),
                name=f"Cluster {cluster_id}",
                text=cluster_df["station_name"],
                hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster_id) + "<extra></extra>"
            ))

    if not clustered_df.empty:
        center_lat = clustered_df["lat"].mean()
        center_lon = clustered_df["lng"].mean()
    else:
        center_lat = 40.7128
        center_lon = -74.0060

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=11,
        mapbox_center={
            "lat": center_lat,
            "lon": center_lon
        },
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=16),
            bordercolor="Black",
            borderwidth=1
        )
    )

    return fig


def create_timeline_map(combined, start_date, end_date, radius_m, categories, max_stations=None):
    """Create animated timeline map visualization"""
    dates = pd.date_range(start_date, end_date).date
    frames = []
    for d in dates:
        df_day = combined[combined["date"] == d].copy()
        
        # Limit stations per day for performance using utility function
        df_day = limit_stations_for_performance(df_day, max_stations)
            
        df_day["diff"] = df_day["departures"] - df_day["arrivals"]
        df_day = df_day.dropna(subset=["lat", "lng"])
        coords = np.radians(df_day[["lat", "lng"]].to_numpy())
        eps = (radius_m / 1000) / 6371.0088
        df_day["cluster"] = DBSCAN(
            eps=eps, min_samples=1,
            algorithm="ball_tree", metric="haversine"
        ).fit_predict(coords)

        agg = (
            df_day.groupby("cluster")
            .agg({
                "lat": "mean", "lng": "mean",
                "arrivals": "sum", "departures": "sum",
                "station_name": lambda names: ", ".join(names[:3]) + ("..." if len(names) > 3 else "")
            })
            .reset_index()
        )
        agg["diff"] = agg["departures"] - agg["arrivals"]
        agg["hover"] = agg["station_name"]

        traces = []
        for name, mask, color in [
            ("More departures", agg["diff"] > 0, "green"),
            ("More arrivals", agg["diff"] < 0, "red"),
            ("Balanced", agg["diff"] == 0, "yellow")
        ]:
            if name in categories:
                sub = agg[mask]
                traces.append(go.Scattermapbox(
                    lat=sub["lat"], lon=sub["lng"], mode="markers",
                    marker=dict(size=12, color=color, opacity=0.8),
                    text=sub["hover"], hovertemplate="%{text}<extra></extra>",
                    name=name, showlegend=False
                ))

        frames.append(go.Frame(data=traces, name=d.strftime('%Y-%m-%d')))

    fig = go.Figure(
        data=frames[0].data if frames else [],
        frames=frames,
        layout=go.Layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=40.7128, lon=-74.0060),
                zoom=11,
                bounds=dict(
                    north=40.9176, south=40.4774,
                    east=-73.7004, west=-74.2591
                )
            ),
            uirevision="station_map",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
                font=dict(size=14),
                bordercolor="black", borderwidth=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=700,

            updatemenus=[dict(
                type="buttons",
                direction="left",
                showactive=False,
                x=0.5,
                y=-0.05,
                xanchor="center",
                yanchor="top",
                pad=dict(t=10, b=5, l=5, r=5),
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )],

            sliders=[dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f.name], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate"
                        }],
                        label=f.name
                    ) for f in frames
                ],
                transition={"duration": 0},
                x=0,
                y=-0.2,
                currentvalue={"prefix": "Date: "},
                pad={"t": 20, "b": 10}
            )]
        )
    )

    return fig


def create_spider_plot_for_month(combined, selected_date, max_stations=None):
    """Create a spider plot showing station metrics for the entire month of the selected date"""
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure()
    
    # Limit stations for performance using utility function
    month_data = limit_stations_for_performance(month_data, max_stations)

    # Calculate station-level metrics for the entire month
    station_metrics = []

    for station_name in month_data['station_name'].unique():
        station_data = month_data[month_data['station_name'] == station_name]

        if len(station_data) > 0:
            daily_balances = (station_data['departures'] - station_data['arrivals']).values

            # Calculate weekday vs weekend difference
            weekday_vals = []
            weekend_vals = []
            for _, day_row in station_data.iterrows():
                day_of_week = pd.to_datetime(day_row['date']).weekday()
                daily_balance = day_row['departures'] - day_row['arrivals']
                if day_of_week < 5:  # Monday=0, Sunday=6
                    weekday_vals.append(daily_balance)
                else:
                    weekend_vals.append(daily_balance)

            weekday_weekend_diff = 0
            if weekday_vals and weekend_vals:
                weekday_weekend_diff = abs(np.mean(weekday_vals) - np.mean(weekend_vals))

            metrics = {
                'station_name': station_name,
                'lat': station_data['lat'].iloc[0],
                'lng': station_data['lng'].iloc[0],
                'avg_net_balance': np.mean(daily_balances),
                'volatility': np.std(daily_balances),
                'range_val': np.max(daily_balances) - np.min(daily_balances),
                'trend_slope': abs(np.polyfit(range(len(daily_balances)), daily_balances, 1)[0]) if len(
                    daily_balances) > 1 else 0,
                'peak_value': np.max(daily_balances),
                'valley_value': abs(np.min(daily_balances)),
                'weekday_weekend_diff': weekday_weekend_diff,
                'consistency': 100 - (np.std(daily_balances) / (abs(np.mean(daily_balances)) + 1) * 100)
            }
            station_metrics.append(metrics)

    if not station_metrics:
        return go.Figure()

    # Create DataFrame and normalize metrics
    stations_df = pd.DataFrame(station_metrics)

    # Limit to reasonable number of stations for visualization
    if len(stations_df) > 100:
        stations_df = stations_df.sample(n=100, random_state=42)

    # Normalize all metrics to 0-1 scale for spider plot
    metric_cols = ['avg_net_balance', 'volatility', 'range_val', 'trend_slope',
                   'peak_value', 'valley_value', 'weekday_weekend_diff', 'consistency']

    for col in metric_cols:
        min_val, max_val = stations_df[col].min(), stations_df[col].max()
        if max_val > min_val:
            stations_df[f'{col}_norm'] = (stations_df[col] - min_val) / (max_val - min_val)
        else:
            stations_df[f'{col}_norm'] = 0.5

    # Create spider plot
    fig_spider = go.Figure()

    # Define spider plot parameters
    n_metrics = len(metric_cols)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)

    # Color stations by their average net balance (red = more departures, green = more arrivals, blue = balanced)
    stations_df['color'] = stations_df['avg_net_balance'].apply(
        lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
    )

    # Plot each station as a spider glyph
    for _, station in stations_df.iterrows():
        # Get normalized values for spider plot
        values = [station[f'{col}_norm'] for col in metric_cols]

        # Add spider lines
        for i in range(n_metrics):
            angle = angles[i]
            value = values[i]
            scaled_value = 0.05 + value * 0.15  # Scale for visibility
            x_end = station['lng'] + scaled_value * np.cos(angle) * 0.01
            y_end = station['lat'] + scaled_value * np.sin(angle) * 0.01

            fig_spider.add_trace(go.Scattermapbox(
                lat=[station['lat'], y_end],
                lon=[station['lng'], x_end],
                mode='lines',
                line=dict(color=station['color'], width=1),
                hoverinfo='skip',
                showlegend=False
            ))

        # Add center point
        fig_spider.add_trace(go.Scattermapbox(
            lat=[station['lat']],
            lon=[station['lng']],
            mode='markers',
            marker=dict(size=3, color=station['color']),
            text=station['station_name'],
            hovertemplate=f"<b>%{{text}}</b><br>Avg Balance: {station['avg_net_balance']:.1f}<extra></extra>",
            showlegend=False
        ))

    # Add legend entries
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#FF6B6B'),
        name='More Departures',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#69DB7C'),
        name='More Arrivals',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#4DABF7'),
        name='Balanced',
        showlegend=True
    ))

    fig_spider.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=stations_df['lat'].mean(),
                lon=stations_df['lng'].mean()
            ),
            zoom=10
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=14)
        )
    )

    return fig_spider


# ── ADDITIONAL VISUALIZATION FUNCTIONS ─────────────────────────────────────────


def create_weekly_seasonal_plot(combined):
    """Create a weekly seasonal plot showing patterns across years"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        df_temp = combined.copy()
        
        # Limit to 200 random stations for performance
        if len(df_temp['station_name'].unique()) > 200:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=200, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['week'] = df_temp['date'].dt.isocalendar().week
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']

        # Calculate weekly totals
        weekly_totals = (
            df_temp.groupby(['year', 'week'])
            .agg({'total_rides': 'sum'})
            .reset_index()
        )

        if weekly_totals.empty:
            return go.Figure()

        # Create polar plot
        fig = go.Figure()

        # Color palette for years
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        for i, year in enumerate(sorted(weekly_totals['year'].unique())):
            year_data = weekly_totals[weekly_totals['year'] == year].sort_values('week')

            # Convert week numbers to angles (0-360 degrees)
            angles = [(week - 1) * 360 / 52 for week in year_data['week']]

            fig.add_trace(go.Scatterpolar(
                r=year_data['total_rides'],
                theta=angles,
                mode='lines+markers',
                name=str(year),
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=6)
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title="Total Rides",
                    visible=True,
                    range=[0, weekly_totals['total_rides'].max() * 1.1]
                ),
                angularaxis=dict(
                    tickvals=[0, 90, 180, 270],
                    ticktext=['Jan', 'Apr', 'Jul', 'Oct'],
                    direction='clockwise',
                    period=360
                )
            ),
            title="Weekly Seasonal Patterns by Year",
            height=600,
            showlegend=True
        )

        return fig

    except Exception as e:
        print(f"Error creating weekly seasonal plot: {e}")
        return go.Figure()


def create_spider_glyph_month(combined):
    """Create spider glyph plot with normalized month number on Y-axis"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import MinMaxScaler

        df_temp = combined.copy()
        
        # Limit to 200 random stations for performance
        if len(df_temp['station_name'].unique()) > 200:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=200, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['year'] = df_temp['date'].dt.year
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']

        # Create month mapping (assuming data months are Sep, Dec, Mar, Jun)
        month_mapping = {9: 1, 12: 2, 3: 3, 6: 4}  # Map to 1-4 for visualization

        # Calculate station metrics by month
        station_monthly = (
            df_temp.groupby(['station_name', 'month'])
            .agg({
                'total_rides': 'sum',
                'departures': 'sum',
                'arrivals': 'sum',
                'net_balance': 'sum',
                'lat': 'first',
                'lng': 'first'
            })
            .reset_index()
        )

        # Add weekday/weekend split
        weekday_data = df_temp[df_temp['day_of_week'] < 5].groupby(['station_name', 'month'])[
            'total_rides'].sum().reset_index()
        weekday_data.columns = ['station_name', 'month', 'weekday_rides']

        weekend_data = df_temp[df_temp['day_of_week'] >= 5].groupby(['station_name', 'month'])[
            'total_rides'].sum().reset_index()
        weekend_data.columns = ['station_name', 'month', 'weekend_rides']

        # Merge data
        station_monthly = station_monthly.merge(weekday_data, on=['station_name', 'month'], how='left')
        station_monthly = station_monthly.merge(weekend_data, on=['station_name', 'month'], how='left')
        station_monthly['weekday_rides'] = station_monthly['weekday_rides'].fillna(0)
        station_monthly['weekend_rides'] = station_monthly['weekend_rides'].fillna(0)

        # Map months to numbers
        station_monthly['month_num'] = station_monthly['month'].map(month_mapping)
        station_monthly = station_monthly.dropna(subset=['month_num'])

        # Limit stations for visualization
        if len(station_monthly) > 300:
            station_monthly = station_monthly.sample(n=300, random_state=42)

        # Normalize X and Y axes using MinMaxScaler for better readability
        scaler_x = MinMaxScaler(feature_range=(0, 100))
        scaler_y = MinMaxScaler(feature_range=(0, 10))
        
        station_monthly['total_rides_norm'] = scaler_x.fit_transform(station_monthly[['total_rides']]).flatten()
        station_monthly['month_num_norm'] = scaler_y.fit_transform(station_monthly[['month_num']]).flatten()

        # Create spider glyph plot
        fig = go.Figure()

        # Color by net balance
        station_monthly['color'] = station_monthly['net_balance'].apply(
            lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
        )

        # Normalize spider arm lengths for consistent visualization
        max_arrivals = station_monthly['arrivals'].max()
        max_departures = station_monthly['departures'].max()
        max_weekday = station_monthly['weekday_rides'].max()
        max_weekend = station_monthly['weekend_rides'].max()

        # Fixed scale for spider arms (normalized to axis scale)
        scale_factor = 2.0  # Arms extend up to 2 units on normalized scale

        for _, station in station_monthly.iterrows():
            x_center = station['total_rides_norm']
            y_center = station['month_num_norm']

            # Calculate normalized spider arm endpoints
            up_length = (station['arrivals'] / max_arrivals) * scale_factor if max_arrivals > 0 else 0
            down_length = (station['departures'] / max_departures) * scale_factor if max_departures > 0 else 0
            left_length = (station['weekday_rides'] / max_weekday) * scale_factor if max_weekday > 0 else 0
            right_length = (station['weekend_rides'] / max_weekend) * scale_factor if max_weekend > 0 else 0

            # Add spider arms
            # Up (arrivals) - Green
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center + up_length],
                mode='lines',
                line=dict(color='#2E8B57', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Down (departures) - Red
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center - down_length],
                mode='lines',
                line=dict(color='#DC143C', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Left (weekday) - Blue
            fig.add_trace(go.Scatter(
                x=[x_center, x_center - left_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color='#1E90FF', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Right (weekend) - Orange
            fig.add_trace(go.Scatter(
                x=[x_center, x_center + right_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color='#FF8C00', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=4, color=station['color']),
                text=station['station_name'],
                hovertemplate=f"<b>%{{text}}</b><br>Total Rides: {station['total_rides']}<br>Month: {station['month']}<br>Net Balance: {station['net_balance']}<extra></extra>",
                showlegend=False
            ))

        # Add spider arm legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#2E8B57', width=3),
                                 name='↑ Arrivals', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#DC143C', width=3),
                                 name='↓ Departures', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#1E90FF', width=3),
                                 name='← Weekday', showlegend=True))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#FF8C00', width=3),
                                 name='→ Weekend', showlegend=True))

        # Add balance legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#FF6B6B'),
                                 name='More Departures', showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#69DB7C'), name='More Arrivals',
                       showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#4DABF7'), name='Balanced',
                       showlegend=True))

        fig.update_layout(
            title="Spider Glyph: Normalized Month vs Total Rides",
            xaxis_title="Normalized Total Rides (0-100 scale)",
            yaxis_title="Normalized Month Number (0-10 scale)",
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        return fig

    except Exception as e:
        print(f"Error creating month spider glyph: {e}")
        return go.Figure()


def create_time_wheel_plot(combined):

        # Map months to numbers
        station_monthly['month_num'] = station_monthly['month'].map(month_mapping)
        station_monthly = station_monthly.dropna(subset=['month_num'])

        # Limit stations for visualization
        if len(station_monthly) > 300:
            station_monthly = station_monthly.sample(n=300, random_state=42)

        # Create spider glyph plot
        fig = go.Figure()

        # Color by net balance
        station_monthly['color'] = station_monthly['net_balance'].apply(
            lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
        )

        # Normalize spider arm lengths with better scaling
        max_arrivals = station_monthly['arrivals'].max()
        max_departures = station_monthly['departures'].max()
        max_weekday = station_monthly['weekday_rides'].max()
        max_weekend = station_monthly['weekend_rides'].max()

        # Use fixed scale instead of proportional to total_rides for better readability
        base_scale = 2000  # Fixed base scale for spider arms
        scale_factor = 0.8  # Scale down spider arms

        for _, station in station_monthly.iterrows():
            x_center = station['total_rides']
            y_center = station['month_num']

            # Calculate spider arm endpoints with fixed scaling
            up_length = (station['arrivals'] / max_arrivals) * scale_factor if max_arrivals > 0 else 0
            down_length = (station['departures'] / max_departures) * scale_factor if max_departures > 0 else 0
            left_length = (station['weekday_rides'] / max_weekday) * scale_factor if max_weekday > 0 else 0
            right_length = (station['weekend_rides'] / max_weekend) * scale_factor if max_weekend > 0 else 0

            # Add spider arms
            # Up (arrivals)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center + up_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Down (departures)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center - down_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Left (weekday)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center - left_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Right (weekend)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center + right_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=4, color=station['color']),
                text=station['station_name'],
                hovertemplate=f"<b>%{{text}}</b><br>Total Rides: {station['total_rides']}<br>Month: {station['month']}<br>Net Balance: {station['net_balance']}<extra></extra>",
                showlegend=False
            ))

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#FF6B6B'),
                                 name='More Departures', showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#69DB7C'), name='More Arrivals',
                       showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#4DABF7'), name='Balanced',
                       showlegend=True))

        fig.update_layout(
            title="Spider Glyph: Month Number vs Total Rides",
            xaxis_title="Total Rides per Month",
            yaxis_title="Month Number (1=Sep, 2=Dec, 3=Mar, 4=Jun)",
            height=600,
            showlegend=True
        )

        return fig



def create_time_wheel_plot(combined):
    """Create a time wheel plot showing activity patterns throughout the day and week"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from math import pi, cos, sin

        df_temp = combined.copy()
        
        # Limit to top stations for performance
        if len(df_temp['station_name'].unique()) > 100:
            top_stations = df_temp.groupby('station_name')['departures'].sum().nlargest(100).index
            df_temp = df_temp[df_temp['station_name'].isin(top_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        
        # Simulate hourly data based on typical bike share patterns
        np.random.seed(42)
        hourly_data = []
        
        for _, row in df_temp.iterrows():
            # Create realistic hourly distribution
            base_pattern = np.array([
                0.02, 0.01, 0.01, 0.01, 0.02, 0.05,  # 0-5: Night/Early morning
                0.08, 0.12, 0.15, 0.10, 0.08, 0.07,  # 6-11: Morning rush/mid-morning
                0.06, 0.05, 0.04, 0.04, 0.05, 0.08,  # 12-17: Lunch/afternoon
                0.12, 0.10, 0.08, 0.06, 0.04, 0.03   # 18-23: Evening rush/night
            ])
            
            # Add day-of-week variation
            dow = row['date'].dayofweek
            if dow >= 5:  # Weekend
                # Smoother pattern for weekends
                base_pattern = np.array([
                    0.03, 0.02, 0.02, 0.02, 0.03, 0.04,  # 0-5: Night
                    0.06, 0.08, 0.10, 0.12, 0.12, 0.11,  # 6-11: Late morning
                    0.10, 0.09, 0.08, 0.07, 0.07, 0.08,  # 12-17: Afternoon
                    0.09, 0.08, 0.07, 0.06, 0.05, 0.04   # 18-23: Evening
                ])
            
            # Normalize to sum to 1
            base_pattern = base_pattern / base_pattern.sum()
            
            # Distribute total rides across hours
            for hour in range(24):
                rides_this_hour = int(row['total_rides'] * base_pattern[hour])
                if rides_this_hour > 0:
                    hourly_data.append({
                        'station_name': row['station_name'],
                        'date': row['date'],
                        'hour': hour,
                        'day_of_week': dow,
                        'rides': rides_this_hour,
                        'lat': row['lat'],
                        'lng': row['lng']
                    })
        
        hourly_df = pd.DataFrame(hourly_data)
        
        if hourly_df.empty:
            return go.Figure().add_annotation(text="No data available for time wheel", 
                                              xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Aggregate by hour and day of week
        time_agg = hourly_df.groupby(['hour', 'day_of_week'])['rides'].sum().reset_index()
        
        # Create polar coordinates for time wheel
        # Hours: 0-23 mapped to 0-2π
        # Days of week: radius from center (0=Monday, 6=Sunday)
        
        fig = go.Figure()
        
        # Create time wheel visualization
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in range(7):
            day_data = time_agg[time_agg['day_of_week'] == day]
            
            if not day_data.empty:
                # Convert hour to angle (0 hour = top, clockwise)
                theta = [(hour * 15 - 90) for hour in day_data['hour']]  # 15 degrees per hour, -90 to start at top
                r = [day + 1] * len(day_data)  # Radius based on day of week
                
                # Scale ride counts for visibility
                max_rides = day_data['rides'].max() if day_data['rides'].max() > 0 else 1
                sizes = day_data['rides'] / max_rides * 30 + 5
                
                fig.add_trace(go.Scatterpolar(
                    r=r,
                    theta=theta,
                    mode='markers',
                    marker=dict(
                        size=sizes,
                        color=colors[day],
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    text=[f"{day_names[day]}<br>Hour: {hour}<br>Rides: {rides}" 
                          for hour, rides in zip(day_data['hour'], day_data['rides'])],
                    hovertemplate='%{text}<extra></extra>',
                    name=day_names[day],
                    showlegend=True
                ))
        
        # Add concentric circles for days
        for day in range(1, 8):
            fig.add_trace(go.Scatterpolar(
                r=[day] * 24,
                theta=list(range(0, 360, 15)),  # Every 15 degrees (hour)
                mode='lines',
                line=dict(color='lightgray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout for polar plot
        fig.update_layout(
            title="Time Wheel: Activity Patterns by Hour and Day of Week",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 8],
                    tickmode='array',
                    tickvals=list(range(1, 8)),
                    ticktext=day_names
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 360, 15)),
                    ticktext=[f"{h:02d}:00" for h in range(24)],
                    direction='clockwise',
                    period=360
                )
            ),
            height=700,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig

    except Exception as e:
        print(f"Error creating time wheel plot: {e}")
        return go.Figure()


def create_spider_glyph_distance(combined):
    """Create spider glyph plot with distance from Manhattan on Y-axis"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        import math

        df_temp = combined.copy()
        
        # Limit to 200 random stations for performance
        if len(df_temp['station_name'].unique()) > 200:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=200, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']

        # Calculate distance from Manhattan center
        manhattan_center_lat = 40.7580
        manhattan_center_lng = -73.9855

        def haversine_distance(lat1, lng1, lat2, lng2):
            lat1, lng1, lat2, lng2 = map(math.radians, [lat1, lng1, lat2, lng2])
            dlat = lat2 - lat1
            dlng = lng2 - lng1
            a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            return c * 6371  # Earth's radius in km

        # Aggregate by station
        station_totals = (
            df_temp.groupby('station_name')
            .agg({
                'total_rides': 'sum',
                'departures': 'sum',
                'arrivals': 'sum',
                'net_balance': 'sum',
                'lat': 'first',
                'lng': 'first'
            })
            .reset_index()
        )

        # Calculate distances
        station_totals['distance_km'] = station_totals.apply(
            lambda row: haversine_distance(row['lat'], row['lng'], manhattan_center_lat, manhattan_center_lng),
            axis=1
        )

        # Add weekday/weekend split
        weekday_data = df_temp[df_temp['day_of_week'] < 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekday_data.columns = ['station_name', 'weekday_rides']

        weekend_data = df_temp[df_temp['day_of_week'] >= 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekend_data.columns = ['station_name', 'weekend_rides']

        # Merge data
        station_totals = station_totals.merge(weekday_data, on='station_name', how='left')
        station_totals = station_totals.merge(weekend_data, on='station_name', how='left')
        station_totals['weekday_rides'] = station_totals['weekday_rides'].fillna(0)
        station_totals['weekend_rides'] = station_totals['weekend_rides'].fillna(0)

        # Limit stations for visualization
        if len(station_totals) > 200:
            station_totals = station_totals.sample(n=200, random_state=42)

        # Create spider glyph plot
        fig = go.Figure()

        # Color by net balance
        station_totals['color'] = station_totals['net_balance'].apply(
            lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
        )

        # Normalize spider arm lengths
        max_arrivals = station_totals['arrivals'].max()
        max_departures = station_totals['departures'].max()
        max_weekday = station_totals['weekday_rides'].max()
        max_weekend = station_totals['weekend_rides'].max()

        scale_factor = 0.001  # Adjust this to control spider size

        for _, station in station_totals.iterrows():
            x_center = station['total_rides']
            y_center = station['distance_km']

            # Calculate spider arm endpoints
            up_length = (station['arrivals'] / max_arrivals) * scale_factor * x_center if max_arrivals > 0 else 0
            down_length = (station[
                               'departures'] / max_departures) * scale_factor * x_center if max_departures > 0 else 0
            left_length = (station['weekday_rides'] / max_weekday) * scale_factor * x_center if max_weekday > 0 else 0
            right_length = (station['weekend_rides'] / max_weekend) * scale_factor * x_center if max_weekend > 0 else 0

            # Add spider arms
            # Up (arrivals)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center + up_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Down (departures)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center - down_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Left (weekday)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center - left_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Right (weekend)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center + right_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=4, color=station['color']),
                text=station['station_name'],
                hovertemplate=f"<b>%{{text}}</b><br>Total Rides: {station['total_rides']}<br>Distance: {station['distance_km']:.1f}km<br>Net Balance: {station['net_balance']}<extra></extra>",
                showlegend=False
            ))

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#FF6B6B'),
                                 name='More Departures', showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#69DB7C'), name='More Arrivals',
                       showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#4DABF7'), name='Balanced',
                       showlegend=True))

        fig.update_layout(
            title="Spider Glyph: Distance from Manhattan vs Total Rides",
            xaxis_title="Total Rides (All Months)",
            yaxis_title="Distance from Manhattan Center (km)",
            height=600,
            showlegend=True
        )

        return fig

    except Exception as e:
        print(f"Error creating distance spider glyph: {e}")
        return go.Figure()


def create_spider_glyph_balance_ratio(combined):
    """Create spider glyph plot with departure/arrival balance ratio on Y-axis"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        df_temp = combined.copy()
        
        # Limit to 200 random stations for performance
        if len(df_temp['station_name'].unique()) > 200:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=200, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']

        # Aggregate by station
        station_totals = (
            df_temp.groupby('station_name')
            .agg({
                'total_rides': 'sum',
                'departures': 'sum',
                'arrivals': 'sum',
                'net_balance': 'sum',
                'lat': 'first',
                'lng': 'first'
            })
            .reset_index()
        )

        # Calculate balance ratio (departures/arrivals)
        station_totals['balance_ratio'] = station_totals['departures'] / (station_totals['arrivals'] + 1)  # Add 1 to avoid division by zero
        
        # Add weekday/weekend split
        weekday_data = df_temp[df_temp['day_of_week'] < 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekday_data.columns = ['station_name', 'weekday_rides']

        weekend_data = df_temp[df_temp['day_of_week'] >= 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekend_data.columns = ['station_name', 'weekend_rides']

        # Merge data
        station_totals = station_totals.merge(weekday_data, on='station_name', how='left')
        station_totals = station_totals.merge(weekend_data, on='station_name', how='left')
        station_totals['weekday_rides'] = station_totals['weekday_rides'].fillna(0)
        station_totals['weekend_rides'] = station_totals['weekend_rides'].fillna(0)

        # Limit stations for visualization
        if len(station_totals) > 200:
            station_totals = station_totals.sample(n=200, random_state=42)

        # Create spider glyph plot
        fig = go.Figure()

        # Color by net balance
        station_totals['color'] = station_totals['net_balance'].apply(
            lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
        )

        # Normalize spider arm lengths
        max_arrivals = station_totals['arrivals'].max()
        max_departures = station_totals['departures'].max()
        max_weekday = station_totals['weekday_rides'].max()
        max_weekend = station_totals['weekend_rides'].max()

        scale_factor = 0.1  # Adjust this to control spider size

        for _, station in station_totals.iterrows():
            x_center = station['total_rides']
            y_center = station['balance_ratio']

            # Calculate spider arm endpoints
            up_length = (station['arrivals'] / max_arrivals) * scale_factor if max_arrivals > 0 else 0
            down_length = (station['departures'] / max_departures) * scale_factor if max_departures > 0 else 0
            left_length = (station['weekday_rides'] / max_weekday) * scale_factor if max_weekday > 0 else 0
            right_length = (station['weekend_rides'] / max_weekend) * scale_factor if max_weekend > 0 else 0

            # Add spider arms
            # Up (arrivals)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center + up_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Down (departures)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center - down_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Left (weekday)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center - left_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Right (weekend)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center + right_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=4, color=station['color']),
                text=station['station_name'],
                hovertemplate=f"<b>%{{text}}</b><br>Total Rides: {station['total_rides']}<br>Balance Ratio: {station['balance_ratio']:.2f}<br>Net Balance: {station['net_balance']}<extra></extra>",
                showlegend=False
            ))

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#FF6B6B'),
                                 name='More Departures', showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#69DB7C'), name='More Arrivals',
                       showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#4DABF7'), name='Balanced',
                       showlegend=True))

        fig.update_layout(
            title="Spider Glyph: Balance Ratio vs Total Rides",
            xaxis_title="Total Rides (All Months)",
            yaxis_title="Balance Ratio (Departures/Arrivals)",
            height=600,
            showlegend=True
        )

        return fig

    except Exception as e:
        print(f"Error creating balance ratio spider glyph: {e}")
        return go.Figure()


def create_spider_glyph_activity_density(combined):
    """Create spider glyph plot with activity density (rides per day) on Y-axis"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np

        df_temp = combined.copy()
        
        # Limit to 200 random stations for performance
        if len(df_temp['station_name'].unique()) > 200:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=200, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']

        # Calculate activity density (average rides per day)
        station_activity = (
            df_temp.groupby('station_name')
            .agg({
                'total_rides': ['sum', 'mean'],  # sum for total, mean for daily average
                'departures': 'sum',
                'arrivals': 'sum',
                'net_balance': 'sum',
                'lat': 'first',
                'lng': 'first',
                'date': 'nunique'  # number of unique days
            })
            .reset_index()
        )

        # Flatten column names
        station_activity.columns = ['station_name', 'total_rides', 'daily_avg_rides', 'departures', 'arrivals', 'net_balance', 'lat', 'lng', 'days_active']
        
        # Calculate activity density as rides per day
        station_activity['activity_density'] = station_activity['total_rides'] / station_activity['days_active']

        # Add weekday/weekend split
        weekday_data = df_temp[df_temp['day_of_week'] < 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekday_data.columns = ['station_name', 'weekday_rides']

        weekend_data = df_temp[df_temp['day_of_week'] >= 5].groupby('station_name')['total_rides'].sum().reset_index()
        weekend_data.columns = ['station_name', 'weekend_rides']

        # Merge data
        station_activity = station_activity.merge(weekday_data, on='station_name', how='left')
        station_activity = station_activity.merge(weekend_data, on='station_name', how='left')
        station_activity['weekday_rides'] = station_activity['weekday_rides'].fillna(0)
        station_activity['weekend_rides'] = station_activity['weekend_rides'].fillna(0)

        # Limit stations for visualization
        if len(station_activity) > 200:
            station_activity = station_activity.sample(n=200, random_state=42)

        # Create spider glyph plot
        fig = go.Figure()

        # Color by net balance
        station_activity['color'] = station_activity['net_balance'].apply(
            lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
        )

        # Normalize spider arm lengths
        max_arrivals = station_activity['arrivals'].max()
        max_departures = station_activity['departures'].max()
        max_weekday = station_activity['weekday_rides'].max()
        max_weekend = station_activity['weekend_rides'].max()

        scale_factor = 5  # Scale factor for spider arms relative to activity density

        for _, station in station_activity.iterrows():
            x_center = station['total_rides']
            y_center = station['activity_density']

            # Calculate spider arm endpoints
            up_length = (station['arrivals'] / max_arrivals) * scale_factor if max_arrivals > 0 else 0
            down_length = (station['departures'] / max_departures) * scale_factor if max_departures > 0 else 0
            left_length = (station['weekday_rides'] / max_weekday) * scale_factor if max_weekday > 0 else 0
            right_length = (station['weekend_rides'] / max_weekend) * scale_factor if max_weekend > 0 else 0

            # Add spider arms
            # Up (arrivals)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center + up_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Down (departures)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center],
                y=[y_center, y_center - down_length],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Left (weekday)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center - left_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Right (weekend)
            fig.add_trace(go.Scatter(
                x=[x_center, x_center + right_length],
                y=[y_center, y_center],
                mode='lines',
                line=dict(color=station['color'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Add center point
            fig.add_trace(go.Scatter(
                x=[x_center],
                y=[y_center],
                mode='markers',
                marker=dict(size=4, color=station['color']),
                text=station['station_name'],
                hovertemplate=f"<b>%{{text}}</b><br>Total Rides: {station['total_rides']}<br>Activity Density: {station['activity_density']:.1f} rides/day<br>Net Balance: {station['net_balance']}<extra></extra>",
                showlegend=False
            ))

        # Add legend
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#FF6B6B'),
                                 name='More Departures', showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#69DB7C'), name='More Arrivals',
                       showlegend=True))
        fig.add_trace(
            go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=10, color='#4DABF7'), name='Balanced',
                       showlegend=True))

        fig.update_layout(
            title="Spider Glyph: Activity Density vs Total Rides",
            xaxis_title="Total Rides (All Months)",
            yaxis_title="Activity Density (Rides per Day)",
            height=600,
            showlegend=True
        )

        return fig

    except Exception as e:
        print(f"Error creating activity density spider glyph: {e}")
        return go.Figure()


# ── ADVANCED MODELS ────────────────────────────────────────────────────────────

def create_arima_forecast(combined, selected_station, forecast_days=7):
    """Create ARIMA forecast for a selected station using only the past 2 weeks"""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd

        fig = go.Figure()  # ensure fig is always defined

        # Get station data
        station_data = combined[combined['station_name'] == selected_station].copy()
        station_data['date'] = pd.to_datetime(station_data['date'])
        station_data = station_data.sort_values('date')
        station_data['net_balance'] = station_data['departures'] - station_data['arrivals']

        if len(station_data) < 7:  # Need minimum data
            return fig, "Insufficient data for ARIMA forecast (need at least 7 days)"

        # Use only the most recent 2 weeks (14 days) for better short-term prediction
        recent_data = station_data.tail(14)

        if len(recent_data) < 7:
            return fig, "Insufficient recent data for ARIMA forecast"

        # NEW: Record training date range
        train_start = recent_data['date'].iloc[0].strftime('%Y-%m-%d')
        train_end = recent_data['date'].iloc[-1].strftime('%Y-%m-%d')

        # Prepare time series from recent data
        ts = recent_data.set_index('date')['net_balance']

        # Fit ARIMA model with parameters optimized for short-term bike share data
        try:
            model = ARIMA(ts, order=(1, 0, 1))
            fitted_model = model.fit()
        except:
            try:
                model = ARIMA(ts, order=(1, 1, 0))
                fitted_model = model.fit()
            except:
                model = ARIMA(ts, order=(0, 1, 1))
                fitted_model = model.fit()

        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        # Calculate confidence intervals
        forecast_ci = fitted_model.get_forecast(steps=forecast_days).conf_int()

        # Create plot
        fig = go.Figure()

        # Historical data (show all available data for context, but with lighter styling)
        fig.add_trace(go.Scatter(
            x=station_data.set_index('date').index,
            y=station_data.set_index('date')['net_balance'].values,
            mode='lines',
            name='All Historical Data',
            line=dict(color='lightgray', width=1),
            opacity=0.4
        ))

        # Recent data used for training (past 2 weeks)
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            mode='lines+markers',
            name='Recent Data (Used for Training)',
            line=dict(color='steelblue', width=2),
            marker=dict(size=6)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8)
        ))

        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_ci.iloc[:, 1],  # Upper bound
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast_ci.iloc[:, 0],  # Lower bound
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            name='Confidence Interval',
            showlegend=True
        ))

        # Calculate date range for zooming (recent data + forecast + small buffer)
        zoom_start = ts.index[0] - pd.Timedelta(days=1)
        zoom_end = forecast_index[-1] + pd.Timedelta(days=1)

        # Calculate y-axis range for better visualization
        recent_values = ts.values
        forecast_values = forecast.values
        all_focus_values = np.concatenate([recent_values, forecast_values])
        y_min = np.min(all_focus_values)
        y_max = np.max(all_focus_values)
        y_buffer = (y_max - y_min) * 0.1

        fig.update_layout(
            title=f"ARIMA Forecast - {selected_station} (Using Past {len(recent_data)} Days)",
            xaxis_title="Date",
            yaxis_title="Net Balance",
            height=500,
            xaxis=dict(
                range=[zoom_start, zoom_end],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                range=[y_min - y_buffer, y_max + y_buffer],
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            )
        )

        return fig, (
            f"ARIMA forecast successful for {forecast_days} days "
            f"(trained on {train_start} → {train_end}, {len(recent_data)} days)"
        )

    except Exception as e:
        return fig, f"ARIMA forecast failed: {str(e)}"


def create_prophet_forecast(combined, selected_station, forecast_days=7):
    """Create Prophet forecast with seasonality"""
    try:
        # Try importing Prophet with proper error handling
        import warnings
        warnings.filterwarnings('ignore')

        try:
            from prophet import Prophet
        except ImportError:
            try:
                from fbprophet import Prophet  # Alternative import name
            except ImportError:
                return go.Figure(), "Prophet library not installed. Please install with: pip install prophet"

        # Get station data
        station_data = combined[combined['station_name'] == selected_station].copy()
        station_data['date'] = pd.to_datetime(station_data['date'])
        station_data = station_data.sort_values('date')
        station_data['net_balance'] = station_data['departures'] - station_data['arrivals']

        if len(station_data) < 14:
            return go.Figure(), "Insufficient data for Prophet forecast"

        # Prepare Prophet format
        prophet_df = station_data[['date', 'net_balance']].rename(columns={'date': 'ds', 'net_balance': 'y'})

        # Fit Prophet model
        model = Prophet(daily_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Create plot
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'],
            y=prophet_df['y'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='steelblue', width=2)
        ))

        # Forecast
        forecast_future = forecast.tail(forecast_days)
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat'],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Uncertainty intervals
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_upper'],
            mode='lines',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat_lower'],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0.3)', width=0),
            name='Uncertainty',
            showlegend=True
        ))

        fig.update_layout(
            title=f"Prophet Forecast - {selected_station}",
            xaxis_title="Date",
            yaxis_title="Net Balance",
            height=400
        )

        return fig, f"Prophet forecast successful for {forecast_days} days"

    except Exception as e:
        return go.Figure(), f"Prophet forecast failed: {str(e)}"


def predict_peak_periods(combined, start_date, end_date=None, use_all_time=False):
    """Predict peak/off-peak periods for all stations with continuous color mapping"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        import plotly.express as px
        
        # Handle date selection logic
        if use_all_time:
            df_filtered = combined.copy()
            date_range_text = "All Time"
        elif end_date is None:
            # Single day analysis
            df_filtered = combined[combined['date'] == start_date].copy()
            date_range_text = start_date.strftime('%B %d, %Y')
        else:
            # Date range analysis
            df_filtered = combined[
                (pd.to_datetime(combined['date']) >= pd.to_datetime(start_date)) & 
                (pd.to_datetime(combined['date']) <= pd.to_datetime(end_date))
            ].copy()
            date_range_text = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"

        if df_filtered.empty:
            return go.Figure().add_annotation(
                text=f"No data available for {date_range_text}",
                xref="paper", yref="paper", x=0.5, y=0.5
            )

        # Calculate activity metrics
        if use_all_time or end_date is not None:
            # Aggregate by station for multi-day analysis
            station_activity = (
                df_filtered.groupby('station_name')
                .agg({
                    'departures': 'sum',
                    'arrivals': 'sum',
                    'lat': 'first',
                    'lng': 'first',
                    'date': 'nunique'  # Number of days with activity
                })
                .reset_index()
            )
            station_activity['total_activity'] = station_activity['departures'] + station_activity['arrivals']
            station_activity['daily_avg_activity'] = station_activity['total_activity'] / station_activity['date']
            activity_col = 'daily_avg_activity'
            activity_label = 'Average Daily Activity'
        else:
            # Single day analysis
            station_activity = df_filtered.copy()
            station_activity['total_activity'] = station_activity['departures'] + station_activity['arrivals']
            activity_col = 'total_activity'
            activity_label = 'Total Activity'

        if station_activity.empty:
            return go.Figure().add_annotation(
                text=f"No stations with activity for {date_range_text}",
                xref="paper", yref="paper", x=0.5, y=0.5
            )

        # Calculate peak intensity score (0-1 scale)
        min_activity = station_activity[activity_col].min()
        max_activity = station_activity[activity_col].max()
        
        if max_activity == min_activity:
            station_activity['peak_intensity'] = 0.5
        else:
            station_activity['peak_intensity'] = (
                (station_activity[activity_col] - min_activity) / (max_activity - min_activity)
            )

        # Create continuous color scale from cold blue to hot red
        station_activity['color_intensity'] = station_activity['peak_intensity']
        
        # Calculate marker sizes based on activity (normalized)
        station_activity['marker_size'] = (
            8 + (station_activity['peak_intensity'] * 12)  # Size between 8-20
        )

        # Create the map with continuous color mapping
        fig = go.Figure()

        # Add stations with continuous color scale
        fig.add_trace(go.Scattermapbox(
            lat=station_activity['lat'],
            lon=station_activity['lng'],
            mode='markers',
            marker=dict(
                size=station_activity['marker_size'],
                color=station_activity['color_intensity'],
                colorscale=[
                    [0.0, '#0066CC'],    # Cold blue for low activity
                    [0.2, '#0099FF'],    # Light blue
                    [0.4, '#66CCFF'],    # Very light blue
                    [0.5, '#FFFFFF'],    # White (neutral)
                    [0.6, '#FFCC66'],    # Light orange
                    [0.8, '#FF9900'],    # Orange
                    [1.0, '#FF3300']     # Hot red for high activity
                ],
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title="Peak Intensity",
                    titleside="right",
                    tickmode="array",
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=["Off-Peak", "Low", "Medium", "High", "Peak"],
                    len=0.7,
                    thickness=15,
                    x=1.02
                ),
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=station_activity['station_name'],
            customdata=np.column_stack((
                station_activity[activity_col].round(1),
                station_activity['peak_intensity'].round(3),
                station_activity['departures'],
                station_activity['arrivals']
            )),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{activity_label}: %{{customdata[0]}}<br>" +
                "Peak Intensity: %{customdata[1]:.1%}<br>" +
                "Departures: %{customdata[2]}<br>" +
                "Arrivals: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))

        # Add reference annotations for peak categories
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=(
                f"<b>Peak Analysis: {date_range_text}</b><br>" +
                f"🔵 Off-Peak: Low activity stations<br>" +
                f"🟡 Medium: Moderate activity stations<br>" +
                f"🔴 Peak: High activity stations<br>" +
                f"Total Stations: {len(station_activity)}"
            ),
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.1)",
            borderwidth=1,
            font=dict(size=12),
            align="left"
        )

        # Update layout
        fig.update_layout(
            title=f"Peak/Off-Peak Station Analysis - {date_range_text}",
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=station_activity['lat'].mean(),
                    lon=station_activity['lng'].mean()
                ),
                zoom=10
            ),
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )

        return fig

    except Exception as e:
        print(f"Error in peak periods analysis: {e}")
        return go.Figure().add_annotation(
            text=f"Error in analysis: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5
        )


def predict_peak_periods_standalone(combined, start_date, end_date=None, use_all_time=False):
    """Predict peak/off-peak periods for all stations with continuous color mapping"""
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        
        # Filter data based on the selected time period
        if use_all_time:
            # Use all available data
            filtered_data = combined.copy()
            period_label = "All Time"
        elif end_date is None:
            # Single day analysis
            filtered_data = combined[combined['date'] == start_date].copy()
            period_label = start_date.strftime('%B %d, %Y')
        else:
            # Date range analysis
            filtered_data = combined[
                (combined['date'] >= start_date) & 
                (combined['date'] <= end_date)
            ].copy()
            period_label = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
        
        if filtered_data.empty:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected period",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"Peak Analysis - {period_label}",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            return fig
        
        # Calculate activity metrics per station
        if use_all_time or end_date is not None:
            # For date ranges or all time: calculate average daily activity
            daily_stats = (
                filtered_data
                .groupby(['station_name', 'date'])
                .agg({
                    'departures': 'sum',
                    'arrivals': 'sum',
                    'lat': 'first',
                    'lng': 'first'
                })
                .reset_index()
            )
            
            # Calculate average daily activity per station
            station_activity = (
                daily_stats
                .groupby('station_name')
                .agg({
                    'departures': 'mean',
                    'arrivals': 'mean',
                    'lat': 'first',
                    'lng': 'first'
                })
                .reset_index()
            )
        else:
            # For single day: use total daily activity
            station_activity = (
                filtered_data
                .groupby('station_name')
                .agg({
                    'departures': 'sum',
                    'arrivals': 'sum',
                    'lat': 'first',
                    'lng': 'first'
                })
                .reset_index()
            )
        
        # Calculate total activity and peak intensity
        station_activity['total_activity'] = station_activity['departures'] + station_activity['arrivals']
        station_activity['net_balance'] = station_activity['departures'] - station_activity['arrivals']
        
        # Remove stations with no activity
        station_activity = station_activity[station_activity['total_activity'] > 0]
        
        if station_activity.empty:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No station activity found for the selected period",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title=f"Peak Analysis - {period_label}",
                height=500,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            return fig
        
        # Calculate peak intensity score (0-100)
        max_activity = station_activity['total_activity'].max()
        min_activity = station_activity['total_activity'].min()
        
        if max_activity > min_activity:
            station_activity['peak_intensity'] = (
                (station_activity['total_activity'] - min_activity) / 
                (max_activity - min_activity) * 100
            )
        else:
            station_activity['peak_intensity'] = 50  # All stations have same activity
        
        # Create continuous color scale from cold blue to hot red
        def intensity_to_color(intensity):
            """Convert intensity (0-100) to color from cold blue to hot red"""
            # Normalize to 0-1
            normalized = intensity / 100.0
            
            # Cold blue (#0066CC) to Hot red (#FF3300)
            if normalized <= 0.5:
                # Blue to Yellow transition
                ratio = normalized * 2
                r = int(0 + (255 - 0) * ratio)
                g = int(102 + (255 - 102) * ratio)
                b = int(204 + (0 - 204) * ratio)
            else:
                # Yellow to Red transition
                ratio = (normalized - 0.5) * 2
                r = int(255)
                g = int(255 + (51 - 255) * ratio)
                b = int(0)
            
            return f"rgb({r},{g},{b})"
        
        # Apply color mapping
        station_activity['color'] = station_activity['peak_intensity'].apply(intensity_to_color)
        
        # Create marker sizes proportional to activity (minimum size 8, maximum size 20)
        max_size = 20
        min_size = 8
        station_activity['marker_size'] = (
            min_size + (station_activity['peak_intensity'] / 100) * (max_size - min_size)
        )
        
        # Create the map
        fig = go.Figure()
        
        # Add station markers with continuous color mapping
        fig.add_trace(go.Scattermapbox(
            lat=station_activity['lat'],
            lon=station_activity['lng'],
            mode='markers',
            marker=dict(
                size=station_activity['marker_size'],
                color=station_activity['color'],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=station_activity['station_name'],
            customdata=np.column_stack([
                station_activity['total_activity'].round(1),
                station_activity['peak_intensity'].round(1),
                station_activity['departures'].round(1),
                station_activity['arrivals'].round(1)
            ]),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                "Activity Level: %{customdata[0]}<br>" +
                "Peak Intensity: %{customdata[1]:.1f}%<br>" +
                "Departures: %{customdata[2]}<br>" +
                "Arrivals: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            name="Station Activity",
            showlegend=False
        ))
        
        # Add color scale legend manually by creating invisible traces
        intensity_levels = [0, 25, 50, 75, 100]
        intensity_labels = ['Cold (Low)', 'Cool', 'Moderate', 'Warm', 'Hot (High)']
        
        for intensity, label in zip(intensity_levels, intensity_labels):
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(
                    size=12,
                    color=intensity_to_color(intensity),
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name=f"{label} ({intensity}%)",
                showlegend=True
            ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=station_activity['lat'].mean(),
                    lon=station_activity['lng'].mean()
                ),
                zoom=11
            ),
            title=f"Peak Activity Analysis - {period_label}",
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1,
                title="Peak Intensity"
            )
        )
        
        return fig
        
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error in peak analysis: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            title="Peak Analysis - Error",
            height=500,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig


def detect_station_anomalies(combined, selected_date, z_threshold=2.5):
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure(), "No data for anomaly detection"

    # Calculate monthly aggregated metrics per station
    station_monthly = (
        month_data
        .groupby('station_name')
        .agg({
            'departures': 'sum',
            'arrivals': 'sum',
            'lat': 'first',
            'lng': 'first'
        })
        .reset_index()
    )

    station_monthly['net_balance'] = station_monthly['departures'] - station_monthly['arrivals']
    station_monthly['total_activity'] = station_monthly['departures'] + station_monthly['arrivals']

    # Calculate Z-scores for the entire month
    station_monthly['net_balance_zscore'] = np.abs(
        (station_monthly['net_balance'] - station_monthly['net_balance'].mean()) / station_monthly['net_balance'].std())
    station_monthly['activity_zscore'] = np.abs(
        (station_monthly['total_activity'] - station_monthly['total_activity'].mean()) / station_monthly[
            'total_activity'].std())

    # Identify anomalies
    station_monthly['is_anomaly'] = (station_monthly['net_balance_zscore'] > z_threshold) | (
            station_monthly['activity_zscore'] > z_threshold)

    # Create map
    fig = go.Figure()

    # Normal stations
    normal_stations = station_monthly[~station_monthly['is_anomaly']]
    if not normal_stations.empty:
        fig.add_trace(go.Scattermapbox(
            lat=normal_stations['lat'],
            lon=normal_stations['lng'],
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.6),
            text=normal_stations['station_name'],
            hovertemplate="<b>%{text}</b><br>Normal Behavior<br>Net Balance: %{customdata[0]}<br>Total Activity: %{customdata[1]}<extra></extra>",
            customdata=normal_stations[['net_balance', 'total_activity']],
            name="Normal"
        ))

    # Anomalous stations
    anomaly_stations = station_monthly[station_monthly['is_anomaly']]
    if not anomaly_stations.empty:
        fig.add_trace(go.Scattermapbox(
            lat=anomaly_stations['lat'],
            lon=anomaly_stations['lng'],
            mode='markers',
            marker=dict(size=12, color='red', opacity=0.9),
            text=anomaly_stations['station_name'],
            hovertemplate="<b>%{text}</b><br>⚠️ Anomaly Detected<br>Net Balance: %{customdata[0]}<br>Total Activity: %{customdata[1]}<br>Net Z-Score: %{customdata[2]:.2f}<br>Activity Z-Score: %{customdata[3]:.2f}<extra></extra>",
            customdata=anomaly_stations[['net_balance', 'total_activity', 'net_balance_zscore', 'activity_zscore']],
            name="Anomalies"
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    anomaly_count = len(anomaly_stations)
    total_count = len(station_monthly)

    return fig, f"Found {anomaly_count} anomalies out of {total_count} stations ({anomaly_count / total_count * 100:.1f}%) for {selected_month.strftime('%B %Y')}"


def create_daily_rides_continuous_plot(combined, start_date, end_date, max_stations=None):
    """Create a continuous line plot showing rides with simulated hourly patterns over time"""
    import random
    import math
    
    # Filter data for the date range
    filtered_data = combined[
        (combined['date'] >= start_date) &
        (combined['date'] <= end_date)
        ].copy()
    
    # Limit stations for performance using utility function
    filtered_data = limit_stations_for_performance(filtered_data, max_stations)

    # Calculate total rides per day (departures + arrivals)
    daily_rides = (
        filtered_data
        .groupby('date')
        .agg({
            'departures': 'sum',
            'arrivals': 'sum'
        })
        .reset_index()
    )

    daily_rides['total_rides'] = daily_rides['departures'] + daily_rides['arrivals']
    daily_rides['net_difference'] = daily_rides['departures'] - daily_rides['arrivals']

    if daily_rides.empty:
        return go.Figure()

    # Generate hourly data with realistic patterns
    hourly_data = []
    
    for _, day_row in daily_rides.iterrows():
        base_date = pd.to_datetime(day_row['date'])
        daily_total = day_row['total_rides']
        daily_net = day_row['net_difference']
        
        # Create hourly distribution pattern (typical bike share pattern)
        hourly_pattern = [
            0.02, 0.01, 0.01, 0.01, 0.02, 0.04,  # 0-5: Night/early morning
            0.08, 0.12, 0.10, 0.08, 0.06, 0.05,  # 6-11: Morning rush
            0.05, 0.05, 0.04, 0.04, 0.05, 0.08,  # 12-17: Afternoon
            0.10, 0.08, 0.06, 0.04, 0.03, 0.02   # 18-23: Evening rush
        ]
        
        # Add some randomness to the pattern
        for hour in range(24):
            timestamp = base_date + pd.Timedelta(hours=hour)
            
            # Calculate hourly rides with some random variation
            base_hourly_rides = daily_total * hourly_pattern[hour]
            variation = random.uniform(0.8, 1.2)  # ±20% variation
            hourly_rides = max(0, int(base_hourly_rides * variation))
            
            # Calculate hourly net difference with pattern
            # Morning: more departures, Evening: more arrivals, with daily net as baseline
            hour_factor = 1.0
            if 6 <= hour <= 9:  # Morning rush - more departures
                hour_factor = 1.5
            elif 17 <= hour <= 20:  # Evening rush - more arrivals
                hour_factor = -1.2
            
            hourly_net = (daily_net / 24) * hour_factor * random.uniform(0.7, 1.3)
            
            hourly_data.append({
                'timestamp': timestamp,
                'hourly_rides': hourly_rides,
                'hourly_net_difference': hourly_net,
                'hour': hour,
                'date': day_row['date']
            })
    
    hourly_df = pd.DataFrame(hourly_data)
    
    if hourly_df.empty:
        return go.Figure()

    # Create the plot
    fig = go.Figure()

    # Add total rides line
    fig.add_trace(go.Scatter(
        x=hourly_df['timestamp'],
        y=hourly_df['hourly_rides'],
        mode='lines',
        name='Total Rides (Hourly)',
        line=dict(color='steelblue', width=2),
        hovertemplate='<b>%{x}</b><br>Hourly Rides: %{y:,}<extra></extra>'
    ))

    # Add markers for daily peaks
    daily_peaks = hourly_df.loc[hourly_df.groupby('date')['hourly_rides'].idxmax()]
    fig.add_trace(go.Scatter(
        x=daily_peaks['timestamp'],
        y=daily_peaks['hourly_rides'],
        mode='markers',
        name='Daily Peaks',
        marker=dict(size=8, color='orange', symbol='star'),
        hovertemplate='<b>Daily Peak</b><br>%{x}<br>Rides: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Continuous Hourly Rides Pattern ({start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')})",
        xaxis_title="Date and Time",
        yaxis_title="Hourly Rides",
        height=500,
        margin=dict(l=20, r=60, t=40, b=20),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickformat='%d/%m %H:%M'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),
        hovermode='x unified'
    )

    return fig


def create_daily_rides_bar_chart(combined, start_date, end_date, max_stations=None):
    """Create a bar chart showing total rides per day for the selected date range"""
    # Filter data for the date range
    filtered_data = combined[
        (combined['date'] >= start_date) &
        (combined['date'] <= end_date)
        ].copy()
    
    # Limit stations for performance using utility function
    filtered_data = limit_stations_for_performance(filtered_data, max_stations)

    # Calculate total rides per day (departures + arrivals)
    daily_rides = (
        filtered_data
        .groupby('date')
        .agg({
            'departures': 'sum',
            'arrivals': 'sum'
        })
        .reset_index()
    )

    daily_rides['total_rides'] = daily_rides['departures'] + daily_rides['arrivals']
    daily_rides['date_str'] = daily_rides['date'].astype(str)

    # Create the bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=daily_rides['date_str'],
        y=daily_rides['total_rides'],
        marker_color='steelblue',
        name='Total Rides',
        hovertemplate='<b>Date: %{x}</b><br>Total Rides: %{y:,}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Daily Total Rides ({start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')})",
        xaxis_title="Date",
        yaxis_title="Total Rides",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            tickangle=45,
            showgrid=True,
            gridwidth=1,
            gridcolor='#444444'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#444444'
        ),
        plot_bgcolor='#2E2E2E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white'),
        title_font=dict(color='white')
    )

    return fig


# ── IMPROVED SPIDER GLYPH VISUALIZATIONS ──────────────────────────────────────

def create_temporal_pattern_spider(combined, max_stations=100):
    """
    Create spider glyph focused on temporal usage patterns
    
    Better than current spider glyphs because it focuses on:
    - Rush Hour Intensity (meaningful for bike share)
    - Midday Activity Level 
    - Weekend vs Weekday Ratio
    - Night Activity patterns
    - Seasonal Variation
    """
    
    df_temp = combined.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
    df_temp['month'] = df_temp['date'].dt.month
    df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
    
    # Limit stations for performance
    if len(df_temp['station_name'].unique()) > max_stations:
        selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=max_stations, random_state=42).tolist()
        df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
    
    station_patterns = []
    
    for station_name in df_temp['station_name'].unique():
        station_data = df_temp[df_temp['station_name'] == station_name]
        
        if len(station_data) < 5:  # Need minimum data
            continue
            
        # Calculate temporal pattern metrics
        weekday_data = station_data[station_data['day_of_week'] < 5]
        weekend_data = station_data[station_data['day_of_week'] >= 5]
        
        # 1. Rush Hour Intensity (simulate based on departures vs arrivals imbalance)
        rush_hour_intensity = abs(station_data['departures'] - station_data['arrivals']).mean()
        
        # 2. Midday Activity (total activity level)
        midday_activity = station_data['total_rides'].mean()
        
        # 3. Weekend vs Weekday Ratio
        weekday_avg = weekday_data['total_rides'].mean() if len(weekday_data) > 0 else 0
        weekend_avg = weekend_data['total_rides'].mean() if len(weekend_data) > 0 else 0
        weekend_weekday_ratio = (weekend_avg / (weekday_avg + 1)) if weekday_avg > 0 else 0
        
        # 4. Night Activity (simulate as low activity vs high activity stations)
        night_activity = station_data['total_rides'].min()  # Minimum as proxy for night activity
        
        # 5. Seasonal Variation (if multiple months available)
        monthly_variation = station_data.groupby('month')['total_rides'].sum().std() if len(station_data['month'].unique()) > 1 else 0
        
        patterns = {
            'station_name': station_name,
            'rush_hour_intensity': rush_hour_intensity,
            'midday_activity': midday_activity,
            'weekend_weekday_ratio': weekend_weekday_ratio,
            'night_activity': night_activity,
            'seasonal_variation': monthly_variation,
            'total_rides': station_data['total_rides'].sum()
        }
        station_patterns.append(patterns)
    
    if not station_patterns:
        return go.Figure()
    
    patterns_df = pd.DataFrame(station_patterns)
    
    # Normalize metrics to 0-1 scale
    metric_cols = ['rush_hour_intensity', 'midday_activity', 'weekend_weekday_ratio', 
                   'night_activity', 'seasonal_variation']
    
    for col in metric_cols:
        min_val, max_val = patterns_df[col].min(), patterns_df[col].max()
        if max_val > min_val:
            patterns_df[f'{col}_norm'] = (patterns_df[col] - min_val) / (max_val - min_val)
        else:
            patterns_df[f'{col}_norm'] = 0.5
    
    # Create regular spider chart (not geographic overlay)
    fig = go.Figure()
    
    # Group stations by activity level for better visualization
    patterns_df['activity_level'] = pd.qcut(patterns_df['total_rides'], 
                                           q=3, labels=['Low', 'Medium', 'High'])
    
    colors = {'Low': '#3498db', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    for activity_level in ['Low', 'Medium', 'High']:
        level_data = patterns_df[patterns_df['activity_level'] == activity_level]
        
        if len(level_data) == 0:
            continue
            
        # Calculate average pattern for this activity level
        avg_pattern = [
            level_data['rush_hour_intensity_norm'].mean(),
            level_data['midday_activity_norm'].mean(),
            level_data['weekend_weekday_ratio_norm'].mean(),
            level_data['night_activity_norm'].mean(),
            level_data['seasonal_variation_norm'].mean()
        ]
        
        # Close the spider web
        avg_pattern.append(avg_pattern[0])
        
        theta_labels = ['Rush Hour<br>Intensity', 'Midday<br>Activity', 'Weekend vs<br>Weekday', 
                       'Night<br>Activity', 'Seasonal<br>Variation', 'Rush Hour<br>Intensity']
        
        fig.add_trace(go.Scatterpolar(
            r=avg_pattern,
            theta=theta_labels,
            fill='toself',
            name=f'{activity_level} Activity Stations',
            line_color=colors[activity_level]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Low', 'Low-Med', 'Medium', 'Med-High', 'High']
            )
        ),
        title="Temporal Pattern Analysis - Station Usage Profiles",
        height=600,
        showlegend=True
    )
    
    return fig


def create_station_role_spider(combined, max_stations=50):
    """
    Create spider glyph to identify station roles in the bike share system
    
    Much more useful than current spider glyphs because it shows:
    - Net Departure Score (departure hub vs arrival hub)
    - Peak Hour Dominance (rush hour vs all-day usage)
    - Consistency Score (reliable vs volatile)
    - Volume Level (high vs low traffic)
    - Balance Score (balanced vs imbalanced)
    """
    
    df_temp = combined.copy()
    df_temp['date'] = pd.to_datetime(df_temp['date'])
    df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
    df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']
    
    # Limit stations for cleaner visualization
    if len(df_temp['station_name'].unique()) > max_stations:
        # Select top stations by total activity
        top_stations = (df_temp.groupby('station_name')['total_rides'].sum()
                       .nlargest(max_stations).index.tolist())
        df_temp = df_temp[df_temp['station_name'].isin(top_stations)]
    
    station_roles = []
    
    for station_name in df_temp['station_name'].unique():
        station_data = df_temp[df_temp['station_name'] == station_name]
        
        if len(station_data) < 3:
            continue
        
        # Calculate role metrics
        total_volume = station_data['total_rides'].sum()
        avg_net_balance = station_data['net_balance'].mean()
        balance_volatility = station_data['net_balance'].std()
        
        # 1. Net Departure Score (0 = arrival hub, 1 = departure hub)
        net_departure_score = (avg_net_balance - station_data['net_balance'].min()) / (
            station_data['net_balance'].max() - station_data['net_balance'].min() + 1)
        
        # 2. Peak Hour Dominance (simulated by variance in daily patterns)
        peak_dominance = balance_volatility / (total_volume / len(station_data) + 1)
        
        # 3. Consistency Score (inverse of volatility)
        consistency = 1 / (balance_volatility + 1)
        
        # 4. Volume Level 
        volume_level = total_volume
        
        # 5. Balance Score (how balanced arrivals and departures are)
        balance_score = 1 / (abs(avg_net_balance) + 1)
        
        roles = {
            'station_name': station_name,
            'net_departure_score': max(0, min(1, net_departure_score)),
            'peak_dominance': peak_dominance,
            'consistency': consistency,
            'volume_level': volume_level,
            'balance_score': balance_score,
            'total_rides': total_volume,
            'avg_net_balance': avg_net_balance
        }
        station_roles.append(roles)
    
    if not station_roles:
        return go.Figure()
    
    roles_df = pd.DataFrame(station_roles)
    
    # Normalize metrics except net_departure_score (already 0-1)
    metric_cols = ['peak_dominance', 'consistency', 'volume_level', 'balance_score']
    
    for col in metric_cols:
        min_val, max_val = roles_df[col].min(), roles_df[col].max()
        if max_val > min_val:
            roles_df[f'{col}_norm'] = (roles_df[col] - min_val) / (max_val - min_val)
        else:
            roles_df[f'{col}_norm'] = 0.5
    
    # Create individual spider plots for top stations by volume
    fig = go.Figure()
    
    # Select top stations by volume for individual spider plots (use all if less than max_stations)
    top_stations = roles_df.nlargest(min(len(roles_df), max_stations), 'total_rides')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (_, station) in enumerate(top_stations.iterrows()):
        
        pattern = [
            station['net_departure_score'],
            station['peak_dominance_norm'],
            station['consistency_norm'],
            station['volume_level_norm'],
            station['balance_score_norm']
        ]
        
        # Close the spider web
        pattern.append(pattern[0])
        
        theta_labels = ['Departure<br>Hub Score', 'Peak Hour<br>Dominance', 'Consistency<br>Score', 
                       'Volume<br>Level', 'Balance<br>Score', 'Departure<br>Hub Score']
        
        fig.add_trace(go.Scatterpolar(
            r=pattern,
            theta=theta_labels,
            name=f"{station['station_name'][:20]}{'...' if len(station['station_name']) > 20 else ''}",
            line_color=colors[i % len(colors)],
            line_width=2
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Low', 'Low-Med', 'Medium', 'Med-High', 'High']
            )
        ),
        title=f"Station Role Analysis - Top {len(top_stations)} Stations by Volume",
        height=700,
        showlegend=True
    )
    
    return fig


def create_cluster_comparison_spider(ts_res, pivot_daily):
    """
    Create spider glyph comparing cluster characteristics
    
    This works with your existing clustering results to show
    meaningful differences between clusters.
    """
    
    if ts_res.empty or pivot_daily.empty:
        return go.Figure()
    
    # Calculate cluster metrics
    cluster_metrics = []
    
    for cluster_id in ts_res['cluster'].unique():
        cluster_stations = ts_res[ts_res['cluster'] == cluster_id]['station_name'].tolist()
        cluster_data = pivot_daily.loc[cluster_stations]
        
        if cluster_data.empty:
            continue
        
        # Calculate meaningful cluster characteristics
        avg_pattern = cluster_data.mean(axis=0)
        
        # 1. Morning Peak Strength (first few days as proxy)
        morning_peak = avg_pattern.head(5).mean()
        
        # 2. Afternoon Pattern (middle days)
        afternoon_pattern = avg_pattern.iloc[5:15].mean() if len(avg_pattern) > 15 else avg_pattern.mean()
        
        # 3. Evening Strength (last few days)
        evening_strength = avg_pattern.tail(5).mean()
        
        # 4. Volatility (standard deviation across days)
        volatility = avg_pattern.std()
        
        # 5. Overall Trend (linear regression slope)
        x_vals = np.arange(len(avg_pattern))
        if len(x_vals) > 1:
            slope = np.polyfit(x_vals, avg_pattern.values, 1)[0]
            trend_strength = abs(slope)
        else:
            trend_strength = 0
        
        metrics = {
            'cluster': cluster_id,
            'stations_count': len(cluster_stations),
            'morning_peak': morning_peak,
            'afternoon_pattern': afternoon_pattern,
            'evening_strength': evening_strength,
            'volatility': volatility,
            'trend_strength': trend_strength
        }
        cluster_metrics.append(metrics)
    
    if not cluster_metrics:
        return go.Figure()
    
    cluster_df = pd.DataFrame(cluster_metrics)
    
    # Normalize metrics
    metric_cols = ['morning_peak', 'afternoon_pattern', 'evening_strength', 'volatility', 'trend_strength']
    
    for col in metric_cols:
        min_val, max_val = cluster_df[col].min(), cluster_df[col].max()
        if max_val > min_val:
            cluster_df[f'{col}_norm'] = (cluster_df[col] - min_val) / (max_val - min_val)
        else:
            cluster_df[f'{col}_norm'] = 0.5
    
    # Create spider plot
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (_, cluster) in cluster_df.iterrows():
        
        pattern = [
            cluster['morning_peak_norm'],
            cluster['afternoon_pattern_norm'],
            cluster['evening_strength_norm'],
            cluster['volatility_norm'],
            cluster['trend_strength_norm']
        ]
        
        # Close the spider web
        pattern.append(pattern[0])
        
        theta_labels = ['Morning<br>Peak', 'Afternoon<br>Activity', 'Evening<br>Strength', 
                       'Volatility', 'Trend<br>Strength', 'Morning<br>Peak']
        
        fig.add_trace(go.Scatterpolar(
            r=pattern,
            theta=theta_labels,
            fill='toself',
            name=f"Cluster {cluster['cluster']} ({cluster['stations_count']} stations)",
            line_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Low', 'Low-Med', 'Medium', 'Med-High', 'High']
            )
        ),
        title="Cluster Comparison - Temporal Pattern Characteristics",
        height=600,
        showlegend=True
    )
    
    return fig


def create_arrivals_departures_spider_plot(combined, selected_date):
    """Create spider plot showing arrivals and departures for the selected date"""
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure()

    # Calculate monthly totals per station
    station_monthly = (
        month_data
        .groupby('station_name')
        .agg({
            'departures': 'sum',
            'arrivals': 'sum',
            'lat': 'first',
            'lng': 'first'
        })
        .reset_index()
    )

    # Limit to reasonable number of stations for visualization
    if len(station_monthly) > 100:
        station_monthly = station_monthly.sample(n=100, random_state=42)

    # Normalize arrivals and departures to 0-1 scale for spider plot
    for col in ['arrivals', 'departures']:
        min_val, max_val = station_monthly[col].min(), station_monthly[col].max()
        if max_val > min_val:
            station_monthly[f'{col}_norm'] = (station_monthly[col] - min_val) / (max_val - min_val)
        else:
            station_monthly[f'{col}_norm'] = 0.5

    # Create spider plot
    fig_spider = go.Figure()

    # Define spider plot parameters (2 metrics: arrivals and departures)
    n_metrics = 2
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)

    # Color stations by their net balance (red = more departures, green = more arrivals, blue = balanced)
    station_monthly['net_balance'] = station_monthly['departures'] - station_monthly['arrivals']
    station_monthly['color'] = station_monthly['net_balance'].apply(
        lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
    )

    # Plot each station as a spider glyph
    for _, station in station_monthly.iterrows():
        # Get normalized values for spider plot [arrivals, departures]
        values = [station['arrivals_norm'], station['departures_norm']]

        # Add spider lines
        for i, (angle, value) in enumerate(zip(angles, values)):
            scaled_value = 0.03 + value * 0.12  # Scale for visibility
            x_end = station['lng'] + scaled_value * np.cos(angle) * 0.01
            y_end = station['lat'] + scaled_value * np.sin(angle) * 0.01

            # Different line styles for arrivals vs departures
            line_style = dict(color=station['color'], width=2) if i == 0 else dict(color=station['color'], width=2,
                                                                                   dash='dash')

            fig_spider.add_trace(go.Scattermapbox(
                lat=[station['lat'], y_end],
                lon=[station['lng'], x_end],
                mode='lines',
                line=line_style,
                hoverinfo='skip',
                showlegend=False
            ))

        # Add center point
        fig_spider.add_trace(go.Scattermapbox(
            lat=[station['lat']],
            lon=[station['lng']],
            mode='markers',
            marker=dict(size=4, color=station['color']),
            text=station['station_name'],
            hovertemplate=f"<b>%{{text}}</b><br>Arrivals: {station['arrivals']}<br>Departures: {station['departures']}<br>Net Balance: {station['net_balance']}<extra></extra>",
            showlegend=False
        ))

    # Add legend entries
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#FF6B6B'),
        name='More Departures',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#69DB7C'),
        name='More Arrivals',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#4DABF7'),
        name='Balanced',
        showlegend=True
    ))

    # Add reference lines legend
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='lines',
        line=dict(color='gray', width=2),
        name='Arrivals',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        name='Departures',
        showlegend=True
    ))

    fig_spider.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(
                lat=station_monthly['lat'].mean(),
                lon=station_monthly['lng'].mean()
            ),
            zoom=10
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=14)
        )
    )

    return fig_spider
