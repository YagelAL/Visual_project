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
    
    # Get unique stations only (since full_df might have daily records)
    unique_stations = full_df.drop_duplicates('station_name')[['station_name', 'lat', 'lng']].copy()
    
    # Ensure we don't exceed the specified maximum for performance
    if len(unique_stations) > max_stations:
        unique_stations = unique_stations.sample(n=max_stations, random_state=42)
        st.info(f"📊 Limited to {max_stations} stations for optimal performance")

    g = gpd.GeoDataFrame(
        unique_stations[['station_name', 'lat', 'lng']],
        geometry=gpd.points_from_xy(unique_stations.lng, unique_stations.lat),
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
    if combined is None or combined.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    
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
            agg['net_balance'] = agg['arrivals'] - agg['departures']  # Fixed: arrivals - departures

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
    
    if pivot_net_filtered is None or pivot_net_filtered.empty:
        return pd.DataFrame(), None
    
    if station_coords is None or station_coords.empty:
        return pd.DataFrame(), None
    
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
    
    if clustered_df is None or clustered_df.empty:
        return go.Figure().add_annotation(
            text="No clustering data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
    
    # Clean the data
    clustered_df = clustered_df.dropna(subset=['lat', 'lng'])
    
    if clustered_df.empty:
        return go.Figure().add_annotation(
            text="No stations with valid coordinates found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )

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

    # Always center on NYC (Manhattan) for consistent view
    center_lat = 40.7128  # NYC latitude
    center_lon = -74.0060  # NYC longitude

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",  # Greyish map style
            center=dict(lat=center_lat, lon=center_lon),
            zoom=10
        ),
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
    # Ensure the date column is datetime
    combined_temp = combined.copy()
    combined_temp['date'] = pd.to_datetime(combined_temp['date'])
    
    dates = pd.date_range(start_date, end_date).date
    frames = []
    for d in dates:
        df_day = combined_temp[combined_temp["date"].dt.date == d].copy()
        
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


def create_tominski_time_wheel(combined):
    """Create a Tominski TimeWheel: Six variable axes arranged circularly around an exposed centered time axis"""
    try:
        import plotly.graph_objects as go
        import numpy as np
        import pandas as pd
        
        df_temp = combined.copy()
        
        # Limit data for performance
        if len(df_temp['station_name'].unique()) > 80:
            selected_stations = pd.Series(df_temp['station_name'].unique()).sample(n=80, random_state=42).tolist()
            df_temp = df_temp[df_temp['station_name'].isin(selected_stations)]
        
        # Ensure date is datetime and extract time dimensions
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['month'] = df_temp['date'].dt.month
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek
        df_temp['day_of_month'] = df_temp['date'].dt.day
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']
        
        # Create simulated hour data for demonstration
        np.random.seed(42)
        df_temp['hour'] = np.random.randint(0, 24, len(df_temp))
        
        # Sample data for cleaner visualization
        if len(df_temp) > 200:
            df_temp = df_temp.sample(n=200, random_state=42)
        
        if df_temp.empty:
            return go.Figure().add_annotation(
                text="No data available for time wheel",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
        
        fig = go.Figure()
        
        # Define the SIX VARIABLE AXES arranged circularly around the time axis
        variable_axes = [
            'total_rides',      # Activity Level
            'departures',       # Outbound Flow
            'arrivals',         # Inbound Flow
            'net_balance',      # Balance
            'month',            # Seasonality
            'day_of_week'       # Weekly Pattern
        ]
        
        n_vars = len(variable_axes)
        
        # Calculate axis angles (evenly spaced around the circle)
        axis_angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
        
        # Axis radius (distance from center)
        axis_radius = 3.5
        
        # Calculate axis positions
        axis_positions = {}
        for i, var in enumerate(variable_axes):
            angle = axis_angles[i]
            axis_positions[var] = {
                'angle': angle,
                'x': axis_radius * np.cos(angle),
                'y': axis_radius * np.sin(angle)
            }
        
        # Axis labels
        axis_labels = {
            'total_rides': 'Activity Level',
            'departures': 'Outbound Flow',
            'arrivals': 'Inbound Flow',
            'net_balance': 'Balance',
            'month': 'Seasonality',
            'day_of_week': 'Weekly Pattern'
        }
        
        # Draw the six variable axes
        for var in variable_axes:
            pos = axis_positions[var]
            
            # Draw axis line from center to edge
            fig.add_trace(go.Scatter(
                x=[0, pos['x']],
                y=[0, pos['y']],
                mode='lines',
                line=dict(color='#2C3E50', width=3),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Add axis labels
            label_x = pos['x'] * 1.15
            label_y = pos['y'] * 1.15
            fig.add_trace(go.Scatter(
                x=[label_x],
                y=[label_y],
                mode='text',
                text=[axis_labels[var]],
                textfont=dict(size=11, color='#2C3E50', family='Arial Black'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Normalize data for each variable to [0, 1] range
        normalized_data = df_temp.copy()
        for var in variable_axes:
            if var in normalized_data.columns:
                min_val = normalized_data[var].min()
                max_val = normalized_data[var].max()
                if max_val > min_val:
                    normalized_data[f'{var}_norm'] = (normalized_data[var] - min_val) / (max_val - min_val)
                else:
                    normalized_data[f'{var}_norm'] = 0.5
        
        # Create CENTRAL TIME AXIS (exposed in center)
        # Extract unique time values for the central axis
        time_values = sorted(df_temp['hour'].unique())
        central_time_radius = 0.8
        
        # Draw central time axis as a circle with hour markings
        time_circle = np.linspace(0, 2*np.pi, 100)
        circle_x = central_time_radius * np.cos(time_circle)
        circle_y = central_time_radius * np.sin(time_circle)
        
        fig.add_trace(go.Scatter(
            x=circle_x,
            y=circle_y,
            mode='lines',
            line=dict(color='#E74C3C', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add hour markers on the central time axis
        for hour in range(0, 24, 4):  # Every 4 hours
            hour_angle = (hour / 24) * 2 * np.pi
            hour_x = central_time_radius * np.cos(hour_angle)
            hour_y = central_time_radius * np.sin(hour_angle)
            
            # Hour marker
            fig.add_trace(go.Scatter(
                x=[hour_x],
                y=[hour_y],
                mode='markers',
                marker=dict(size=8, color='#E74C3C'),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Hour label
            label_x = hour_x * 1.3
            label_y = hour_y * 1.3
            fig.add_trace(go.Scatter(
                x=[label_x],
                y=[label_y],
                mode='text',
                text=[f'{hour:02d}h'],
                textfont=dict(size=8, color='#E74C3C'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add central time axis label
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='text',
            text=['TIME<br>AXIS'],
            textfont=dict(size=12, color='#E74C3C', family='Arial Black'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Create data visualization connecting time axis to variable axes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA726', '#FF9999', '#66B2FF', '#99FF99', '#E74C3C']
        
        # Group data by hour for better visualization
        for hour in sorted(df_temp['hour'].unique()):
            hour_data = normalized_data[normalized_data['hour'] == hour]
            
            if len(hour_data) < 2:
                continue
                
            # Calculate mean values for this hour
            hour_means = {}
            for var in variable_axes:
                if f'{var}_norm' in hour_data.columns:
                    hour_means[var] = hour_data[f'{var}_norm'].mean()
                else:
                    hour_means[var] = 0.5
            
            # Position on central time axis
            time_angle = (hour / 24) * 2 * np.pi
            time_x = central_time_radius * np.cos(time_angle)
            time_y = central_time_radius * np.sin(time_angle)
            
            # Draw connections from time axis to variable axes
            for var in variable_axes:
                pos = axis_positions[var]
                norm_val = hour_means[var]
                
                # Position along the variable axis (from center outward)
                distance = 1.0 + norm_val * 2.0  # Range from 1.0 to 3.0
                
                var_x = distance * np.cos(pos['angle'])
                var_y = distance * np.sin(pos['angle'])
                
                # Create smooth curve from time axis to variable axis
                # Control point for bezier curve
                control_x = (time_x + var_x) / 2 * 0.5
                control_y = (time_y + var_y) / 2 * 0.5
                
                # Bezier curve
                t = np.linspace(0, 1, 50)
                curve_x = (1-t)**2 * time_x + 2*(1-t)*t * control_x + t**2 * var_x
                curve_y = (1-t)**2 * time_y + 2*(1-t)*t * control_y + t**2 * var_y
                
                # Color based on hour
                color_idx = hour % len(colors)
                line_color = colors[color_idx]
                
                # Add the connection curve
                fig.add_trace(go.Scatter(
                    x=curve_x,
                    y=curve_y,
                    mode='lines',
                    line=dict(
                        color=line_color,
                        width=1.5,
                        smoothing=1.3
                    ),
                    opacity=0.4,
                    showlegend=False,
                    hovertemplate=f'Hour: {hour:02d}<br>' +
                                 f'Variable: {axis_labels[var]}<br>' +
                                 f'Value: {norm_val:.2f}<extra></extra>'
                ))
        
        # Add axis scale markers on variable axes
        for var in variable_axes:
            pos = axis_positions[var]
            
            # Add tick marks along each axis
            for tick in [0.25, 0.5, 0.75, 1.0]:
                tick_distance = 1.0 + tick * 2.0
                tick_x = tick_distance * np.cos(pos['angle'])
                tick_y = tick_distance * np.sin(pos['angle'])
                
                fig.add_trace(go.Scatter(
                    x=[tick_x],
                    y=[tick_y],
                    mode='markers',
                    marker=dict(size=3, color='#34495E'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add outer boundary circle
        outer_circle = np.linspace(0, 2*np.pi, 100)
        outer_x = 4.0 * np.cos(outer_circle)
        outer_y = 4.0 * np.sin(outer_circle)
        
        fig.add_trace(go.Scatter(
            x=outer_x,
            y=outer_y,
            mode='lines',
            line=dict(color='#BDC3C7', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="Tominski TimeWheel: Six Variable Axes around Central Time Axis",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1,
                range=[-5, 5]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-5, 5]
            ),
            height=700,
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating Tominski time wheel: {e}")
        return go.Figure().add_annotation(
            text=f"Error creating time wheel: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="red")
        )


def create_parallel_coordinates_plot(combined):
    """Create a parallel coordinates plot showing multi-dimensional station characteristics"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        
        # Check for empty data
        if combined.empty:
            return go.Figure().add_annotation(
                text="No data available for parallel coordinates analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14)
            )
        
        # Prepare data with basic safety checks
        df = combined.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['total_rides'] = df['departures'] + df['arrivals']
        df['net_balance'] = df['arrivals'] - df['departures']
        
        # Calculate simple station-level metrics
        station_stats = []
        
        for station in df['station_name'].unique()[:50]:  # Limit to first 50 stations for performance
            station_data = df[df['station_name'] == station]
            
            if len(station_data) < 3:  # Skip stations with very little data
                continue
            
            # Calculate basic metrics with safety checks
            total_activity = max(1, station_data['total_rides'].sum())  # Ensure minimum value
            avg_balance = station_data['net_balance'].mean()
            balance_volatility = max(0.1, station_data['net_balance'].std())  # Ensure minimum variation
            
            # Activity consistency
            activity_std = station_data['total_rides'].std()
            activity_mean = station_data['total_rides'].mean()
            consistency = max(0.1, min(1.0, 1 / (1 + (activity_std / (activity_mean + 1)))))
            
            station_stats.append({
                'station_name': station,
                'total_activity': total_activity,
                'avg_balance': avg_balance,
                'balance_volatility': balance_volatility,
                'consistency': consistency
            })
        
        if len(station_stats) < 5:
            return go.Figure().add_annotation(
                text="Insufficient data for parallel coordinates analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14)
            )
        
        # Convert to DataFrame and handle edge cases
        station_df = pd.DataFrame(station_stats)
        
        # Replace any infinite values with finite values
        station_df = station_df.replace([np.inf, -np.inf], 0)
        station_df = station_df.fillna(0)
        
        # Ensure we have some variation in each dimension
        for col in ['total_activity', 'avg_balance', 'balance_volatility', 'consistency']:
            if station_df[col].std() < 0.001:  # Very small variation
                # Add small random variation to prevent flat lines
                station_df[col] = station_df[col] + np.random.normal(0, 0.1, len(station_df))
        
        # Create dimensions for parallel coordinates - simplified approach
        dimensions = [
            dict(
                label="Total Activity",
                values=station_df['total_activity'].tolist()
            ),
            dict(
                label="Average Balance",
                values=station_df['avg_balance'].tolist()
            ),
            dict(
                label="Balance Volatility",
                values=station_df['balance_volatility'].tolist()
            ),
            dict(
                label="Consistency",
                values=station_df['consistency'].tolist()
            )
        ]
        
        # Create color scale based on total activity
        activity_min = station_df['total_activity'].min()
        activity_max = station_df['total_activity'].max()
        
        if activity_max > activity_min:
            colors = (station_df['total_activity'] - activity_min) / (activity_max - activity_min)
        else:
            colors = [0.5] * len(station_df)
        
        # Create the parallel coordinates plot - minimal configuration
        fig = go.Figure()
        
        fig.add_trace(go.Parcoords(
            line=dict(
                color=colors.tolist(),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activity Level")
            ),
            dimensions=dimensions
        ))
        
        fig.update_layout(
            title="Station Characteristics Analysis",
            height=600,
            margin=dict(l=80, r=80, t=60, b=40)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in parallel coordinates: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text="Error creating parallel coordinates plot",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="red")
        )


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


def create_time_wheel_plot(combined, selected_week=None):
    """Create a comprehensive cyclic time wheel with connecting lines and week selector"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        from math import pi, cos, sin

        if combined is None or combined.empty:
            return go.Figure().add_annotation(
                text="No data available for time wheel visualization",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )

        # Prepare data
        df_temp = combined.copy()
        
        # Limit stations for performance
        if len(df_temp['station_name'].unique()) > 80:
            top_stations = df_temp.groupby('station_name')['departures'].sum().nlargest(80).index
            df_temp = df_temp[df_temp['station_name'].isin(top_stations)]
            
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        df_temp['day_of_week'] = df_temp['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df_temp['week'] = df_temp['date'].dt.isocalendar().week
        df_temp['total_rides'] = df_temp['departures'] + df_temp['arrivals']
        df_temp['net_balance'] = df_temp['departures'] - df_temp['arrivals']
        
        # Filter by selected week if provided
        if selected_week is not None:
            df_temp = df_temp[df_temp['week'] == selected_week]
            if df_temp.empty:
                return go.Figure().add_annotation(
                    text=f"No data available for week {selected_week}",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=16)
                )

        # Generate realistic hourly patterns based on actual bike share behavior
        np.random.seed(42)
        hourly_expanded_data = []
        
        for _, row in df_temp.iterrows():
            total_daily_rides = row['total_rides']
            
            # Create realistic hourly distribution patterns
            if row['day_of_week'] < 5:  # Weekdays
                hourly_pattern = np.array([
                    0.005, 0.002, 0.001, 0.001, 0.003, 0.015,  # 0-5: Night/early morning
                    0.045, 0.085, 0.120, 0.075, 0.055, 0.048,  # 6-11: Morning rush & mid-morning
                    0.052, 0.048, 0.042, 0.045, 0.058, 0.095,  # 12-17: Afternoon
                    0.135, 0.085, 0.065, 0.045, 0.025, 0.012   # 18-23: Evening rush & night
                ])
            else:  # Weekends
                hourly_pattern = np.array([
                    0.008, 0.003, 0.002, 0.002, 0.005, 0.012,  # 0-5: Night/early morning
                    0.025, 0.045, 0.065, 0.085, 0.095, 0.105,  # 6-11: Gradual morning increase
                    0.110, 0.095, 0.085, 0.075, 0.070, 0.075,  # 12-17: Steady afternoon
                    0.085, 0.070, 0.055, 0.040, 0.025, 0.015   # 18-23: Evening decline
                ])
            
            # Normalize pattern to sum to 1
            hourly_pattern = hourly_pattern / hourly_pattern.sum()
            
            # Generate hourly ride counts
            for hour in range(24):
                hour_rides = int(total_daily_rides * hourly_pattern[hour])
                if hour_rides > 0:  # Only include hours with activity
                    hourly_expanded_data.append({
                        'station_name': row['station_name'],
                        'date': row['date'],
                        'hour': hour,
                        'day_of_week': row['day_of_week'],
                        'rides_this_hour': hour_rides,
                        'net_balance_ratio': row['net_balance'] / max(1, total_daily_rides)
                    })

        if not hourly_expanded_data:
            return go.Figure().add_annotation(
                text="No hourly data could be generated",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )

        hourly_df = pd.DataFrame(hourly_expanded_data)
        
        # Aggregate data by hour and day of week
        time_aggregated = (
            hourly_df.groupby(['hour', 'day_of_week'])
            .agg({
                'rides_this_hour': 'sum',
                'net_balance_ratio': 'mean'
            })
            .reset_index()
        )

        # Create the polar time wheel plot
        fig = go.Figure()
        
        # Define colors for each day of the week
        day_colors = [
            '#FF6B6B',  # Monday - Red
            '#4ECDC4',  # Tuesday - Teal
            '#45B7D1',  # Wednesday - Blue
            '#96CEB4',  # Thursday - Green
            '#FFEAA7',  # Friday - Yellow
            '#DDA0DD',  # Saturday - Plum
            '#98D8C8'   # Sunday - Mint
        ]
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        # Create traces for each day of the week (only bubbles, no lines)
        for day_idx in range(7):
            day_data = time_aggregated[time_aggregated['day_of_week'] == day_idx]
            if day_data.empty:
                continue
            day_data = day_data.sort_values('hour')
            angles = [(hour * 15) - 90 for hour in day_data['hour']]
            base_radius = 1 + (day_idx * 0.5)
            max_rides = time_aggregated['rides_this_hour'].max()
            if max_rides > 0:
                radius_values = [base_radius + (rides / max_rides) * 1.5 for rides in day_data['rides_this_hour']]
            else:
                radius_values = [base_radius] * len(day_data)
            if max_rides > 0:
                bubble_sizes = [8 + (rides / max_rides) * 27 for rides in day_data['rides_this_hour']]
            else:
                bubble_sizes = [12] * len(day_data)
            fig.add_trace(go.Scatterpolar(
                r=radius_values,
                theta=angles,
                mode='markers',
                name=day_names[day_idx],
                marker=dict(
                    size=bubble_sizes,
                    color=day_colors[day_idx],
                    opacity=0.85,
                    line=dict(
                        color='white',
                        width=2
                    ),
                    sizemode='diameter'
                ),
                hovertemplate=(
                    f"<b>{day_names[day_idx]}</b><br>" +
                    "Hour: %{customdata[0]:02d}:00<br>" +
                    "Activity Level: %{customdata[1]:.0f} rides<br>" +
                    "Radius Position: %{r:.1f}<br>" +
                    "<extra></extra>"
                ),
                customdata=list(zip(day_data['hour'], day_data['rides_this_hour']))
            ))

        # Add concentric reference circles for days (more subtle)
        circle_theta = list(range(0, 360, 10))  # Every 10 degrees for smoother circles
        
        for day_idx in range(7):
            base_radius = 1 + (day_idx * 0.5)
            circle_r = [base_radius] * len(circle_theta)
            fig.add_trace(go.Scatterpolar(
                r=circle_r,
                theta=circle_theta,
                mode='lines',
                line=dict(color='lightgray', width=0.5, dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add hour markers around the wheel
        hour_labels = []
        hour_positions = []
        for hour in range(0, 24, 2):  # Every 2 hours
            angle = (hour * 15) - 90
            hour_labels.append(f"{hour:02d}:00")
            hour_positions.append(angle)

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    title="Activity Level by Day of Week",
                    visible=True,
                    range=[0, 6],
                    tickmode='array',
                    tickvals=[1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25],
                    ticktext=day_names,
                    gridcolor='lightgray',
                    gridwidth=0.5,
                    showline=True,
                    linewidth=1,
                    linecolor='gray'
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=hour_positions,
                    ticktext=hour_labels,
                    direction='clockwise',
                    rotation=90,
                    gridcolor='lightgray',
                    gridwidth=0.5
                )
            ),
            title={
                'text': f"Cyclic Time Wheel: Activity by Day and Hour",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            height=850,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=16, color='white'),
                bgcolor='rgba(30,30,30,0.95)',
                bordercolor='black',
                borderwidth=2,
                title=dict(
                    text="Days of Week",
                    font=dict(size=16, color='white')
                )
            ),
            margin=dict(l=20, r=180, t=120, b=60),
            annotations=[
                dict(
                    text="📊 Bubble Size = Hourly Activity Level",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    font=dict(size=13, color="white"),
                    bgcolor="rgba(30,30,30,0.95)",
                    bordercolor="black",
                    borderwidth=1
                ),
                dict(
                    text="🕐 Hours flow clockwise from midnight (top)",
                    xref="paper", yref="paper",
                    x=0.02, y=0.05,
                    showarrow=False,
                    font=dict(size=12, color="white"),
                    bgcolor="rgba(30,30,30,0.85)"
                )
            ]
        )

        return fig
        
    except Exception as e:
        print(f"Error creating time wheel: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure().add_annotation(
            text=f"Error creating time wheel: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="red")
        )


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
    """Create ARIMA forecast for a selected station using more historical data for better confidence intervals"""
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

        if len(station_data) < 14:  # Need minimum data
            return fig, "Insufficient data for ARIMA forecast (need at least 14 days)"

        # Use more data for better confidence intervals: 4-6 weeks (28-42 days) or all available data if less
        available_days = len(station_data)
        if available_days >= 42:
            recent_data = station_data.tail(42)  # Use 6 weeks
        elif available_days >= 28:
            recent_data = station_data.tail(28)  # Use 4 weeks
        else:
            recent_data = station_data  # Use all available data

        if len(recent_data) < 14:
            return fig, "Insufficient recent data for ARIMA forecast"

        # Record training date range
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
            title=f"ARIMA Forecast - {selected_station} (Using Past {len(recent_data)} Days - More Data = Narrower CI)",
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


def predict_peak_periods(combined, start_date, end_date=None, use_all_time=False):
    """Predict peak/off-peak periods for all stations with continuous color mapping"""
    try:
        import plotly.graph_objects as go
        import pandas as pd
        import numpy as np
        import plotly.express as px
        
        # Debug: Check input data
        if combined is None or combined.empty:
            return go.Figure().add_annotation(
                text="No data provided for peak analysis",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
        
        # Debug: Check required columns
        required_cols = ['date', 'station_name', 'departures', 'arrivals', 'lat', 'lng']
        missing_cols = [col for col in required_cols if col not in combined.columns]
        if missing_cols:
            return go.Figure().add_annotation(
                text=f"Missing required columns: {missing_cols}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )
        
        # Ensure date column is datetime
        combined_temp = combined.copy()
        combined_temp['date'] = pd.to_datetime(combined_temp['date'])
        
        # Debug: Check date conversion
        print(f"Date range in data: {combined_temp['date'].min()} to {combined_temp['date'].max()}")
        print(f"Input start_date: {start_date}, type: {type(start_date)}")
        if end_date:
            print(f"Input end_date: {end_date}, type: {type(end_date)}")
        
        # Handle date selection logic
        if use_all_time:
            df_filtered = combined_temp.copy()
            date_range_text = "All Time"
        elif end_date is None:
            # Single day analysis - ensure start_date is a date object
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
            df_filtered = combined_temp[combined_temp['date'].dt.date == start_date].copy()
            date_range_text = start_date.strftime('%B %d, %Y')
        else:
            # Date range analysis - ensure dates are date objects
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
            if hasattr(end_date, 'date'):
                end_date = end_date.date()
            df_filtered = combined_temp[
                (combined_temp['date'].dt.date >= start_date) & 
                (combined_temp['date'].dt.date <= end_date)
            ].copy()
            date_range_text = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"

        # Debug: Check filtered data
        print(f"Filtered data shape: {df_filtered.shape}")
        print(f"Date range text: {date_range_text}")

        if df_filtered.empty:
            return go.Figure().add_annotation(
                text=f"No data available for {date_range_text}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
            )

        # Calculate activity metrics
        if use_all_time or end_date is not None:
            # Aggregate by station for multi-day analysis
            print("Performing multi-day analysis...")
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
            print("Performing single day analysis...")
            station_activity = df_filtered.copy()
            station_activity['total_activity'] = station_activity['departures'] + station_activity['arrivals']
            activity_col = 'total_activity'
            activity_label = 'Total Activity'

        # Debug: Check activity calculation
        print(f"Station activity shape: {station_activity.shape}")
        print(f"Activity column: {activity_col}")
        if not station_activity.empty:
            print(f"Activity range: {station_activity[activity_col].min()} to {station_activity[activity_col].max()}")
        
        if station_activity.empty:
            return go.Figure().add_annotation(
                text=f"No stations with activity for {date_range_text}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=16)
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

        # Add stations with simple color mapping (avoiding coloraxis system)
        # Convert intensity values to discrete color categories
        def get_color_and_category(intensity):
            if intensity <= 0.2:
                return '#08519c', 'Non-Peak'  # Dark blue
            elif intensity <= 0.4:
                return '#3182bd', 'Low'       # Medium blue
            elif intensity <= 0.6:
                return '#6baed6', 'Medium'    # Light blue
            elif intensity <= 0.8:
                return '#9ecae1', 'High'      # Very light blue
            else:
                return '#c6dbef', 'Peak'      # Lightest blue
        
        # Apply color mapping
        station_activity['color'] = station_activity['color_intensity'].apply(lambda x: get_color_and_category(x)[0])
        station_activity['category'] = station_activity['color_intensity'].apply(lambda x: get_color_and_category(x)[1])
        
        # Add stations with discrete colors (no coloraxis needed)
        fig.add_trace(go.Scattermapbox(
            lat=station_activity['lat'],
            lon=station_activity['lng'],
            mode='markers',
            marker=dict(
                size=station_activity['marker_size'],
                color=station_activity['color'],
                opacity=0.8,
                line=dict(color='white', width=1)
            ),
            text=station_activity['station_name'],
            customdata=np.column_stack((
                station_activity[activity_col].round(1),
                station_activity['peak_intensity'].round(3),
                station_activity['departures'],
                station_activity['arrivals'],
                station_activity['category']
            )),
            hovertemplate=(
                "<b>%{text}</b><br>" +
                f"{activity_label}: %{{customdata[0]}}<br>" +
                "Peak Intensity: %{customdata[1]:.1%}<br>" +
                "Category: %{customdata[4]}<br>" +
                "Departures: %{customdata[2]}<br>" +
                "Arrivals: %{customdata[3]}<br>" +
                "<extra></extra>"
            ),
            showlegend=False
        ))
        
        # Add manual legend entries for peak categories
        legend_colors = {
            'Non-Peak': '#08519c',
            'Low': '#3182bd', 
            'Medium': '#6baed6',
            'High': '#9ecae1',
            'Peak': '#c6dbef'
        }
        
        for category, color in legend_colors.items():
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=category,
                showlegend=True,
                hoverinfo='skip'
            ))

        # Add reference annotations for peak categories
        fig.add_annotation(
            x=0.02, y=0.98,
            xref="paper", yref="paper",
            text=(
                f"<b>Peak Analysis: {date_range_text}</b><br>" +
                f"🔵 Non-Peak: Very low activity stations<br>" +
                f"🟢 Low: Below average activity stations<br>" +
                f"🟡 Medium: Average activity stations<br>" +
                f"🟠 High: Above average activity stations<br>" +
                f"🔴 Peak: Very high activity stations<br>" +
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
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1
            )
        )

        print(f"Successfully created peak analysis with {len(station_activity)} stations")
        return fig

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in peak periods analysis: {e}")
        print(f"Full error details: {error_details}")
        return go.Figure().add_annotation(
            text=f"Error in analysis: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="red")
        )


def predict_peak_periods_standalone(combined, start_date, end_date=None, use_all_time=False):
    """Predict peak/off-peak periods for all stations with 5-level discrete color mapping"""
    try:
        # Simple data filtering
        if use_all_time:
            data = combined.copy()
            title_period = "All Time"
        elif end_date is None:
            data = combined[combined['date'] == start_date].copy()
            title_period = f"Single Day - {start_date}"
        else:
            data = combined[(combined['date'] >= start_date) & (combined['date'] <= end_date)].copy()
            title_period = f"Date Range - {start_date} to {end_date}"
        
        if data.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Peak Analysis - No Data", height=500)
            return fig
        
        # Calculate station activity
        station_stats = data.groupby('station_name').agg({
            'departures': 'sum',
            'arrivals': 'sum',
            'lat': 'first',
            'lng': 'first'
        }).reset_index()
        
        # Total activity and intensity calculation
        station_stats['total_activity'] = station_stats['departures'] + station_stats['arrivals']
        station_stats = station_stats[station_stats['total_activity'] > 0]
        
        if station_stats.empty:
            fig = go.Figure()
            fig.add_annotation(text="No station activity found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Peak Analysis - No Activity", height=500)
            return fig
        
        # Normalize to 0-100 scale
        min_activity = station_stats['total_activity'].min()
        max_activity = station_stats['total_activity'].max()
        
        if max_activity > min_activity:
            station_stats['intensity'] = ((station_stats['total_activity'] - min_activity) / (max_activity - min_activity)) * 100
        else:
            station_stats['intensity'] = 50.0
        
        # 5-level discrete color mapping
        def get_color(intensity):
            if intensity < 20:
                return '#0066CC'  # Cold Blue
            elif intensity < 40:
                return '#3399FF'  # Cool Blue  
            elif intensity < 60:
                return '#FFFF66'  # Yellow
            elif intensity < 80:
                return '#FF9933'  # Orange
            else:
                return '#FF3300'  # Hot Red
        
        station_stats['color'] = station_stats['intensity'].apply(get_color)
        station_stats['size'] = 8 + (station_stats['intensity'] / 100) * 12  # Size 8-20
        
        # Create map
        fig = go.Figure()
        
        # Add stations
        fig.add_trace(go.Scattermapbox(
            lat=station_stats['lat'],
            lon=station_stats['lng'],
            mode='markers',
            marker=dict(
                size=station_stats['size'],
                color=station_stats['color'],
                opacity=0.8
            ),
            text=station_stats['station_name'],
            hovertemplate="<b>%{text}</b><br>Activity: %{customdata[0]}<br>Intensity: %{customdata[1]:.1f}%<extra></extra>",
            customdata=np.column_stack([station_stats['total_activity'], station_stats['intensity']]),
            showlegend=False
        ))
        
        # Add legend entries
        colors = ['#0066CC', '#3399FF', '#FFFF66', '#FF9933', '#FF3300']
        labels = ['Non-Peak', 'Low Activity', 'Medium Activity', 'High Activity', 'Peak']
        
        for color, label in zip(colors, labels):
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None],
                mode='markers',
                marker=dict(size=12, color=color),
                name=label,
                showlegend=True
            ))
        
        # Layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=station_stats['lat'].mean(), lon=station_stats['lng'].mean()),
                zoom=11
            ),
            title=f"Peak Activity Analysis - {title_period}",
            height=600,
            margin=dict(l=0, r=0, t=40, b=0),
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(title="Peak Analysis - Error", height=500)
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
        title=f"Daily rides-hourly ({start_date.strftime('%d/%m/%y')} - {end_date.strftime('%d/%m/%y')})",
        xaxis_title="Date and Time",
        yaxis_title="Hourly Rides",
        height=500,
        margin=dict(l=20, r=60, t=80, b=30),
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
            y=1.05,
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
        showlegend=True,
        dragmode=False
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
        showlegend=True,
        dragmode=False
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
