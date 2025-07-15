import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from libpysal.weights import W
import time
from spopt.region.skater import Skater
from libpysal.weights import KNN
import geopandas as gpd
from libpysal.weights import Queen
from spopt.region import RegionKMeansHeuristic
from shapely.geometry import Point
from pyproj import Transformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import date
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.graph_objects as go
import time
from copy import copy
from libpysal.weights import W
from pyproj import Transformer
from spopt.region import RegionKMeansHeuristic
import streamlit as st

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


# ── CORE FUNCTIONS ─────────────────────────────────────────────────────────────

@st.cache_data
def prepare_geodata_and_weights(full_df):
    """
    Prepare geodata optimized for balanced zones with sufficient points
    """
    # Keep more stations for meaningful balanced zones - aim for 100-200 points
    if len(full_df) > 200:
        # Sample to get roughly 150-200 stations for better zone balancing
        sample_size = min(200, max(150, int(len(full_df) * 0.3)))
        full_df = full_df.sample(n=sample_size, random_state=42)
        st.info(f"📊 Sampled {len(full_df)} stations for zone balancing")

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


def run_spatial_balanced_clustering(df_day, n_clusters, proj_gdf, global_w):
    """
    ArcGIS-inspired Balanced Zones algorithm using open-source tools
    Creates spatially contiguous zones balanced by net_balance variable
    """
    t0 = time.time()

    # 1) join & compute net_balance
    day_stats = df_day.set_index('station_name')[['departures', 'arrivals']]
    day_gdf = proj_gdf.join(day_stats, on='station_name', how='inner').copy()
    day_gdf['net_balance'] = day_gdf['departures'] - day_gdf['arrivals']

    if day_gdf['net_balance'].abs().sum() == 0:
        st.write(f"⏱ TOTAL (no imbalance): {time.time() - t0:.3f}s")
        return day_gdf.to_crs("EPSG:4326"), pd.DataFrame()

    # Sample large datasets for optimal balanced zones performance
    original_size = len(day_gdf)
    if len(day_gdf) > 150:
        # Keep more stations for meaningful balance - aim for 100-150 stations
        target_size = min(150, max(100, len(day_gdf) // 2))
        day_gdf = day_gdf.sample(n=target_size, random_state=42)
        st.info(f"🎯 Using {len(day_gdf)}/{original_size} stations for balanced zones")
    elif len(day_gdf) < 30:
        st.warning(f"⚠️ Only {len(day_gdf)} stations available - need at least 30 for meaningful balanced zones")
        # Use all available stations
        st.info(f"🎯 Using all {len(day_gdf)} stations")

    st.info("🎯 Running ArcGIS-inspired Balanced Zones algorithm...")

    # 2) ArcGIS-style Balanced Zones Implementation
    t2 = time.time()

    # Build spatial connectivity matrix
    idx = list(day_gdf.index)
    idx_set = set(idx)

    # Create adjacency matrix for spatial contiguity
    from scipy.sparse import lil_matrix
    n_stations = len(day_gdf)
    station_to_idx = {station_idx: i for i, station_idx in enumerate(idx)}

    adjacency = lil_matrix((n_stations, n_stations))

    for i, station_idx in enumerate(idx):
        if station_idx in global_w.neighbors:
            for neighbor_idx in global_w.neighbors[station_idx]:
                if neighbor_idx in idx_set:
                    j = station_to_idx[neighbor_idx]
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1  # Ensure symmetry

    adjacency = adjacency.tocsr()

    # ArcGIS Balanced Zones core algorithm
    best_zones = None
    best_balance_score = float('inf')

    # Multiple attempts with different strategies
    for attempt in range(10):
        np.random.seed(attempt)

        # Step 1: Create initial seed zones using spatial distribution
        zones = np.full(n_stations, -1)  # -1 means unassigned
        zone_balances = np.zeros(n_clusters)
        zone_sizes = np.zeros(n_clusters)

        # Select spatially distributed seeds
        coords = day_gdf[['X', 'Y']].values
        from sklearn.cluster import KMeans
        seed_kmeans = KMeans(n_clusters=n_clusters, random_state=attempt, n_init=1)
        seed_clusters = seed_kmeans.fit_predict(coords)

        # Assign initial seeds - one station per zone
        for zone_id in range(n_clusters):
            zone_stations = np.where(seed_clusters == zone_id)[0]
            if len(zone_stations) > 0:
                # Pick the station closest to cluster center
                center = seed_kmeans.cluster_centers_[zone_id]
                distances = np.sum((coords[zone_stations] - center) ** 2, axis=1)
                seed_station = zone_stations[np.argmin(distances)]

                zones[seed_station] = zone_id
                zone_balances[zone_id] = day_gdf.iloc[seed_station]['net_balance']
                zone_sizes[zone_id] = 1

        # Step 2: Grow zones using ArcGIS-style region growing
        unassigned = np.where(zones == -1)[0]

        while len(unassigned) > 0:
            best_assignment = None
            best_score = float('inf')

            # For each unassigned station
            for station_idx in unassigned:
                station_balance = day_gdf.iloc[station_idx]['net_balance']

                # Find adjacent zones
                adjacent_zones = set()
                station_neighbors = adjacency[station_idx].nonzero()[1]

                for neighbor_idx in station_neighbors:
                    if zones[neighbor_idx] != -1:
                        adjacent_zones.add(zones[neighbor_idx])

                # If no adjacent zones, find nearest zone
                if not adjacent_zones:
                    assigned_stations = np.where(zones != -1)[0]
                    if len(assigned_stations) > 0:
                        station_coord = coords[station_idx:station_idx + 1]
                        assigned_coords = coords[assigned_stations]
                        distances = np.sum((assigned_coords - station_coord) ** 2, axis=1)
                        nearest_station = assigned_stations[np.argmin(distances)]
                        adjacent_zones.add(zones[nearest_station])

                # Evaluate assignment to each adjacent zone
                for zone_id in adjacent_zones:
                    # ArcGIS balance criterion: minimize deviation from target
                    total_balance = np.sum(np.abs(zone_balances))
                    target_balance_per_zone = total_balance / n_clusters if total_balance > 0 else 0

                    # Calculate balance score if we assign this station to this zone
                    new_zone_balance = zone_balances[zone_id] + station_balance
                    balance_deviation = abs(new_zone_balance)

                    # Size constraint: prefer balanced zone sizes
                    size_penalty = max(0, zone_sizes[zone_id] - (n_stations // n_clusters)) * 10

                    score = balance_deviation + size_penalty

                    if score < best_score:
                        best_score = score
                        best_assignment = (station_idx, zone_id, station_balance)

            # Make the best assignment
            if best_assignment:
                station_idx, zone_id, station_balance = best_assignment
                zones[station_idx] = zone_id
                zone_balances[zone_id] += station_balance
                zone_sizes[zone_id] += 1
                unassigned = unassigned[unassigned != station_idx]
            else:
                # Fallback: assign remaining stations to smallest zones
                for station_idx in unassigned:
                    smallest_zone = np.argmin(zone_sizes)
                    zones[station_idx] = smallest_zone
                    zone_balances[smallest_zone] += day_gdf.iloc[station_idx]['net_balance']
                    zone_sizes[smallest_zone] += 1
                break

        # Step 3: Local optimization phase (ArcGIS-style boundary refinement)
        improved = True
        iterations = 0
        while improved and iterations < 20:
            improved = False
            iterations += 1

            # Check each station for potential reassignment
            for station_idx in range(n_stations):
                current_zone = zones[station_idx]
                current_balance = day_gdf.iloc[station_idx]['net_balance']

                # Find adjacent zones
                adjacent_zones = set()
                station_neighbors = adjacency[station_idx].nonzero()[1]

                for neighbor_idx in station_neighbors:
                    adjacent_zones.add(zones[neighbor_idx])

                adjacent_zones.discard(current_zone)  # Remove current zone

                # Don't leave a zone empty
                if zone_sizes[current_zone] <= 1:
                    continue

                # Try reassigning to adjacent zones
                for new_zone in adjacent_zones:
                    # Calculate balance improvement
                    old_balance_score = abs(zone_balances[current_zone]) + abs(zone_balances[new_zone])

                    new_current_balance = zone_balances[current_zone] - current_balance
                    new_target_balance = zone_balances[new_zone] + current_balance
                    new_balance_score = abs(new_current_balance) + abs(new_target_balance)

                    if new_balance_score < old_balance_score:
                        # Make the reassignment
                        zones[station_idx] = new_zone
                        zone_balances[current_zone] -= current_balance
                        zone_balances[new_zone] += current_balance
                        zone_sizes[current_zone] -= 1
                        zone_sizes[new_zone] += 1
                        improved = True
                        break

        # Score this solution
        balance_score = np.sum(np.abs(zone_balances))

        if balance_score < best_balance_score:
            best_balance_score = balance_score
            best_zones = zones.copy()

    # Assign results
    day_gdf['cluster'] = best_zones
    st.write(f"⏱ ArcGIS-style Balanced Zones: {time.time() - t2:.3f}s")

    # 3) Calculate centroids and zone statistics
    t3 = time.time()
    cent = (
        day_gdf
        .groupby('cluster')
        .agg(
            centroid_x=('X', 'mean'),
            centroid_y=('Y', 'mean'),
            net_balance=('net_balance', 'sum'),
            station_count=('station_name', 'count'),
            departures=('departures', 'sum'),
            arrivals=('arrivals', 'sum')
        )
        .reset_index()
    )

    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True) \
        .transform(cent['centroid_x'], cent['centroid_y'])
    cent['centroid_lng'], cent['centroid_lat'] = lon, lat
    st.write(f"⏱ centroids+proj: {time.time() - t3:.3f}s")

    # Show ArcGIS-style balance quality metrics
    zone_balances = cent['net_balance'].values
    total_imbalance = np.sum(np.abs(zone_balances))
    max_zone_imbalance = np.max(np.abs(zone_balances))
    balanced_zones = np.sum(np.abs(zone_balances) < 20)

    st.success(
        f"🎯 ArcGIS-style Balance: total_imbalance={total_imbalance:.0f}, max_zone={max_zone_imbalance:.0f}, balanced_zones={balanced_zones}/{n_clusters}")

    st.write(f"⏱ TOTAL ArcGIS-inspired clustering: {time.time() - t0:.3f}s")

    return day_gdf.to_crs("EPSG:4326"), cent[[
        'cluster', 'centroid_lat', 'centroid_lng',
        'net_balance', 'station_count', 'departures', 'arrivals'
    ]]


# ── Load processed data ────────────────────────────────────────────────────────
@st.cache_data
def load_processed_data():
    data = {}
    for code in months:
        fp = f"processed_{code}.csv"
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            # drop the bogus 2024-08-31 row
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


# ── Time Series Clustering ─────────────────────────────────────────────────────
@st.cache_data
def perform_time_series_clustering(pivot_net_filtered, n_clusters, station_coords):
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


@st.cache_data
def aggregate_weekly_by_station(combined, year, week):
    df = combined.copy()
    df['date'] = pd.to_datetime(df['date'])

    df['iso_year'] = df['date'].dt.isocalendar().year
    df['iso_week'] = df['date'].dt.isocalendar().week

    wk = df[(df['iso_year'] == year) & (df['iso_week'] == week)]
    if wk.empty:
        return wk.iloc[0:0]

    agg = (
        wk
        .groupby(['station_name', 'lat', 'lng'], as_index=False)
        .agg({'departures': 'sum', 'arrivals': 'sum'})
    )
    agg['net_balance'] = agg['departures'] - agg['arrivals']
    return agg


@st.cache_data
def prepare_daily_time_series_data(combined, selected_month):
    """
    Prepare daily time series data for a specific month
    """
    df2 = combined.copy()
    df2['date'] = pd.to_datetime(df2['date'])

    # Filter by selected month
    month_data = df2[df2['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return pd.DataFrame(), pd.DataFrame(), []

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


# ── Static‐Map visualizer ───────────────────────────────────────────────────────
def create_map_visualization(df_day, radius_m, categories):
    df_day = df_day.copy()
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


# ── Time Series Clustering Map ─────────────────────────────────────────────────
def create_time_series_cluster_map(clustered_df):
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
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="Black",
            borderwidth=1
        )
    )

    return fig


def create_spatial_cluster_map(cluster_results, centroids_df=None):
    if cluster_results.empty:
        return go.Figure()

    colors = px.colors.qualitative.Set1
    fig = go.Figure()

    # station‐point traces
    for cluster_id in sorted(cluster_results['cluster'].unique()):
        cluster_data = cluster_results[cluster_results['cluster'] == cluster_id]
        fig.add_trace(go.Scattermapbox(
            lat=cluster_data['lat'],
            lon=cluster_data['lng'],
            mode='markers',
            marker=dict(
                size=8,
                color=colors[(cluster_id - 1) % len(colors)],
                opacity=0.6
            ),
            hoverinfo='none',
            showlegend=False
        ))

    # centroid styling via two layers
    OUTLINE_PAD = 4
    INNER_SIZE = 16
    OUTLINE_SIZE = INNER_SIZE + OUTLINE_PAD

    if centroids_df is not None and not centroids_df.empty:
        for _, centroid in centroids_df.iterrows():
            cid = int(centroid['cluster'])
            net = centroid['net_balance']
            lat, lon = centroid['centroid_lat'], centroid['centroid_lng']

            # 1) black outline layer
            fig.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lon],
                mode='markers',
                marker=dict(
                    size=OUTLINE_SIZE,
                    color='black',
                    opacity=1
                ),
                hoverinfo='none',
                showlegend=False
            ))

            # 2) inner colored dot with hover
            fig.add_trace(go.Scattermapbox(
                lat=[lat], lon=[lon],
                mode='markers',
                marker=dict(
                    size=INNER_SIZE,
                    color=colors[(cid - 1) % len(colors)],
                    opacity=0.9
                ),
                hovertemplate=(
                    f"<b>Cluster {cid}</b><br>"
                    f"Total Net: {net:+.0f}<extra></extra>"
                ),
                name=f"Cluster {cid}"
            ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=10
        ),
        legend=dict(
            title="Centroids",
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )

    return fig


# ── Timeline‐Map visualizer ─────────────────────────────────────────────────────
def create_timeline_map(combined, start_date, end_date, radius_m, categories):
    dates = pd.date_range(start_date, end_date).date
    frames = []
    for d in dates:
        df_day = combined[combined["date"] == d].copy()
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.title("NYC Citibike Station Balance Visualization")

    data = load_processed_data()
    if not data:
        st.error("No data loaded.")
        return

    combined = pd.concat(data.values(), ignore_index=True)
    dates = sorted(pd.to_datetime(combined["date"]).dt.date.unique())

    all_stations_df = combined[['station_name', 'lat', 'lng']].drop_duplicates()

    # Prepare geodata once for the entire app
    proj_gdf, global_w = prepare_geodata_and_weights(all_stations_df)

    # Map mode selection
    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Static Map", "Timeline Map"])

    # Add clear cache button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and refresh the app"):
        st.cache_data.clear()
        st.rerun()

    if mode == "Static Map":
        # Static Map Mode controls
        st.sidebar.header("Main Map Date")
        sel_date = st.sidebar.date_input("", value=dates[0], min_value=dates[0], max_value=dates[-1])

        st.sidebar.header("Map Settings")
        radius_m = st.sidebar.slider("Radius (m):", 100, 200, 100, 10)
        show_dep = st.sidebar.checkbox("More departures", True)
        show_arr = st.sidebar.checkbox("More arrivals", True)
        show_bal = st.sidebar.checkbox("Balanced", True)

        categories = [
            name for name, chk in zip(
                ["More departures", "More arrivals", "Balanced"],
                [show_dep, show_arr, show_bal]
            ) if chk
        ]

        st.subheader(f"Main Map — {sel_date.strftime('%d/%m/%y')}")
        df_day = combined[combined["date"] == sel_date]
        if df_day.empty:
            st.warning("No data for this date.")
        else:
            fig = create_map_visualization(df_day, radius_m, categories)
            st.plotly_chart(fig, use_container_width=True)

        # Daily Time-Series Clustering by Month
        st.subheader("Daily Time Series Clustering")

        # Month selection dropdown
        col1, col2 = st.columns([1, 2])
        with col1:
            # Get available months from the data
            available_months = sorted(pd.to_datetime(combined['date']).dt.to_period('M').unique())
            month_options = {str(month): month.strftime('%B %Y') for month in available_months}

            selected_month_str = st.selectbox(
                "Select Month:",
                options=list(month_options.keys()),
                format_func=lambda x: month_options[x],
                key="month_select"
            )
            selected_month = pd.Period(selected_month_str)

        with col2:
            ts_k = st.selectbox("Number of Clusters:", list(range(1, 7)), index=2, key="ts_k")

        # Prepare daily time series data for selected month
        pivot_daily, coords, day_info = prepare_daily_time_series_data(combined, selected_month)

        if not pivot_daily.empty and day_info:
            # Display the selected month info
            start_day = day_info[0]['date'].strftime('%d/%m/%y')
            end_day = day_info[-1]['date'].strftime('%d/%m/%y')
            st.write(f"**Analyzing daily patterns: {start_day} - {end_day}** ({len(day_info)} days)")

            # Perform clustering
            ts_res, kmeans_model = perform_time_series_clustering(pivot_daily, ts_k, coords)

            # Show the clustering map
            st.plotly_chart(create_time_series_cluster_map(ts_res), use_container_width=True)

            # Show cluster characteristics
            st.markdown("### Daily Cluster Characteristics")

            if not ts_res.empty and len(pivot_daily.columns) > 0:
                # Create cluster analysis
                cluster_data = []

                # Day labels are already in DD/MM format
                day_labels = list(pivot_daily.columns)

                # Calculate average patterns for each cluster
                for cluster_id in sorted(ts_res['cluster'].unique()):
                    cluster_stations = ts_res[ts_res['cluster'] == cluster_id]['station_name'].tolist()
                    cluster_series = pivot_daily.loc[pivot_daily.index.isin(cluster_stations)]

                    if not cluster_series.empty:
                        avg_pattern = cluster_series.mean(axis=0).values

                        # Calculate trend line using linear regression
                        from sklearn.linear_model import LinearRegression
                        x_values = np.arange(len(avg_pattern)).reshape(-1, 1)
                        lr = LinearRegression()
                        lr.fit(x_values, avg_pattern)
                        trend_line = lr.predict(x_values)
                        trend_slope = lr.coef_[0]

                        cluster_data.append({
                            'cluster': cluster_id,
                            'stations': len(cluster_stations),
                            'pattern': avg_pattern,
                            'trend_line': trend_line,
                            'trend_slope': trend_slope,
                            'max_value': avg_pattern.max(),
                            'min_value': avg_pattern.min(),
                            'volatility': avg_pattern.std(),
                            'trend': 'Increasing' if trend_slope > 0 else 'Decreasing',
                            'all_station_data': cluster_series.values
                        })

                # 1. Cluster Data Distribution - Combined Whisker Plot
                st.markdown("### Cluster Data Distribution")

                fig_whisker = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

                for i, cluster_info in enumerate(cluster_data):
                    cluster_id = cluster_info['cluster']
                    all_data = cluster_info['all_station_data']
                    flat_data = all_data.flatten()

                    fig_whisker.add_trace(go.Box(
                        y=flat_data,
                        name=f"Cluster {cluster_id} ({cluster_info['stations']} stations)",
                        boxpoints='outliers',
                        marker_color=colors[i % len(colors)],
                        jitter=0.3,
                        pointpos=-1.8
                    ))

                fig_whisker.update_layout(
                    title=f"Net Balance Distribution - All Clusters ({month_options[selected_month_str]})",
                    yaxis_title="Net Balance",
                    xaxis_title="Cluster",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_whisker, use_container_width=True)

                # 2. Daily Patterns with Trend Lines - Normalized
                st.markdown("### Daily Patterns with Trend Analysis (Normalized)")

                # Add normalization option
                col1, col2 = st.columns([3, 1])
                with col2:
                    normalization_type = st.selectbox(
                        "Normalization:",
                        ["Z-Score (Mean=0, Std=1)", "Min-Max (0-100)", "Raw Values"],
                        index=0,
                        key="norm_type"
                    )

                fig_patterns = go.Figure()

                # Add zero reference line (always at 0 for normalized, actual 0 for raw)
                if normalization_type == "Raw Values":
                    ref_line_y = 0
                    ref_label = "Zero Reference"
                    y_title = "Net Balance"
                else:
                    ref_line_y = 0
                    ref_label = "Mean Reference" if "Z-Score" in normalization_type else "Mid Reference"
                    y_title = "Normalized Net Balance"

                fig_patterns.add_hline(
                    y=ref_line_y,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text=ref_label,
                    annotation_position="bottom right"
                )

                for i, cluster_info in enumerate(cluster_data):
                    color = colors[i % len(colors)]
                    cluster_id = cluster_info['cluster']

                    # Get the pattern and apply normalization
                    original_pattern = cluster_info['pattern']

                    if normalization_type == "Z-Score (Mean=0, Std=1)":
                        if np.std(original_pattern) > 0:
                            normalized_pattern = (original_pattern - np.mean(original_pattern)) / np.std(
                                original_pattern)
                        else:
                            normalized_pattern = original_pattern - np.mean(original_pattern)
                    elif normalization_type == "Min-Max (0-100)":
                        min_val, max_val = np.min(original_pattern), np.max(original_pattern)
                        if max_val > min_val:
                            normalized_pattern = (original_pattern - min_val) / (max_val - min_val) * 100
                        else:
                            normalized_pattern = np.full_like(original_pattern, 50)
                    else:  # Raw Values
                        normalized_pattern = original_pattern

                    # Calculate trend line on normalized data
                    x_values = np.arange(len(normalized_pattern))
                    if len(x_values) > 1:
                        trend_slope = np.polyfit(x_values, normalized_pattern, 1)[0]
                        trend_line = np.polyval(
                            [trend_slope, np.mean(normalized_pattern) - trend_slope * np.mean(x_values)], x_values)
                    else:
                        trend_line = normalized_pattern
                        trend_slope = 0

                    # Add actual pattern line
                    fig_patterns.add_trace(go.Scatter(
                        x=day_labels,
                        y=normalized_pattern,
                        mode='lines+markers',
                        name=f"Cluster {cluster_id} Data",
                        line=dict(color=color, width=3),
                        marker=dict(size=6),
                        legendgroup=f"cluster_{cluster_id}",
                        showlegend=True
                    ))

                    # Add trend line
                    fig_patterns.add_trace(go.Scatter(
                        x=day_labels,
                        y=trend_line,
                        mode='lines',
                        name=f"Cluster {cluster_id} Trend (slope: {trend_slope:.2f})",
                        line=dict(color=color, width=2, dash='dot'),
                        legendgroup=f"cluster_{cluster_id}",
                        showlegend=True
                    ))

                # Update title based on normalization
                if normalization_type == "Z-Score (Mean=0, Std=1)":
                    title_suffix = "Z-Score Normalized (Mean=0, Std=1)"
                elif normalization_type == "Min-Max (0-100)":
                    title_suffix = "Min-Max Normalized (0-100 Scale)"
                else:
                    title_suffix = "Raw Values"

                fig_patterns.update_layout(
                    title=f"Daily Net Balance Patterns - {title_suffix} ({month_options[selected_month_str]})",
                    xaxis_title="Day (DD/MM)",
                    yaxis_title=y_title,
                    height=500,
                    hovermode='x unified',
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                )
                st.plotly_chart(fig_patterns, use_container_width=True)

                # Add explanation of normalization benefits
                with st.expander("ℹ️ Understanding Normalization"):
                    st.markdown("""
                    **Why Normalize for Better Behavioral Understanding?**

                    **🎯 Z-Score Normalization (Recommended):**
                    - Centers each cluster around 0 (mean)
                    - Shows relative deviations from typical behavior
                    - Makes pattern shapes directly comparable
                    - Reveals which days are "unusually high/low" for each cluster

                    **📊 Min-Max Normalization:**
                    - Scales all clusters to 0-100 range
                    - Preserves pattern shapes
                    - Good for comparing relative trends

                    **🔍 What This Reveals:**
                    - **Pattern Synchronization**: Do clusters follow similar daily rhythms?
                    - **Volatility Differences**: Which clusters are more/less variable?
                    - **Trend Consistency**: Are trends similar when scale is removed?
                    - **Behavioral Types**: Weekend vs weekday dominant patterns
                    """)

                # 3. Spatial Spider Plot - All Stations
                st.markdown("### Spatial Spider Plot - All Stations")

                # Time period selection
                col1, col2 = st.columns([1, 3])
                with col1:
                    time_scope = st.selectbox(
                        "Time Scope:",
                        ["Selected Month", "All Time"],
                        index=0,
                        key="time_scope"
                    )

                # Prepare data based on time scope
                if time_scope == "All Time":
                    # Use all available data
                    df_temp = combined.copy()
                    df_temp['date'] = pd.to_datetime(df_temp['date'])

                    # Aggregate all data by station
                    all_time_agg = (
                        df_temp
                        .groupby(['station_name'], as_index=False)
                        .agg({
                            'departures': 'sum',
                            'arrivals': 'sum',
                            'lat': 'first',
                            'lng': 'first'
                        })
                    )
                    all_time_agg['net_balance'] = all_time_agg['departures'] - all_time_agg['arrivals']

                    # Calculate additional metrics for all time
                    station_metrics = []
                    for _, station_row in all_time_agg.iterrows():
                        station_name = station_row['station_name']
                        station_data = df_temp[df_temp['station_name'] == station_name]

                        if len(station_data) > 1:
                            daily_balances = (station_data['departures'] - station_data['arrivals']).values

                            # Calculate weekday vs weekend difference
                            weekday_vals = []
                            weekend_vals = []
                            for _, day_row in station_data.iterrows():
                                day_of_week = pd.to_datetime(day_row['date']).weekday()
                                daily_balance = day_row['departures'] - day_row['arrivals']
                                if day_of_week < 5:
                                    weekday_vals.append(daily_balance)
                                else:
                                    weekend_vals.append(daily_balance)

                            weekday_weekend_diff = 0
                            if weekday_vals and weekend_vals:
                                weekday_weekend_diff = abs(np.mean(weekday_vals) - np.mean(weekend_vals))

                            metrics = {
                                'station_name': station_name,
                                'lat': station_row['lat'],
                                'lng': station_row['lng'],
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

                    time_label = "All Time"

                else:
                    # Use selected month data
                    if not pivot_daily.empty:
                        station_metrics = []
                        for station_name in pivot_daily.index:
                            if station_name in coords['station_name'].values:
                                station_coord = coords[coords['station_name'] == station_name].iloc[0]
                                station_data = pivot_daily.loc[station_name].values

                                # Calculate weekday vs weekend difference
                                weekday_weekend_diff = 0
                                if len(day_info) > 0:
                                    weekday_values = []
                                    weekend_values = []

                                    for i, day_detail in enumerate(day_info):
                                        day_of_week = day_detail['date'].weekday()
                                        if i < len(station_data):
                                            if day_of_week < 5:
                                                weekday_values.append(station_data[i])
                                            else:
                                                weekend_values.append(station_data[i])

                                    if weekday_values and weekend_values:
                                        weekday_avg = np.mean(weekday_values)
                                        weekend_avg = np.mean(weekend_values)
                                        weekday_weekend_diff = abs(weekday_avg - weekend_avg)

                                metrics = {
                                    'station_name': station_name,
                                    'lat': station_coord['lat'],
                                    'lng': station_coord['lng'],
                                    'avg_net_balance': np.mean(station_data),
                                    'volatility': np.std(station_data),
                                    'range_val': np.max(station_data) - np.min(station_data),
                                    'trend_slope': abs(np.polyfit(range(len(station_data)), station_data, 1)[0]),
                                    'peak_value': np.max(station_data),
                                    'valley_value': abs(np.min(station_data)),
                                    'weekday_weekend_diff': weekday_weekend_diff,
                                    'consistency': 100 - (np.std(station_data) / (abs(np.mean(station_data)) + 1) * 100)
                                }
                                station_metrics.append(metrics)
                    else:
                        station_metrics = []

                    time_label = month_options[selected_month_str]

                if station_metrics:
                    # Create DataFrame and normalize metrics
                    stations_df = pd.DataFrame(station_metrics)

                    # Normalize all metrics to 0-1 scale for spider plot
                    metric_cols = ['avg_net_balance', 'volatility', 'range_val', 'trend_slope',
                                   'peak_value', 'valley_value', 'weekday_weekend_diff', 'consistency']

                    for col in metric_cols:
                        min_val, max_val = stations_df[col].min(), stations_df[col].max()
                        if max_val > min_val:
                            stations_df[f'{col}_norm'] = (stations_df[col] - min_val) / (max_val - min_val)
                        else:
                            stations_df[f'{col}_norm'] = 0.5

                    # Create spatial spider plot
                    fig_spider = go.Figure()

                    # Define spider plot parameters
                    n_metrics = len(metric_cols)
                    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)

                    # Color stations by cluster if available
                    if not ts_res.empty:
                        # Merge with cluster information
                        stations_df = stations_df.merge(
                            ts_res[['station_name', 'cluster']],
                            on='station_name',
                            how='left'
                        )
                        cluster_colors = {
                            1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c',
                            4: '#d62728', 5: '#9467bd', 6: '#8c564b'
                        }
                    else:
                        stations_df['cluster'] = 1
                        cluster_colors = {1: '#1f77b4'}

                    # Plot each station as a spider glyph
                    for _, station in stations_df.iterrows():
                        # Get normalized values for spider plot
                        values = [station[f'{col}_norm'] for col in metric_cols]

                        cluster_id = station.get('cluster', 1)
                        color = cluster_colors.get(cluster_id, '#1f77b4')

                        # Add spider lines
                        for i in range(n_metrics):
                            angle = angles[i]
                            value = values[i]
                            scaled_value = 0.1 + value * 0.4
                            x_end = station['lng'] + scaled_value * np.cos(angle) * 0.01
                            y_end = station['lat'] + scaled_value * np.sin(angle) * 0.01

                            fig_spider.add_trace(go.Scattermapbox(
                                lat=[station['lat'], y_end],
                                lon=[station['lng'], x_end],
                                mode='lines',
                                line=dict(color=color, width=1),
                                hoverinfo='skip',
                                showlegend=False
                            ))

                        # Add center point
                        fig_spider.add_trace(go.Scattermapbox(
                            lat=[station['lat']],
                            lon=[station['lng']],
                            mode='markers',
                            marker=dict(size=4, color=color),
                            text=station['station_name'],
                            hovertemplate=f"<b>%{{text}}</b><br>Cluster: {cluster_id}<extra></extra>",
                            showlegend=False
                        ))

                    # Add legend for clusters
                    for cluster_id, color in cluster_colors.items():
                        if cluster_id in stations_df['cluster'].values:
                            fig_spider.add_trace(go.Scattermapbox(
                                lat=[None], lon=[None],
                                mode='markers',
                                marker=dict(size=10, color=color),
                                name=f'Cluster {cluster_id}',
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
                        title=f"Spatial Spider Plot - All Stations ({time_label})",
                        height=700,
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="left",
                            x=0
                        )
                    )

                    st.plotly_chart(fig_spider, use_container_width=True)

                    # Show spider plot legend
                    st.markdown("#### Spider Plot Dimensions")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **📊 Each spider arm represents:**
                        - **Avg Net Balance**: Overall demand pattern
                        - **Volatility**: Day-to-day variation
                        - **Range**: Spread of values
                        - **Trend Slope**: Rate of change
                        """)
                    with col2:
                        st.markdown("""
                        **📈 Spider arm meanings:**
                        - **Peak Value**: Maximum demand
                        - **Valley Value**: Maximum deficit  
                        - **Weekday-Weekend Diff**: Usage pattern difference
                        - **Consistency**: Predictability measure
                        """)

                else:
                    st.warning("No station data available for spider plot.")

                # 4. Statistics and Insights
                col1, col2 = st.columns([2, 1])
                with col2:
                    # Cluster summary statistics
                    summary_data = []
                    for cluster_info in cluster_data:
                        summary_data.append({
                            'Cluster': cluster_info['cluster'],
                            'Stations': cluster_info['stations'],
                            'Max': f"{cluster_info['max_value']:.0f}",
                            'Min': f"{cluster_info['min_value']:.0f}",
                            'Volatility': f"{cluster_info['volatility']:.0f}",
                            'Trend Slope': f"{cluster_info['trend_slope']:.1f}",
                            'Direction': cluster_info['trend']
                        })

                    summary_df = pd.DataFrame(summary_data)
                    st.write("**Daily Cluster Statistics:**")
                    st.dataframe(summary_df, use_container_width=True)

                # 5. Simplified Cluster Insights
                st.markdown("### Daily Cluster Insights")

                insights_cols = st.columns(len(cluster_data))
                for i, cluster_info in enumerate(cluster_data):
                    with insights_cols[i]:
                        st.markdown(f"**Cluster {cluster_info['cluster']}**")
                        st.metric("Stations", cluster_info['stations'])
                        st.metric("Peak Balance", f"{cluster_info['max_value']:.0f}")
                        st.metric("Trend Slope", f"{cluster_info['trend_slope']:.1f}")

                        # Determine cluster behavior for daily patterns
                        if cluster_info['volatility'] > 30:
                            behavior = "🌊 Highly Variable"
                        elif cluster_info['volatility'] > 10:
                            behavior = "📈 Moderately Variable"
                        else:
                            behavior = "📊 Stable"

                        # Trend interpretation
                        if abs(cluster_info['trend_slope']) < 0.5:
                            trend_desc = "🔄 Stable"
                        elif cluster_info['trend_slope'] > 0:
                            trend_desc = "📈 Growing Demand"
                        else:
                            trend_desc = "📉 Declining Demand"

                        st.write(f"**Behavior:** {behavior}")
                        st.write(f"**Trend:** {trend_desc}")

            else:
                # Show basic cluster summary if no time series data
                cluster_summary = ts_res.groupby("cluster").size().reset_index(name="station_count")
                st.write("**Cluster Summary:**")
                st.dataframe(cluster_summary, use_container_width=True)

        else:
            st.warning(f"No data available for {month_options.get(selected_month_str, 'selected month')}.")

        # 6. Additional Model Suggestions
        st.markdown("### 🔬 Additional Analysis Models")

        with st.expander("📊 **Enhanced Spider Glyph Plot - Additional Data Suggestions**"):
            st.markdown("""
            **🎯 Current Spider Glyph Dimensions:**
            - Avg Net Balance, Volatility, Range, Trend Slope, Peak/Valley Values, Weekday-Weekend Diff, Consistency

            **📈 Additional Data to Enhance Spider Glyph:**

            **Temporal Patterns:**
            - **Peak Hour Intensity**: Concentration of activity in rush hours
            - **Seasonality Score**: How much the pattern varies by season
            - **Consistency Index**: How predictable the daily pattern is

            **Spatial Context:**
            - **Distance to City Center**: Proximity to Manhattan core
            - **Neighborhood Density**: Population/business density around station
            - **Transit Connectivity**: Number of nearby subway/bus stops
            - **Land Use Type**: Residential vs Commercial vs Mixed ratio

            **Operational Metrics:**
            - **Capacity Utilization**: How often stations are full/empty
            - **Turnover Rate**: How quickly bikes are picked up after arrival
            - **Weather Sensitivity**: How much weather affects this cluster

            **User Behavior:**
            - **Trip Duration Patterns**: Average trip length from these stations
            - **User Type Mix**: Tourist vs Subscriber ratio
            - **Return Rate**: How often bikes return to same station
            """)

        with st.expander("🤖 Machine Learning Models You Could Add"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                **📈 Forecasting Models:**
                - **ARIMA/SARIMA**: Predict future net balance patterns
                - **Prophet**: Seasonal forecasting with holidays
                - **LSTM Neural Networks**: Deep learning for sequence prediction
                - **Vector Autoregression (VAR)**: Multi-station forecasting

                **🏷️ Classification Models:**
                - **Station Type Classification**: Classify stations (residential, business, tourist)
                - **Peak/Off-Peak Prediction**: Predict high-demand periods
                - **Weather Impact Analysis**: How weather affects each cluster
                """)

            with col2:
                st.markdown("""
                **🔍 Pattern Discovery:**
                - **Anomaly Detection**: Identify unusual station behavior
                - **Seasonal Decomposition**: Separate trend, seasonal, residual components
                - **Dynamic Time Warping**: Find similar pattern shapes
                - **Change Point Detection**: Identify when behavior patterns change

                **🗺️ Spatial Models:**
                - **Spatial Autocorrelation**: How nearby stations influence each other
                - **Geographically Weighted Regression**: Location-based modeling
                - **Network Analysis**: Station-to-station flow optimization
                """)

        with st.expander("📊 Advanced Visualizations You Could Add"):
            st.markdown("""
            **Interactive Dashboards:**
            - **3D Surface Plots**: Time × Space × Net Balance
            - **Sankey Diagrams**: Flow between station clusters
            - **Heatmaps**: Hourly patterns across clusters
            - **Network Graphs**: Station connectivity and influence

            **Predictive Analytics:**
            - **Forecasting Dashboard**: Next 7-day predictions per cluster
            - **What-if Scenarios**: Impact of adding/removing stations
            - **Optimization Models**: Optimal bike redistribution strategies

            **Real-time Features:**
            - **Live Data Integration**: Real-time station status
            - **Alert Systems**: Automatic imbalance notifications
            - **Mobile Dashboard**: Responsive design for field operations
            """)

            # 3. Spatial Spider Plot - All Stations
            st.markdown("### Spatial Spider Plot - All Stations")

            # Time period selection
            col1, col2 = st.columns([1, 3])
            with col1:
                time_scope = st.selectbox(
                    "Time Scope:",
                    ["Selected Month", "All Time"],
                    index=0,
                    key="time_scope"
                )

            # Prepare data based on time scope
            if time_scope == "All Time":
                # Use all available data
                all_time_pivot, all_coords, _ = prepare_daily_time_series_data(combined,
                                                                               combined['date'].dt.to_period('M').iloc[
                                                                                   0])  # Get structure
                # Recalculate with all data
                df_temp = combined.copy()
                df_temp['date'] = pd.to_datetime(df_temp['date'])

                # Aggregate all data by station
                all_time_agg = (
                    df_temp
                    .groupby(['station_name'], as_index=False)
                    .agg({
                        'departures': 'sum',
                        'arrivals': 'sum',
                        'lat': 'first',
                        'lng': 'first'
                    })
                )
                all_time_agg['net_balance'] = all_time_agg['departures'] - all_time_agg['arrivals']

                # Calculate additional metrics for all time
                station_metrics = []
                for _, station_row in all_time_agg.iterrows():
                    station_name = station_row['station_name']
                    station_data = df_temp[df_temp['station_name'] == station_name]

                    if len(station_data) > 1:
                        daily_balances = (station_data['departures'] - station_data['arrivals']).values

                        # Calculate weekday vs weekend difference
                        weekday_vals = []
                        weekend_vals = []
                        for _, day_row in station_data.iterrows():
                            day_of_week = pd.to_datetime(day_row['date']).weekday()
                            daily_balance = day_row['departures'] - day_row['arrivals']
                            if day_of_week < 5:
                                weekday_vals.append(daily_balance)
                            else:
                                weekend_vals.append(daily_balance)

                        weekday_weekend_diff = 0
                        if weekday_vals and weekend_vals:
                            weekday_weekend_diff = abs(np.mean(weekday_vals) - np.mean(weekend_vals))

                        metrics = {
                            'station_name': station_name,
                            'lat': station_row['lat'],
                            'lng': station_row['lng'],
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

                time_label = "All Time"

            else:
                # Use selected month data
                if not pivot_daily.empty:
                    station_metrics = []
                    for station_name in pivot_daily.index:
                        if station_name in coords['station_name'].values:
                            station_coord = coords[coords['station_name'] == station_name].iloc[0]
                            station_data = pivot_daily.loc[station_name].values

                            # Calculate weekday vs weekend difference
                            weekday_weekend_diff = 0
                            if len(day_info) > 0:
                                weekday_values = []
                                weekend_values = []

                                for i, day_detail in enumerate(day_info):
                                    day_of_week = day_detail['date'].weekday()
                                    if i < len(station_data):
                                        if day_of_week < 5:
                                            weekday_values.append(station_data[i])
                                        else:
                                            weekend_values.append(station_data[i])

                                if weekday_values and weekend_values:
                                    weekday_avg = np.mean(weekday_values)
                                    weekend_avg = np.mean(weekend_values)
                                    weekday_weekend_diff = abs(weekday_avg - weekend_avg)

                            metrics = {
                                'station_name': station_name,
                                'lat': station_coord['lat'],
                                'lng': station_coord['lng'],
                                'avg_net_balance': np.mean(station_data),
                                'volatility': np.std(station_data),
                                'range_val': np.max(station_data) - np.min(station_data),
                                'trend_slope': abs(np.polyfit(range(len(station_data)), station_data, 1)[0]),
                                'peak_value': np.max(station_data),
                                'valley_value': abs(np.min(station_data)),
                                'weekday_weekend_diff': weekday_weekend_diff,
                                'consistency': 100 - (np.std(station_data) / (abs(np.mean(station_data)) + 1) * 100)
                            }
                            station_metrics.append(metrics)
                else:
                    station_metrics = []

                time_label = month_options[selected_month_str]

            if station_metrics:
                # Create DataFrame and normalize metrics
                stations_df = pd.DataFrame(station_metrics)

                # Normalize all metrics to 0-1 scale for spider plot
                metric_cols = ['avg_net_balance', 'volatility', 'range_val', 'trend_slope',
                               'peak_value', 'valley_value', 'weekday_weekend_diff', 'consistency']

                for col in metric_cols:
                    min_val, max_val = stations_df[col].min(), stations_df[col].max()
                    if max_val > min_val:
                        stations_df[f'{col}_norm'] = (stations_df[col] - min_val) / (max_val - min_val)
                    else:
                        stations_df[f'{col}_norm'] = 0.5

                # Create spatial spider plot
                fig_spider = go.Figure()

                # Define spider plot parameters
                n_metrics = len(metric_cols)
                angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)

                # Color stations by cluster if available
                if not ts_res.empty:
                    # Merge with cluster information
                    stations_df = stations_df.merge(
                        ts_res[['station_name', 'cluster']],
                        on='station_name',
                        how='left'
                    )
                    cluster_colors = {
                        1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c',
                        4: '#d62728', 5: '#9467bd', 6: '#8c564b'
                    }
                else:
                    stations_df['cluster'] = 1
                    cluster_colors = {1: '#1f77b4'}

                # Plot each station as a spider glyph
                for _, station in stations_df.iterrows():
                    # Get normalized values for spider plot
                    values = [station[f'{col}_norm'] for col in metric_cols]

                    # Convert polar to cartesian coordinates
                    spider_x = []
                    spider_y = []
                    for i, (angle, value) in enumerate(zip(angles, values)):
                        # Scale value for visibility (0.1 to 0.5 range)
                        scaled_value = 0.1 + value * 0.4
                        x_offset = scaled_value * np.cos(angle)
                        y_offset = scaled_value * np.sin(angle)
                        spider_x.extend(
                            [station['lng'], station['lng'] + x_offset * 0.01, station['lng']])  # Scale for map
                        spider_y.extend([station['lat'], station['lat'] + y_offset * 0.01, station['lat']])

                    cluster_id = station.get('cluster', 1)
                    color = cluster_colors.get(cluster_id, '#1f77b4')

                    # Add spider lines
                    for i in range(n_metrics):
                        angle = angles[i]
                        value = values[i]
                        scaled_value = 0.1 + value * 0.4
                        x_end = station['lng'] + scaled_value * np.cos(angle) * 0.01
                        y_end = station['lat'] + scaled_value * np.sin(angle) * 0.01

                        fig_spider.add_trace(go.Scattermapbox(
                            lat=[station['lat'], y_end],
                            lon=[station['lng'], x_end],
                            mode='lines',
                            line=dict(color=color, width=1),
                            hoverinfo='skip',
                            showlegend=False
                        ))

                    # Add center point
                    fig_spider.add_trace(go.Scattermapbox(
                        lat=[station['lat']],
                        lon=[station['lng']],
                        mode='markers',
                        marker=dict(size=4, color=color),
                        text=station['station_name'],
                        hovertemplate=f"<b>%{{text}}</b><br>Cluster: {cluster_id}<extra></extra>",
                        showlegend=False
                    ))

                # Add legend for clusters
                for cluster_id, color in cluster_colors.items():
                    if cluster_id in stations_df['cluster'].values:
                        fig_spider.add_trace(go.Scattermapbox(
                            lat=[None], lon=[None],
                            mode='markers',
                            marker=dict(size=10, color=color),
                            name=f'Cluster {cluster_id}',
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
                    title=f"Spatial Spider Plot - All Stations ({time_label})",
                    height=700,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0
                    )
                )

                st.plotly_chart(fig_spider, use_container_width=True)

                # Show spider plot legend
                st.markdown("#### Spider Plot Dimensions")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                        **📊 Each spider arm represents:**
                        - **Avg Net Balance**: Overall demand pattern
                        - **Volatility**: Day-to-day variation
                        - **Range**: Spread of values
                        - **Trend Slope**: Rate of change
                        """)
                with col2:
                    st.markdown("""
                        **📈 Spider arm meanings:**
                        - **Peak Value**: Maximum demand
                        - **Valley Value**: Maximum deficit  
                        - **Weekday-Weekend Diff**: Usage pattern difference
                        - **Consistency**: Predictability measure
                        """)

            else:
                st.warning("No station data available for spider plot.")
            st.markdown("### Daily Patterns with Trend Analysis")

            fig_patterns = go.Figure()

            # Add zero reference line first (so it appears behind)
            fig_patterns.add_hline(
                y=0,
                line_dash="dash",
                line_color="gray",
                annotation_text="Zero Reference",
                annotation_position="bottom right"
            )

            for i, cluster_info in enumerate(cluster_data):
                color = colors[i % len(colors)]
                cluster_id = cluster_info['cluster']

                # Add actual pattern line
                fig_patterns.add_trace(go.Scatter(
                    x=day_labels,
                    y=cluster_info['pattern'],
                    mode='lines+markers',
                    name=f"Cluster {cluster_id} Data",
                    line=dict(color=color, width=3),
                    marker=dict(size=6),
                    legendgroup=f"cluster_{cluster_id}",
                    showlegend=True
                ))

                # Add trend line
                fig_patterns.add_trace(go.Scatter(
                    x=day_labels,
                    y=cluster_info['trend_line'],
                    mode='lines',
                    name=f"Cluster {cluster_id} Trend (slope: {cluster_info['trend_slope']:.1f})",
                    line=dict(color=color, width=2, dash='dot'),
                    legendgroup=f"cluster_{cluster_id}",
                    showlegend=True
                ))

            fig_patterns.update_layout(
                title=f"Daily Net Balance Patterns with Trend Lines ({month_options[selected_month_str]})",
                xaxis_title="Day (DD/MM)",
                yaxis_title="Average Net Balance",
                height=500,
                hovermode='x unified',
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            st.plotly_chart(fig_patterns, use_container_width=True)

            with col2:
                # Cluster summary statistics
                summary_data = []
                for cluster_info in cluster_data:
                    summary_data.append({
                        'Cluster': cluster_info['cluster'],
                        'Stations': cluster_info['stations'],
                        'Max': f"{cluster_info['max_value']:.0f}",
                        'Min': f"{cluster_info['min_value']:.0f}",
                        'Volatility': f"{cluster_info['volatility']:.0f}",
                        'Trend Slope': f"{cluster_info['trend_slope']:.1f}",
                        'Direction': cluster_info['trend']
                    })

                summary_df = pd.DataFrame(summary_data)
                st.write("**Daily Cluster Statistics:**")
                st.dataframe(summary_df, use_container_width=True)

            # 5. Additional Model Suggestions
            st.markdown("### 🔬 Additional Analysis Models")

            with st.expander("📊 **Enhanced Star Glyph Plot - Additional Data Suggestions**"):
                st.markdown("""
                    **🎯 Current Star Glyph Dimensions:**
                    - Station Count, Avg Net Balance, Volatility, Range, Trend Slope, Peak/Valley Values, Data Spread

                    **📈 Additional Data to Enhance Star Glyph:**

                    **Temporal Patterns:**
                    - **Weekday vs Weekend Behavior**: Average difference between weekday/weekend patterns
                    - **Peak Hour Intensity**: Concentration of activity in rush hours
                    - **Seasonality Score**: How much the pattern varies by season
                    - **Consistency Index**: How predictable the daily pattern is

                    **Spatial Context:**
                    - **Distance to City Center**: Proximity to Manhattan core
                    - **Neighborhood Density**: Population/business density around station
                    - **Transit Connectivity**: Number of nearby subway/bus stops
                    - **Land Use Type**: Residential vs Commercial vs Mixed ratio

                    **Operational Metrics:**
                    - **Capacity Utilization**: How often stations are full/empty
                    - **Turnover Rate**: How quickly bikes are picked up after arrival
                    - **Maintenance Frequency**: How often station needs attention
                    - **Weather Sensitivity**: How much weather affects this cluster

                    **User Behavior:**
                    - **Trip Duration Patterns**: Average trip length from these stations
                    - **User Type Mix**: Tourist vs Subscriber ratio
                    - **Return Rate**: How often bikes return to same station
                    - **Multi-modal Usage**: Integration with public transit
                    """)

            with st.expander("🤖 Machine Learning Models You Could Add"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("""
                        **📈 Forecasting Models:**
                        - **ARIMA/SARIMA**: Predict future net balance patterns
                        - **Prophet**: Seasonal forecasting with holidays
                        - **LSTM Neural Networks**: Deep learning for sequence prediction
                        - **Vector Autoregression (VAR)**: Multi-station forecasting

                        **🏷️ Classification Models:**
                        - **Station Type Classification**: Classify stations (residential, business, tourist)
                        - **Peak/Off-Peak Prediction**: Predict high-demand periods
                        - **Weather Impact Analysis**: How weather affects each cluster
                        """)

                with col2:
                    st.markdown("""
                        **🔍 Pattern Discovery:**
                        - **Anomaly Detection**: Identify unusual station behavior
                        - **Seasonal Decomposition**: Separate trend, seasonal, residual components
                        - **Dynamic Time Warping**: Find similar pattern shapes
                        - **Change Point Detection**: Identify when behavior patterns change

                        **🗺️ Spatial Models:**
                        - **Spatial Autocorrelation**: How nearby stations influence each other
                        - **Geographically Weighted Regression**: Location-based modeling
                        - **Network Analysis**: Station-to-station flow optimization
                        """)

            with st.expander("📊 Advanced Visualizations You Could Add"):
                st.markdown("""
                    **Interactive Dashboards:**
                    - **3D Surface Plots**: Time × Space × Net Balance
                    - **Sankey Diagrams**: Flow between station clusters
                    - **Heatmaps**: Hourly patterns across clusters
                    - **Network Graphs**: Station connectivity and influence

                    **Predictive Analytics:**
                    - **Forecasting Dashboard**: Next 7-day predictions per cluster
                    - **What-if Scenarios**: Impact of adding/removing stations
                    - **Optimization Models**: Optimal bike redistribution strategies

                    **Real-time Features:**
                    - **Live Data Integration**: Real-time station status
                    - **Alert Systems**: Automatic imbalance notifications
                    - **Mobile Dashboard**: Responsive design for field operations
                    """)

            # 5. Simplified Cluster Insights
            st.markdown("### Daily Cluster Insights")

            insights_cols = st.columns(len(cluster_data))
            for i, cluster_info in enumerate(cluster_data):
                with insights_cols[i]:
                    st.markdown(f"**Cluster {cluster_info['cluster']}**")
                    st.metric("Stations", cluster_info['stations'])
                    st.metric("Peak Balance", f"{cluster_info['max_value']:.0f}")
                    st.metric("Trend Slope", f"{cluster_info['trend_slope']:.1f}")

                    # Determine cluster behavior for daily patterns
                    if cluster_info['volatility'] > 30:
                        behavior = "🌊 Highly Variable"
                    elif cluster_info['volatility'] > 10:
                        behavior = "📈 Moderately Variable"
                    else:
                        behavior = "📊 Stable"

                    # Trend interpretation
                    if abs(cluster_info['trend_slope']) < 0.5:
                        trend_desc = "🔄 Stable"
                    elif cluster_info['trend_slope'] > 0:
                        trend_desc = "📈 Growing Demand"
                    else:
                        trend_desc = "📉 Declining Demand"

                    st.write(f"**Behavior:** {behavior}")
                    st.write(f"**Trend:** {trend_desc}")

            else:
            # Show basic cluster summary if no time series data
            cluster_summary = ts_res.groupby("cluster").size().reset_index(name="station_count")
            st.write("**Cluster Summary:**")
            st.dataframe(cluster_summary, use_container_width=True)

        else:
        st.warning(f"No data available for {month_options.get(selected_month_str, 'selected month')}.")

        else:
        # Timeline mode controls
        st.sidebar.header("Timeline Options")
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

        st.sidebar.header("Select Date Range")
        start_date = st.sidebar.date_input("Start date:", value=dates[0], min_value=dates[0], max_value=dates[-1])
        end_date = st.sidebar.date_input("End date:",
                                         value=min(dates[-1], dates[0] + pd.Timedelta(days=6)),
                                         min_value=dates[0], max_value=dates[-1])

        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        # Animated Timeline Map
        st.subheader(f"Timeline Map — {start_date.strftime('%d/%m/%y')} to {end_date.strftime('%d/%m/%y')}")
        try:
            fig_timeline = create_timeline_map(combined, start_date, end_date, radius_m, categories)
            st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_map")
        except Exception as e:
            st.error(f"Error creating timeline map: {e}")

        # Daily Spatial Balanced Clustering
        st.sidebar.header("Balanced Clustering (RegionKMeans)")
        selected_date = st.sidebar.date_input("Select Date for Clustering", min_value=dates[0], max_value=dates[-1],
                                              value=dates[0])
        n_clusters = st.sidebar.slider("Number of clusters for spatial analysis", 2, 5, value=3)

        df_day = combined[combined['date'] == selected_date].copy()
        df_day = df_day.dropna(subset=['lat', 'lng', 'departures', 'arrivals'])
        if not df_day.empty:
            df_day['net_balance'] = df_day['departures'] - df_day['arrivals']
            gdf, centroids_df = run_spatial_balanced_clustering(
                df_day, n_clusters, proj_gdf, global_w
            )
            st.subheader(f"Balanced Zones on {selected_date}")
            st.plotly_chart(create_spatial_cluster_map(gdf, centroids_df), use_container_width=True)

            st.markdown("### Zone Statistics")
            if not centroids_df.empty:
                st.dataframe(centroids_df[['cluster', 'station_count', 'net_balance', 'departures', 'arrivals']])

        # Weather Data
        st.subheader("Weather Data")
        try:
            weather_data = load_weather(start_date, end_date)
            if not weather_data.empty:
                col1, col2 = st.columns(2)
                with col1:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=weather_data["date"],
                        y=weather_data["temperature"],
                        mode="lines+markers",
                        name="Temperature",
                        line=dict(width=2),
                    ))
                    fig_temp.update_layout(title="Daily Max Temperature", xaxis_title="Date", yaxis_title="Temp (°C)",
                                           height=300)
                    st.plotly_chart(fig_temp, use_container_width=True)
                with col2:
                    fig_hum = go.Figure()
                    fig_hum.add_trace(go.Scatter(
                        x=weather_data["date"],
                        y=weather_data["humidity"],
                        mode="lines+markers",
                        name="Humidity",
                        line=dict(width=2),
                    ))
                    fig_hum.update_layout(title="Daily Max Humidity", xaxis_title="Date", yaxis_title="Humidity (%)",
                                          height=300)
                    st.plotly_chart(fig_hum, use_container_width=True)
            else:
                st.warning("No weather data available for this date range.")
        except Exception as e:
            st.error(f"Error loading weather data: {e}")

if __name__ == "__main__":
    main()