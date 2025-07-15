import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
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

# ── Month mappings ─────────────────────────────────────────────────────────────
months = {
    "202409": "September 2024",
    "202412": "December 2024",
    "202503": "March 2025",
    "202506": "June 2025"
}

# ── DATA LOADING AND PREPARATION ───────────────────────────────────────────────

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

# ── WEATHER DATA ───────────────────────────────────────────────────────────────

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

# ── CLUSTERING FUNCTIONS ───────────────────────────────────────────────────────

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
                        station_coord = coords[station_idx:station_idx+1]
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
    
    st.success(f"🎯 ArcGIS-style Balance: total_imbalance={total_imbalance:.0f}, max_zone={max_zone_imbalance:.0f}, balanced_zones={balanced_zones}/{n_clusters}")

    st.write(f"⏱ TOTAL ArcGIS-inspired clustering: {time.time() - t0:.3f}s")

    return day_gdf.to_crs("EPSG:4326"), cent[[
        'cluster', 'centroid_lat', 'centroid_lng',
        'net_balance', 'station_count', 'departures', 'arrivals'
    ]]
