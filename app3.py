import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
from kneed import KneeLocator
from pyproj import Transformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from datetime import date
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

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
    # Prepare data for time series clustering
    ts_data = pivot_net_filtered.values

    # Normalize the time series using standard scaler
    scaler = StandardScaler()
    ts_data_scaled = scaler.fit_transform(ts_data)

    # Use K-means on the time series data directly
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(ts_data_scaled)

    # Create results dataframe
    results = pd.DataFrame({
        'station_name': pivot_net_filtered.index,
        'cluster': cluster_labels
    })

    # Add coordinates
    results = results.merge(station_coords, on='station_name', how='left')
    results = results.dropna(subset=['lat', 'lng'])

    return results, kmeans


# ── Improved Spatial K-means Clustering (Balanced) ──────────────────────────────────────
@st.cache_data
def perform_spatial_balanced_clustering(combined, selected_date, n_clusters):
    # filter for the one day
    df_day = combined[combined["date"] == selected_date].copy()
    df_day = df_day.dropna(subset=["lat", "lng", "departures", "arrivals"])
    if df_day.empty:
        return pd.DataFrame(), None

    # compute net balance
    df_day['net_balance'] = df_day['departures'] - df_day['arrivals']

    # spatial features → scale
    X_spatial = df_day[['lat', 'lng']].to_numpy()
    spatial_scaler = StandardScaler()
    X_spatial_scaled = spatial_scaler.fit_transform(X_spatial)

    # K-means +1 shift
    spatial_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    spatial_clusters = spatial_kmeans.fit_predict(X_spatial_scaled) + 1
    df_day['cluster'] = spatial_clusters

    # recover centroids in original lat/lng
    centroids_scaled = spatial_kmeans.cluster_centers_
    centroids_original = spatial_scaler.inverse_transform(centroids_scaled)
    centroids_df = pd.DataFrame({
        'cluster': range(1, n_clusters + 1),
        'centroid_lat': centroids_original[:, 0],
        'centroid_lng': centroids_original[:, 1]
    })

    # cluster stats
    cluster_stats = df_day.groupby('cluster').agg({
        'net_balance': 'sum',
        'departures': 'sum',
        'arrivals': 'sum',
        'station_name': 'count'
    }).reset_index()

    # merge stats (fill missing clusters)
    centroids_df = (
        centroids_df
        .merge(cluster_stats, on='cluster', how='left')
        .fillna({'net_balance': 0, 'departures': 0, 'arrivals': 0, 'station_name': 0})
    )

    return df_day, centroids_df


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
def create_time_series_cluster_map(cluster_results):
    if cluster_results.empty:
        return go.Figure()

    # Create color palette for clusters
    colors = px.colors.qualitative.Set3

    fig = go.Figure()

    for cluster_id in sorted(cluster_results['cluster'].unique()):
        cluster_data = cluster_results[cluster_results['cluster'] == cluster_id]

        fig.add_trace(go.Scattermapbox(
            lat=cluster_data['lat'],
            lon=cluster_data['lng'],
            mode='markers',
            marker=dict(
                size=10,
                color=colors[cluster_id % len(colors)],
                opacity=0.8
            ),
            text=cluster_data['station_name'],
            hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster_id) + "<extra></extra>",
            name=f"Cluster {cluster_id}"
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=11,
            bounds=dict(north=40.9176, south=40.4774,
                        east=-73.7004, west=-74.2591)
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(size=14),
            bordercolor="black", borderwidth=1
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    return fig


# ── Improved Spatial K-means Clustering Map ─────────────────────────────────────────────
def create_spatial_cluster_map(cluster_results, centroids_df=None):
    if cluster_results.empty:
        return go.Figure()

    colors = px.colors.qualitative.Set1
    fig = go.Figure()

    # — station‐point traces unchanged —
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

    # — centroid styling via two layers —
    OUTLINE_PAD = 4    # smaller pad
    INNER_SIZE   = 16  # reduced inner diameter
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
            zoom=10  # zoomed out a bit
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

@st.cache_data
def perform_arcgis_style_kmeans(agg_df, n_clusters, crs_from="EPSG:4326", crs_to="EPSG:3857"):
    """
    agg_df: DataFrame with ['station_name','lat','lng','net_balance',…]
    n_clusters: desired number of spatial clusters
    crs_from: input CRS of lat/lng (4326)
    crs_to: projected CRS for planar K-means (3857 or your local UTM)
    """
    if agg_df.empty:
        return agg_df, pd.DataFrame()

    # 1) drop nulls
    df = agg_df.dropna(subset=['lat','lng']).copy()

    # 2) project lat/lng → X, Y in meters
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    xs, ys = transformer.transform(df['lng'].values, df['lat'].values)
    df['X'] = xs
    df['Y'] = ys

    # 3) build feature matrix (here purely spatial; add net_balance if you like)
    X_feat = df[['X','Y']].to_numpy()

    # 4) scale if desired (optional for pure spatial)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_feat)

    # 5) run K-Means → labels 0…K-1 → shift to 1…K
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled) + 1
    df['cluster'] = labels

    # 6) get centroids back into projected space then invert to lat/lng
    centers_proj = km.cluster_centers_
    # undo scaling
    centers_feat = scaler.inverse_transform(centers_proj)
    cent_x, cent_y = centers_feat[:,0], centers_feat[:,1]
    # inverse-project
    lon_c, lat_c = transformer.transform(cent_x, cent_y, direction="INVERSE")
    cent_df = pd.DataFrame({
        'cluster': list(range(1, n_clusters+1)),
        'centroid_lat': lat_c,
        'centroid_lng': lon_c
    })

    # 7) compute per-cluster stats and merge (keeps 1…K)
    stats = (
        df.groupby('cluster')
          .agg(
            net_balance=('net_balance','sum'),
            departures=('departures','sum'),
            arrivals=('arrivals','sum'),
            station_count=('station_name','count')
          )
          .reset_index()
    )
    cent_df = (
        cent_df
        .merge(stats, on='cluster', how='left')
        .fillna({'net_balance':0,'departures':0,'arrivals':0,'station_count':0})
    )

    return df, cent_df

@st.cache_data
def perform_monthly_spatial_balanced_clustering(agg_df, n_clusters):
    """
    Run the exact 'balanced' KMeans you used daily, but on a monthly aggregate.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Output of aggregate_monthly_by_station(combined, year, month),
        with columns ['station_name','lat','lng','departures','arrivals','net_balance'].
    n_clusters : int
        Number of clusters to create.

    Returns
    -------
    df : pd.DataFrame
        Same as agg_df plus a 'cluster' column (1…n_clusters).
    centroids_df : pd.DataFrame
        DataFrame with one row per cluster, columns:
        ['cluster','centroid_lat','centroid_lng','net_balance','departures','arrivals','station_name'].
    """
    # 1) copy + guard
    df = agg_df.dropna(subset=["lat","lng","departures","arrivals"]).copy()
    if df.empty:
        return df, pd.DataFrame()

    # 2) calculate net_balance (already in agg_df, but just to be safe)
    df['net_balance'] = df['departures'] - df['arrivals']

    # 3) scale spatial coords
    X = df[['lat','lng']].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4) run KMeans → labels 0…K-1 → shift to 1…K
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled) + 1
    df['cluster'] = labels

    # 5) recover centroids in original lat/lng
    centers = scaler.inverse_transform(km.cluster_centers_)
    centroids_df = pd.DataFrame({
        'cluster': list(range(1, n_clusters+1)),
        'centroid_lat': centers[:,0],
        'centroid_lng': centers[:,1]
    })

    # 6) compute cluster stats
    stats = (
        df.groupby('cluster')
          .agg(
            net_balance=('net_balance','sum'),
            departures=('departures','sum'),
            arrivals=('arrivals','sum'),
            station_name=('station_name','count')
          )
          .reset_index()
    )

    # 7) merge so every cluster 1…K appears
    centroids_df = (
        centroids_df
        .merge(stats, on='cluster', how='left')
        .fillna({'net_balance':0,'departures':0,'arrivals':0,'station_name':0})
    )

    return df, centroids_df

@st.cache_data
def aggregate_weekly_by_station(combined, year, week):
    # 1) copy and ensure datetime64
    df = combined.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 2) now .dt will work
    df['iso_year'] = df['date'].dt.isocalendar().year
    df['iso_week'] = df['date'].dt.isocalendar().week

    # 3) filter the target week
    wk = df[(df['iso_year'] == year) & (df['iso_week'] == week)]
    if wk.empty:
        return wk.iloc[0:0]

    # 4) aggregate
    agg = (
        wk
        .groupby(['station_name','lat','lng'], as_index=False)
        .agg({'departures':'sum','arrivals':'sum'})
    )
    agg['net_balance'] = agg['departures'] - agg['arrivals']
    return agg

@st.cache_data
def prepare_weekly_time_series_data(combined):
    df2 = combined.copy()
    # 1) ensure datetime64
    df2['date'] = pd.to_datetime(df2['date'])

    # 2) attach ISO-week Period
    df2['year_week'] = df2['date'].dt.to_period('W')

    # 3) get the true min date as a Python date
    min_date = df2['date'].min().date()

    # 4) keep only weeks whose start_time.date() ≥ min_date
    valid_weeks = [
        p for p in sorted(df2['year_week'].unique())
        if p.start_time.date() >= min_date   # ← both sides are date()
    ]

    series_list = []
    for yw in valid_weeks:
        year, weeknum = yw.year, yw.week
        agg = aggregate_weekly_by_station(combined, year, weeknum)
        ser = agg.groupby('station_name')['net_balance'].sum()
        ser.name = yw
        series_list.append(ser)

    if not series_list:
        return pd.DataFrame(), pd.DataFrame()

    pivot = pd.concat(series_list, axis=1).fillna(0)
    station_coords = (
        combined
        .drop_duplicates('station_name')[['station_name','lat','lng']]
        .reset_index(drop=True)
    )

    return pivot, station_coords

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
    dates = sorted(combined["date"].unique())

    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Static Map", "Timeline Map"])

    if mode == "Static Map":
        # ── 1) Main Map ────────────────────────────────────────
        st.sidebar.header("Main Map Date")
        sel_date = st.sidebar.date_input(
            "", value=dates[0], min_value=dates[0], max_value=dates[-1]
        )
        st.subheader(f"Main Map — {sel_date.strftime('%d/%m/%y')}")
        df_day = combined[combined["date"] == sel_date]
        if df_day.empty:
            st.warning("No data for this date.")
        else:
            fig = create_map_visualization(
                df_day,
                st.sidebar.slider("Radius (m):", 100, 200, 100, 10),
                [
                    name
                    for name, chk in zip(
                        ["More departures", "More arrivals", "Balanced"],
                        [
                            st.sidebar.checkbox("More departures", True),
                            st.sidebar.checkbox("More arrivals", True),
                            st.sidebar.checkbox("Balanced", True),
                        ],
                    )
                    if chk
                ],
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── 2) Weekly Time-Series Clustering ──────────────────────
        st.subheader("Weekly Time Series Clustering")
        ts_k = st.selectbox("Clusters:", list(range(1, 7)), index=2, key="ts_k")
        pivot_wk, coords = prepare_weekly_time_series_data(combined)
        if not pivot_wk.empty:
            ts_res, _ = perform_time_series_clustering(pivot_wk, ts_k, coords)
            st.plotly_chart(create_time_series_cluster_map(ts_res), use_container_width=True)
            st.dataframe(ts_res.groupby("cluster").size().reset_index(name="count"))
        else:
            st.warning("Insufficient weekly data.")

    else:
        # ── Timeline mode ─────────────────────────────────────────
        st.sidebar.header("Timeline Options")
        radius_m = st.sidebar.slider("Clustering radius (m):", 100, 200, 100, 10)
        show_dep = st.sidebar.checkbox("More departures", True)
        show_arr = st.sidebar.checkbox("More arrivals", True)
        show_bal = st.sidebar.checkbox("Balanced", True)
        categories = [
            name
            for name, chk in zip(
                ["More departures", "More arrivals", "Balanced"],
                [show_dep, show_arr, show_bal],
            )
            if chk
        ]

        st.sidebar.header("Select Date Range")
        start_date = st.sidebar.date_input(
            "Start date:", value=dates[0], min_value=dates[0], max_value=dates[-1]
        )
        end_date = st.sidebar.date_input(
            "End date:",
            value=min(dates[-1], dates[0] + pd.Timedelta(days=6)),
            min_value=dates[0], max_value=dates[-1],
        )

        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        # ── 1) Animated Timeline Map ────────────────────────────
        st.subheader(
            f"Timeline Map — {start_date.strftime('%d/%m/%y')} to {end_date.strftime('%d/%m/%y')}"
        )
        try:
            fig_timeline = create_timeline_map(combined, start_date, end_date, radius_m, categories)
            st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_map")
        except Exception as e:
            st.error(f"Error creating timeline map: {e}")

        # ── Daily Spatial Auto K-Means Clustering ───────────────────────────────
        st.subheader("Daily Spatial K-Means Clustering (Auto-K)")

        daily_date = st.date_input(
            "Select date for spatial clustering:",
            value=dates[0],
            min_value=dates[0],
            max_value=dates[-1],
            key="auto_daily_date"
        )

        df_day = combined[combined["date"] == daily_date].copy()
        df_day = df_day.dropna(subset=["lat", "lng", "departures", "arrivals"])

        if df_day.empty or len(df_day) < 3:
            st.warning(f"No sufficient data for {daily_date}.")
        else:
            # Standardize coordinates
            coords = df_day[["lat", "lng"]].values
            scaler = StandardScaler()
            coords_scaled = scaler.fit_transform(coords)

            # Calculate inertia for different k values
            inertias = []
            max_k = min(15, len(df_day) - 1)
            k_range = range(2, max_k + 1)

            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(coords_scaled)
                inertias.append(km.inertia_)

            # Use KneeLocator to find the elbow
            kn = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
            best_k = kn.elbow if kn.elbow else 5  # fallback to 5 if no knee found

            # Fit final model with best_k
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            df_day["cluster"] = kmeans.fit_predict(coords_scaled) + 1

            # Get centroids in original scale
            centroids = scaler.inverse_transform(kmeans.cluster_centers_)
            centroids_df = pd.DataFrame({
                "cluster": range(1, best_k + 1),
                "centroid_lat": centroids[:, 0],
                "centroid_lng": centroids[:, 1]
            })
            net_balance_df = df_day.groupby("cluster")["net_balance"].sum().reset_index()
            centroids_df = centroids_df.merge(net_balance_df, on="cluster", how="left")
            # Calculate net balance
            df_day["net_balance"] = df_day["departures"] - df_day["arrivals"]

            fig_daily_spatial = create_spatial_cluster_map(df_day, centroids_df)
            st.plotly_chart(fig_daily_spatial, use_container_width=True)

            stats = df_day.groupby('cluster').agg(
                Stations=('station_name', 'count'),
                NetBalance=('net_balance', 'sum'),
                Departures=('departures', 'sum'),
                Arrivals=('arrivals', 'sum')
            ).reset_index()
            st.markdown(f"**Daily Spatial Cluster Stats (k={best_k}):**")
            st.dataframe(stats, use_container_width=True)

        # ── 3) Weather Data ───────────────────────────────────────
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
                    fig_temp.update_layout(
                        title="Daily Max Temperature",
                        xaxis_title="Date",
                        yaxis_title="Temp (°C)",
                        height=300,
                    )
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
                    fig_hum.update_layout(
                        title="Daily Max Humidity",
                        xaxis_title="Date",
                        yaxis_title="Humidity (%)",
                        height=300,
                    )
                    st.plotly_chart(fig_hum, use_container_width=True)
            else:
                st.warning("No weather data available for this date range.")
        except Exception as e:
            st.error(f"Error loading weather data: {e}")
if __name__ == "__main__":
    main()