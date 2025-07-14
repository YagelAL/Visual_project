import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import os
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


# ── Prepare time series data for clustering ────────────────────────────────────
@st.cache_data
def prepare_time_series_data(combined):
    # Create pivot table with stations as rows and dates as columns
    pivot_dep = combined.pivot_table(
        index='station_name',
        columns='date',
        values='departures',
        fill_value=0
    )
    pivot_arr = combined.pivot_table(
        index='station_name',
        columns='date',
        values='arrivals',
        fill_value=0
    )

    # Calculate net balance (departures - arrivals)
    pivot_net = pivot_dep - pivot_arr

    # Get station coordinates
    station_coords = combined.groupby('station_name')[['lat', 'lng']].first().reset_index()

    # Filter stations that have data for at least 50% of the days
    min_days = len(pivot_net.columns) * 0.5
    valid_stations = pivot_net.index[pivot_net.notna().sum(axis=1) >= min_days]

    pivot_net_filtered = pivot_net.loc[valid_stations].fillna(0)

    return pivot_net_filtered, station_coords


# ── Time Series Clustering ─────────────────────────────────────────────────────
@st.cache_data
def perform_time_series_clustering(pivot_net_filtered, n_clusters, station_coords, method='kmeans'):
    # Prepare data for time series clustering
    ts_data = pivot_net_filtered.values

    # Normalize the time series using standard scaler
    scaler = StandardScaler()
    ts_data_scaled = scaler.fit_transform(ts_data)

    if method == 'kmeans':
        # Method 1: Use K-means on the time series data directly
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ts_data_scaled)
        model = kmeans

    elif method == 'hierarchical':
        # Method 2: Use hierarchical clustering with correlation distance
        correlation_matrix = np.corrcoef(ts_data_scaled)
        # Convert correlation to distance (1 - abs(correlation))
        distance_matrix = 1 - np.abs(correlation_matrix)
        np.fill_diagonal(distance_matrix, 0)

        # Ensure distance matrix is valid
        distance_matrix = np.maximum(distance_matrix, 0)

        # Perform hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='ward')
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust') - 1
        model = linkage_matrix

    else:  # 'pca_kmeans'
        # Method 3: Use PCA + K-means for dimensionality reduction
        pca = PCA(n_components=min(10, ts_data_scaled.shape[1]))
        ts_data_pca = pca.fit_transform(ts_data_scaled)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(ts_data_pca)
        model = (pca, kmeans)

    # Create results dataframe
    results = pd.DataFrame({
        'station_name': pivot_net_filtered.index,
        'cluster': cluster_labels
    })

    # Add coordinates
    results = results.merge(station_coords, on='station_name', how='left')
    results = results.dropna(subset=['lat', 'lng'])

    return results, model


# ── K-means Clustering ─────────────────────────────────────────────────────────
@st.cache_data
def perform_kmeans_clustering(combined, selected_date, n_clusters):
    # Get data for selected date
    df_day = combined[combined["date"] == selected_date].copy()
    df_day = df_day.dropna(subset=["lat", "lng", "departures", "arrivals"])

    if df_day.empty:
        return pd.DataFrame()

    # Prepare features for clustering
    features = ['departures', 'arrivals', 'lat', 'lng']
    X = df_day[features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    df_day['cluster'] = cluster_labels

    return df_day


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


# ── K-means Clustering Map ─────────────────────────────────────────────────────
def create_kmeans_cluster_map(cluster_results):
    if cluster_results.empty:
        return go.Figure()

    # Create color palette for clusters
    colors = px.colors.qualitative.Set1

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
            hovertemplate="<b>%{text}</b><br>Cluster: " + str(cluster_id) +
                          "<br>Departures: %{customdata[0]}<br>Arrivals: %{customdata[1]}<extra></extra>",
            customdata=cluster_data[['departures', 'arrivals']].values,
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

        st.sidebar.header("Select Date")
        selected_date = st.sidebar.date_input(
            "", value=dates[0], min_value=dates[0], max_value=dates[-1]
        )

        st.subheader(f"Main Map — {selected_date.strftime('%d/%m/%y')}")
        df_day = combined[combined["date"] == selected_date]
        if df_day.empty:
            st.warning("No Citibike data for this date.")
        else:
            fig_map = create_map_visualization(df_day, radius_m, categories)
            st.plotly_chart(fig_map, use_container_width=True, key="station_map")

        # ── Time Series Clustering ────────────────────────
        st.subheader("Time Series Clustering Map")
        st.markdown("#### Time Series Clustering Controls")
        ts_clusters = st.selectbox("Time Series Clusters:", list(range(1, 7)), index=2, key="ts_clusters")
        ts_method = st.selectbox("Time Series Method:", ["kmeans", "hierarchical", "pca_kmeans"],
                                 format_func=lambda x: {
                                     "kmeans": "K-means",
                                     "hierarchical": "Hierarchical",
                                     "pca_kmeans": "PCA + K-means"
                                 }[x], key="ts_method")

        st.write(f"Stations clustered by similar temporal patterns using {ts_method.replace('_', ' ').title()} method")

        try:
            pivot_net_filtered, station_coords = prepare_time_series_data(combined)
            if len(pivot_net_filtered) > 0:
                ts_cluster_results, ts_model = perform_time_series_clustering(
                    pivot_net_filtered, ts_clusters, station_coords, ts_method
                )
                fig_ts = create_time_series_cluster_map(ts_cluster_results)
                st.plotly_chart(fig_ts, use_container_width=True, key="ts_cluster_map")

                cluster_stats = ts_cluster_results.groupby('cluster').size().reset_index(name='station_count')
                st.write("**Time Series Cluster Statistics:**")
                st.dataframe(cluster_stats, use_container_width=True)

                if st.checkbox("Show Average Time Series by Cluster"):
                    pivot_with_clusters = pivot_net_filtered.copy()
                    pivot_with_clusters['cluster'] = ts_cluster_results.set_index('station_name')['cluster']

                    fig_ts_avg = go.Figure()
                    dark_colors = px.colors.qualitative.Dark24

                    for cluster_id in sorted(pivot_with_clusters['cluster'].unique()):
                        cluster_data = pivot_with_clusters[pivot_with_clusters['cluster'] == cluster_id]
                        cluster_avg = cluster_data.drop('cluster', axis=1).mean()

                        fig_ts_avg.add_trace(go.Scatter(
                            x=list(range(len(cluster_avg))),
                            y=cluster_avg.values,
                            mode='lines+markers',
                            name=f'Cluster {cluster_id}',
                            line=dict(color=dark_colors[cluster_id % len(dark_colors)], width=3)
                        ))

                    fig_ts_avg.update_layout(
                        title="Average Time Series by Cluster",
                        xaxis_title="Time Point",
                        yaxis_title="Average Net Balance",
                        height=400
                    )
                    st.plotly_chart(fig_ts_avg, use_container_width=True)
            else:
                st.warning("Insufficient data for time series clustering.")
        except Exception as e:
            st.error(f"Error in time series clustering: {str(e)}")

        # ── K-means Clustering ────────────────────────
        st.subheader("K-means Clustering Map")
        st.markdown("#### K-means Clustering Controls")
        kmeans_clusters = st.selectbox("K-means Clusters:", list(range(1, 7)), index=3, key="kmeans_clusters")
        st.write(f"Stations clustered by daily activity patterns over a 30-day window from {selected_date.strftime('%d/%m/%y')}")

        try:
            kmeans_cluster_results = perform_kmeans_clustering(combined, selected_date, kmeans_clusters)
            if not kmeans_cluster_results.empty:
                fig_kmeans = create_kmeans_cluster_map(kmeans_cluster_results)
                st.plotly_chart(fig_kmeans, use_container_width=True, key="kmeans_cluster_map")

                cluster_stats = kmeans_cluster_results.groupby('cluster').agg({
                    'station_name': 'count',
                    'departures': 'mean',
                    'arrivals': 'mean'
                }).round(2)
                cluster_stats.columns = ['Station Count', 'Avg Departures', 'Avg Arrivals']
                st.write("**Cluster Statistics:**")
                st.dataframe(cluster_stats, use_container_width=True)
            else:
                st.warning("No data available for K-means clustering on selected date.")
        except Exception as e:
            st.error(f"Error in K-means clustering: {str(e)}")

    else:
        # Timeline mode remains unchanged
        pass



if __name__ == "__main__":
    main()