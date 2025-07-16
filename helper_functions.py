import requests
import os
from libpysal.weights import KNN
import geopandas as gpd
from sklearn.cluster import DBSCAN, KMeans
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from datetime import date
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
    """Load and combine all processed monthly data files"""
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

def create_map_visualization(df_day, radius_m, categories):
    """Create static map visualization with DBSCAN clustering"""
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


def create_timeline_map(combined, start_date, end_date, radius_m, categories):
    """Create animated timeline map visualization"""
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


def predict_peak_periods(combined, selected_date):
    """Predict peak/off-peak periods for all stations"""
    df_day = combined[combined['date'] == selected_date].copy()
    df_day['total_activity'] = df_day['departures'] + df_day['arrivals']

    if df_day.empty:
        return go.Figure()

    # Define peak threshold (top 25% of activity)
    peak_threshold = df_day['total_activity'].quantile(0.75)
    df_day['period_type'] = df_day['total_activity'].apply(
        lambda x: 'Peak' if x >= peak_threshold else 'Off-Peak'
    )

    # Create map
    fig = go.Figure()

    colors = {'Peak': 'red', 'Off-Peak': 'blue'}
    sizes = {'Peak': 12, 'Off-Peak': 8}

    for period_type in ['Peak', 'Off-Peak']:
        data = df_day[df_day['period_type'] == period_type]

        fig.add_trace(go.Scattermapbox(
            lat=data['lat'],
            lon=data['lng'],
            mode='markers',
            marker=dict(
                size=sizes[period_type],
                color=colors[period_type],
                opacity=0.7
            ),
            text=data['station_name'],
            hovertemplate=f"<b>%{{text}}</b><br>{period_type} Period<br>Total Activity: %{{customdata}}<extra></extra>",
            customdata=data['total_activity'],
            name=period_type
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

    return fig


def create_weather_impact_analysis(combined, weather_data):
    """Analyze weather impact on station usage"""
    if weather_data.empty:
        return go.Figure(), "No weather data available"

    # Merge weather with bike data
    combined_weather = combined.merge(weather_data, on='date', how='inner')

    if combined_weather.empty:
        return go.Figure(), "No matching weather data"

    # Calculate daily totals per station
    station_weather = (
        combined_weather
        .groupby(['station_name', 'date', 'temperature', 'humidity'])
        .agg({
            'departures': 'sum',
            'arrivals': 'sum',
            'lat': 'first',
            'lng': 'first'
        })
        .reset_index()
    )

    station_weather['total_rides'] = station_weather['departures'] + station_weather['arrivals']

    # Calculate correlation per station
    weather_correlations = []
    for station in station_weather['station_name'].unique():
        station_data = station_weather[station_weather['station_name'] == station]

        if len(station_data) > 5:  # Need minimum data points
            temp_corr = np.corrcoef(station_data['temperature'], station_data['total_rides'])[0, 1]
            humidity_corr = np.corrcoef(station_data['humidity'], station_data['total_rides'])[0, 1]

            weather_correlations.append({
                'station_name': station,
                'lat': station_data['lat'].iloc[0],
                'lng': station_data['lng'].iloc[0],
                'temp_correlation': temp_corr if not np.isnan(temp_corr) else 0,
                'humidity_correlation': humidity_corr if not np.isnan(humidity_corr) else 0,
                'weather_sensitivity': abs(temp_corr) + abs(humidity_corr) if not (
                        np.isnan(temp_corr) or np.isnan(humidity_corr)) else 0
            })

    if not weather_correlations:
        return go.Figure(), "Insufficient data for weather analysis"

    corr_df = pd.DataFrame(weather_correlations)

    # Create scatter plot
    fig = go.Figure()

    # Color by weather sensitivity
    fig.add_trace(go.Scattermapbox(
        lat=corr_df['lat'],
        lon=corr_df['lng'],
        mode='markers',
        marker=dict(
            size=10,
            color=corr_df['weather_sensitivity'],
            colorscale='Viridis',
            colorbar=dict(title="Weather Sensitivity"),
            opacity=0.8
        ),
        text=corr_df['station_name'],
        hovertemplate="<b>%{text}</b><br>" +
                      "Weather Sensitivity: %{marker.color:.3f}<br>" +
                      "Temp Correlation: %{customdata[0]:.3f}<br>" +
                      "Humidity Correlation: %{customdata[1]:.3f}<extra></extra>",
        customdata=corr_df[['temp_correlation', 'humidity_correlation']],
        name="Stations"
    ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=10
        ),
        height=500,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig, f"Weather analysis completed for {len(corr_df)} stations"


def detect_station_anomalies(combined, selected_date, z_threshold=2.5):
    """Detect anomalous station behavior using Z-score for entire month"""
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


def create_daily_rides_bar_chart(combined, start_date, end_date):
    """Create a bar chart showing total rides per day for the selected date range"""
    # Filter data for the date range
    filtered_data = combined[
        (combined['date'] >= start_date) &
        (combined['date'] <= end_date)
        ].copy()

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


def create_parallel_coordinates_plot(combined, selected_date):
    """
    Create a parallel coordinates plot showing multiple station metrics
    for comprehensive multi-dimensional analysis
    """
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure(), "No data available for the selected month"

    # Calculate comprehensive station metrics
    station_metrics = []

    for station_name in month_data['station_name'].unique():
        station_data = month_data[month_data['station_name'] == station_name]

        if len(station_data) > 0:
            # Basic metrics
            daily_departures = station_data['departures'].values
            daily_arrivals = station_data['arrivals'].values
            daily_balances = daily_departures - daily_arrivals
            total_activity = daily_departures + daily_arrivals

            # Temporal analysis
            station_data_with_dow = station_data.copy()
            station_data_with_dow['day_of_week'] = pd.to_datetime(station_data_with_dow['date']).dt.dayofweek
            station_data_with_dow['is_weekend'] = station_data_with_dow['day_of_week'].isin([5, 6])

            weekday_data = station_data_with_dow[~station_data_with_dow['is_weekend']]
            weekend_data = station_data_with_dow[station_data_with_dow['is_weekend']]

            # Calculate metrics
            avg_departures = np.mean(daily_departures)
            avg_arrivals = np.mean(daily_arrivals)
            avg_net_balance = np.mean(daily_balances)
            avg_total_activity = np.mean(total_activity)

            # Variability metrics
            balance_volatility = np.std(daily_balances)
            activity_volatility = np.std(total_activity)

            # Range metrics
            balance_range = np.max(daily_balances) - np.min(daily_balances)
            activity_range = np.max(total_activity) - np.min(total_activity)

            # Trend analysis
            if len(daily_balances) > 1:
                balance_trend = np.polyfit(range(len(daily_balances)), daily_balances, 1)[0]
                activity_trend = np.polyfit(range(len(total_activity)), total_activity, 1)[0]
            else:
                balance_trend = 0
                activity_trend = 0

            # Peak analysis
            max_departures = np.max(daily_departures)
            max_arrivals = np.max(daily_arrivals)
            max_deficit = abs(np.min(daily_balances)) if np.min(daily_balances) < 0 else 0
            max_surplus = np.max(daily_balances) if np.max(daily_balances) > 0 else 0

            # Weekday vs Weekend difference
            weekday_weekend_diff = 0
            if not weekday_data.empty and not weekend_data.empty:
                weekday_avg = np.mean(weekday_data['departures'] - weekday_data['arrivals'])
                weekend_avg = np.mean(weekend_data['departures'] - weekend_data['arrivals'])
                weekday_weekend_diff = abs(weekday_avg - weekend_avg)

            # Consistency measure (inverse of coefficient of variation)
            consistency = 100 - (balance_volatility / (abs(avg_net_balance) + 1) * 100)

            # Activity intensity (normalized by max possible activity)
            activity_intensity = avg_total_activity / (np.max(total_activity) + 1) * 100

            # Imbalance ratio (how skewed towards departures or arrivals)
            imbalance_ratio = abs(avg_departures - avg_arrivals) / (avg_departures + avg_arrivals + 1) * 100

            metrics = {
                'Station': station_name,
                'Lat': station_data['lat'].iloc[0],
                'Lng': station_data['lng'].iloc[0],
                'Avg_Departures': avg_departures,
                'Avg_Arrivals': avg_arrivals,
                'Avg_Net_Balance': avg_net_balance,
                'Avg_Total_Activity': avg_total_activity,
                'Balance_Volatility': balance_volatility,
                'Activity_Volatility': activity_volatility,
                'Balance_Range': balance_range,
                'Activity_Range': activity_range,
                'Balance_Trend': balance_trend,
                'Activity_Trend': activity_trend,
                'Max_Departures': max_departures,
                'Max_Arrivals': max_arrivals,
                'Max_Deficit': max_deficit,
                'Max_Surplus': max_surplus,
                'Weekday_Weekend_Diff': weekday_weekend_diff,
                'Consistency': consistency,
                'Activity_Intensity': activity_intensity,
                'Imbalance_Ratio': imbalance_ratio
            }

            station_metrics.append(metrics)

    if not station_metrics:
        return go.Figure(), "No station metrics calculated"

    # Create DataFrame
    df_metrics = pd.DataFrame(station_metrics)

    # Limit to reasonable number of stations for visualization
    if len(df_metrics) > 150:
        df_metrics = df_metrics.sample(n=150, random_state=42)
        st.info(f"📊 Sampled {len(df_metrics)} stations for parallel coordinates visualization")

    # Select key metrics for parallel coordinates
    parallel_metrics = [
        'Avg_Net_Balance',
        'Avg_Total_Activity',
        'Balance_Volatility',
        'Activity_Intensity',
        'Imbalance_Ratio',
        'Consistency',
        'Balance_Range',
        'Weekday_Weekend_Diff'
    ]

    # Create color coding based on station type
    df_metrics['Station_Type'] = df_metrics['Avg_Net_Balance'].apply(
        lambda x: 'Departure Hub' if x > 5 else 'Arrival Hub' if x < -5 else 'Balanced'
    )

    # Color mapping
    color_map = {
        'Departure Hub': '#FF6B6B',
        'Arrival Hub': '#4ECDC4',
        'Balanced': '#45B7D1'
    }

    df_metrics['Color'] = df_metrics['Station_Type'].map(color_map)

    # Create parallel coordinates plot
    fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df_metrics['Avg_Net_Balance'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(
                title="Average Net Balance",
                titleside="top",
                tickmode="linear",
                tick0=df_metrics['Avg_Net_Balance'].min(),
                dtick=(df_metrics['Avg_Net_Balance'].max() - df_metrics['Avg_Net_Balance'].min()) / 6
            )
        ),
        dimensions=[
            dict(
                range=[df_metrics[metric].min(), df_metrics[metric].max()],
                label=metric.replace('_', ' '),
                values=df_metrics[metric]
            ) for metric in parallel_metrics
        ],
        labelangle=45,
        labelside="bottom"
    )
    )

    fig.update_layout(
        title=f"Station Performance Parallel Coordinates - {selected_month.strftime('%B %Y')}",
        height=700,
        margin=dict(l=80, r=80, t=80, b=120),
        font=dict(size=12),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    return fig, f"Parallel coordinates plot created for {len(df_metrics)} stations"


def create_interactive_parallel_coordinates(combined, selected_date):
    """
    Create an interactive parallel coordinates plot with Plotly Express
    """
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure(), "No data available for the selected month"

    # Calculate station metrics (same as above but simplified)
    station_metrics = []

    for station_name in month_data['station_name'].unique():
        station_data = month_data[month_data['station_name'] == station_name]

        if len(station_data) > 2:  # Need minimum data
            daily_departures = station_data['departures'].values
            daily_arrivals = station_data['arrivals'].values
            daily_balances = daily_departures - daily_arrivals
            total_activity = daily_departures + daily_arrivals

            metrics = {
                'Station': station_name,
                'Net_Balance': np.mean(daily_balances),
                'Total_Activity': np.mean(total_activity),
                'Volatility': np.std(daily_balances),
                'Range': np.max(daily_balances) - np.min(daily_balances),
                'Peak_Departures': np.max(daily_departures),
                'Peak_Arrivals': np.max(daily_arrivals),
                'Station_Type': 'Departure Hub' if np.mean(daily_balances) > 5 else 'Arrival Hub' if np.mean(
                    daily_balances) < -5 else 'Balanced'
            }

            station_metrics.append(metrics)

    if not station_metrics:
        return go.Figure(), "No station metrics calculated"

    df_metrics = pd.DataFrame(station_metrics)

    # Limit stations for performance
    if len(df_metrics) > 100:
        df_metrics = df_metrics.sample(n=100, random_state=42)

    # Create parallel coordinates plot with Plotly Express
    fig = px.parallel_coordinates(
        df_metrics,
        dimensions=['Net_Balance', 'Total_Activity', 'Volatility', 'Range', 'Peak_Departures', 'Peak_Arrivals'],
        color='Net_Balance',
        color_continuous_scale='RdYlBu_r',
        labels={
            'Net_Balance': 'Net Balance',
            'Total_Activity': 'Total Activity',
            'Volatility': 'Volatility',
            'Range': 'Range',
            'Peak_Departures': 'Peak Departures',
            'Peak_Arrivals': 'Peak Arrivals'
        },
        title=f"Station Metrics Parallel Coordinates - {selected_month.strftime('%B %Y')}"
    )

    fig.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
        font=dict(size=12)
    )

    return fig, f"Interactive parallel coordinates created for {len(df_metrics)} stations"


def create_parallel_coordinates_with_clustering(combined, selected_date, n_clusters=4):
    """
    Create parallel coordinates plot with K-means clustering overlay
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure(), "No data available for the selected month"

    # Calculate station metrics
    station_metrics = []

    for station_name in month_data['station_name'].unique():
        station_data = month_data[month_data['station_name'] == station_name]

        if len(station_data) > 2:
            daily_departures = station_data['departures'].values
            daily_arrivals = station_data['arrivals'].values
            daily_balances = daily_departures - daily_arrivals
            total_activity = daily_departures + daily_arrivals

            metrics = {
                'Station': station_name,
                'Net_Balance': np.mean(daily_balances),
                'Total_Activity': np.mean(total_activity),
                'Volatility': np.std(daily_balances),
                'Range': np.max(daily_balances) - np.min(daily_balances),
                'Peak_Departures': np.max(daily_departures),
                'Peak_Arrivals': np.max(daily_arrivals),
                'Consistency': 100 - (np.std(daily_balances) / (abs(np.mean(daily_balances)) + 1) * 100)
            }

            station_metrics.append(metrics)

    if not station_metrics:
        return go.Figure(), "No station metrics calculated"

    df_metrics = pd.DataFrame(station_metrics)

    # Limit stations for performance
    if len(df_metrics) > 100:
        df_metrics = df_metrics.sample(n=100, random_state=42)

    # Prepare data for clustering
    feature_columns = ['Net_Balance', 'Total_Activity', 'Volatility', 'Range', 'Peak_Departures', 'Peak_Arrivals',
                       'Consistency']
    features = df_metrics[feature_columns].values

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    df_metrics['Cluster'] = cluster_labels + 1  # Start from 1 instead of 0

    # Create parallel coordinates plot with cluster coloring
    fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df_metrics['Cluster'],
            colorscale='Set3',
            showscale=True,
            colorbar=dict(
                title="Cluster",
                titleside="top",
                tickmode="array",
                tickvals=list(range(1, n_clusters + 1)),
                ticktext=[f"Cluster {i}" for i in range(1, n_clusters + 1)]
            )
        ),
        dimensions=[
            dict(
                range=[df_metrics[col].min(), df_metrics[col].max()],
                label=col.replace('_', ' '),
                values=df_metrics[col]
            ) for col in feature_columns
        ],
        labelangle=45,
        labelside="bottom"
    )
    )

    fig.update_layout(
        title=f"Station Clustering Parallel Coordinates - {selected_month.strftime('%B %Y')} ({n_clusters} Clusters)",
        height=700,
        margin=dict(l=80, r=80, t=80, b=120),
        font=dict(size=12)
    )

    # Calculate cluster statistics
    cluster_stats = df_metrics.groupby('Cluster').agg({
        'Net_Balance': ['mean', 'std'],
        'Total_Activity': ['mean', 'std'],
        'Volatility': ['mean', 'std']
    }).round(2)

    return fig, f"Clustered parallel coordinates created for {len(df_metrics)} stations in {n_clusters} clusters", cluster_stats

# Function to add to helper_functions.py


def render_parallel_coordinates_section(combined, selected_date):
    """
    Render the parallel coordinates analysis section
    """
    st.subheader("📊 Parallel Coordinates Analysis")

    # Create tabs for different parallel coordinates views
    tab1, tab2, tab3 = st.tabs(["📈 Basic Metrics", "🎯 Interactive View", "🔍 Clustered Analysis"])

    with tab1:
        st.markdown("**Multi-dimensional station performance visualization**")

        with st.spinner("Creating parallel coordinates plot..."):
            try:
                fig, message = create_parallel_coordinates_plot(combined, selected_date)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(message)

                    # Add explanation
                    with st.expander("📖 How to Read This Plot"):
                        st.markdown("""
                        **Parallel Coordinates Plot Guide:**

                        - **Each line** represents one bike station
                        - **Each vertical axis** represents a different metric
                        - **Line color** indicates the average net balance (red = more departures, blue = more arrivals)
                        - **Parallel lines** indicate stations with similar patterns
                        - **Intersecting lines** show stations that rank differently across metrics

                        **Key Insights:**
                        - Look for clusters of similar lines
                        - Identify outlier stations that deviate from patterns
                        - Compare how stations perform across different dimensions
                        """)
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error creating parallel coordinates plot: {e}")

    with tab2:
        st.markdown("**Interactive parallel coordinates with simplified metrics**")

        with st.spinner("Creating interactive plot..."):
            try:
                fig, message = create_interactive_parallel_coordinates(combined, selected_date)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(message)
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error creating interactive plot: {e}")

    with tab3:
        st.markdown("**Parallel coordinates with K-means clustering**")

        n_clusters = st.slider("Number of Clusters:", 2, 6, 4, key="parallel_clusters")

        with st.spinner("Creating clustered analysis..."):
            try:
                fig, message, cluster_stats = create_parallel_coordinates_with_clustering(combined, selected_date,
                                                                                          n_clusters)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(message)

                    # Show cluster statistics
                    st.subheader("Cluster Statistics")
                    st.dataframe(cluster_stats)

                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Error creating clustered analysis: {e}")


def create_spider_plot_for_month(combined, selected_date):
    """Create a spider scatter plot showing station metrics"""
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

    # Take top 200 stations by total activity
    station_monthly['total_activity'] = station_monthly['departures'] + station_monthly['arrivals']
    top_stations = station_monthly.nlargest(200, 'total_activity')

    # Min-max scale the data for better spider arm visibility
    top_stations['total_activity_scaled'] = (top_stations['total_activity'] - top_stations['total_activity'].min()) / (
                top_stations['total_activity'].max() - top_stations['total_activity'].min())
    top_stations['departure_ratio_scaled'] = (top_stations['departure_ratio'] - top_stations[
        'departure_ratio'].min()) / (top_stations['departure_ratio'].max() - top_stations['departure_ratio'].min())

    # Calculate key metrics
    top_stations['net_balance'] = top_stations['departures'] - top_stations['arrivals']
    top_stations['arrival_ratio'] = top_stations['arrivals'] / top_stations['total_activity']
    top_stations['balance_magnitude'] = abs(top_stations['net_balance'])

    # Min-max scale all metrics for spider arms (0-1 scale)
    metrics = ['total_activity', 'departure_ratio', 'arrival_ratio', 'balance_magnitude']
    for metric in metrics:
        min_val = top_stations[metric].min()
        max_val = top_stations[metric].max()
        if max_val > min_val:
            top_stations[f'{metric}_norm'] = (top_stations[metric] - min_val) / (max_val - min_val)
        else:
            top_stations[f'{metric}_norm'] = 0.5

    # Create scatter plot
    fig = go.Figure()

    # Define spider arm angles (4 directions: up, right, down, left)
    angles = [np.pi / 2, 0, -np.pi / 2, np.pi]  # 90, 0, -90, 180 degrees
    metric_names = ['Total Activity', 'Departure Ratio', 'Arrival Ratio', 'Balance Magnitude']

    # Color stations by net balance
    top_stations['color'] = top_stations['net_balance'].apply(
        lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
    )

    # Create spider arms for each station
    for _, station in top_stations.iterrows():
        station_x = station['total_activity_scaled']
        station_y = station['departure_ratio_scaled']

        # Get normalized values for spider arms
        values = [station[f'{metric}_norm'] for metric in metrics]

        # Create spider arms
        for i, (angle, value, metric_name) in enumerate(zip(angles, values, metric_names)):
            # Calculate arm length (scale by value) - make them more visible
            arm_length = 100 + value * 300  # Longer base length + more scaling

            # Calculate end point
            end_x = station_x + arm_length * np.cos(angle)
            end_y = station_y + arm_length * np.sin(angle)

            # Add spider arm line
            fig.add_trace(go.Scatter(
                x=[station_x, end_x],
                y=[station_y, end_y],
                mode='lines',
                line=dict(color=station['color'], width=1.5),
                hoverinfo='skip',
                showlegend=False
            ))

            # Add small marker at end of arm
            fig.add_trace(go.Scatter(
                x=[end_x],
                y=[end_y],
                mode='markers',
                marker=dict(size=3, color=station['color'], symbol='circle'),
                hoverinfo='skip',
                showlegend=False
            ))

    # Add center points (stations)
    for color_val, color_name in [('#FF6B6B', 'More Departures'), ('#69DB7C', 'More Arrivals'),
                                  ('#4DABF7', 'Balanced')]:
        color_stations = top_stations[top_stations['color'] == color_val]

        if len(color_stations) > 0:
            fig.add_trace(go.Scatter(
                x=color_stations['total_activity_scaled'],
                y=color_stations['departure_ratio_scaled'],
                mode='markers',
                marker=dict(size=6, color=color_val, symbol='circle'),
                text=color_stations['station_name'],
                hovertemplate='<b>%{text}</b><br>' +
                              'Total Activity: %{customdata[0]:.0f}<br>' +
                              'Departure Ratio: %{customdata[1]:.2f}<br>' +
                              'Net Balance: %{customdata[2]:.0f}<br>' +
                              'Balance Magnitude: %{customdata[3]:.0f}<extra></extra>',
                customdata=color_stations[
                    ['total_activity', 'departure_ratio', 'net_balance', 'balance_magnitude']].values,
                name=color_name,
                showlegend=True
            ))

    fig.update_layout(
        title=f"Spider Scatter Plot - Station Metrics ({selected_month.strftime('%B %Y')})",
        xaxis_title="Total Activity (Min-Max Scaled: 0-1)",
        yaxis_title="Departure Ratio (Min-Max Scaled: 0-1)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(40,40,40,0.9)",  # Dark background
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white", size=12)
        ),
        annotations=[
            dict(
                text="<b>SPIDER SCATTER PLOT EXPLANATION:</b><br><br>" +
                     "<b>X-axis:</b> Total Activity (Min-Max Scaled)<br>" +
                     "<b>Y-axis:</b> Departure Ratio (Min-Max Scaled)<br><br>" +
                     "<b>Spider Arms (from each station point):</b><br>" +
                     "↑ <b>Up:</b> Total Activity (min-max scaled)<br>" +
                     "→ <b>Right:</b> Departure Ratio (min-max scaled)<br>" +
                     "↓ <b>Down:</b> Arrival Ratio (min-max scaled)<br>" +
                     "← <b>Left:</b> Balance Magnitude (min-max scaled)<br><br>" +
                     "<b>Station Colors:</b><br>" +
                     "🔴 Red = More Departures than Arrivals<br>" +
                     "🟢 Green = More Arrivals than Departures<br>" +
                     "🔵 Blue = Balanced (Equal Arrivals/Departures)<br><br>" +
                     "<b>Scaling:</b> All metrics normalized to 0-1 range<br>" +
                     "<b>Showing:</b> Top 200 stations by total activity",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(40,40,40,0.9)",  # Dark background
                bordercolor="white",
                borderwidth=1,
                font=dict(size=9, color="white")
            )
        ]
    )

    return fig


def create_arrivals_departures_spider_plot(combined, selected_date):
    """Create spider scatter plot comparing arrivals vs departures"""
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

    # Take top 200 stations by total activity
    station_monthly['total_activity'] = station_monthly['departures'] + station_monthly['arrivals']
    top_stations = station_monthly.nlargest(200, 'total_activity')

    # Calculate metrics
    top_stations['net_balance'] = top_stations['departures'] - top_stations['arrivals']

    # Min-max scale arrivals and departures for spider arms
    max_arrivals = top_stations['arrivals'].max()
    max_departures = top_stations['departures'].max()
    min_arrivals = top_stations['arrivals'].min()
    min_departures = top_stations['departures'].min()

    if max_arrivals > min_arrivals:
        top_stations['arrivals_norm'] = (top_stations['arrivals'] - min_arrivals) / (max_arrivals - min_arrivals)
    else:
        top_stations['arrivals_norm'] = 0.5

    if max_departures > min_departures:
        top_stations['departures_norm'] = (top_stations['departures'] - min_departures) / (
                    max_departures - min_departures)
    else:
        top_stations['departures_norm'] = 0.5

    # Also scale X and Y axis data
    top_stations['arrivals_scaled'] = (top_stations['arrivals'] - min_arrivals) / (max_arrivals - min_arrivals)
    top_stations['departures_scaled'] = (top_stations['departures'] - min_departures) / (
                max_departures - min_departures)

    # Create scatter plot
    fig = go.Figure()

    # Define spider arm angles (2 directions: up for departures, down for arrivals)
    angles = [np.pi / 2, -np.pi / 2]  # Up and down

    # Color stations by net balance
    top_stations['color'] = top_stations['net_balance'].apply(
        lambda x: '#FF6B6B' if x > 0 else '#69DB7C' if x < 0 else '#4DABF7'
    )

    # Create spider arms for each station
    for _, station in top_stations.iterrows():
        station_x = station['arrivals_scaled']
        station_y = station['departures_scaled']

        # Spider arms: up for departures, down for arrivals
        values = [station['departures_norm'], station['arrivals_norm']]

        # Create spider arms
        for i, (angle, value) in enumerate(zip(angles, values)):
            # Calculate arm length - make them more visible
            arm_length = 80 + value * 150  # Longer base length + more scaling

            # Calculate end point
            end_x = station_x + arm_length * np.cos(angle)
            end_y = station_y + arm_length * np.sin(angle)

            # Different line styles for arrivals vs departures
            line_width = 2 if i == 0 else 1  # Thick for departures, thin for arrivals

            # Add spider arm line
            fig.add_trace(go.Scatter(
                x=[station_x, end_x],
                y=[station_y, end_y],
                mode='lines',
                line=dict(color=station['color'], width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))

            # Add small marker at end of arm
            fig.add_trace(go.Scatter(
                x=[end_x],
                y=[end_y],
                mode='markers',
                marker=dict(size=3, color=station['color'], symbol='circle'),
                hoverinfo='skip',
                showlegend=False
            ))

    # Add center points (stations)
    for color_val, color_name in [('#FF6B6B', 'More Departures'), ('#69DB7C', 'More Arrivals'),
                                  ('#4DABF7', 'Balanced')]:
        color_stations = top_stations[top_stations['color'] == color_val]

        if len(color_stations) > 0:
            fig.add_trace(go.Scatter(
                x=color_stations['arrivals_scaled'],
                y=color_stations['departures_scaled'],
                mode='markers',
                marker=dict(size=6, color=color_val, symbol='circle'),
                text=color_stations['station_name'],
                hovertemplate='<b>%{text}</b><br>' +
                              'Arrivals: %{customdata[0]}<br>' +
                              'Departures: %{customdata[1]}<br>' +
                              'Net Balance: %{customdata[2]}<extra></extra>',
                customdata=color_stations[['arrivals', 'departures', 'net_balance']].values,
                name=color_name,
                showlegend=True
            ))

    # Add reference lines for line styles
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='gray', width=3),
        name='Departures (thick lines)',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='gray', width=1),
        name='Arrivals (thin lines)',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Arrivals vs Departures Spider Scatter Plot ({selected_month.strftime('%B %Y')})",
        xaxis_title="Arrivals (Min-Max Scaled: 0-1)",
        yaxis_title="Departures (Min-Max Scaled: 0-1)",
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(40,40,40,0.9)",  # Dark background
            bordercolor="white",
            borderwidth=1,
            font=dict(color="white", size=12)
        ),
        annotations=[
            dict(
                text="<b>ARRIVALS vs DEPARTURES SPIDER PLOT:</b><br><br>" +
                     "<b>X-axis:</b> Arrivals (Min-Max Scaled)<br>" +
                     "<b>Y-axis:</b> Departures (Min-Max Scaled)<br><br>" +
                     "<b>Spider Arms (from each station point):</b><br>" +
                     "↑ <b>Up:</b> Departures (thick lines, min-max scaled)<br>" +
                     "↓ <b>Down:</b> Arrivals (thin lines, min-max scaled)<br><br>" +
                     "<b>Station Colors:</b><br>" +
                     "🔴 Red = More Departures than Arrivals<br>" +
                     "🟢 Green = More Arrivals than Departures<br>" +
                     "🔵 Blue = Balanced (Equal Arrivals/Departures)<br><br>" +
                     "<b>Diagonal Line:</b> Perfect balance (scaled Y=X)<br>" +
                     "<b>Above Diagonal:</b> More departures than arrivals<br>" +
                     "<b>Below Diagonal:</b> More arrivals than departures<br><br>" +
                     "<b>Scaling:</b> All values normalized to 0-1 range<br>" +
                     "<b>Showing:</b> Top 200 stations by total activity",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                xanchor="left", yanchor="top",
                bgcolor="rgba(40,40,40,0.9)",  # Dark background
                bordercolor="white",
                borderwidth=1,
                font=dict(size=9, color="white")
            )
        ]
    )

    return fig