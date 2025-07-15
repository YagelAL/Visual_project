from turtle import st
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
        import streamlit as st



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


# ── WEATHER DATA ───────────────────────────────────────────────────────────────

@st.cache_data
def load_weather(start_date: date, end_date: date):
    """Fetch weather data from Open-Meteo API"""
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


def create_spider_plot_for_month(combined, selected_date):
    """Create a spider plot showing station metrics for the entire month of the selected date"""
    # Get the month period for the selected date
    selected_month = pd.Period(selected_date, freq='M')

    # Filter data for the selected month
    df_month = combined.copy()
    df_month['date'] = pd.to_datetime(df_month['date'])
    month_data = df_month[df_month['date'].dt.to_period('M') == selected_month]

    if month_data.empty:
        return go.Figure()

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

    # Color stations by their average net balance (blue = negative, red = positive)
    stations_df['color'] = stations_df['avg_net_balance'].apply(
        lambda x: '#FF6B6B' if x > 0 else '#4DABF7' if x < 0 else '#69DB7C'
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
        marker=dict(size=10, color='#4DABF7'),
        name='More Arrivals',
        showlegend=True
    ))
    fig_spider.add_trace(go.Scattermapbox(
        lat=[None], lon=[None],
        mode='markers',
        marker=dict(size=10, color='#69DB7C'),
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

    return fig


# ── ADVANCED MODELS ────────────────────────────────────────────────────────────

def create_arima_forecast(combined, selected_station, forecast_days=7):
    """Create ARIMA forecast for a selected station"""
    try:
        from statsmodels.tsa.arima.model import ARIMA

        # Get station data
        station_data = combined[combined['station_name'] == selected_station].copy()
        station_data['date'] = pd.to_datetime(station_data['date'])
        station_data = station_data.sort_values('date')
        station_data['net_balance'] = station_data['departures'] - station_data['arrivals']

        if len(station_data) < 14:  # Need minimum data
            return go.Figure(), "Insufficient data for ARIMA forecast"

        # Prepare time series
        ts = station_data.set_index('date')['net_balance']

        # Fit ARIMA model (simple auto-selection)
        model = ARIMA(ts, order=(1, 1, 1))
        fitted_model = model.fit()

        # Generate forecast
        forecast = fitted_model.forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=ts.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        # Create plot
        fig = go.Figure()

        # Historical data
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=ts.values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='steelblue', width=2)
        ))

        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_index,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title=f"ARIMA Forecast - {selected_station}",
            xaxis_title="Date",
            yaxis_title="Net Balance",
            height=400
        )

        return fig, f"ARIMA forecast successful for {forecast_days} days"

    except Exception as e:
        return go.Figure(), f"ARIMA forecast failed: {str(e)}"


def create_prophet_forecast(combined, selected_station, forecast_days=7):
    """Create Prophet forecast with seasonality"""
    try:
        from prophet import Prophet

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
    """Detect anomalous station behavior using Z-score"""
    df_day = combined[combined['date'] == selected_date].copy()
    df_day['net_balance'] = df_day['departures'] - df_day['arrivals']
    df_day['total_activity'] = df_day['departures'] + df_day['arrivals']

    if df_day.empty:
        return go.Figure(), "No data for anomaly detection"

    # Calculate Z-scores
    df_day['net_balance_zscore'] = np.abs(
        (df_day['net_balance'] - df_day['net_balance'].mean()) / df_day['net_balance'].std())
    df_day['activity_zscore'] = np.abs(
        (df_day['total_activity'] - df_day['total_activity'].mean()) / df_day['total_activity'].std())

    # Identify anomalies
    df_day['is_anomaly'] = (df_day['net_balance_zscore'] > z_threshold) | (df_day['activity_zscore'] > z_threshold)

    # Create map
    fig = go.Figure()

    # Normal stations
    normal_stations = df_day[~df_day['is_anomaly']]
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
    anomaly_stations = df_day[df_day['is_anomaly']]
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
    total_count = len(df_day)

    return fig, f"Found {anomaly_count} anomalies out of {total_count} stations ({anomaly_count / total_count * 100:.1f}%)"


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