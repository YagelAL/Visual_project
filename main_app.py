
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.express as px
import datetime
from helper_functions import (
    load_processed_data,
    prepare_daily_time_series_data,
    perform_time_series_clustering,
    create_map_visualization,
    create_time_series_cluster_map,
    create_timeline_map,
    create_daily_rides_bar_chart,
    create_daily_rides_continuous_plot,
    create_spider_glyph_month,
    create_spider_glyph_activity_density,
    create_time_wheel_plot,
    create_temporal_pattern_spider,
    create_station_role_spider,
    create_arima_forecast,
    predict_peak_periods_standalone,
    detect_station_anomalies,
    create_tominski_time_wheel,
    create_parallel_coordinates_plot
)

MAX_STATIONS_GLOBAL = 200
MAX_STATIONS_COMPLEX_VIZ = 100
MAX_STATIONS_SIMPLE_VIZ = 200

st.set_page_config(
    page_title="NYC Citibike Station Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    if "max_stations_simple_viz" not in st.session_state:
        st.session_state.max_stations_simple_viz = MAX_STATIONS_SIMPLE_VIZ
    if "max_stations_complex_viz" not in st.session_state:
        st.session_state.max_stations_complex_viz = MAX_STATIONS_COMPLEX_VIZ

    st.title("NYC Citibike Station Visualization")

    data = load_processed_data()
    if not data:
        st.error("No data loaded.")
        return

    combined = pd.concat(data.values(), ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date']).dt.date
    all_dates = sorted(combined['date'].unique())
    march_dates = [d for d in all_dates if d.month == 3 and d.year == 2025]
    if march_dates:
        default_month_start = min(march_dates)
        default_month_end = max(march_dates)
        dates = sorted(march_dates)
    else:
        default_month_start = all_dates[0] if all_dates else None
        default_month_end = all_dates[-1] if all_dates else None
        dates = sorted(all_dates)
    all_stations_df = combined[['station_name', 'lat', 'lng']].drop_duplicates()

    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Main Map", "Timeline", "Models"])
    if mode == "Main Map":
        render_main_map_mode(combined, dates)
    elif mode == "Timeline":
        render_timeline_mode(combined, dates, st.session_state)
    elif mode == "Models":
        render_models_map_mode(combined, all_dates, st.session_state)
def render_main_map_mode(combined, dates):
    """Main map interface with station activity analysis and visualizations"""
    
    # Date selection for all analyses
    # Default to first date in March 2025 if available
    march_2025_dates = [d for d in dates if d.month == 3 and d.year == 2025]
    default_date = march_2025_dates[0] if march_2025_dates else dates[0]
    sel_date = st.date_input("Select Date (March 2025 only):", value=default_date, min_value=dates[0], max_value=dates[-1])

    # ── STATION ACTIVITY CLUSTERING MAP ────────────────────────────────────────
    st.header("Station Activity Map")
    
    # Map controls in compact layout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
    with col1:
        radius_m = st.slider("Radius (m):", 100, 200, 100, 10, key="main_map_radius")
    with col2:
        show_dep = st.checkbox("More departures", True, key="main_map_dep")
    with col3:
        show_arr = st.checkbox("More arrivals", True, key="main_map_arr")
    with col4:
        show_bal = st.checkbox("Balanced", True, key="main_map_bal")

    # Build category filter from checkboxes
    categories = [
        name for name, checked in zip(
            ["More departures", "More arrivals", "Balanced"],
            [show_dep, show_arr, show_bal]
        ) if checked
    ]
    
    # Generate and display clustering map
    df_day = combined[combined["date"] == sel_date]
    if df_day.empty:
        st.warning("No data for this date.")
    else:
        fig = create_map_visualization(df_day, radius_m, categories)
        st.plotly_chart(fig, use_container_width=True)

    # ── DAILY RIDES OVERVIEW ───────────────────────────────────────────────────
    st.header("Daily Rides Overview")

    # Chart type and date range selection
    chart_type = st.radio("Select chart type:", ["Hourly plot", "Daily Bar Chart"], horizontal=True)

    # Load all available cleaned and processed files for daily rides overview
    import glob, os
    cleaned_files = glob.glob("rides_*_cleaned.csv")
    processed_files = glob.glob("processed_*.csv")

    # Build a mapping from month/year to file
    file_map = {}
    for f in cleaned_files:
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) >= 4:
            mm = parts[1]
            yyyy = parts[2]
            key = f"{mm}_{yyyy}"
            file_map[key] = f
    for f in processed_files:
        base = os.path.basename(f)
        name = base.replace("processed_","").replace(".csv","")
        if len(name) == 6:
            mm = name[:2]
            yyyy = name[2:]
            key = f"{mm}_{yyyy}"
            if key not in file_map:
                file_map[key] = f

    # Collect all available dates from all files
    available_dates = []
    date_file_map = {}
    for key, fpath in file_map.items():
        try:
            df = pd.read_csv(fpath)
            df['date'] = pd.to_datetime(df['date']).dt.date
            for d in sorted(df['date'].unique()):
                available_dates.append(d)
                date_file_map[d] = fpath
        except Exception:
            continue
    available_dates = sorted(set(available_dates))

    if available_dates:
        # Default to first date in March 2025 if available
        march_2025_dates = [d for d in available_dates if d.month == 3 and d.year == 2025]
        default_overview_start = march_2025_dates[0] if march_2025_dates else available_dates[0]
        default_overview_end = march_2025_dates[-1] if march_2025_dates else available_dates[-1]
        col1, col2 = st.columns(2)
        with col1:
            overview_start = st.date_input("Overview Start Date:", value=default_overview_start, min_value=available_dates[0], max_value=available_dates[-1], key="overview_start")
        with col2:
            overview_end = st.date_input("Overview End Date:", value=default_overview_end, min_value=available_dates[0], max_value=available_dates[-1], key="overview_end")

        # Find file for each date in range and concatenate
        date_range = [d for d in available_dates if overview_start <= d <= overview_end]
        dfs = []
        for d in date_range:
            fpath = date_file_map.get(d)
            if fpath:
                try:
                    df = pd.read_csv(fpath)
                    df['date'] = pd.to_datetime(df['date']).dt.date
                    dfs.append(df[df['date'] == d])
                except Exception:
                    continue
        if dfs:
            filtered_all = pd.concat(dfs, ignore_index=True)
            if chart_type == "Hourly plot":
                daily_chart = create_daily_rides_continuous_plot(filtered_all, overview_start, overview_end, max_stations=st.session_state.max_stations_simple_viz)
            else:
                daily_chart = create_daily_rides_bar_chart(filtered_all, overview_start, overview_end, max_stations=st.session_state.max_stations_simple_viz)
            st.plotly_chart(daily_chart, use_container_width=True)
        else:
            st.warning("No data available for selected date range.")
    else:
        st.warning("No data available for daily rides overview.")

    # ── TIME SERIES CLUSTERING ANALYSIS ────────────────────────────────────────
    st.header("Daily Time Series Clustering")
    render_daily_time_series_section(combined, st.session_state)

    # ── BEHAVIORAL PATTERN ANALYSIS ────────────────────────────────────────────
    st.header("Spider Glyph Analysis")
    
    # Temporal usage patterns
    st.subheader("Temporal Usage Patterns")
    try:
        temporal_spider = create_temporal_pattern_spider(combined, max_stations=st.session_state.max_stations_complex_viz)
        if temporal_spider.data:
            st.plotly_chart(temporal_spider, use_container_width=True)
        else:
            st.info("No data available for temporal pattern analysis.")
    except Exception as e:
        st.error(f"Error creating temporal pattern spider: {e}")
    
    # Station role identification
    st.subheader("Station Role Analysis")
    try:
        role_spider = create_station_role_spider(combined, max_stations=10)
        if role_spider.data:
            st.plotly_chart(role_spider, use_container_width=True)
        else:
            st.info("No data available for station role analysis.")
    except Exception as e:
        st.error(f"Error creating station role spider: {e}")

    # ── TEMPORAL CYCLING PATTERNS ──────────────────────────────────────────────
    st.header("Daily Cycling Patterns - Time Wheel")

    # Gather all available cleaned and processed files for cyclic wheel
    import glob, os
    cleaned_files = glob.glob("rides_*_cleaned.csv")
    processed_files = glob.glob("processed_*.csv")

    # Build a mapping from month/year to file
    file_map = {}
    for f in cleaned_files:
        # Expect format rides_MM_YYYY_cleaned.csv
        base = os.path.basename(f)
        parts = base.split('_')
        if len(parts) >= 4:
            mm = parts[1]
            yyyy = parts[2]
            key = f"{mm}_{yyyy}"
            file_map[key] = f
    for f in processed_files:
        # Expect format processed_MMYYYY.csv
        base = os.path.basename(f)
        name = base.replace("processed_","").replace(".csv","")
        if len(name) == 6:
            mm = name[:2]
            yyyy = name[2:]
            key = f"{mm}_{yyyy}"
            if key not in file_map:
                file_map[key] = f

    # Collect all available dates from all files
    available_dates = []
    date_file_map = {}
    for key, fpath in file_map.items():
        try:
            df = pd.read_csv(fpath)
            df['date'] = pd.to_datetime(df['date']).dt.date
            for d in sorted(df['date'].unique()):
                available_dates.append(d)
                date_file_map[d] = fpath
        except Exception:
            continue
    available_dates = sorted(set(available_dates))

    if available_dates:
        # Find first date in March 2025, fallback to first available date
        march_2025_dates = [d for d in available_dates if d.month == 3 and d.year == 2025]
        default_date = march_2025_dates[0] if march_2025_dates else available_dates[0]
        selected_date = st.date_input(
            "Select Start Date:",
            value=default_date,
            min_value=available_dates[0],
            max_value=available_dates[-1],
            help="Choose the first day to visualize (shows 7 days from this date)"
        )
        # Get file for selected_date
        file_for_selected = date_file_map.get(selected_date)
        if file_for_selected:
            try:
                df_selected = pd.read_csv(file_for_selected)
                df_selected['date'] = pd.to_datetime(df_selected['date']).dt.date
                date_range = [selected_date + datetime.timedelta(days=i) for i in range(7)]
                filtered = df_selected[df_selected['date'].isin(date_range)]
                if filtered.empty:
                    st.warning("No data available for selected week.")
                else:
                    time_wheel_fig = create_time_wheel_plot(filtered)
                    st.plotly_chart(time_wheel_fig, use_container_width=True)
            except Exception:
                st.warning(f"Could not load file for selected date: {file_for_selected}")
        else:
            st.warning("No file found for selected date.")
    else:
        st.warning("No data available for cyclic time wheel.")

    # ── DETAILED STATION METRICS ───────────────────────────────────────────────
    # Move Station Analysis Plots to end and update titles
    # ...existing code...

    # ── MULTI-DIMENSIONAL ANALYSIS ─────────────────────────────────────────────
    # ...existing code...

    # ── ALL-TIME STATION ANALYSIS ─────────────────────────────────────────────
    st.header("Station Analysis Plots (All Time)")
    tab1, tab2 = st.tabs(["Monthly Spider Glyph (All Time)", "Activity Scatter Plot (All Time)"])
    with tab1:
        try:
            spider_fig = create_spider_glyph_month(combined)
            st.plotly_chart(spider_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating month spider glyph: {e}")
    with tab2:
        try:
            scatter_fig = create_spider_glyph_activity_density(combined)
            st.plotly_chart(scatter_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating activity scatter plot: {e}")

    # ── ALL-TIME MULTI-DIMENSIONAL ANALYSIS ───────────────────────────────────
    st.header("Multi-Dimensional Station Analysis (All Time)")
    try:
        parallel_fig = create_parallel_coordinates_plot(combined)
        st.plotly_chart(parallel_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating parallel coordinates plot: {e}")

    # ── TOMINSKI TIME WHEEL VISUALIZATION ──────────────────────────────────────
    st.header("Tominski Time Wheel")
    
    try:
        tominski_fig = create_tominski_time_wheel(combined)
        st.plotly_chart(tominski_fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating Tominski time wheel: {e}")


def render_daily_time_series_section(combined, session_state):
    """Time series clustering analysis for selected date ranges"""
    
    # Use March 2025 as default month for clustering analysis
    combined['date'] = pd.to_datetime(combined['date']).dt.date
    march_dates = [d for d in combined['date'].unique() if d.month == 3 and d.year == 2025]
    if march_dates:
        dates = sorted(march_dates)
    else:
        dates = sorted(combined["date"].unique())
    # Default to first date in March 2025 if available
    march_2025_dates = [d for d in dates if d.month == 3 and d.year == 2025]
    default_start = march_2025_dates[0] if march_2025_dates else dates[0]
    default_end = min(default_start + datetime.timedelta(days=14), dates[-1]) if dates else None
    
    # Date range and cluster selection
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input("Start Date:", value=default_start, min_value=dates[0], max_value=dates[-1], key="ts_start_date")
    with col2:
        end_date = st.date_input("End Date:", value=default_end, min_value=dates[0], max_value=dates[-1], key="ts_end_date")
    with col3:
        ts_k = st.selectbox("Clusters:", list(range(1, 7)), index=2, key="ts_k")

    if start_date > end_date:
        st.error("Start date must be before or equal to end date")
        return

    # Generate time series clustering
    pivot_daily, coords, day_info = prepare_daily_time_series_data(combined, start_date, end_date, max_stations=session_state.max_stations_complex_viz)

    if not pivot_daily.empty and day_info:
        # Display analysis period
        start_day = day_info[0]['date'].strftime('%d/%m/%y')
        end_day = day_info[-1]['date'].strftime('%d/%m/%y')
        st.write(f"**Analyzing daily patterns: {start_day} - {end_day}** ({len(day_info)} days)")

        # Perform clustering and visualization
        ts_res, kmeans_model = perform_time_series_clustering(pivot_daily, ts_k, coords)
        st.plotly_chart(create_time_series_cluster_map(ts_res), use_container_width=True)

        # Detailed cluster analysis
        date_range_str = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
        render_cluster_analysis(ts_res, pivot_daily, day_info, date_range_str, combined)
    else:
        st.warning(f"No data available for the selected date range.")


def render_cluster_analysis(ts_res, pivot_daily, day_info, date_range_str, combined):
    """Generate detailed analysis of clustering results with statistical visualizations"""

    if not ts_res.empty and len(pivot_daily.columns) > 0:
        # Prepare cluster statistics
        cluster_data = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        day_labels = list(pivot_daily.columns)  # Already in DD/MM format

        # Calculate cluster patterns and statistics
        for cluster_id in sorted(ts_res['cluster'].unique()):
            cluster_stations = ts_res[ts_res['cluster'] == cluster_id]['station_name'].tolist()
            cluster_station_data = pivot_daily.loc[cluster_stations]
            
            if not cluster_station_data.empty:
                avg_pattern = cluster_station_data.mean(axis=0).values
                cluster_data.append({
                    'cluster': cluster_id,
                    'stations': len(cluster_stations),
                    'pattern': avg_pattern,
                    'all_station_data': cluster_station_data.values
                })

        # Visualization 1: Data distribution analysis
        st.subheader("Cluster Data Distribution")
        render_cluster_distribution_plot(cluster_data, date_range_str, colors)

        # Visualization 2: Temporal pattern analysis
        st.subheader("Daily Usage Patterns")
        render_daily_patterns_plot(cluster_data, day_labels, date_range_str, colors)
    else:
        # Fallback: basic cluster summary
        cluster_summary = ts_res.groupby("cluster").size().reset_index(name="station_count")
        st.dataframe(cluster_summary, use_container_width=True)


def render_cluster_distribution_plot(cluster_data, date_range_str, colors):
    """Generate box plot visualization of cluster data distributions"""
    
    fig_whisker = go.Figure()

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
        title=f"Cluster Data Distribution - {date_range_str}",
        yaxis_title="Net Balance",
        xaxis_title="Cluster",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_whisker, use_container_width=True)


def render_daily_patterns_plot(cluster_data, day_labels, date_range_str, colors):
    """Generate normalized daily pattern visualization with trend analysis"""
    
    fig_patterns = go.Figure()

    # Add reference line at zero for normalized data
    fig_patterns.add_hline(y=0, line_dash="dash", line_color="gray", 
                          annotation_text="Mean Reference", annotation_position="bottom right")

    for i, cluster_info in enumerate(cluster_data):
        color = colors[i % len(colors)]
        cluster_id = cluster_info['cluster']
        original_pattern = cluster_info['pattern']

        # Apply Z-Score normalization
        if np.std(original_pattern) > 0:
            normalized_pattern = (original_pattern - np.mean(original_pattern)) / np.std(original_pattern)
        else:
            normalized_pattern = np.zeros_like(original_pattern)

        # Calculate linear trend
        x_values = np.arange(len(normalized_pattern))
        if len(x_values) > 1:
            trend_slope, trend_intercept = np.polyfit(x_values, normalized_pattern, 1)
            trend_line = trend_slope * x_values + trend_intercept
        else:
            trend_slope = 0
            trend_line = normalized_pattern

        # Add pattern data trace
        fig_patterns.add_trace(go.Scatter(
            x=day_labels, y=normalized_pattern, mode='lines+markers',
            name=f"Cluster {cluster_id} Data", line=dict(color=color, width=3),
            marker=dict(size=6), legendgroup=f"cluster_{cluster_id}", showlegend=True
        ))

        # Add trend line
        fig_patterns.add_trace(go.Scatter(
            x=day_labels, y=trend_line, mode='lines',
            name=f"Cluster {cluster_id} Trend (slope: {trend_slope:.2f})",
            line=dict(color=color, width=2, dash='dot'),
            legendgroup=f"cluster_{cluster_id}", showlegend=True
        ))

    fig_patterns.update_layout(
        title=f"Daily Net Balance Patterns - {date_range_str}",
        xaxis_title="Day (DD/MM)", yaxis_title="Z-Score Normalized Net Balance",
        height=500, hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig_patterns, use_container_width=True)


def render_timeline_mode(combined, dates, session_state):
    """Timeline mode: animated map visualization with date range controls"""
    
    # Sidebar controls for timeline parameters
    st.sidebar.header("Timeline Options")
    radius_m = st.sidebar.slider("Clustering radius (m):", 100, 200, 100, 10, key="timeline_radius")

    st.sidebar.header("Select Date Range")
    # Default to first date in March 2025 if available
    march_2025_dates = [d for d in dates if d.month == 3 and d.year == 2025]
    default_start = march_2025_dates[0] if march_2025_dates else dates[0]
    default_end = dates[-1]
    start_date = st.sidebar.date_input("Start date (March 2025 only):", value=default_start, min_value=dates[0], max_value=dates[-1])
    end_date = st.sidebar.date_input("End date (March 2025 only):", value=default_end, min_value=dates[0], max_value=dates[-1])

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    # Category selection for station types
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        show_dep = st.checkbox("More departures", True, key="timeline_dep")
    with col2:
        show_arr = st.checkbox("More arrivals", True, key="timeline_arr")
    with col3:
        show_bal = st.checkbox("Balanced", True, key="timeline_bal")

    categories = [name for name, checked in zip(["More departures", "More arrivals", "Balanced"], [show_dep, show_arr, show_bal]) if checked]

    # Animated timeline visualization
    st.subheader(f"Timeline Map — {start_date.strftime('%d/%m/%y')} to {end_date.strftime('%d/%m/%y')}")
    try:
        fig_timeline = create_timeline_map(combined, start_date, end_date, radius_m, categories, max_stations=session_state.max_stations_simple_viz)
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_map")
    except Exception as e:
        st.error(f"Error creating timeline map: {e}")

    # Daily activity summary
    st.subheader("Daily Ride Counts")
    try:
        fig_daily_rides = create_daily_rides_bar_chart(combined, start_date, end_date, max_stations=session_state.max_stations_simple_viz)
        st.plotly_chart(fig_daily_rides, use_container_width=True, key="daily_rides_chart")
    except Exception as e:
        st.error(f"Error creating daily rides chart: {e}")


def render_models_map_mode(combined, dates, session_state):
    """Advanced analytics: ARIMA forecasting, peak analysis, and anomaly detection"""
    
    st.subheader(" Analytics Models")
    
    analysis_date = dates[0] if dates else date.today()

    # ── INTERACTIVE STATION SELECTION ──────────────────────────────────────────
    st.subheader("Select Station from Map")

    available_stations = sorted(combined['station_name'].unique())
    station_coords = combined[['station_name', 'lat', 'lng']].drop_duplicates().reset_index(drop=True)

    # Initialize selected station in session state
    if 'selected_station_models' not in st.session_state:
        st.session_state.selected_station_models = available_stations[0]

    # Create interactive station selection map
    fig_station_map = create_station_selection_map(station_coords, st.session_state.selected_station_models)
    selected_points = st.plotly_chart(fig_station_map, use_container_width=True, key="station_selection_map", on_select="rerun")
    
    # Handle map interactions
    handle_station_selection(selected_points, station_coords, available_stations)

    # Station selection controls
    render_station_selection_controls(available_stations)
    selected_station = st.session_state.selected_station_models

    st.markdown("---")

    # ── ARIMA FORECASTING MODEL ─────────────────────────────────────────────────
    st.subheader(" ARIMA Forecast")
    
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    with col1:
        forecast_days_arima = st.slider("Forecast Days:", 3, 14, 7, key="arima_days")
    with col2:
        st.success(f"**Selected Station:**\n{selected_station}")

    # Run ARIMA analysis using June 2025 for training and forecast for dates after June 2025
    with st.spinner("Running ARIMA model..."):
        try:
            # Filter for June 2025
            combined['date'] = pd.to_datetime(combined['date'])
            june_start = pd.to_datetime('2025-06-01')
            june_end = pd.to_datetime('2025-06-30')
            june_data = combined[(combined['date'] >= june_start) & (combined['date'] <= june_end)]
            forecast_days = forecast_days_arima
            if june_data.empty:
                st.warning("No data available for June 2025 for ARIMA analysis.")
            else:
                fig_arima, message_arima = create_arima_forecast(june_data, selected_station, forecast_days)
                st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.error(f"Error in ARIMA forecast: {e}")

    st.markdown("---")

    # ── PEAK/OFF-PEAK ANALYSIS ──────────────────────────────────────────────────
    st.subheader("Peak/Off-Peak Analysis")
    
    analysis_start, analysis_end, use_all_time = render_peak_analysis_controls(dates, analysis_date)
    
    # Generate peak analysis using processed_*.csv files
    import glob, os
    processed_files = glob.glob(os.path.join(os.path.dirname(__file__), "processed_*.csv"))
    dfs = []
    for f in processed_files:
        try:
            df = pd.read_csv(f)
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            dfs.append(df)
        except Exception:
            continue
    if dfs:
        processed_all = pd.concat(dfs, ignore_index=True)
        # Filter for selected date range
        if use_all_time:
            filtered = processed_all
        elif analysis_end is not None:
            filtered = processed_all[(processed_all['date'] >= analysis_start) & (processed_all['date'] <= analysis_end)]
        else:
            filtered = processed_all[processed_all['date'] == analysis_start]

        # Limit to 200 points per day
        filtered_limited = filtered.groupby('date').apply(lambda x: x.head(200)).reset_index(drop=True)

        with st.spinner("Analyzing peak periods from processed files (max 200 points/day)..."):
            try:
                fig_peak = predict_peak_periods_standalone(filtered_limited, analysis_start, analysis_end, use_all_time)
                st.plotly_chart(fig_peak, use_container_width=True)
            except Exception as e:
                st.error(f"Error in peak analysis: {e}")
    else:
        st.warning("No processed data available for peak analysis.")

    st.markdown("---")

    # ── ANOMALY DETECTION ───────────────────────────────────────────────────────
    st.subheader(" Monthly Anomaly Detection")
    z_threshold, analysis_date_anomaly = render_anomaly_controls(combined, dates)
    # Filter combined to only the selected month for anomaly detection
    selected_month = pd.Period(analysis_date_anomaly, freq='M')
    combined_month = combined.copy()
    combined_month['date'] = pd.to_datetime(combined_month['date']).dt.date
    month_start = selected_month.start_time.date()
    month_end = selected_month.end_time.date()
    combined_month = combined_month[(combined_month['date'] >= month_start) & (combined_month['date'] <= month_end)]
    # Run anomaly detection
    with st.spinner("Detecting anomalies..."):
        try:
            fig_anomaly, message_anomaly = detect_station_anomalies(combined_month, analysis_date_anomaly, z_threshold)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            st.info(message_anomaly)
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")


def create_station_selection_map(station_coords, selected_station):
    """Create interactive map for station selection"""
    fig_station_map = go.Figure()

    # Add all stations
    fig_station_map.add_trace(go.Scattermapbox(
        lat=station_coords['lat'], lon=station_coords['lng'], mode='markers',
        marker=dict(size=6, color='black', opacity=0.8),
        text=station_coords['station_name'], customdata=station_coords['station_name'],
        hovertemplate="<b>%{text}</b><br>Click to select this station<extra></extra>",
        name="Available Stations", showlegend=False
    ))

    # Highlight selected station
    if selected_station in station_coords['station_name'].values:
        selected_coords = station_coords[station_coords['station_name'] == selected_station]
        if not selected_coords.empty:
            fig_station_map.add_trace(go.Scattermapbox(
                lat=selected_coords['lat'], lon=selected_coords['lng'], mode='markers',
                marker=dict(size=12, color='red', opacity=1.0),
                text=selected_coords['station_name'],
                hovertemplate="<b>%{text}</b><br>Selected Station<extra></extra>",
                name="Selected Station", showlegend=False
            ))

    fig_station_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=40.7128, lon=-74.0060), zoom=11),
        height=450, margin=dict(l=0, r=0, t=0, b=0), showlegend=False
    )
    return fig_station_map


def handle_station_selection(selected_points, station_coords, available_stations):
    """Handle station selection from map clicks"""
    if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
        if selected_points.selection.get('points'):
            clicked_point = selected_points.selection['points'][0]
            
            point_index = clicked_point.get('pointIndex') or clicked_point.get('point_index')
            
            if 'customdata' in clicked_point:
                clicked_station = clicked_point['customdata']
                if clicked_station in available_stations:
                    st.session_state.selected_station_models = clicked_station
                    st.rerun()
            
            if point_index is not None and point_index < len(station_coords):
                clicked_station = station_coords.iloc[point_index]['station_name']
                st.session_state.selected_station_models = clicked_station
                st.rerun()


def render_station_selection_controls(available_stations):
    """Render station selection dropdown and info display"""
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_station_dropdown = st.selectbox(
            "Select Station:", available_stations,
            index=available_stations.index(st.session_state.selected_station_models) if st.session_state.selected_station_models in available_stations else 0,
            key="station_dropdown"
        )
        
        if selected_station_dropdown != st.session_state.selected_station_models:
            st.session_state.selected_station_models = selected_station_dropdown

    with col2:
        st.success(f"**Selected Station:**\n{st.session_state.selected_station_models}")


def filter_data_from_june(combined):
    """Filter data to only include records from June onwards"""
    combined_filtered = combined.copy()
    combined_filtered['date'] = pd.to_datetime(combined_filtered['date'])
    data_year = combined_filtered['date'].dt.year.min()
    june_start = pd.to_datetime(f'{data_year}-06-01')
    return combined_filtered[combined_filtered['date'] >= june_start]


def render_peak_analysis_controls(dates, analysis_date):
    """Render controls for peak analysis mode selection"""
    col1, col2 = st.columns(2)
    
    # Default to first date in March 2025 if available
    march_2025_dates = [d for d in dates if d.month == 3 and d.year == 2025]
    default_analysis_date = march_2025_dates[0] if march_2025_dates else dates[0]
    with col1:
        analysis_mode = st.selectbox("Analysis Mode:", ["Single Day", "Date Range", "All Time"], help="Choose the time period for peak analysis")
    
    with col2:
        if analysis_mode == "Single Day":
            analysis_start = st.date_input("Select Date:", value=default_analysis_date, min_value=dates[0], max_value=dates[-1], key="peak_single")
            analysis_end = None
        elif analysis_mode == "Date Range":
            analysis_start = st.date_input("Start Date:", value=default_analysis_date, min_value=dates[0], max_value=dates[-1], key="peak_range_start")
            analysis_end = st.date_input("End Date:", value=min(default_analysis_date + datetime.timedelta(days=7), dates[-1]), min_value=analysis_start, max_value=dates[-1], key="peak_range_end")
        else:  # All Time
            analysis_start = dates[0]
            analysis_end = dates[-1]
    
    use_all_time = (analysis_mode == "All Time")
    return analysis_start, analysis_end, use_all_time


def render_anomaly_controls(combined, dates):
    col1, col2 = st.columns([1, 2])
    with col1:
        z_threshold = st.slider("Z-Score Threshold:", 2.0, 3.0, 2.5, 0.1)

    # Month selection for anomaly detection (allow any available month)
    available_periods = sorted({pd.Period(d, freq='M') for d in dates})
    month_options = {str(p): p.strftime('%B %Y') for p in available_periods}
    selected_month_str = st.selectbox("Select Month:", options=list(month_options.keys()), format_func=lambda x: month_options[x], key="anomaly_month_select")
    selected_month = pd.Period(selected_month_str)

    # Get representative date from selected month
    month_dates = [d for d in dates if pd.Period(d, freq='M') == selected_month]
    analysis_date = month_dates[0] if month_dates else dates[0]
    return z_threshold, analysis_date


if __name__ == "__main__":
    main()
    # Run the main application