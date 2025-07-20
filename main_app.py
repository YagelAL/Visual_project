import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.express as px
import datetime

# Import helper functions
from helper_functions import (
    months,
    load_processed_data,
    prepare_geodata_and_weights,
    prepare_daily_time_series_data,
    perform_time_series_clustering,
    create_map_visualization,
    create_time_series_cluster_map,
    create_timeline_map,
    create_daily_rides_bar_chart,
    create_daily_rides_continuous_plot,
    create_spider_glyph_month,
    create_spider_glyph_distance,
    create_spider_glyph_balance_ratio,
    create_spider_glyph_activity_density,
    create_time_wheel_plot,
    create_spider_plot_for_month,
    create_temporal_pattern_spider,
    create_station_role_spider,
    create_cluster_comparison_spider,
    create_arima_forecast,
    predict_peak_periods_standalone,
    detect_station_anomalies,
    create_arrivals_departures_spider_plot,
    create_weekly_seasonal_plot,
    create_tominski_time_wheel,
    create_parallel_coordinates_plot
)

# ── APPLICATION CONFIGURATION ──────────────────────────────────────────────────
# Performance Settings - Adjust these to use more or less data
MAX_STATIONS_GLOBAL = 200  # Global maximum for main data loading
MAX_STATIONS_COMPLEX_VIZ = 100  # For complex visualizations (spider plots, etc.)
MAX_STATIONS_SIMPLE_VIZ = 200  # For simple visualizations (maps, charts)

# You can increase these values for better analysis at the cost of performance:
# - For small datasets or powerful hardware: try 500-1000+
# - For medium datasets: try 300-500  
# - For large datasets or slower hardware: keep at 200 or lower

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Citibike Station Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    # Initialize session state for max stations if not set
    if "max_stations_simple_viz" not in st.session_state:
        st.session_state.max_stations_simple_viz = MAX_STATIONS_SIMPLE_VIZ
    if "max_stations_complex_viz" not in st.session_state:
        st.session_state.max_stations_complex_viz = MAX_STATIONS_COMPLEX_VIZ

    st.title("NYC Citibike Station Visualization")

    # Load data with default settings
    data = load_processed_data()
    if not data:
        st.error("No data loaded.")
        return

    combined = pd.concat(data.values(), ignore_index=True)
    dates = sorted(pd.to_datetime(combined["date"]).dt.date.unique())

    all_stations_df = combined[['station_name', 'lat', 'lng']].drop_duplicates()

    # Map mode selection
    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Main Map", "Timeline", "Models"])
    if mode == "Main Map":
        # Main Map with date picker above it
        sel_date = st.date_input("Select Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])

        # 1. STATION ACTIVITY MAP
        st.header("Station Activity Map")

        # Map Settings
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])
        with col1:
            radius_m = st.slider("Radius (m):", 100, 200, 100, 10, key="main_map_radius")
        with col2:
            show_dep = st.checkbox("More departures", True, key="main_map_dep")
        with col3:
            show_arr = st.checkbox("More arrivals", True, key="main_map_arr")
        with col4:
            show_bal = st.checkbox("Balanced", True, key="main_map_bal")
        with col5:
            pass  # Empty space

        categories = [
            name for name, chk in zip(
                ["More departures", "More arrivals", "Balanced"],
                [show_dep, show_arr, show_bal]
            ) if chk
        ]
        
        # Display the map
        df_day = combined[combined["date"] == sel_date]
        if df_day.empty:
            st.warning("No data for this date.")
        else:
            fig = create_map_visualization(df_day, radius_m, categories)
            st.plotly_chart(fig, use_container_width=True)

        # ...existing code...





        # 2. DAILY RIDES OVERVIEW
        st.header("Daily Rides Overview")

        # Add some spacing
        st.markdown("")
        st.markdown("")
        
        # Chart type selection
        chart_type = st.radio(
            "Select chart type:",
            ["Hourly plot", "Daily Bar Chart"],
            horizontal=True
        )
        
        # Date range selection for daily overview
        col1, col2 = st.columns(2)
        with col1:
            overview_start = st.date_input(
                "Overview Start Date", 
                value=dates[0],
                min_value=dates[0],
                max_value=dates[-1]
            )
        with col2:
            overview_end = st.date_input(
                "Overview End Date", 
                value=min(dates[0] + datetime.timedelta(days=14), dates[-1]),
                min_value=dates[0],
                max_value=dates[-1]
            )
        
        if overview_start <= overview_end:
            if chart_type == "Hourly plot":
                daily_chart = create_daily_rides_continuous_plot(
                    combined, 
                    overview_start, 
                    overview_end, 
                    max_stations=st.session_state.max_stations_simple_viz
                )
            else:
                daily_chart = create_daily_rides_bar_chart(
                    combined, 
                    overview_start, 
                    overview_end, 
                    max_stations=st.session_state.max_stations_simple_viz
                )
            st.plotly_chart(daily_chart, use_container_width=True)
        else:
            st.error("Start date must be before or equal to end date")

        # 3. DAILY TIME-SERIES CLUSTERING
        st.header("Daily Time Series Clustering")
        render_daily_time_series_section(combined, st.session_state)

        # 4. IMPROVED SPIDER GLYPH ANALYSIS
        st.header("Spider Glyph Analysis")

        
        # Temporal Pattern Spider
        st.subheader("Temporal Usage Patterns")
        
        try:
            temporal_spider = create_temporal_pattern_spider(combined, max_stations=st.session_state.max_stations_complex_viz)
            if temporal_spider.data:
                st.plotly_chart(temporal_spider, use_container_width=True)
            else:
                st.info("No data available for temporal pattern analysis.")
        except Exception as e:
            st.error(f"Error creating temporal pattern spider: {e}")
        
        # Station Role Spider
        st.subheader("Station Role Analysis")
        
        try:
            role_spider = create_station_role_spider(combined, max_stations=10)
            if role_spider.data:
                st.plotly_chart(role_spider, use_container_width=True)
            else:
                st.info("No data available for station role analysis.")
        except Exception as e:
            st.error(f"Error creating station role spider: {e}")

        # 5. STATION ANALYSIS PLOTS
        st.header("Station Analysis Plots")
        # Tab selection for station analysis plots
        tab1, tab2 = st.tabs(["Monthly Spider Glyph", "Activity Scatter Plot"])
        
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
                
        # 6. CYCLIC TIME WHEEL (Separate Plot)
        st.header("pattern over time daily - Cyclic Time Wheel")
        
        if combined is not None and not combined.empty:
            combined_temp = combined.copy()
            combined_temp['date'] = pd.to_datetime(combined_temp['date'])
            available_dates = sorted(combined_temp['date'].dt.date.unique())
            if available_dates:
                selected_date = st.date_input(
                    "Select Start Date:",
                    value=available_dates[0],
                    min_value=available_dates[0],
                    max_value=available_dates[-1],
                    help="Choose the first day to visualize (shows 7 days from this date)"
                )
                date_range = [selected_date + datetime.timedelta(days=i) for i in range(7)]
                filtered = combined_temp[combined_temp['date'].dt.date.isin(date_range)]
                if filtered.empty:
                    st.warning("No data available for selected week.")
                else:
                    time_wheel_fig = create_time_wheel_plot(filtered)
                    st.plotly_chart(time_wheel_fig, use_container_width=True)
            else:
                st.warning("No date data available")
        else:
            st.warning("No data available for cyclic time wheel.")

        # 6. PARALLEL COORDINATES PLOT
        st.header("Multi-Dimensional Station Analysis")
        
        try:
            parallel_fig = create_parallel_coordinates_plot(combined)
            st.plotly_chart(parallel_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating parallel coordinates plot: {e}")

        # 7. TOMINSKI TIME WHEEL PLOT
        st.header("Tominski Time Wheel")
        
        try:
            tominski_fig = create_tominski_time_wheel(combined)
            st.plotly_chart(tominski_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating Tominski time wheel: {e}")

    elif mode == "Timeline":
        render_timeline_mode(combined, dates, st.session_state)
    
    elif mode == "Models":
        render_models_map_mode(combined, dates, st.session_state)


def render_daily_time_series_section(combined, session_state):
    """Render the daily time series clustering section"""
    # Date range selection
    dates = sorted(pd.to_datetime(combined["date"]).dt.date.unique())
    
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_date = st.date_input(
            "Start Date:",
            value=dates[0],
            min_value=dates[0],
            max_value=dates[-1],
            key="ts_start_date"
        )
    with col2:
        end_date = st.date_input(
            "End Date:",
            value=min(dates[0] + datetime.timedelta(days=14), dates[-1]),
            min_value=dates[0],
            max_value=dates[-1],
            key="ts_end_date"
        )
    with col3:
        ts_k = st.selectbox("Clusters:", list(range(1, 7)), index=2, key="ts_k")

    if start_date > end_date:
        st.error("Start date must be before or equal to end date")
        return

    # Prepare daily time series data for selected date range
    pivot_daily, coords, day_info = prepare_daily_time_series_data(combined, start_date, end_date, max_stations=session_state.max_stations_complex_viz)

    if not pivot_daily.empty and day_info:
        # Display the selected date range info
        start_day = day_info[0]['date'].strftime('%d/%m/%y')
        end_day = day_info[-1]['date'].strftime('%d/%m/%y')
        st.write(f"**Analyzing daily patterns: {start_day} - {end_day}** ({len(day_info)} days)")

        # Perform clustering
        ts_res, kmeans_model = perform_time_series_clustering(pivot_daily, ts_k, coords)

        # Show the clustering map
        st.plotly_chart(create_time_series_cluster_map(ts_res), use_container_width=True)

        # Show cluster characteristics
        date_range_str = f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
        render_cluster_analysis(ts_res, pivot_daily, day_info, date_range_str, combined)

    else:
        st.warning(f"No data available for the selected date range.")


def render_cluster_analysis(ts_res, pivot_daily, day_info, date_range_str, combined):
    """Render detailed cluster analysis section"""

    if not ts_res.empty and len(pivot_daily.columns) > 0:
        # Create cluster analysis
        cluster_data = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Day labels are already in DD/MM format
        day_labels = list(pivot_daily.columns)

        # Calculate average patterns for each cluster
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

        # 1. Cluster Data Distribution - Combined Whisker Plot
        st.subheader("Cluster Data Distribution")

        render_cluster_distribution_plot(cluster_data, date_range_str, colors)

        # 2. Daily Patterns with Trend Lines (Z-Score only)
        st.subheader("Daily Usage Patterns")
   
        render_daily_patterns_plot(cluster_data, day_labels, date_range_str, colors)

    else:
        # Show basic cluster summary if no time series data
        cluster_summary = ts_res.groupby("cluster").size().reset_index(name="station_count")
        st.dataframe(cluster_summary, use_container_width=True)


def render_cluster_distribution_plot(cluster_data, date_range_str, colors):
    """Render cluster data distribution whisker plot"""
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
    """Render daily patterns with Z-Score normalization"""
    fig_patterns = go.Figure()

    # Add zero reference line
    fig_patterns.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Mean Reference",
        annotation_position="bottom right"
    )

    for i, cluster_info in enumerate(cluster_data):
        color = colors[i % len(colors)]
        cluster_id = cluster_info['cluster']

        # Get the pattern and apply Z-Score normalization
        original_pattern = cluster_info['pattern']

        if np.std(original_pattern) > 0:
            normalized_pattern = (original_pattern - np.mean(original_pattern)) / np.std(original_pattern)
        else:
            normalized_pattern = np.zeros_like(original_pattern)

        # Calculate trend line on normalized data
        x_values = np.arange(len(normalized_pattern))
        if len(x_values) > 1:
            trend_slope, trend_intercept = np.polyfit(x_values, normalized_pattern, 1)
            trend_line = trend_slope * x_values + trend_intercept
        else:
            trend_slope = 0
            trend_line = normalized_pattern

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

    fig_patterns.update_layout(
        title=f"Daily Net Balance Patterns - {date_range_str}",
        xaxis_title="Day (DD/MM)",
        yaxis_title="Z-Score Normalized Net Balance",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig_patterns, use_container_width=True)


def render_timeline_mode(combined, dates, session_state):
    """Render the timeline mode interface"""
    # Timeline mode controls in sidebar (radius only)
    st.sidebar.header("Timeline Options")
    radius_m = st.sidebar.slider("Clustering radius (m):", 100, 200, 100, 10, key="timeline_radius")

    st.sidebar.header("Select Date Range")
    start_date = st.sidebar.date_input("Start date:", value=dates[0], min_value=dates[0], max_value=dates[-1])
    end_date = st.sidebar.date_input("End date:",
                                     value=min(dates[-1], dates[0] + pd.Timedelta(days=6)),
                                     min_value=dates[0], max_value=dates[-1])

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    # Station category controls above the map
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        show_dep = st.checkbox("More departures", True, key="timeline_dep")
    with col2:
        show_arr = st.checkbox("More arrivals", True, key="timeline_arr")
    with col3:
        show_bal = st.checkbox("Balanced", True, key="timeline_bal")
    with col4:
        pass  # Empty space

    categories = [
        name for name, chk in zip(
            ["More departures", "More arrivals", "Balanced"],
            [show_dep, show_arr, show_bal]
        ) if chk
    ]

    # 1) Animated Timeline Map
    st.subheader(f"Timeline Map — {start_date.strftime('%d/%m/%y')} to {end_date.strftime('%d/%m/%y')}")
    try:
        fig_timeline = create_timeline_map(combined, start_date, end_date, radius_m, categories, max_stations=session_state.max_stations_simple_viz)
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_map")
    except Exception as e:
        st.error(f"Error creating timeline map: {e}")

    # 2) Daily Rides Bar Chart
    st.subheader("Daily Ride Counts")
    try:
        fig_daily_rides = create_daily_rides_bar_chart(combined, start_date, end_date, max_stations=session_state.max_stations_simple_viz)
        st.plotly_chart(fig_daily_rides, use_container_width=True, key="daily_rides_chart")
    except Exception as e:
        st.error(f"Error creating daily rides chart: {e}")


def render_models_map_mode(combined, dates, session_state):
    """Render the advanced models map interface - showing all models together"""
    st.subheader(" Analytics Models")
    
    # Initialize analysis_date with a default value
    analysis_date = dates[0] if dates else date.today()

    # ═══════════════════════════════════════════════════════════════════════════════
    # STATION SELECTION MAP
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("Select Station from Map")

    # Create station selection map
    available_stations = sorted(combined['station_name'].unique())
    station_coords = combined[['station_name', 'lat', 'lng']].drop_duplicates().reset_index(drop=True)

    # Initialize session state for selected station if not exists
    if 'selected_station_models' not in st.session_state:
        st.session_state.selected_station_models = available_stations[0]

    # Create station selection map (simplified)
    fig_station_map = go.Figure()

    # Add all stations to the map
    fig_station_map.add_trace(go.Scattermapbox(
        lat=station_coords['lat'],
        lon=station_coords['lng'],
        mode='markers',
        marker=dict(
            size=6,
            color='black',
            opacity=0.8
        ),
        text=station_coords['station_name'],
        customdata=station_coords['station_name'],
        hovertemplate="<b>%{text}</b><br>Click to select this station<extra></extra>",
        name="Available Stations",
        showlegend=False
    ))

    # Highlight selected station
    if st.session_state.selected_station_models in station_coords['station_name'].values:
        selected_coords = station_coords[station_coords['station_name'] == st.session_state.selected_station_models]
        if not selected_coords.empty:
            fig_station_map.add_trace(go.Scattermapbox(
                lat=selected_coords['lat'],
                lon=selected_coords['lng'],
                mode='markers',
                marker=dict(
                    size=12,
                    color='red',
                    opacity=1.0
                ),
                text=selected_coords['station_name'],
                hovertemplate="<b>%{text}</b><br>Selected Station<extra></extra>",
                name="Selected Station",
                showlegend=False
            ))

    fig_station_map.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=40.7128, lon=-74.0060),
            zoom=11
        ),
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )

    # Display the interactive map
    selected_points = st.plotly_chart(fig_station_map, use_container_width=True, key="station_selection_map", on_select="rerun")
    
    # Handle map click selection - more robust handling
    if selected_points and hasattr(selected_points, 'selection') and selected_points.selection:
        if selected_points.selection.get('points'):
            clicked_point = selected_points.selection['points'][0]
            
            # Try different ways to get the point index
            point_index = None
            if 'pointIndex' in clicked_point:
                point_index = clicked_point['pointIndex']
            elif 'point_index' in clicked_point:
                point_index = clicked_point['point_index']
            elif 'customdata' in clicked_point:
                # If we stored station name in customdata
                clicked_station = clicked_point['customdata']
                if clicked_station in available_stations:
                    st.session_state.selected_station_models = clicked_station
                    st.rerun()
            
            if point_index is not None and point_index < len(station_coords):
                clicked_station = station_coords.iloc[point_index]['station_name']
                st.session_state.selected_station_models = clicked_station
                st.rerun()

    # Station selection controls
    col1, col2 = st.columns([2, 1])
    with col1:
        # Dropdown as primary selection method
        selected_station_dropdown = st.selectbox(
            "Select Station:",
            available_stations,
            index=available_stations.index(
                st.session_state.selected_station_models) if st.session_state.selected_station_models in available_stations else 0,
            key="station_dropdown"
        )

        # Update session state if dropdown changes
        if selected_station_dropdown != st.session_state.selected_station_models:
            st.session_state.selected_station_models = selected_station_dropdown

    with col2:
        # Show selected station info with better styling
        selected_info = f"**Selected Station:**\n{st.session_state.selected_station_models}"
        st.success(selected_info)

        # Show station coordinates
        if st.session_state.selected_station_models in station_coords['station_name'].values:
            coords = station_coords[station_coords['station_name'] == st.session_state.selected_station_models].iloc[0]
            st.caption(f"Location: {coords['lat']:.4f}, {coords['lng']:.4f}")

    selected_station = st.session_state.selected_station_models

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. ARIMA FORECAST
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("ARIMA Forecast")


    # Forecast days control moved to main area
    col1, col2, col3 = st.columns([1, 1.5, 1.5])
    with col1:
        forecast_days_arima = st.slider("Forecast Days:", 3, 14, 7, key="arima_days")
    with col2:
        selected_info = f"**Selected Station:**\n{selected_station}"
        st.success(selected_info)
    with col3:
        pass  # Empty column for spacing

    with st.spinner("Running ARIMA model..."):
        try:
            # Filter data to only use data from first of June onwards
            combined_filtered = combined.copy()
            combined_filtered['date'] = pd.to_datetime(combined_filtered['date'])
            
            # Determine the year from the data and set first of June of that year
            data_year = combined_filtered['date'].dt.year.min()
            june_start = pd.to_datetime(f'{data_year}-06-01')
            combined_filtered = combined_filtered[combined_filtered['date'] >= june_start]
            
            if combined_filtered.empty:
                st.warning("No data available from first of June onwards for ARIMA analysis.")
            else:
                fig_arima, message_arima = create_arima_forecast(combined_filtered, selected_station, forecast_days_arima)
                st.plotly_chart(fig_arima, use_container_width=True)
        except Exception as e:
            st.error(f"Error in ARIMA forecast: {e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. PEAK/OFF-PEAK PREDICTION  
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("Peak/Off-Peak Analysis")
    
    # Simple date selection
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_mode = st.selectbox(
            "Analysis Mode:",
            ["Single Day", "Date Range", "All Time"],
            help="Choose the time period for peak analysis"
        )
    
    with col2:
        if analysis_mode == "Single Day":
            analysis_start = st.date_input(
                "Select Date:",
                value=analysis_date,
                min_value=dates[0],
                max_value=dates[-1],
                key="peak_single"
            )
            analysis_end = None
        elif analysis_mode == "Date Range":
            analysis_start = st.date_input(
                "Start Date:",
                value=analysis_date,
                min_value=dates[0],
                max_value=dates[-1],
                key="peak_range_start"
            )
            analysis_end = st.date_input(
                "End Date:",
                value=min(analysis_date + datetime.timedelta(days=7), dates[-1]),
                min_value=analysis_start,
                max_value=dates[-1],
                key="peak_range_end"
            )
        else:  # All Time
            analysis_start = dates[0]
            analysis_end = dates[-1]
    
   
    # Run peak analysis
    with st.spinner("Analyzing peak periods..."):
        try:
            use_all_time = (analysis_mode == "All Time")
            fig_peak = predict_peak_periods_standalone(
                combined, 
                analysis_start, 
                analysis_end, 
                use_all_time
            )
            st.plotly_chart(fig_peak, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error in peak analysis: {e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. ANOMALY DETECTION
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader(" Monthly Anomaly Detection")


    # Z-Score threshold setting above the anomaly detection (shorter slider)
    col1, col2 = st.columns([1, 2])
    with col1:
        z_threshold = st.slider("Z-Score Threshold (Anomaly Detection):", 2.0, 3.0, 2.5, 0.1)
    with col2:
        pass

    # Month selector
    available_months = sorted(pd.to_datetime(combined['date']).dt.to_period('M').unique())
    month_options = {str(month): month.strftime('%B %Y') for month in available_months}
    selected_month_str = st.selectbox(
        "Select Month:",
        options=list(month_options.keys()),
        format_func=lambda x: month_options[x],
        key="anomaly_month_select"
    )
    selected_month = pd.Period(selected_month_str)

    # Get a date from the selected month for anomaly detection
    month_dates = [d for d in dates if pd.Period(d, freq='M') == selected_month]
    if month_dates:
        analysis_date = month_dates[0]
    else:
        analysis_date = dates[0]

    with st.spinner("Detecting anomalies..."):
        try:
            fig_anomaly, message_anomaly = detect_station_anomalies(combined, analysis_date, z_threshold)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            st.info(message_anomaly)
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")


if __name__ == "__main__":
    main()


