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
    create_prophet_forecast,
    predict_peak_periods_standalone,
    detect_station_anomalies,
    create_arrivals_departures_spider_plot,
    create_weekly_seasonal_plot
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
    st.title("NYC Citibike Station Visualization")

    # Initialize session state for performance settings
    if 'max_stations_global' not in st.session_state:
        st.session_state.max_stations_global = MAX_STATIONS_GLOBAL
    if 'max_stations_simple_viz' not in st.session_state:
        st.session_state.max_stations_simple_viz = MAX_STATIONS_SIMPLE_VIZ
    if 'max_stations_complex_viz' not in st.session_state:
        st.session_state.max_stations_complex_viz = MAX_STATIONS_COMPLEX_VIZ
    if 'use_test_data' not in st.session_state:
        st.session_state.use_test_data = False

    # Data source selection
    st.sidebar.header("📊 Data Source")
    use_test_data = st.sidebar.checkbox(
        "Use Test Data (for presentations)", 
        value=st.session_state.use_test_data,
        help="Use lightweight synthetic data instead of CSV files. Perfect for demos and presentations!"
    )
    st.session_state.use_test_data = use_test_data

    # Load data with configurable limit
    data = load_processed_data(
        max_stations=st.session_state.max_stations_global,
        use_test_data=st.session_state.use_test_data
    )
    if not data:
        st.error("No data loaded.")
        return

    combined = pd.concat(data.values(), ignore_index=True)
    dates = sorted(pd.to_datetime(combined["date"]).dt.date.unique())

    all_stations_df = combined[['station_name', 'lat', 'lng']].drop_duplicates()

    # Map mode selection
    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Main Map", "Timeline Map", "Models Map"])
    
    # Performance configuration
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Performance Settings")
    
    # Advanced settings toggle
    show_advanced = st.sidebar.checkbox("Show Advanced Settings", help="Adjust data limits for performance tuning")
    
    if show_advanced:
        st.sidebar.markdown("**Data Limits:**")
        
        # Allow users to override the default settings
        global_limit = st.sidebar.slider(
            "Global Data Limit:", 
            min_value=50, 
            max_value=2000, 
            value=st.session_state.max_stations_global,
            step=50,
            help="Maximum stations for main data loading. Higher = more data but slower performance."
        )
        
        simple_viz_limit = st.sidebar.slider(
            "Simple Visualizations:", 
            min_value=50, 
            max_value=1000, 
            value=st.session_state.max_stations_simple_viz,
            step=50,
            help="Maximum stations for maps and basic charts."
        )
        
        complex_viz_limit = st.sidebar.slider(
            "Complex Visualizations:", 
            min_value=25, 
            max_value=500, 
            value=st.session_state.max_stations_complex_viz,
            step=25,
            help="Maximum stations for clustering and spider plots."
        )
        
        # Update session state if user changed them
        if global_limit != st.session_state.max_stations_global:
            st.session_state.max_stations_global = global_limit
            st.cache_data.clear()  # Clear cache when limits change
            
        if simple_viz_limit != st.session_state.max_stations_simple_viz:
            st.session_state.max_stations_simple_viz = simple_viz_limit
            
        if complex_viz_limit != st.session_state.max_stations_complex_viz:
            st.session_state.max_stations_complex_viz = complex_viz_limit
        
        st.sidebar.info(f"Current limits: {global_limit}/{simple_viz_limit}/{complex_viz_limit}")
        
    else:
        st.sidebar.info(f"Using default limits: {st.session_state.max_stations_global}/{st.session_state.max_stations_simple_viz}/{st.session_state.max_stations_complex_viz}")

    # Add clear cache button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Clear Cache", help="Clear cached data and refresh the app"):
        st.cache_data.clear()
        st.rerun()

    if mode == "Main Map":
        # Static Map Mode controls
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

        # Main Map with date picker above it
        sel_date = st.date_input("Select Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])

        # 1. STATION ACTIVITY MAP
        st.header("📍 Station Activity Map")
        df_day = combined[combined["date"] == sel_date]
        if df_day.empty:
            st.warning("No data for this date.")
        else:
            fig = create_map_visualization(df_day, radius_m, categories, max_stations=st.session_state.max_stations_simple_viz)
            st.plotly_chart(fig, use_container_width=True)

        # 2. DAILY RIDES OVERVIEW
        st.header("📊 Daily Rides Overview")
        st.write("Shows total bike rides and patterns over time with simulated hourly granularity")
        
        # Chart type selection
        chart_type = st.radio(
            "Select chart type:",
            ["Continuous Time Series (Hourly)", "Daily Bar Chart"],
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
            if chart_type == "Continuous Time Series (Hourly)":
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
        st.header("🔄 Daily Time Series Clustering")
        render_daily_time_series_section(combined, st.session_state)

        # 4. SPIDER GLYPH PLOTS
        st.header("🕸️ Advanced Spider Glyph Visualizations")

        # Create tabs for different spider glyph approaches
        spider_tab1, spider_tab2 = st.tabs([
            "🔄 Improved Spider Glyphs", 
            "📊 Reorganized Plots"
        ])

        with spider_tab1:
            st.markdown("### 🎯 Improved Spider Glyph Analysis")
            st.markdown("""
            **Better spider glyphs designed specifically for bike share insights:**
            - **Temporal Patterns**: Rush hour intensity, weekend vs weekday patterns
            - **Station Roles**: Departure hubs, arrival destinations, balanced stations
            """)
            
            # Temporal Pattern Spider
            st.subheader("⏰ Temporal Usage Patterns")
            with st.expander("ℹ️ How to read temporal patterns", expanded=False):
                st.markdown("""
                **🕷️ This spider shows station usage patterns grouped by activity level:**
                - **Rush Hour Intensity**: How much stations favor departure vs arrival times
                - **Midday Activity**: Overall activity during non-peak hours
                - **Weekend vs Weekday**: Ratio showing leisure vs commuter usage
                - **Night Activity**: Base level activity during low-demand periods
                - **Seasonal Variation**: How much patterns change across months
                
                **🎨 Colors represent station activity levels:**
                - **🔴 Red**: High-activity stations (busy hubs)
                - **🟠 Orange**: Medium-activity stations (neighborhood stations)
                - **🔵 Blue**: Low-activity stations (peripheral locations)
                """)
            
            try:
                temporal_spider = create_temporal_pattern_spider(combined, max_stations=st.session_state.max_stations_complex_viz)
                if temporal_spider.data:
                    st.plotly_chart(temporal_spider, use_container_width=True)
                else:
                    st.info("No data available for temporal pattern analysis.")
            except Exception as e:
                st.error(f"Error creating temporal pattern spider: {e}")
            
            # Station Role Spider
            st.subheader("🎭 Station Role Analysis")
            with st.expander("ℹ️ How to read station roles", expanded=False):
                st.markdown("""
                **📊 This spider identifies different station roles in the bike share system:**
                - **Departure Hub Score**: 0 = arrival destination, 1 = departure hub
                - **Peak Hour Dominance**: How much activity concentrates in rush hours
                - **Consistency Score**: How predictable daily patterns are
                - **Volume Level**: Overall traffic level compared to other stations
                - **Balance Score**: How well-balanced arrivals and departures are
                
                **🔍 Use this to identify:**
                - **Departure Hubs**: Residential areas, transit stations
                - **Arrival Destinations**: Business districts, tourist areas
                - **Balanced Stations**: Mixed-use areas, transfer points
                """)
            
            try:
                role_spider = create_station_role_spider(combined, max_stations=10)
                if role_spider.data:
                    st.plotly_chart(role_spider, use_container_width=True)
                else:
                    st.info("No data available for station role analysis.")
            except Exception as e:
                st.error(f"Error creating station role spider: {e}")

        with spider_tab2:
            st.markdown("### 📊 Spider Glyph Visualizations")
            
            # Y-axis selection
            y_axis_option = st.selectbox(
                "Choose Y-axis for Spider Glyph:",
                ["Month", "Distance from Manhattan", "Balance Ratio", "Activity Density", "Time Wheel"],
                help="Different Y-axis options provide different insights into station characteristics"
            )
            
            if y_axis_option == "Month":
                st.subheader("📅 Spider Glyph by Month")
                try:
                    spider_fig = create_spider_glyph_month(combined)
                    st.plotly_chart(spider_fig, use_container_width=True)
                    st.info("Y-axis shows normalized month mapping with improved scaling for better readability")
                except Exception as e:
                    st.error(f"Error creating month spider glyph: {e}")
                
            elif y_axis_option == "Distance from Manhattan":
                st.subheader("📍 Spider Glyph by Distance from Manhattan")
                try:
                    spider_fig = create_spider_glyph_distance(combined)
                    st.plotly_chart(spider_fig, use_container_width=True)
                    st.info("Y-axis shows distance in kilometers from Manhattan center (Times Square)")
                except Exception as e:
                    st.error(f"Error creating distance spider glyph: {e}")
                
            elif y_axis_option == "Balance Ratio":
                st.subheader("⚖️ Spider Glyph by Balance Ratio")
                try:
                    spider_fig = create_spider_glyph_balance_ratio(combined)
                    st.plotly_chart(spider_fig, use_container_width=True)
                    st.info("Y-axis shows departure/arrival balance ratio. Values >1 indicate more departures than arrivals")
                except Exception as e:
                    st.error(f"Error creating balance ratio spider glyph: {e}")
                
            elif y_axis_option == "Activity Density":
                st.subheader("🚀 Spider Glyph by Activity Density")
                try:
                    spider_fig = create_spider_glyph_activity_density(combined)
                    st.plotly_chart(spider_fig, use_container_width=True)
                    st.info("Y-axis shows average rides per day, indicating how busy each station is")
                except Exception as e:
                    st.error(f"Error creating activity density spider glyph: {e}")
                    
            elif y_axis_option == "Time Wheel":
                st.subheader("🕐 Time Wheel Visualization")
                try:
                    time_wheel_fig = create_time_wheel_plot(combined)
                    st.plotly_chart(time_wheel_fig, use_container_width=True)
                    st.info("Polar plot showing activity patterns by hour (angle) and day of week (radius). Each point represents activity level at that time.")
                    with st.expander("ℹ️ How to read the Time Wheel", expanded=False):
                        st.markdown("""
                        **🕐 Time Wheel Interpretation:**
                        - **Angle (Clock Position)**: Hour of day (12 o'clock = midnight, 3 o'clock = 6am, etc.)
                        - **Distance from Center**: Day of week (inner = Monday, outer = Sunday)
                        - **Point Size**: Relative activity level at that hour/day combination
                        - **Colors**: Different days of the week for easy identification
                        
                        **🔍 Patterns to Look For:**
                        - **Rush Hour Spikes**: Large points during morning (7-9am) and evening (5-7pm)
                        - **Weekend Differences**: Different patterns on Saturday/Sunday (outer rings)
                        - **Night Activity**: Smaller points during late night hours (10pm-5am)
                        """)
                except Exception as e:
                    st.error(f"Error creating time wheel plot: {e}")

    elif mode == "Timeline Map":
        render_timeline_mode(combined, dates, st.session_state)
    
    elif mode == "Models Map":
        render_models_map_mode(combined, dates, st.session_state)


def render_daily_time_series_section(combined, session_state):
    """Render the daily time series clustering section"""
    st.subheader("Daily Time Series Clustering")

    # Month selection dropdown
    col1, col2 = st.columns([2, 1])
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
        ts_k = st.selectbox("Clusters:", list(range(1, 7)), index=2, key="ts_k")

    # Prepare daily time series data for selected month
    pivot_daily, coords, day_info = prepare_daily_time_series_data(combined, selected_month, max_stations=session_state.max_stations_complex_viz)

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
        render_cluster_analysis(ts_res, pivot_daily, day_info, month_options, selected_month_str, combined)

    else:
        st.warning(f"No data available for {month_options.get(selected_month_str, 'selected month')}.")


def render_cluster_analysis(ts_res, pivot_daily, day_info, month_options, selected_month_str, combined):
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
        render_cluster_distribution_plot(cluster_data, month_options, selected_month_str, colors)

        # 2. Daily Patterns with Trend Lines (Z-Score only)
        render_daily_patterns_plot(cluster_data, day_labels, month_options, selected_month_str, colors)

        # 3. Cluster Statistics and Insights (centered)
        render_cluster_statistics(cluster_data)

    else:
        # Show basic cluster summary if no time series data
        cluster_summary = ts_res.groupby("cluster").size().reset_index(name="station_count")
        st.dataframe(cluster_summary, use_container_width=True)


def render_cluster_distribution_plot(cluster_data, month_options, selected_month_str, colors):
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
        title=f"Cluster Data Distribution - {month_options[selected_month_str]}",
        yaxis_title="Net Balance",
        xaxis_title="Cluster",
        height=400,
        showlegend=True
    )
    st.plotly_chart(fig_whisker, use_container_width=True)


def render_daily_patterns_plot(cluster_data, day_labels, month_options, selected_month_str, colors):
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
        title=f"Daily Net Balance Patterns - {month_options[selected_month_str]}",
        xaxis_title="Day (DD/MM)",
        yaxis_title="Z-Score Normalized Net Balance",
        height=500,
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    st.plotly_chart(fig_patterns, use_container_width=True)


def render_cluster_statistics(cluster_data):
    """Render cluster statistics and insights"""
    # Center the statistics table
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Cluster summary statistics
        summary_data = []
        for cluster_info in cluster_data:
            avg_pattern = cluster_info['pattern']
            summary_data.append({
                'Cluster': cluster_info['cluster'],
                'Stations': cluster_info['stations'],
                'Avg Net Balance': f"{np.mean(avg_pattern):.1f}",
                'Volatility (Std)': f"{np.std(avg_pattern):.1f}",
                'Max Daily': f"{np.max(avg_pattern):.1f}",
                'Min Daily': f"{np.min(avg_pattern):.1f}"
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

    # Cluster insights
    insights_cols = st.columns(len(cluster_data))
    for i, cluster_info in enumerate(cluster_data):
        with insights_cols[i]:
            avg_pattern = cluster_info['pattern']
            avg_net = np.mean(avg_pattern)
            volatility = np.std(avg_pattern)
            
            if avg_net > 10:
                pattern_type = "🔴 Departure Hub"
            elif avg_net < -10:
                pattern_type = "🟢 Arrival Hub"
            else:
                pattern_type = "🔵 Balanced"
            
            stability = "📈 Volatile" if volatility > 50 else "📊 Stable"
            
            st.metric(
                label=f"Cluster {cluster_info['cluster']}",
                value=f"{cluster_info['stations']} stations",
                delta=f"{avg_net:.1f} avg balance"
            )
            st.write(f"**{pattern_type}**")
            st.write(f"**{stability}**")


def render_timeline_mode(combined, dates, session_state):
    """Render the timeline mode interface"""
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
    st.subheader("Advanced Analytics Models")

    # ═══════════════════════════════════════════════════════════════════════════════
    # STATION SELECTION MAP
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("🗺️ Select Station from Map")

    # Create station selection map
    available_stations = sorted(combined['station_name'].unique())
    station_coords = combined[['station_name', 'lat', 'lng']].drop_duplicates().reset_index(drop=True)

    # Initialize session state for selected station if not exists
    if 'selected_station_models' not in st.session_state:
        st.session_state.selected_station_models = available_stations[0]

    # Create interactive station selection map
    fig_station_map = go.Figure()

    # Add all stations to the map with more subtle styling
    fig_station_map.add_trace(go.Scattermapbox(
        lat=station_coords['lat'],
        lon=station_coords['lng'],
        mode='markers',
        marker=dict(
            size=6,
            color='rgba(70, 130, 180, 0.6)',  # Steel blue with transparency
            opacity=0.8
        ),
        text=station_coords['station_name'],
        hovertemplate="<b>%{text}</b><br>Click to select<extra></extra>",
        name="Available Stations",
        showlegend=False,
        customdata=station_coords['station_name']  # Store station names for click events
    ))

    # Highlight selected station with more prominent but not overwhelming styling
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
        showlegend=False  # Hide legend for cleaner look
    )

    # Display the map and capture click events
    map_click = st.plotly_chart(
        fig_station_map,
        use_container_width=True,
        key="station_selection_map",
        on_select="rerun"
    )

    # Handle map click events
    if map_click and map_click.get('selection') and map_click['selection'].get('points'):
        clicked_points = map_click['selection']['points']
        if clicked_points:
            clicked_station = clicked_points[0].get('customdata')
            if clicked_station and clicked_station != st.session_state.selected_station_models:
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
            key="station_dropdown",
            help="Choose a station for forecasting analysis"
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
            st.caption(f"📍 {coords['lat']:.4f}, {coords['lng']:.4f}")

    # Improved usage instructions
    st.info(
        "💡 **How to select a station:** Use the dropdown menu above or click on any station marker in the map. The selected station will be highlighted in red.")

    selected_station = st.session_state.selected_station_models

    st.markdown("---")

    # Global settings in sidebar
    st.sidebar.header("Global Settings")

    # Date settings for different analyses
    analysis_date = st.sidebar.date_input("Analysis Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])

    # Anomaly detection settings
    z_threshold = st.sidebar.slider("Z-Score Threshold (Anomaly Detection):", 1.5, 4.0, 2.5, 0.1)

    # ═══════════════════════════════════════════════════════════════════════════════
    # 1. ARIMA FORECAST
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("🔮 ARIMA Forecast")

    # Forecast days control moved to main area
    col1, col2 = st.columns([1, 3])
    with col1:
        forecast_days_arima = st.slider("Forecast Days:", 3, 14, 7, key="arima_days")
    with col2:
        st.write(f"**Station:** {selected_station}")

    with st.spinner("Running ARIMA model..."):
        try:
            fig_arima, message_arima = create_arima_forecast(combined, selected_station, forecast_days_arima)
            st.plotly_chart(fig_arima, use_container_width=True)
            st.info(message_arima)
        except Exception as e:
            st.error(f"Error in ARIMA forecast: {e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 2. PROPHET FORECAST
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("📈 Prophet Forecast")

    # Forecast days control moved to main area
    col1, col2 = st.columns([1, 3])
    with col1:
        forecast_days_prophet = st.slider("Forecast Days:", 3, 14, 7, key="prophet_days")
    with col2:
        st.write(f"**Station:** {selected_station}")

    with st.spinner("Running Prophet model..."):
        try:
            fig_prophet, message_prophet = create_prophet_forecast(combined, selected_station, forecast_days_prophet)
            st.plotly_chart(fig_prophet, use_container_width=True)
            st.info(message_prophet)
        except Exception as e:
            st.error(f"Error in Prophet forecast: {e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 3. PEAK/OFF-PEAK PREDICTION
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader("⏰ Peak/Off-Peak Analysis with Continuous Intensity")
    
    # Enhanced date selection for peak analysis
    st.markdown("**📅 Select Analysis Period:**")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        analysis_mode = st.radio(
            "Analysis Mode:",
            ["Single Day", "Date Range", "All Time"],
            help="Choose the time period for peak analysis"
        )
    
    with col2:
        if analysis_mode == "Single Day":
            peak_start_date = st.date_input(
                "Analysis Date:",
                value=analysis_date,
                min_value=dates[0],
                max_value=dates[-1],
                key="peak_single_date"
            )
            peak_end_date = None
            
        elif analysis_mode == "Date Range":
            peak_start_date = st.date_input(
                "Start Date:",
                value=analysis_date,
                min_value=dates[0],
                max_value=dates[-1],
                key="peak_start_date"
            )
            
        else:  # All Time
            peak_start_date = dates[0]
            peak_end_date = dates[-1]
    
    with col3:
        if analysis_mode == "Date Range":
            # Calculate max end date (up to 1 month from start)
            max_end_date = min(
                peak_start_date + datetime.timedelta(days=31),
                dates[-1]
            )
            peak_end_date = st.date_input(
                "End Date:",
                value=min(
                    peak_start_date + datetime.timedelta(days=7),
                    max_end_date
                ),
                min_value=peak_start_date,
                max_value=max_end_date,
                key="peak_end_date"
            )
        else:
            st.write("")  # Empty placeholder

    # Analysis information
    if analysis_mode == "Single Day":
        st.info(f"🔍 Analyzing peak patterns for {peak_start_date.strftime('%B %d, %Y')}")
    elif analysis_mode == "Date Range":
        days_diff = (peak_end_date - peak_start_date).days + 1
        st.info(f"🔍 Analyzing peak patterns over {days_diff} days ({peak_start_date.strftime('%b %d')} - {peak_end_date.strftime('%b %d, %Y')})")
    else:
        st.info(f"🔍 Analyzing peak patterns across all available data ({dates[0].strftime('%b %d, %Y')} - {dates[-1].strftime('%b %d, %Y')})")

    with st.spinner("Analyzing peak periods with continuous intensity mapping..."):
        try:
            use_all_time = (analysis_mode == "All Time")
            fig_peak = predict_peak_periods_standalone(
                combined, 
                peak_start_date, 
                peak_end_date, 
                use_all_time
            )
            st.plotly_chart(fig_peak, use_container_width=True)
            
            # Add explanation
            with st.expander("ℹ️ How to read the Peak Analysis", expanded=False):
                st.markdown("""
                **🌡️ Continuous Color Mapping:**
                - **🔵 Cold Blue**: Off-peak stations with low activity
                - **🟡 Yellow**: Medium activity stations
                - **🔴 Hot Red**: Peak stations with high activity
                - **Marker Size**: Proportional to activity level
                
                **📊 Peak Intensity Score:**
                - Calculated as normalized activity level (0-100%)
                - Takes into account the selected time period
                - For date ranges: Uses average daily activity
                - For single days: Uses total daily activity
                
                **🎯 Use Cases:**
                - **Single Day**: Identify daily peak patterns
                - **Date Range**: Find consistently busy stations
                - **All Time**: Discover overall system hotspots
                """)
                
        except Exception as e:
            st.error(f"Error in peak period analysis: {e}")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════════
    # 4. ANOMALY DETECTION
    # ═══════════════════════════════════════════════════════════════════════════════
    st.subheader(f"🚨 Monthly Anomaly Detection - {analysis_date.strftime('%B %Y')}")

    st.write(f"**Z-Score Threshold:** {z_threshold}")
    st.info("ℹ️ This analysis uses the entire month's data to detect anomalous station behavior patterns.")

    with st.spinner("Detecting anomalies..."):
        try:
            fig_anomaly, message_anomaly = detect_station_anomalies(combined, analysis_date, z_threshold)
            st.plotly_chart(fig_anomaly, use_container_width=True)
            st.info(message_anomaly)
        except Exception as e:
            st.error(f"Error in anomaly detection: {e}")


if __name__ == "__main__":
    main()
