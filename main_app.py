import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import date
import plotly.express as px

# Import helper functions
from helper_functions import (
    months,
    load_processed_data,
    prepare_geodata_and_weights,
    prepare_daily_time_series_data,
    load_weather,
    perform_time_series_clustering,
    create_map_visualization,
    create_time_series_cluster_map,
    create_timeline_map,
    create_daily_rides_bar_chart,
    create_spider_plot_for_month,
    create_arima_forecast,
    create_prophet_forecast,
    predict_peak_periods,
    create_weather_impact_analysis,
    detect_station_anomalies
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Citibike Station Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    st.title("NYC Citibike Station Visualization")

    # Load data
    data = load_processed_data()
    if not data:
        st.error("No data loaded.")
        return

    combined = pd.concat(data.values(), ignore_index=True)
    dates = sorted(pd.to_datetime(combined["date"]).dt.date.unique())

    all_stations_df = combined[['station_name', 'lat', 'lng']].drop_duplicates()

    # Map mode selection
    st.sidebar.header("Map Mode")
    mode = st.sidebar.radio("Choose view:", ["Main Map", "Timeline Map", "Models Map"])

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

        df_day = combined[combined["date"] == sel_date]
        if df_day.empty:
            st.warning("No data for this date.")
        else:
            fig = create_map_visualization(df_day, radius_m, categories)
            st.plotly_chart(fig, use_container_width=True)

        # Spider Plot for the entire month
        st.subheader(f"Monthly Station Analysis - {sel_date.strftime('%B %Y')}")
        try:
            fig_spider = create_spider_plot_for_month(combined, sel_date)
            if fig_spider.data:
                st.plotly_chart(fig_spider, use_container_width=True)

                # Add spider plot explanation
                with st.expander("📊 How to Read the Spider Plot"):
                    st.markdown("""
                    **Each station is represented by a spider-like visualization showing 8 key metrics:**

                    **📊 Spider Arms (longer = higher value):**
                    - **Avg Net Balance**: Overall demand pattern
                    - **Volatility**: Day-to-day variation  
                    - **Range**: Spread between max/min values
                    - **Trend Slope**: Rate of change over time
                    - **Peak Value**: Maximum daily demand
                    - **Valley Value**: Maximum daily deficit
                    - **Weekday-Weekend Diff**: Usage pattern difference
                    - **Consistency**: Predictability measure (longer = more predictable)

                    **🎨 Colors:**
                    - **🔴 Red**: Stations with more departures (net positive)
                    - **🔵 Blue**: Stations with more arrivals (net negative) 
                    - **🟢 Green**: Balanced stations (near zero net)
                    """)
            else:
                st.info("No data available for spider plot in this month.")
        except Exception as e:
            st.error(f"Error creating spider plot: {e}")

        # Daily Time-Series Clustering by Month
        render_daily_time_series_section(combined)

    elif mode == "Timeline Map":
        # Timeline mode
        render_timeline_mode(combined, dates)

    else:  # Models Map
        # Models mode
        render_models_map_mode(combined, dates)


def render_daily_time_series_section(combined):
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
        render_cluster_analysis(ts_res, pivot_daily, day_info, month_options, selected_month_str)

    else:
        st.warning(f"No data available for {month_options.get(selected_month_str, 'selected month')}.")


def render_cluster_analysis(ts_res, pivot_daily, day_info, month_options, selected_month_str):
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
            cluster_series = pivot_daily.loc[pivot_daily.index.isin(cluster_stations)]

            if not cluster_series.empty:
                avg_pattern = cluster_series.mean(axis=0).values

                # Calculate trend line using linear regression
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
        render_cluster_distribution_plot(cluster_data, month_options, selected_month_str, colors)

        # 2. Daily Patterns with Trend Lines (Z-Score only)
        render_daily_patterns_plot(cluster_data, day_labels, month_options, selected_month_str, colors)

        # 3. Cluster Statistics and Insights (centered)
        render_cluster_statistics(cluster_data)

        # 4. Model Enhancement Suggestions
        render_model_suggestions()

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
            normalized_pattern = original_pattern - np.mean(original_pattern)

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
        st.dataframe(summary_df, use_container_width=True)

    # Simplified Cluster Insights
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


def render_model_suggestions():
    """Render model enhancement suggestions"""
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


def render_timeline_mode(combined, dates):
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
        fig_timeline = create_timeline_map(combined, start_date, end_date, radius_m, categories)
        st.plotly_chart(fig_timeline, use_container_width=True, key="timeline_map")
    except Exception as e:
        st.error(f"Error creating timeline map: {e}")

    # 2) Daily Rides Bar Chart
    st.subheader("Daily Ride Counts")
    try:
        fig_daily_rides = create_daily_rides_bar_chart(combined, start_date, end_date)
        st.plotly_chart(fig_daily_rides, use_container_width=True, key="daily_rides_chart")
    except Exception as e:
        st.error(f"Error creating daily rides chart: {e}")


def render_models_map_mode(combined, dates):
    """Render the advanced models map interface"""
    st.subheader("Advanced Analytics Models")

    # Model selection
    st.sidebar.header("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model:",
        ["ARIMA Forecast", "Prophet Forecast", "Peak/Off-Peak Prediction", "Weather Impact Analysis",
         "Anomaly Detection"]
    )

    if selected_model in ["ARIMA Forecast", "Prophet Forecast"]:
        # Station selection for forecasting models
        st.sidebar.header("Forecast Settings")
        available_stations = sorted(combined['station_name'].unique())
        selected_station = st.sidebar.selectbox("Select Station:", available_stations)
        forecast_days = st.sidebar.slider("Forecast Days:", 3, 14, 7)

        if selected_model == "ARIMA Forecast":
            st.subheader(f"ARIMA Forecast - {selected_station}")

            with st.spinner("Running ARIMA model..."):
                try:
                    fig, message = create_arima_forecast(combined, selected_station, forecast_days)
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"ARIMA forecast error: {e}")

        elif selected_model == "Prophet Forecast":
            st.subheader(f"Prophet Forecast - {selected_station}")

            with st.spinner("Running Prophet model..."):
                try:
                    fig, message = create_prophet_forecast(combined, selected_station, forecast_days)
                    if fig.data:
                        st.plotly_chart(fig, use_container_width=True)
                        st.success(message)
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Prophet forecast error: {e}")

    elif selected_model == "Peak/Off-Peak Prediction":
        st.sidebar.header("Peak Analysis Settings")
        analysis_date = st.sidebar.date_input("Analysis Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])

        st.subheader(f"Peak/Off-Peak Analysis - {analysis_date.strftime('%d/%m/%y')}")

        with st.spinner("Analyzing peak periods..."):
            try:
                fig = predict_peak_periods(combined, analysis_date)
                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)

                    # Show statistics
                    df_day = combined[combined['date'] == analysis_date]
                    if not df_day.empty:
                        df_day['total_activity'] = df_day['departures'] + df_day['arrivals']
                        peak_threshold = df_day['total_activity'].quantile(0.75)
                        peak_stations = len(df_day[df_day['total_activity'] >= peak_threshold])
                        total_stations = len(df_day)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Stations", total_stations)
                        with col2:
                            st.metric("Peak Stations", peak_stations)
                        with col3:
                            st.metric("Peak Threshold", f"{peak_threshold:.0f} rides")
                else:
                    st.warning("No data available for peak analysis")
            except Exception as e:
                st.error(f"Peak analysis error: {e}")

    elif selected_model == "Weather Impact Analysis":
        st.sidebar.header("Weather Analysis Settings")
        start_date = st.sidebar.date_input("Start Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])
        end_date = st.sidebar.date_input("End Date:", value=min(dates[-1], dates[0] + pd.Timedelta(days=30)),
                                         min_value=dates[0], max_value=dates[-1])

        st.subheader(f"Weather Impact Analysis")

        if start_date > end_date:
            st.error("Start date must be before end date.")
            return

        with st.spinner("Loading weather data and analyzing impact..."):
            try:
                weather_data = load_weather(start_date, end_date)
                fig, message = create_weather_impact_analysis(combined, weather_data)

                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                    st.success(message)
                else:
                    st.error(message)
            except Exception as e:
                st.error(f"Weather analysis error: {e}")

    elif selected_model == "Anomaly Detection":
        st.sidebar.header("Anomaly Detection Settings")
        anomaly_date = st.sidebar.date_input("Analysis Date:", value=dates[0], min_value=dates[0], max_value=dates[-1])
        z_threshold = st.sidebar.slider("Z-Score Threshold:", 1.5, 4.0, 2.5, 0.1)

        st.subheader(f"Anomaly Detection - {anomaly_date.strftime('%d/%m/%y')}")

        with st.spinner("Detecting anomalies..."):
            try:
                fig, message = detect_station_anomalies(combined, anomaly_date, z_threshold)

                if fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                    st.info(message)

                    # Show threshold info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Z-Score Threshold", f"{z_threshold}")
                    with col2:
                        st.metric("Sensitivity", "Higher" if z_threshold < 2.5 else "Lower")
                else:
                    st.warning("No data available for anomaly detection")
            except Exception as e:
                st.error(f"Anomaly detection error: {e}")


if __name__ == "__main__":
    main()