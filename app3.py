import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import os

# Set page config
st.set_page_config(
    page_title="NYC Citibike Station Balance Visualization",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directory path
base_dir = r"C:\Users\nacha\OneDrive\Desktop\school stuff\year 3 semester 1\visualization\vis proj\second try"

# Month mappings
months = {
    "202409": "September 2024",
    "202412": "December 2024", 
    "202503": "March 2025",
    "202506": "June 2025"
}

@st.cache_data
def load_processed_data():
    """Load all processed data files"""
    data = {}
    for month_code, month_name in months.items():
        file_path = os.path.join(base_dir, f"processed_{month_code}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date']).dt.date
            data[month_code] = df
        else:
            st.warning(f"Data file not found for {month_name}: {file_path}")
    return data

def create_map_visualization(df, selected_date):
    """Create map visualization showing station gains/losses"""
    
    # Filter data for selected date
    date_data = df[df['date'] == selected_date].copy()
    
    if len(date_data) == 0:
        st.error(f"No data available for {selected_date}")
        return None
    
    # Create categories for visualization
    date_data['category'] = date_data['net_balance'].apply(
        lambda x: 'Gained' if x > 0 else 'Lost' if x < 0 else 'Neutral'
    )
    
    # Create colors and sizes
    date_data['color'] = date_data['category'].map({
        'Gained': 'green',
        'Lost': 'red', 
        'Neutral': 'gray'
    })
    
    # Size based on absolute balance with better scaling and caps
    abs_balance = np.abs(date_data['net_balance'])
    # Cap the maximum size to prevent huge circles
    max_balance = np.percentile(abs_balance, 95)  # Use 95th percentile as max
    capped_balance = np.minimum(abs_balance, max_balance)
    # Scale sizes: minimum 4, maximum 20
    date_data['size'] = 4 + (capped_balance / max_balance * 16) if max_balance > 0 else 4
    
    # Create hover text
    date_data['hover_text'] = date_data.apply(
        lambda row: f"<b>{row['station_name']}</b><br>"
                   f"Net Balance: {int(row['net_balance']):+d} bikes<br>"
                   f"Arrivals: {int(row['arrivals'])}<br>"
                   f"Departures: {int(row['departures'])}<br>"
                   f"Status: {row['category']}", 
        axis=1
    )
    
    # Create the map
    fig = go.Figure()
    
    # Add points for each category
    for category in ['Gained', 'Lost', 'Neutral']:
        category_data = date_data[date_data['category'] == category]
        if len(category_data) > 0:
            fig.add_trace(
                go.Scattermapbox(
                    lat=category_data['lat'],
                    lon=category_data['lng'],
                    mode='markers',
                    marker=dict(
                        size=category_data['size'],
                        color=category_data['color'],
                        opacity=0.7,
                        sizemode='diameter'
                    ),
                    text=category_data['hover_text'],
                    hovertemplate='%{text}<extra></extra>',
                    name=f'{category} ({len(category_data)} stations)',
                    showlegend=True
                )
            )
    
    # Update layout
    fig.update_layout(
        mapbox=dict(
            accesstoken='pk.eyJ1Ijoia2xvb20iLCJhIjoiY21jd2FiMWMyMDBlaDJsc2VxaW50Z2ttcCJ9.eYdMHhxN1no9KMtxEUZDGg',
            style='mapbox://styles/mapbox/streets-v11',
            center=dict(lat=40.7589, lon=-73.9851),  # NYC center
            zoom=11
        ),
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def main():
    st.title("🚲 NYC Citibike Station Balance Visualization")
    st.markdown("Visualize which stations gained or lost bikes at the end of each day")
    
    # Load data
    with st.spinner("Loading data..."):
        processed_data = load_processed_data()
    
    if not processed_data:
        st.error("No processed data found. Please run the preprocessing script first.")
        return
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Month selection
    available_months = list(processed_data.keys())
    month_options = {code: months[code] for code in available_months}
    
    selected_month = st.sidebar.selectbox(
        "Select Month:",
        options=available_months,
        format_func=lambda x: month_options[x],
        index=0
    )
    
    # Get data for selected month
    month_data = processed_data[selected_month]
    
    # Date selection
    available_dates = sorted(month_data['date'].unique())
    
    selected_date = st.sidebar.selectbox(
        "Select Date:",
        options=available_dates,
        format_func=lambda x: x.strftime("%Y-%m-%d (%A)"),
        index=len(available_dates)//2  # Default to middle date
    )
    
    # Display summary statistics
    st.sidebar.subheader("Summary Statistics")
    date_summary = month_data[month_data['date'] == selected_date]
    
    if len(date_summary) > 0:
        total_stations = len(date_summary)
        gained_stations = len(date_summary[date_summary['net_balance'] > 0])
        lost_stations = len(date_summary[date_summary['net_balance'] < 0])
        neutral_stations = len(date_summary[date_summary['net_balance'] == 0])
        
        st.sidebar.metric("Total Active Stations", total_stations)
        st.sidebar.metric("Stations That Gained", gained_stations, 
                         delta=f"{gained_stations/total_stations*100:.1f}%")
        st.sidebar.metric("Stations That Lost", lost_stations,
                         delta=f"{lost_stations/total_stations*100:.1f}%")
        st.sidebar.metric("Neutral Stations", neutral_stations)
        
        total_net_balance = date_summary['net_balance'].sum()
        st.sidebar.metric("Total Net Balance", f"{int(total_net_balance):+d} bikes")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader(f"Station Balance Map - {months[selected_month]}")
        st.caption(f"Date: {selected_date.strftime('%Y-%m-%d (%A)')}")
        
        # Create and display map
        fig = create_map_visualization(month_data, selected_date)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Legend")
        st.markdown("""
        🟢 **Green circles**: Stations that gained bikes  
        🔴 **Red circles**: Stations that lost bikes  
        ⚫ **Gray circles**: Neutral stations (no net change)
        
        **Circle size** represents the magnitude of change.
        """)
        
        # Show top/bottom stations
        if len(date_summary) > 0:
            st.subheader("Top Gainers")
            top_gainers = date_summary.nlargest(5, 'net_balance')[['station_name', 'net_balance']]
            for _, row in top_gainers.iterrows():
                st.write(f"• {row['station_name']}: +{int(row['net_balance'])}")
            
            st.subheader("Top Losers")
            top_losers = date_summary.nsmallest(5, 'net_balance')[['station_name', 'net_balance']]
            for _, row in top_losers.iterrows():
                st.write(f"• {row['station_name']}: {int(row['net_balance'])}")
    
    # Additional analysis section
    st.subheader("Monthly Trends")
    
    # Create daily summary chart
    daily_summary = month_data.groupby('date').agg({
        'net_balance': ['sum', 'count'],
        'arrivals': 'sum',
        'departures': 'sum'
    }).reset_index()
    
    daily_summary.columns = ['date', 'total_net_balance', 'active_stations', 'total_arrivals', 'total_departures']
    
    # Plot daily trends
    fig_trend = go.Figure()
    
    fig_trend.add_trace(
        go.Scatter(
            x=daily_summary['date'],
            y=daily_summary['total_net_balance'],
            mode='lines+markers',
            name='Daily Net Balance',
            line=dict(color='blue', width=2)
        )
    )
    
    fig_trend.update_layout(
        title=f"Daily Net Balance Trend - {months[selected_month]}",
        xaxis_title="Date",
        yaxis_title="Net Balance (bikes)",
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Show data table
    with st.expander("View Raw Data"):
        st.dataframe(
            date_summary[['station_name', 'net_balance', 'arrivals', 'departures']].sort_values('net_balance', ascending=False),
            use_container_width=True
        )

if __name__ == "__main__":
    main()