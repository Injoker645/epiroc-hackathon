"""
Lane Explorer Page
Interactive map-based lane exploration and analysis using state-based routes.
Includes YoY comparisons.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_raw_data, compute_lane_statistics
from utils.map_utils import create_lane_map, add_legend_to_map
from utils.yoy_utils import calculate_yoy_metrics, filter_last_n_years, format_delta

st.set_page_config(
    page_title="Lane Explorer",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("# 🗺️ Lane Explorer")
st.markdown("Explore shipping routes by state and their performance metrics on an interactive map.")

# Load data
@st.cache_data
def get_data():
    raw_data = load_raw_data()
    lane_stats = compute_lane_statistics(raw_data)
    return raw_data, lane_stats

try:
    raw_data, lane_stats = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you have the raw data file in the Dataset folder.")
    st.stop()

# Sidebar filters
st.sidebar.markdown("### Filters")

# State filter
all_origin_states = sorted(lane_stats['origin_state'].dropna().unique().tolist())
selected_origin_state = st.sidebar.selectbox(
    "Origin State",
    options=["All"] + all_origin_states
)

# Filter destination states based on origin
if selected_origin_state != "All":
    dest_options = sorted(
        lane_stats[lane_stats['origin_state'] == selected_origin_state]['dest_state'].unique().tolist()
    )
else:
    dest_options = sorted(lane_stats['dest_state'].dropna().unique().tolist())

selected_dest_state = st.sidebar.selectbox(
    "Destination State",
    options=["All"] + dest_options
)

# Performance filter
performance_filter = st.sidebar.selectbox(
    "Performance Level",
    ["All", "High (>80%)", "Medium (40-80%)", "Low (<40%)"]
)

# Minimum shipments filter
min_shipments = st.sidebar.slider(
    "Minimum Shipments",
    min_value=1,
    max_value=500,
    value=10
)

# Apply filters
filtered_stats = lane_stats[lane_stats['total_shipments'] >= min_shipments].copy()

if selected_origin_state != "All":
    filtered_stats = filtered_stats[filtered_stats['origin_state'] == selected_origin_state]
if selected_dest_state != "All":
    filtered_stats = filtered_stats[filtered_stats['dest_state'] == selected_dest_state]

if performance_filter == "High (>80%)":
    filtered_stats = filtered_stats[filtered_stats['on_time_rate'] >= 0.8]
elif performance_filter == "Medium (40-80%)":
    filtered_stats = filtered_stats[
        (filtered_stats['on_time_rate'] >= 0.4) & 
        (filtered_stats['on_time_rate'] < 0.8)
    ]
elif performance_filter == "Low (<40%)":
    filtered_stats = filtered_stats[filtered_stats['on_time_rate'] < 0.4]

st.sidebar.markdown(f"**{len(filtered_stats)}** routes shown")

# Filter raw data for YoY calculations
filtered_raw = raw_data.copy()
if selected_origin_state != "All":
    filtered_raw = filtered_raw[filtered_raw['origin_state'] == selected_origin_state]
if selected_dest_state != "All":
    filtered_raw = filtered_raw[filtered_raw['dest_state'] == selected_dest_state]

# YoY Metrics (uses data's max date as reference)
yoy = calculate_yoy_metrics(filtered_raw, window_days=90)
periods = yoy['periods']

st.markdown("### 📊 Performance Overview (YoY Comparison)")
st.caption(f"📅 {periods['current_start'].strftime('%b %d')} - {periods['current_end'].strftime('%b %d, %Y')} vs same period last year")

col1, col2, col3, col4 = st.columns(4)

with col1:
    volume_delta = yoy.get('volume_change_pct')
    delta_str, delta_color = format_delta(volume_delta, suffix='% YoY', inverse=False, decimals=0)
    st.metric(
        "Shipments (90d)",
        f"{yoy['current_count']:,}",
        delta=delta_str if volume_delta is not None else None,
        delta_color=delta_color
    )

with col2:
    on_time_delta = yoy.get('on_time_rate_delta')
    current_otr = yoy.get('current_on_time_rate', 0)
    delta_str, delta_color = format_delta(on_time_delta, suffix='% YoY', inverse=False, decimals=1)
    st.metric(
        "On-Time Rate",
        f"{current_otr:.0f}%",
        delta=delta_str if on_time_delta is not None else None,
        delta_color=delta_color
    )

with col3:
    late_delta = yoy.get('late_rate_delta')
    current_late = yoy.get('current_late_rate', 0)
    delta_str, delta_color = format_delta(late_delta, suffix='% YoY', inverse=True, decimals=1)
    st.metric(
        "Late Rate",
        f"{current_late:.0f}%",
        delta=delta_str if late_delta is not None else None,
        delta_color=delta_color
    )

with col4:
    transit_delta = yoy.get('avg_transit_delta')
    current_transit = yoy.get('current_avg_transit', 0)
    delta_str, delta_color = format_delta(transit_delta, suffix='d YoY', inverse=True, decimals=1)
    st.metric(
        "Avg Transit",
        f"{current_transit:.1f} days",
        delta=delta_str if transit_delta is not None else None,
        delta_color=delta_color
    )

st.markdown("---")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Interactive Route Map")
    
    try:
        from streamlit_folium import st_folium
        
        # Route selection for highlighting
        route_options = filtered_stats['lane_state_pair'].tolist()
        selected_route = st.selectbox(
            "Highlight Route (optional)",
            options=["None"] + route_options,
            help="Select a route to highlight on the map"
        )
        
        selected_route_id = None if selected_route == "None" else selected_route
        
        # Create and display map
        with st.spinner("Generating map..."):
            m = create_lane_map(filtered_stats, selected_route_id)
            add_legend_to_map(m)
        
        st_data = st_folium(m, width=None, height=500, key="main_map")
        
    except ImportError:
        st.warning("Map requires streamlit-folium. Install with: `pip install streamlit-folium`")
        st.info("Showing data in table format below.")

with col2:
    st.markdown("### Quick Stats")
    
    total_routes = len(filtered_stats)
    avg_on_time = filtered_stats['on_time_rate'].mean() * 100 if len(filtered_stats) > 0 else 0
    total_shipments = filtered_stats['total_shipments'].sum()
    
    st.metric("Total Routes", f"{total_routes:,}")
    st.metric("Avg On-Time Rate", f"{avg_on_time:.1f}%")
    st.metric("Total Shipments", f"{total_shipments:,}")
    
    st.markdown("---")
    st.markdown("### Performance Distribution")
    
    perf_dist = pd.DataFrame({
        'Category': ['High (>80%)', 'Medium (40-80%)', 'Low (<40%)'],
        'Count': [
            len(filtered_stats[filtered_stats['on_time_rate'] >= 0.8]),
            len(filtered_stats[(filtered_stats['on_time_rate'] >= 0.4) & (filtered_stats['on_time_rate'] < 0.8)]),
            len(filtered_stats[filtered_stats['on_time_rate'] < 0.4])
        ]
    })
    st.bar_chart(perf_dist.set_index('Category'))
    
    # Top performing routes
    if len(filtered_stats) > 0:
        st.markdown("---")
        st.markdown("### 🏆 Top Routes")
        top_routes = filtered_stats.nlargest(5, 'on_time_rate')[['lane_state_pair', 'on_time_rate', 'total_shipments']]
        for _, row in top_routes.iterrows():
            otr = row['on_time_rate'] * 100
            st.markdown(f"**{row['lane_state_pair']}**: {otr:.0f}% ({row['total_shipments']} shipments)")

# Route details table
st.markdown("---")
st.markdown("### Route Details (Last 12 Months)")

# Prepare display dataframe
display_df = filtered_stats[[
    'lane_state_pair', 'origin_state', 'dest_state',
    'total_shipments', 'avg_transit_hours', 'on_time_rate', 'avg_distance'
]].copy()

display_df['on_time_rate'] = (display_df['on_time_rate'] * 100).round(1)
display_df['avg_transit_hours'] = display_df['avg_transit_hours'].round(1)
display_df['avg_distance'] = display_df['avg_distance'].round(0)

display_df.columns = ['Route', 'Origin', 'Destination', 'Shipments', 
                      'Avg Hours', 'On-Time %', 'Avg Miles']

# Sort options
col_sort, col_order = st.columns([2, 1])
with col_sort:
    sort_col = st.selectbox(
        "Sort by",
        options=['Shipments', 'On-Time %', 'Avg Hours', 'Avg Miles'],
        index=0
    )
with col_order:
    sort_asc = st.checkbox("Ascending", value=False)

display_df = display_df.sort_values(sort_col, ascending=sort_asc)

st.dataframe(
    display_df,
    hide_index=True,
    use_container_width=True,
    height=400
)

# Download option
csv = display_df.to_csv(index=False)
st.download_button(
    label="📥 Download Route Data",
    data=csv,
    file_name="route_statistics.csv",
    mime="text/csv"
)
