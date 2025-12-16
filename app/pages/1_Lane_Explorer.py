"""
Lane Explorer Page
Interactive map-based lane exploration and analysis.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_raw_data, compute_lane_statistics
from utils.map_utils import create_lane_map, create_single_lane_map, add_legend_to_map

st.set_page_config(
    page_title="Lane Explorer",
    page_icon="🗺️",
    layout="wide"
)

st.markdown("# 🗺️ Lane Explorer")
st.markdown("Explore shipping lanes and their performance metrics on an interactive map.")

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
    st.stop()

# Sidebar filters
st.sidebar.markdown("### Filters")

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

if performance_filter == "High (>80%)":
    filtered_stats = filtered_stats[filtered_stats['on_time_rate'] >= 0.8]
elif performance_filter == "Medium (40-80%)":
    filtered_stats = filtered_stats[
        (filtered_stats['on_time_rate'] >= 0.4) & 
        (filtered_stats['on_time_rate'] < 0.8)
    ]
elif performance_filter == "Low (<40%)":
    filtered_stats = filtered_stats[filtered_stats['on_time_rate'] < 0.4]

st.sidebar.markdown(f"**{len(filtered_stats)}** lanes shown")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Interactive Lane Map")
    
    try:
        from streamlit_folium import st_folium
        
        # Lane selection for highlighting
        selected_lane = st.selectbox(
            "Highlight Lane (optional)",
            options=["None"] + filtered_stats['lane_zip3_pair'].tolist(),
            help="Select a lane to highlight on the map"
        )
        
        selected_lane_id = None
        if selected_lane != "None":
            selected_lane_id = filtered_stats[
                filtered_stats['lane_zip3_pair'] == selected_lane
            ]['lane_id'].iloc[0]
        
        # Create and display map
        with st.spinner("Generating map..."):
            m = create_lane_map(filtered_stats, selected_lane_id)
            add_legend_to_map(m)
        
        st_data = st_folium(m, width=None, height=500, key="main_map")
        
    except ImportError:
        st.warning("Map requires streamlit-folium. Install with: `pip install streamlit-folium`")
        st.info("Showing data in table format below.")

with col2:
    st.markdown("### Quick Stats")
    
    total_lanes = len(filtered_stats)
    avg_on_time = filtered_stats['on_time_rate'].mean() * 100
    total_shipments = filtered_stats['total_shipments'].sum()
    
    st.metric("Total Lanes", f"{total_lanes:,}")
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

# Lane details table
st.markdown("---")
st.markdown("### Lane Details")

# Prepare display dataframe
display_df = filtered_stats[[
    'lane_zip3_pair', 'origin_zip_3d', 'dest_zip_3d',
    'total_shipments', 'avg_transit_hours', 'on_time_rate', 'avg_distance'
]].copy()

display_df['on_time_rate'] = (display_df['on_time_rate'] * 100).round(1)
display_df['avg_transit_hours'] = display_df['avg_transit_hours'].round(1)
display_df['avg_distance'] = display_df['avg_distance'].round(0)

display_df.columns = ['Lane', 'Origin', 'Destination', 'Shipments', 
                      'Avg Hours', 'On-Time %', 'Avg Miles']

# Sort options
sort_col = st.selectbox(
    "Sort by",
    options=['Shipments', 'On-Time %', 'Avg Hours', 'Avg Miles'],
    index=0
)
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
    label="Download Lane Data",
    data=csv,
    file_name="lane_statistics.csv",
    mime="text/csv"
)

