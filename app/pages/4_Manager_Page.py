"""
Manager Page
Lana data for managers to review the performance of a specific lane.
"""

import sys
from pathlib import Path
# Add app directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

import streamlit as st
from streamlit_echarts import st_echarts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_loader import load_raw_data
from utils.prediction_utils import format_duration, get_status_emoji

st.set_page_config(
    page_title="Manager Page",
    page_icon= "📊",
    layout="wide"
)

st.markdown("# 📊 Manager Page")
st.markdown("Use this page to analyze lane performance based on historical shipment data.")

# Custom CSS
st.markdown("""
<style>
    .filter-box {
        background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .carrier-card {
        background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #30363d;
    }
    .carrier-card.best {
        border-left-color: #ffd700;
        background: linear-gradient(90deg, rgba(255, 215, 0, 0.05) 0%, #1e2329 100%);
    }
</style>
""", unsafe_allow_html=True)


# Load data
@st.cache_data
def get_data():
    df = load_raw_data()
    # Ensure we have the necessary columns
    if 'origin_zip_3d' not in df.columns:
        df['origin_zip_3d'] = df['origin_zip'].astype(str).str[:3]
    if 'dest_zip_3d' not in df.columns:
        df['dest_zip_3d'] = df['dest_zip'].astype(str).str[:3]
    if 'distance_bucket' not in df.columns:
        # Create distance buckets
        bins = [0, 100, 250, 500, 1000, 2000, float('inf')]
        labels = ['0-100mi', '100-250mi', '250-500mi', '500-1000mi', '1000-2000mi', '2000+mi']
        df['distance_bucket'] = pd.cut(df['customer_distance'], bins=bins, labels=labels)

    # Create route_key if not present
    if 'route_key' not in df.columns:
        df['route_key'] = (
                df['origin_zip_3d'].astype(str) + '_' +
                df['dest_zip_3d'].astype(str) + '_' +
                df['distance_bucket'].astype(str)
        )

    return df


try:
    raw_data = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()


# =====================
# DYNAMIC FILTERS
# =====================
st.markdown("### 🔍 Select Route")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Origin ZIP3 dropdown
    origin_zip3_options = sorted(raw_data['origin_zip_3d'].dropna().unique().tolist())
    selected_origin = st.selectbox(
        "📍 Origin ZIP3",
        options=['All'] + origin_zip3_options,
        help="3-digit ZIP code of origin"
    )

# Filter destinations based on origin
if selected_origin != 'All':
    filtered_for_dest = raw_data[raw_data['origin_zip_3d'] == selected_origin]
else:
    filtered_for_dest = raw_data

with col2:
    dest_zip3_options = sorted(filtered_for_dest['dest_zip_3d'].dropna().unique().tolist())
    selected_dest = st.selectbox(
        "📍 Destination ZIP3",
        options=['All'] + dest_zip3_options,
        help="3-digit ZIP code of destination"
    )

# Filter lane_id based on origin + dest
if selected_origin != 'All':
    filtered_for_lane = filtered_for_dest.copy()
else:
    filtered_for_lane = raw_data.copy()
if selected_dest != 'All':
    filtered_for_lane = filtered_for_lane[filtered_for_lane['dest_zip_3d'] == selected_dest]

with col3:
    lane_id_options = sorted(filtered_for_lane['lane_id'].dropna().unique().tolist()) if 'lane_id' in filtered_for_lane.columns else []
    selected_lane_id = st.selectbox(
        "🛣️ Lane ID",
        options=['All'] + lane_id_options,
        help="Lane ID (origin-dest pair)"
    )

with col4:
    distance_bucket_options = sorted(filtered_for_lane['distance_bucket'].dropna().unique().tolist(),
                                      key=lambda x: ['0-100mi', '100-250mi', '250-500mi', '500-1000mi', '1000-2000mi', '2000+mi'].index(str(x)) if str(x) in ['0-100mi', '100-250mi', '250-500mi', '500-1000mi', '1000-2000mi', '2000+mi'] else 99)
    selected_distance = st.selectbox(
        "📏 Distance Bucket",
        options=['All'] + [str(x) for x in distance_bucket_options],
        help="Distance range for the shipment"
    )

# Apply all filters
filtered_data = raw_data.copy()
if selected_origin != 'All':
    filtered_data = filtered_data[filtered_data['origin_zip_3d'] == selected_origin]
if selected_dest != 'All':
    filtered_data = filtered_data[filtered_data['dest_zip_3d'] == selected_dest]
if selected_lane_id != 'All' and 'lane_id' in filtered_data.columns:
    filtered_data = filtered_data[filtered_data['lane_id'] == selected_lane_id]
if selected_distance != 'All':
    filtered_data = filtered_data[filtered_data['distance_bucket'].astype(str) == selected_distance]

# Show filter summary
route_desc = []
if selected_origin != 'All':
    route_desc.append(f"Origin: {selected_origin}xx")
if selected_dest != 'All':
    route_desc.append(f"Dest: {selected_dest}xx")
if selected_distance != 'All':
    route_desc.append(f"Distance: {selected_distance}")

if route_desc:
    st.info(f"🛤️ Route Filter: {' → '.join(route_desc)} | **{len(filtered_data):,} historical shipments**")
    st.markdown("### 📋 Filtered Results")
    st.dataframe(filtered_data)

    # ========== New: Quarterly delay rate statistics and visualization ==========
    if 'otd_designation' in filtered_data.columns and 'actual_delivery' in filtered_data.columns:
        # Only keep rows with delivery date
        filtered_data = filtered_data.dropna(subset=['actual_delivery'])
        # Convert to datetime
        filtered_data['actual_delivery'] = pd.to_datetime(filtered_data['actual_delivery'])
        # Calculate quarter
        filtered_data['quarter'] = filtered_data['actual_delivery'].dt.to_period('Q').astype(str)
        # Delay rate (not On Time is considered delay)
        filtered_data['is_delay'] = filtered_data['otd_designation'] != 'On Time'
        # Group by quarter for delay rate and count
        delay_by_quarter = filtered_data.groupby('quarter').agg(
            delay_rate=('is_delay', 'mean'),
            order_count=('is_delay', 'count')
        ).reset_index()
        delay_by_quarter['delay_rate'] = (delay_by_quarter['delay_rate'] * 100).round(2)

        # Calculate thresholds
        delay_80 = np.percentile(delay_by_quarter['delay_rate'], 80) if len(delay_by_quarter) > 0 else 0
        order_median = np.median(delay_by_quarter['order_count']) if len(delay_by_quarter) > 0 else 0

        # Assign quadrant color
        def get_color(row):
            if row['delay_rate'] >= delay_80 and row['order_count'] >= order_median:
                return '#e74c3c'  # High delay, high order (red)
            elif row['delay_rate'] >= delay_80 and row['order_count'] < order_median:
                return '#f39c12'  # High delay, low order (orange)
            elif row['delay_rate'] < delay_80 and row['order_count'] >= order_median:
                return '#3498db'  # Low delay, high order (blue)
            else:
                return '#27ae60'  # Low delay, low order (green)
        delay_by_quarter['color'] = delay_by_quarter.apply(get_color, axis=1)

        # ECharts config for bar chart
        option_delay = {
            "title": {"text": "Quarterly Delay Rate (%)"},
            "tooltip": {},
            "xAxis": {"type": "category", "data": delay_by_quarter['quarter'].tolist()},
            "yAxis": {"type": "value", "min": 0, "max": 100},
            "series": [{
                "data": [float(x) for x in delay_by_quarter['delay_rate']],
                "type": "bar",
                "label": {"show": True, "position": "top", "formatter": "{c}%"}
            }]
        }

        # ECharts config for scatter plot
        scatter_data = [
            {
                "value": [float(row['delay_rate']), int(row['order_count'])],
                "itemStyle": {"color": row['color']},
                "name": row['quarter']
            }
            for _, row in delay_by_quarter.iterrows()
        ]
        scatter_option = {
            "title": {"text": "Quarterly Delay Rate vs Order Count"},
            "tooltip": {"trigger": "item", "formatter": "Quarter: {b}<br/>Delay Rate: {c0}%<br/>Order Count: {c1}"},
            "xAxis": {
                "name": "Delay Rate (%)",
                "min": 0,
                "max": 100
            },
            "yAxis": {
                "name": "Order Count",
                "min": 0,
                "max": int(delay_by_quarter['order_count'].max() * 1.1) if len(delay_by_quarter) > 0 else 1
            },
            "series": [
                {
                    "type": "scatter",
                    "data": scatter_data,
                    "symbolSize": 18,
                    "label": {"show": True, "formatter": "{b}"}
                },
                # Vertical line (delay 80%)
                {
                    "type": "line",
                    "markLine": {
                        "silent": True,
                        "data": [
                            {"xAxis": float(delay_80)}
                        ],
                        "lineStyle": {"type": "dashed", "color": "#888"}
                    }
                },
                # Horizontal line (order median)
                {
                    "type": "line",
                    "markLine": {
                        "silent": True,
                        "data": [
                            {"yAxis": float(order_median)}
                        ],
                        "lineStyle": {"type": "dashed", "color": "#888"}
                    }
                }
            ]
        }

        # Toggle between bar and scatter
        chart_type = st.radio(
            "Select chart type:",
            ["Bar Chart (Delay Rate by Quarter)", "Scatter Plot (Delay Rate vs Order Count)"],
            horizontal=True
        )
        if chart_type == "Bar Chart (Delay Rate by Quarter)":
            st_echarts(options=option_delay, height="400px")
        else:
            st_echarts(options=scatter_option, height="500px")
    else:
        st.warning("Data missing otd_designation or actual_delivery fields, cannot calculate delay rate.")
    # ========== END ==========
else:
    st.info(f"🌐 All routes selected | **{len(filtered_data):,} historical shipments**")

# ========== New: Comments Section ==========
st.markdown("---")
st.markdown("### 💬 Human Note (Comments Section)")
if 'comments' not in st.session_state:
    st.session_state['comments'] = []

with st.form("comment_form", clear_on_submit=True):
    comment_input = st.text_area("Write your comment:", max_chars=300)
    submitted = st.form_submit_button("Submit Comment")
    if submitted and comment_input.strip():
        st.session_state['comments'].append({
            'text': comment_input.strip(),
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        st.success("Comment submitted!")

if st.session_state['comments']:
    st.markdown("#### 📝 All Comments:")
    for c in reversed(st.session_state['comments']):
        st.markdown(f"- {c['text']}  ")
        st.caption(f"🕒 {c['time']}")
else:
    st.info("No comments yet, be the first to share your thoughts!")
