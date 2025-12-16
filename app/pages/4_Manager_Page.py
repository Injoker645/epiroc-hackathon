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
from utils.yoy_utils import calculate_yoy_metrics, filter_last_n_years

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

# 1. Origin State
with col1:
    origin_states = sorted(raw_data['origin_state'].dropna().unique().tolist())
    selected_origin = st.selectbox(
        "📍 Origin State",
        options=['All'] + origin_states,
        key="origin_state",
        help="Select the origin state (abbreviation)"
    )

# Filter data for next dropdown
if selected_origin != 'All':
    filtered_for_dest = raw_data[raw_data['origin_state'] == selected_origin]
else:
    filtered_for_dest = raw_data

# 2. Destination State (filtered by origin)
with col2:
    dest_states = sorted(filtered_for_dest['dest_state'].dropna().unique().tolist())
    selected_dest = st.selectbox(
        "📍 Destination State",
        options=['All'] + dest_states,
        key="dest_state",
        help="Select the destination state (abbreviation)"
    )

# Filter lane_id based on origin + dest
if selected_origin != 'All':
    filtered_for_lane = filtered_for_dest.copy()
else:
    filtered_for_lane = raw_data.copy()
if selected_dest != 'All':
    filtered_for_lane = filtered_for_lane[filtered_for_lane['dest_state'] == selected_dest]

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
    filtered_data = filtered_data[filtered_data['origin_state'] == selected_origin]
if selected_dest != 'All':
    filtered_data = filtered_data[filtered_data['dest_state'] == selected_dest]
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

        # ========== New: Delay Rate by Carrier Mode (Stacked Bar) ==========
        delay_by_mode = None
        option_mode = None
        if 'carrier_mode' in filtered_data.columns:
            # Group by quarter and carrier_mode
            delay_by_mode = filtered_data.groupby(['quarter', 'carrier_mode']).agg(
                delay_rate=('is_delay', 'mean'),
                order_count=('is_delay', 'count')
            ).reset_index()
            delay_by_mode['delay_rate'] = (delay_by_mode['delay_rate'] * 100).round(2)
            # Pivot for ECharts
            mode_list = delay_by_mode['carrier_mode'].unique().tolist()
            quarter_list = delay_by_quarter['quarter'].tolist()
            series = []
            for mode in mode_list:
                mode_data = [
                    float(delay_by_mode[(delay_by_mode['quarter'] == q) & (delay_by_mode['carrier_mode'] == mode)]['delay_rate'].values[0])
                    if ((delay_by_mode['quarter'] == q) & (delay_by_mode['carrier_mode'] == mode)).any() else 0
                    for q in quarter_list
                ]
                series.append({
                    "name": mode,
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": True, "position": "top", "formatter": "{c}%"},
                    "data": mode_data
                })
            option_mode = {
                "title": {"text": "Delay Rate by Carrier Mode (Quarterly)", "left": "center"},
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "legend": {"data": mode_list, "top": 30},
                "xAxis": {"type": "category", "data": quarter_list},
                "yAxis": {"type": "value", "min": 0, "max": 100, "name": "Delay Rate (%)"},
                "series": series
            }
        # ========== END Delay Rate by Carrier Mode ==========

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

        # Toggle between bar, scatter, and carrier mode charts
        chart_type = st.radio(
            "Select chart type:",
            [
                "Bar Chart (Delay Rate by Quarter)",
                "Scatter Plot (Delay Rate vs Order Count)",
                "Delay Rate by Carrier Mode (Quarterly)"
            ],
            horizontal=True
        )
        if chart_type == "Bar Chart (Delay Rate by Quarter)":
            st_echarts(options=option_delay, height="400px")
        elif chart_type == "Scatter Plot (Delay Rate vs Order Count)":
            st_echarts(options=scatter_option, height="500px")
        elif chart_type == "Delay Rate by Carrier Mode (Quarterly)":
            if option_mode:
                st_echarts(options=option_mode, height="500px")
            else:
                st.info("No carrier mode data available for this filter.")
    else:
        st.warning("Data missing otd_designation or actual_delivery fields, cannot calculate delay rate.")
    # ========== END ==========
else:
    st.info(f"🌐 All routes selected | **{len(filtered_data):,} historical shipments**")

# ========== New: Key Performance Metrics (YoY Comparison) ==========
if route_desc:
    st.info(f"🛤️ Route Filter: {' → '.join(route_desc)} | **{len(filtered_data):,} historical shipments**")
    # st.markdown("### 📋 Filtered Results")
    # st.dataframe(filtered_data)

    # ========== Key Performance Metrics (YoY Comparison) ==========
    if len(filtered_data) > 0:
        yoy = calculate_yoy_metrics(filtered_data, window_days=90)
        periods = yoy['periods']
        st.markdown("---")
        st.caption(f"📅 Comparing: Last 90 days ({periods['current_start'].strftime('%b %d')} - {periods['current_end'].strftime('%b %d, %Y')}) vs Same period last year ({periods['prior_start'].strftime('%b %d')} - {periods['prior_end'].strftime('%b %d, %Y')})")
        st.markdown("### 📊 Key Performance Metrics (YoY Comparison)")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            volume_delta = yoy.get('volume_change_pct')
            st.metric(
                "Shipments (90d)",
                f"{yoy['current_count']:,}",
                delta=f"{volume_delta:+.1f}% YoY" if volume_delta is not None else None,
                delta_color="normal"
            )
        with col2:
            on_time_delta = yoy.get('on_time_rate_delta')
            current_otr = yoy.get('current_on_time_rate', 0)
            st.metric(
                "On-Time Rate",
                f"{current_otr:.1f}%",
                delta=f"{on_time_delta:+.1f}% YoY" if on_time_delta is not None else None,
                delta_color="normal"
            )
        with col3:
            late_delta = yoy.get('late_rate_delta')
            current_late = yoy.get('current_late_rate', 0)
            st.metric(
                "Late Rate",
                f"{current_late:.1f}%",
                delta=f"{late_delta:+.1f}% YoY" if late_delta is not None else None,
                delta_color="inverse"
            )
        with col4:
            transit_delta = yoy.get('avg_transit_delta')
            current_transit = yoy.get('current_avg_transit', 0)
            st.metric(
                "Avg Transit Days",
                f"{current_transit:.1f}",
                delta=f"{transit_delta:+.1f}d YoY" if transit_delta is not None else None,
                delta_color="inverse"
            )
    # ========== END Key Performance Metrics ==========
# ========== Key Factors from Historical Records ==========
st.markdown("---")
st.markdown("### 🗝️ Possible Factors in Historical Records")
# Use last 1 year for history, fallback to all if too few
history_data = filter_last_n_years(filtered_data, years=1)
if len(history_data) < 50:
    history_data = filtered_data.copy()
# Prepare key factor column (reuse logic from ETA Simulator)
avg_transit = history_data['actual_transit_days'].mean() if 'actual_transit_days' in history_data.columns and len(history_data) > 0 else 0
avg_distance = history_data['customer_distance'].mean() if 'customer_distance' in history_data.columns and len(history_data) > 0 else 0
def get_key_factor(row):
    factors = []
    transit = row.get('actual_transit_days', avg_transit)
    distance = row.get('customer_distance', avg_distance)
    if transit > avg_transit * 1.3:
        if distance > avg_distance * 1.2:
            factors.append(f"Long distance ({distance:.0f}mi)")
        carrier = str(row.get('carrier_pseudo', ''))[:8]
        if carrier:
            factors.append(f"Carrier: {carrier}")
        mode = row.get('carrier_mode', '')
        if mode == 'LTL':
            factors.append("LTL mode (slower)")
    elif transit < avg_transit * 0.7:
        if distance < avg_distance * 0.8:
            factors.append(f"Short distance ({distance:.0f}mi)")
        mode = row.get('carrier_mode', '')
        if mode in ['TL Dry', 'TL Flatbed']:
            factors.append(f"{mode} (direct)")
    else:
        if distance > avg_distance:
            factors.append(f"Distance: {distance:.0f}mi")
        else:
            carrier = str(row.get('carrier_pseudo', ''))[:10]
            factors.append(f"Carrier: {carrier}")
    return factors[0] if factors else "Standard"
if 'actual_transit_days' in history_data.columns and 'customer_distance' in history_data.columns:
    history_data = history_data.copy()
    history_data['Key Factor'] = history_data.apply(get_key_factor, axis=1)
    from collections import Counter
    key_factors = history_data['Key Factor'].dropna().astype(str).tolist()
    factor_counts = Counter(key_factors)
    if factor_counts:
        top_factors = factor_counts.most_common(10)
        other_count = sum([v for k, v in factor_counts.items() if (k, v) not in top_factors])
        labels = [k for k, v in top_factors]
        values = [v for k, v in top_factors]
        if other_count > 0:
            labels.append('Other')
            values.append(other_count)
        donut_option = {
            "title": {"text": "Key Factor Distribution", "left": "center"},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"orient": "vertical", "left": "left", "top": "middle", "data": labels},
            "series": [
                {
                    "name": "Key Factor",
                    "type": "pie",
                    "radius": ["50%", "80%"],
                    "avoidLabelOverlap": False,
                    "label": {"show": True, "position": "outside"},
                    "emphasis": {"label": {"show": True, "fontSize": 16, "fontWeight": "bold"}},
                    "labelLine": {"show": True},
                    "data": [
                        {"value": v, "name": k} for k, v in zip(labels, values)
                    ]
                }
            ]
        }
        # --- Layout: Donut chart left, comments right ---
        donut_col, comment_col = st.columns([1.2, 1])
        with donut_col:
            st_echarts(options=donut_option, height="400px")
            st.caption("Donut chart of key factors for filtered shipments")
        with comment_col:
            # Show filter combination above comment box
            filter_label = f"Filter: Origin={selected_origin}, Dest={selected_dest}, Lane={selected_lane_id}, Distance={selected_distance}"
            st.markdown(f"<div style='font-size:0.95rem;color:#888;margin-bottom:0.5rem'><b>{filter_label}</b></div>", unsafe_allow_html=True)
            # Comment system (scoped to filter)
            def get_filter_key():
                return f"comments_{str(selected_origin)}_{str(selected_dest)}_{str(selected_lane_id)}_{str(selected_distance)}"
            key_factor_filter_id = get_filter_key()
            if key_factor_filter_id not in st.session_state:
                st.session_state[key_factor_filter_id] = []
            with st.form(f"comment_form_{key_factor_filter_id}", clear_on_submit=True):
                comment_input = st.text_area("Write your comment about the key factors:", max_chars=300)
                submitted = st.form_submit_button("Submit Comment")
                if submitted and comment_input.strip():
                    st.session_state[key_factor_filter_id].append({
                        'text': comment_input.strip(),
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    st.success("Comment submitted!")
            if st.session_state[key_factor_filter_id]:
                st.markdown("#### 📝 All Comments for Key Factors:")
                for c in reversed(st.session_state[key_factor_filter_id]):
                    st.markdown(f"- {c['text']}  ")
                    st.caption(f"🕒 {c['time']}")
            else:
                st.info("No comments yet for these key factors, be the first to share your thoughts!")
    else:
        st.info("No key factor data available for this filter.")
else:
    st.info("Not enough data to extract key factors.")
# ========== END Key Factors ==========
