"""
ETA Simulator Page
What-If analysis for carrier selection and ETA prediction.
Uses dynamic cascading filters: Origin State → Destination State → Route → Distance Bucket
Includes YoY comparisons and paginated historical records.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_raw_data
from utils.prediction_utils import format_duration, get_status_emoji
from utils.yoy_utils import calculate_yoy_metrics, format_delta, filter_last_n_years, get_yoy_periods

st.set_page_config(
    page_title="ETA Simulator",
    page_icon="🎯",
    layout="wide"
)

st.markdown("# 🎯 What-If ETA Simulator")
st.markdown("Predict delivery ETAs and compare carrier options using dynamic route filtering.")

# Custom CSS
st.markdown("""
<style>
    .filter-section {
        background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .route-summary {
        background: linear-gradient(145deg, #1a1f24 0%, #22282e 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 3px solid #667eea;
    }
    .yoy-positive { color: #34a853; font-weight: 600; }
    .yoy-negative { color: #ea4335; font-weight: 600; }
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
    
    # Ensure distance_bucket exists
    if 'distance_bucket' not in df.columns:
        bins = [0, 100, 250, 500, 1000, 2000, float('inf')]
        labels = ['0-100mi', '100-250mi', '250-500mi', '500-1000mi', '1000-2000mi', '2000+mi']
        df['distance_bucket'] = pd.cut(df['customer_distance'], bins=bins, labels=labels)
    
    # Create lane_state_pair_distance_bucket if not present
    if 'lane_state_pair_distance_bucket' not in df.columns and 'lane_state_pair' in df.columns:
        df['lane_state_pair_distance_bucket'] = df['lane_state_pair'] + '_' + df['distance_bucket'].astype(str)
    
    return df

try:
    raw_data = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you've run the ETL pipeline first to generate the processed data.")
    st.stop()

# Initialize session state for pagination
if 'show_all_history' not in st.session_state:
    st.session_state.show_all_history = False

# =====================
# DYNAMIC CASCADING FILTERS
# =====================
st.markdown("### 🔍 Route Selection")
st.markdown("*Filters update dynamically as you select each option*")

col1, col2, col3, col4 = st.columns(4)

# 1. Origin State
with col1:
    origin_states = sorted(raw_data['origin_state'].dropna().unique().tolist())
    selected_origin = st.selectbox(
        "📍 Origin State",
        options=['All'] + origin_states,
        key="origin_state"
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
        key="dest_state"
    )

# Filter for route dropdown
if selected_dest != 'All':
    filtered_for_route = filtered_for_dest[filtered_for_dest['dest_state'] == selected_dest]
else:
    filtered_for_route = filtered_for_dest

# 3. Lane ID (specific route identifier)
with col3:
    lanes = sorted(filtered_for_route['lane_id'].dropna().unique().tolist())
    selected_lane = st.selectbox(
        "🛤️ Lane ID",
        options=['All'] + lanes,
        key="lane_id",
        help="Specific lane identifier"
    )

# Filter for distance bucket
if selected_lane != 'All':
    filtered_for_distance = filtered_for_route[filtered_for_route['lane_id'] == selected_lane]
else:
    filtered_for_distance = filtered_for_route

# 4. Distance Bucket (filtered by route)
with col4:
    distance_bucket_order = ['0-100mi', '100-250mi', '250-500mi', '500-1000mi', '1000-2000mi', '2000+mi']
    distance_buckets = filtered_for_distance['distance_bucket'].dropna().unique().tolist()
    distance_buckets_sorted = sorted(
        distance_buckets,
        key=lambda x: distance_bucket_order.index(str(x)) if str(x) in distance_bucket_order else 99
    )
    selected_distance = st.selectbox(
        "📏 Distance Bucket",
        options=['All'] + [str(x) for x in distance_buckets_sorted],
        key="distance_bucket"
    )

# Apply all filters
filtered_data = raw_data.copy()
if selected_origin != 'All':
    filtered_data = filtered_data[filtered_data['origin_state'] == selected_origin]
if selected_dest != 'All':
    filtered_data = filtered_data[filtered_data['dest_state'] == selected_dest]
if selected_lane != 'All':
    filtered_data = filtered_data[filtered_data['lane_id'] == selected_lane]
if selected_distance != 'All':
    filtered_data = filtered_data[filtered_data['distance_bucket'].astype(str) == selected_distance]

# Build route description
route_parts = []
if selected_origin != 'All':
    route_parts.append(selected_origin)
if selected_dest != 'All':
    route_parts.append(selected_dest)

if route_parts:
    route_display = " → ".join(route_parts)
    if selected_lane != 'All':
        route_display += f" (Lane: {selected_lane[:12]}...)" if len(selected_lane) > 12 else f" (Lane: {selected_lane})"
    if selected_distance != 'All':
        route_display += f" [{selected_distance}]"
elif selected_lane != 'All':
    route_display = f"Lane: {selected_lane[:20]}..." if len(selected_lane) > 20 else f"Lane: {selected_lane}"
else:
    route_display = "All Routes"

# =====================
# ROUTE SUMMARY WITH YoY
# =====================
st.markdown("---")
st.markdown("### 📊 Route Summary")

# Calculate YoY metrics for filtered data (uses data's max date as reference)
yoy = calculate_yoy_metrics(filtered_data, window_days=90)
periods = yoy['periods']

st.caption(f"📅 YoY: {periods['current_start'].strftime('%b %d')} - {periods['current_end'].strftime('%b %d, %Y')} vs same period last year")

# Summary metrics with YoY
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Selection",
        route_display if len(route_display) < 20 else route_display[:17] + "...",
    )

with col2:
    volume_delta = yoy.get('volume_change_pct')
    # Shipments up = green (normal), down = red (inverse)
    st.metric(
        "Shipments (90d)",
        f"{yoy['current_count']:,}",
        delta=f"{volume_delta:+.0f}% YoY" if volume_delta is not None else None,
        delta_color="normal"  # Green when positive
    )

with col3:
    on_time_delta = yoy.get('on_time_rate_delta')
    current_otr = yoy.get('current_on_time_rate', 0)
    # On-time rate up = green (normal)
    st.metric(
        "On-Time Rate",
        f"{current_otr:.0f}%",
        delta=f"{on_time_delta:+.1f}% YoY" if on_time_delta is not None else None,
        delta_color="normal"  # Green when positive
    )

with col4:
    transit_delta = yoy.get('avg_transit_delta')
    current_transit = yoy.get('current_avg_transit', 0)
    # Avg transit down = green (inverse: positive delta shows red, negative shows green)
    st.metric(
        "Avg Transit",
        f"{current_transit:.1f} days",
        delta=f"{transit_delta:+.1f}d YoY" if transit_delta is not None else None,
        delta_color="inverse"  # Green when negative (faster is better)
    )

# Additional YoY metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    late_delta = yoy.get('late_rate_delta')
    current_late = yoy.get('current_late_rate', 0)
    # Late rate down = green (inverse)
    st.metric(
        "Late Rate",
        f"{current_late:.0f}%",
        delta=f"{late_delta:+.1f}% YoY" if late_delta is not None else None,
        delta_color="inverse"  # Green when negative (fewer late is better)
    )

with col2:
    std_delta = yoy.get('transit_std_delta')
    current_std = yoy.get('current_transit_std', 0)
    # Variability down = green (inverse)
    st.metric(
        "Transit Variability",
        f"±{current_std:.1f} days",
        delta=f"{std_delta:+.1f}d YoY" if std_delta is not None else None,
        delta_color="inverse"  # Green when negative (less variable is better)
    )

with col3:
    st.metric("Total (All Time)", f"{len(filtered_data):,}")

with col4:
    unique_carriers = filtered_data['carrier_pseudo'].nunique()
    st.metric("Active Carriers", f"{unique_carriers}")

# =====================
# HISTORICAL RECORDS (Default: Last 1 Year)
# =====================
if len(filtered_data) > 0:
    st.markdown("---")
    
    # Default to last 1 year
    if st.session_state.show_all_history:
        history_data = filtered_data.copy()
        history_label = "all time"
    else:
        history_data = filter_last_n_years(filtered_data, years=1)
        history_label = "last 12 months"
    
    with st.expander(f"📋 Historical Records ({len(history_data):,} shipments, {history_label})", expanded=False):
        # Prepare display dataframe
        display_cols = [
            'actual_ship', 'actual_delivery', 'origin_state', 'dest_state',
            'carrier_pseudo', 'carrier_mode', 'actual_transit_days', 
            'otd_designation', 'customer_distance'
        ]
        available_cols = [c for c in display_cols if c in history_data.columns]
        
        history_df = history_data[available_cols].copy()
        
        # Add "Key Factor" column - explain why this shipment took longer/shorter
        # Compare each shipment to the average and identify the main differentiator
        avg_transit = history_data['actual_transit_days'].mean() if len(history_data) > 0 else 0
        avg_distance = history_data['customer_distance'].mean() if len(history_data) > 0 else 0
        
        def get_key_factor(row):
            """Identify the key factor explaining this shipment's transit time."""
            factors = []
            
            transit = row.get('actual_transit_days', avg_transit)
            distance = row.get('customer_distance', avg_distance)
            
            # Check if transit is unusual
            if transit > avg_transit * 1.3:
                # Longer than average - find why
                if distance > avg_distance * 1.2:
                    factors.append(f"Long distance ({distance:.0f}mi)")
                carrier = str(row.get('carrier_pseudo', ''))[:8]
                if carrier:
                    factors.append(f"Carrier: {carrier}")
                mode = row.get('carrier_mode', '')
                if mode == 'LTL':
                    factors.append("LTL mode (slower)")
            elif transit < avg_transit * 0.7:
                # Faster than average
                if distance < avg_distance * 0.8:
                    factors.append(f"Short distance ({distance:.0f}mi)")
                mode = row.get('carrier_mode', '')
                if mode in ['TL Dry', 'TL Flatbed']:
                    factors.append(f"{mode} (direct)")
            else:
                # Normal - just show the top factor
                if distance > avg_distance:
                    factors.append(f"Distance: {distance:.0f}mi")
                else:
                    carrier = str(row.get('carrier_pseudo', ''))[:10]
                    factors.append(f"Carrier: {carrier}")
            
            return factors[0] if factors else "Standard"
        
        history_df['Key Factor'] = history_df.apply(get_key_factor, axis=1)
        
        history_df = history_df.sort_values('actual_ship', ascending=False)
        
        # Format columns
        if 'actual_ship' in history_df.columns:
            history_df['actual_ship'] = pd.to_datetime(history_df['actual_ship']).dt.strftime('%Y-%m-%d')
        if 'actual_delivery' in history_df.columns:
            history_df['actual_delivery'] = pd.to_datetime(history_df['actual_delivery']).dt.strftime('%Y-%m-%d')
        if 'actual_transit_days' in history_df.columns:
            history_df['actual_transit_days'] = history_df['actual_transit_days'].round(1)
        if 'customer_distance' in history_df.columns:
            history_df['customer_distance'] = history_df['customer_distance'].round(0).astype(int)
        
        # Rename for display
        col_renames = {
            'actual_ship': 'Ship Date',
            'actual_delivery': 'Delivery Date', 
            'origin_state': 'Origin',
            'dest_state': 'Dest',
            'carrier_pseudo': 'Carrier',
            'carrier_mode': 'Mode',
            'actual_transit_days': 'Transit Days',
            'otd_designation': 'Status',
            'customer_distance': 'Miles'
        }
        history_df = history_df.rename(columns=col_renames)
        
        # Reorder columns to put Key Factor near Transit Days
        cols_order = ['Ship Date', 'Delivery Date', 'Origin', 'Dest', 'Carrier', 'Mode', 
                      'Transit Days', 'Key Factor', 'Status', 'Miles']
        cols_order = [c for c in cols_order if c in history_df.columns]
        history_df = history_df[cols_order]
        
        # Display in scrollable container (limit rows for performance)
        max_rows = 500
        st.dataframe(
            history_df.head(max_rows),
            use_container_width=True,
            height=400
        )
        
        # Show more options
        if len(history_df) > max_rows:
            st.caption(f"Showing first {max_rows} of {len(history_df):,} records")
        
        # Toggle to show older records
        total_records = len(filtered_data)
        records_in_year = len(filter_last_n_years(filtered_data, years=1))
        older_records = total_records - records_in_year
        
        if older_records > 0 and not st.session_state.show_all_history:
            st.markdown("---")
            if st.button(f"📂 Show More ({older_records:,} older records beyond 12 months)", key="show_more"):
                st.session_state.show_all_history = True
                st.rerun()
        elif st.session_state.show_all_history:
            if st.button("📁 Show Only Last 12 Months", key="show_less"):
                st.session_state.show_all_history = False
                st.rerun()

# =====================
# SHIPMENT DETAILS
# =====================
st.markdown("---")
st.markdown("### ⏰ Shipment Details")

col1, col2, col3 = st.columns(3)

with col1:
    ship_date = st.date_input(
        "📅 Ship Date",
        value=datetime.now().date()
    )

with col2:
    ship_time = st.time_input(
        "🕐 Ship Time",
        value=datetime.now().replace(hour=10, minute=0, second=0).time()
    )

with col3:
    goal_days = st.number_input(
        "🎯 Goal Transit Days",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=0.5,
        help="Target transit days (0 = no target)"
    )

# =====================
# GENERATE RECOMMENDATIONS
# =====================
if st.button("🚀 Generate Carrier Recommendations", type="primary", use_container_width=True):
    if len(filtered_data) == 0:
        st.warning("No shipment data for this route. Try broader filters.")
    else:
        ship_datetime = datetime.combine(ship_date, ship_time)
        goal = goal_days if goal_days > 0 else None
        
        with st.spinner("Analyzing carrier performance..."):
            # Use last year data for recommendations (more relevant)
            recent_filtered = filter_last_n_years(filtered_data, years=1)
            if len(recent_filtered) < 10:
                recent_filtered = filtered_data  # Fall back to all data if not enough recent
            
            # Group by carrier and mode
            carrier_stats = recent_filtered.groupby(['carrier_pseudo', 'carrier_mode']).agg({
                'actual_transit_days': ['mean', 'std', 'count'],
                'otd_designation': lambda x: (x == 'On Time').mean()
            }).reset_index()
            
            # Flatten columns
            carrier_stats.columns = ['carrier', 'mode', 'avg_days', 'std_days', 'sample_size', 'on_time_rate']
            
            # Convert to hours
            carrier_stats['avg_hours'] = carrier_stats['avg_days'] * 24
            carrier_stats['std_hours'] = carrier_stats['std_days'] * 24
            
            # Filter to carriers with enough samples
            carrier_stats = carrier_stats[carrier_stats['sample_size'] >= 3]
            
            if len(carrier_stats) == 0:
                st.warning("Not enough data for reliable carrier recommendations. Try broader filters.")
            else:
                # Sort by average hours
                carrier_stats = carrier_stats.sort_values('avg_hours')
                
                # Build recommendations
                recommendations = []
                for i, row in carrier_stats.iterrows():
                    eta_hours = row['avg_hours']
                    std_hours = row['std_hours'] if pd.notna(row['std_hours']) else eta_hours * 0.2
                    
                    eta_datetime = ship_datetime + timedelta(hours=eta_hours)
                    
                    # Determine status
                    if goal:
                        goal_hours = goal * 24
                        if eta_hours < goal_hours * 0.9:
                            status = 'early'
                        elif eta_hours <= goal_hours * 1.1:
                            status = 'on_time'
                        else:
                            status = 'late'
                    else:
                        # Use on-time rate as proxy
                        if row['on_time_rate'] >= 0.7:
                            status = 'on_time'
                        elif row['on_time_rate'] >= 0.5:
                            status = 'risk'
                        else:
                            status = 'late'
                    
                    # Confidence based on sample size and variance
                    sample_confidence = min(row['sample_size'] / 30, 1.0)
                    variance_confidence = max(0, 1 - (std_hours / (eta_hours + 1)))
                    confidence = (sample_confidence * 0.5 + variance_confidence * 0.5)
                    
                    recommendations.append({
                        'carrier': row['carrier'],
                        'mode': row['mode'],
                        'eta_hours': eta_hours,
                        'eta_datetime': eta_datetime,
                        'eta_formatted': format_duration(eta_hours),
                        'std_hours': std_hours,
                        'status': status,
                        'confidence': confidence,
                        'on_time_rate': row['on_time_rate'],
                        'sample_size': int(row['sample_size']),
                        'is_best': False
                    })
                
                # Mark best option
                if recommendations:
                    recommendations[0]['is_best'] = True
                
                st.session_state.recommendations = recommendations
                st.session_state.ship_datetime = ship_datetime
                st.session_state.goal_days = goal
                st.session_state.route_display = route_display

# =====================
# DISPLAY RESULTS
# =====================
if 'recommendations' in st.session_state and st.session_state.recommendations:
    recommendations = st.session_state.recommendations
    ship_datetime = st.session_state.ship_datetime
    route_display = st.session_state.get('route_display', 'Selected Route')
    
    st.markdown("---")
    st.markdown(f"### 📦 Carrier Options for {route_display}")
    st.caption(f"Ship: {ship_datetime.strftime('%A, %B %d, %Y at %I:%M %p')} | Based on last 12 months performance")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    best = next((r for r in recommendations if r['is_best']), recommendations[0])
    
    with col1:
        st.metric("🏆 Best ETA", best['eta_formatted'])
    with col2:
        st.metric("📊 Options", len(recommendations))
    with col3:
        avg_hours = sum(r['eta_hours'] for r in recommendations) / len(recommendations)
        st.metric("📈 Avg ETA", format_duration(avg_hours))
    with col4:
        on_time_count = sum(1 for r in recommendations if r['status'] in ['on_time', 'early'])
        st.metric("✅ On-Time", f"{on_time_count}/{len(recommendations)}")
    
    st.markdown("---")
    
    # Carrier cards
    for i, rec in enumerate(recommendations):
        is_best = rec['is_best']
        status = rec['status']
        
        with st.container():
            cols = st.columns([0.5, 2.5, 2, 1.5, 2, 1.5])
            
            # Rank
            with cols[0]:
                if is_best:
                    st.markdown("### 🏆")
                else:
                    st.markdown(f"### {i+1}")
            
            # Carrier info
            with cols[1]:
                carrier_display = rec['carrier'][:15] + "..." if len(rec['carrier']) > 15 else rec['carrier']
                st.markdown(f"**{carrier_display}**")
                st.caption(f"{rec['mode']} | {rec['sample_size']} shipments")
            
            # ETA
            with cols[2]:
                st.markdown(f"**{rec['eta_formatted']}**")
                st.caption(rec['eta_datetime'].strftime('%b %d, %I:%M %p'))
            
            # Status
            with cols[3]:
                emoji = get_status_emoji(status)
                if status == 'on_time':
                    st.success(f"{emoji} On-Time")
                elif status == 'early':
                    st.info(f"{emoji} Early")
                elif status == 'late':
                    st.error(f"{emoji} Late")
                else:
                    st.warning(f"{emoji} Risk")
            
            # Confidence
            with cols[4]:
                conf_pct = int(rec['confidence'] * 100)
                st.progress(rec['confidence'])
                st.caption(f"{conf_pct}% confidence")
            
            # On-time rate
            with cols[5]:
                otr = rec['on_time_rate'] * 100
                color = "green" if otr >= 70 else "orange" if otr >= 50 else "red"
                st.markdown(f":{color}[{otr:.0f}% OTR]")
            
            # Details expander
            with st.expander("View Details"):
                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.metric("Avg Transit", f"{rec['eta_hours']:.1f} hrs")
                with detail_cols[1]:
                    st.metric("Std Dev", f"±{rec['std_hours']:.1f} hrs")
                with detail_cols[2]:
                    st.metric("Sample Size", f"{rec['sample_size']}")
                with detail_cols[3]:
                    st.metric("On-Time Rate", f"{rec['on_time_rate']*100:.1f}%")
                
                # Show confidence interval
                lower = max(0, rec['eta_hours'] - 1.96 * rec['std_hours'])
                upper = rec['eta_hours'] + 1.96 * rec['std_hours']
                st.caption(f"95% CI: {format_duration(lower)} - {format_duration(upper)}")
            
            st.markdown("---")
    
    # Download
    results_df = pd.DataFrame([{
        'Carrier': r['carrier'],
        'Mode': r['mode'],
        'ETA Hours': round(r['eta_hours'], 1),
        'ETA DateTime': r['eta_datetime'],
        'Status': r['status'],
        'Confidence': round(r['confidence'], 2),
        'On-Time Rate': round(r['on_time_rate'], 2),
        'Sample Size': r['sample_size'],
        'Is Best': r['is_best']
    } for r in recommendations])
    
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        data=csv,
        file_name=f"carrier_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
