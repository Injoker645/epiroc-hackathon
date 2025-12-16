"""
ETA Simulator Page
What-If analysis for carrier selection and ETA prediction.
Combined shipment details form with comprehensive filtering and SHAP-based explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import re

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_raw_data, load_model, load_processed_features
from utils.prediction_utils import format_duration, get_status_emoji
from utils.yoy_utils import calculate_yoy_metrics, filter_last_n_years
from utils.shap_key_factors import get_key_factor_for_recommendation, batch_compute_key_factors

st.set_page_config(
    page_title="ETA Simulator",
    page_icon="🎯",
    layout="wide"
)

st.markdown("# 🎯 What-If ETA Simulator")
st.markdown("Configure shipment parameters and get optimized carrier recommendations with SHAP explanations.")

# Custom CSS
st.markdown("""
<style>
    .shipment-section {
        background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid #30363d;
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
    .key-factor-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        font-size: 0.85rem;
        font-weight: 600;
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
    
    return df

try:
    raw_data = get_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you've run the ETL pipeline first to generate the processed data.")
    st.stop()

# Initialize session state
if 'show_all_history' not in st.session_state:
    st.session_state.show_all_history = False

# Helper functions
def is_weekend(date):
    """Check if date is a weekend."""
    return date.weekday() >= 5

def parse_range_input(input_str, min_val=0, max_val=float('inf')):
    """Parse range input like '40-2000' or single value."""
    if not input_str or input_str.strip() == '':
        return None
    
    input_str = input_str.strip()
    
    # Check for range
    if '-' in input_str:
        parts = input_str.split('-')
        if len(parts) == 2:
            try:
                low = float(parts[0].strip())
                high = float(parts[1].strip())
                return (low, high)
            except:
                return None
    
    # Single value
    try:
        val = float(input_str)
        return (val, val)
    except:
        return None

def matches_range(value, range_tuple):
    """Check if value matches range."""
    if range_tuple is None:
        return True
    low, high = range_tuple
    return low <= value <= high

# =====================
# SHIPMENT DETAILS SECTION
# =====================
st.markdown("### 📦 Shipment Details")
st.markdown("*Configure all shipment parameters below. Only Origin and Destination are required.*")

with st.container():
    col1, col2 = st.columns(2)
    
    # Origin State (required, multiple selection)
    with col1:
        origin_states = sorted(raw_data['origin_state'].dropna().unique().tolist())
        selected_origins = st.multiselect(
            "📍 Origin State(s) *",
            options=origin_states,
            help="Select one or more origin states (required)"
        )
    
    # Destination State (required, multiple selection)
    with col2:
        dest_states = sorted(raw_data['dest_state'].dropna().unique().tolist())
        selected_dests = st.multiselect(
            "📍 Destination State(s) *",
            options=dest_states,
            help="Select one or more destination states (required)"
        )
    
    # Lane ID (optional, multiple selection with range input)
    col1, col2 = st.columns(2)
    with col1:
        all_lanes = sorted(raw_data['lane_id'].dropna().unique().tolist())
        selected_lanes = st.multiselect(
            "🛤️ Lane ID(s)",
            options=all_lanes,
            help="Select specific lane IDs (optional)"
        )
    
    with col2:
        lane_range_input = st.text_input(
            "🛤️ Lane ID Range",
            value="",
            placeholder="e.g., 40-2000 (type range or leave blank)",
            help="Enter a range like '40-2000' to filter lanes by ID"
        )
        lane_range = parse_range_input(lane_range_input)
    
    # Distance (optional, free entry decimal)
    col1, col2 = st.columns(2)
    with col1:
        distance_input = st.text_input(
            "📏 Distance (miles)",
            value="",
            placeholder="Enter distance in miles (decimal allowed, e.g., 245.5)",
            help="Optional: Enter specific distance in miles"
        )
        try:
            distance_value = float(distance_input) if distance_input.strip() else None
        except:
            distance_value = None
            if distance_input.strip():
                st.warning("Invalid distance value. Please enter a number.")
    
    with col2:
        distance_range_input = st.text_input(
            "📏 Distance Range (miles)",
            value="",
            placeholder="e.g., 40-2000 (type range or leave blank)",
            help="Enter a range like '40-2000' to filter by distance"
        )
        distance_range = parse_range_input(distance_range_input)
    
    # Carrier Mode (optional, multiple selection)
    carrier_modes = sorted(raw_data['carrier_mode'].dropna().unique().tolist())
    selected_modes = st.multiselect(
        "🚚 Carrier Mode(s)",
        options=carrier_modes,
        help="Select carrier modes (LTL, TL Dry, TL Flatbed) - optional"
    )
    
    st.markdown("---")
    
    # Ship Date & Time
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        today = datetime.now().date()
        # Default to today, but allow future dates only
        ship_date = st.date_input(
            "📅 Ship Date *",
            value=today,
            min_value=today,
            help="Select ship date (no weekends, no past dates)"
        )
        
        # Check for weekend
        if is_weekend(ship_date):
            st.warning("⚠️ Selected date is a weekend. Please choose a weekday.")
            ship_date = None
    
    with col2:
        ship_date_flexible = st.checkbox(
            "Flexible Date",
            value=False,
            help="Allow flexible ship date"
        )
    
    with col3:
        # Ship time - if today, can't be in past
        now = datetime.now()
        default_time = now.replace(hour=10, minute=0, second=0, microsecond=0).time()
        
        if ship_date == today:
            min_time = now.time()
        else:
            min_time = datetime.min.time()
        
        ship_time = st.time_input(
            "🕐 Ship Time *",
            value=default_time,
            help="Select ship time (if today, must be future time)"
        )
        
        # Validate time if today
        if ship_date == today and ship_time < now.time():
            st.warning("⚠️ Ship time must be in the future for today's date.")
            ship_time = None
    
    with col4:
        ship_time_flexible = st.checkbox(
            "Flexible Time",
            value=False,
            help="Allow flexible ship time"
        )
    
    # Goal Transit Days (required, integer only)
    col1, col2 = st.columns([2, 1])
    with col1:
        goal_days = st.number_input(
            "🎯 Goal Transit Days *",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            help="Target transit days (required, integer only)"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing

# Validate required fields
if not selected_origins or not selected_dests:
    st.error("⚠️ Please select at least one Origin State and one Destination State.")
    st.stop()

if ship_date is None or ship_time is None:
    st.error("⚠️ Please select a valid ship date and time.")
    st.stop()

# =====================
# FILTER DATA BASED ON SELECTIONS
# =====================
filtered_data = raw_data.copy()

# Filter by origin states
if selected_origins:
    filtered_data = filtered_data[filtered_data['origin_state'].isin(selected_origins)]

# Filter by destination states
if selected_dests:
    filtered_data = filtered_data[filtered_data['dest_state'].isin(selected_dests)]

# Filter by lane IDs
if selected_lanes:
    filtered_data = filtered_data[filtered_data['lane_id'].isin(selected_lanes)]
elif lane_range:
    # Filter by lane ID range (if lane_id is numeric)
    try:
        filtered_data['lane_id_numeric'] = pd.to_numeric(filtered_data['lane_id'], errors='coerce')
        low, high = lane_range
        filtered_data = filtered_data[
            (filtered_data['lane_id_numeric'] >= low) & 
            (filtered_data['lane_id_numeric'] <= high)
        ]
        filtered_data = filtered_data.drop(columns=['lane_id_numeric'])
    except:
        pass

# Filter by distance
if distance_value is not None:
    # Use specific distance value
    filtered_data = filtered_data[
        (filtered_data['customer_distance'] >= distance_value * 0.95) &
        (filtered_data['customer_distance'] <= distance_value * 1.05)
    ]
elif distance_range:
    low, high = distance_range
    filtered_data = filtered_data[
        (filtered_data['customer_distance'] >= low) &
        (filtered_data['customer_distance'] <= high)
    ]

# Filter by carrier modes
if selected_modes:
    filtered_data = filtered_data[filtered_data['carrier_mode'].isin(selected_modes)]

if len(filtered_data) == 0:
    st.warning("No shipment data matches your filters. Please adjust your selections.")
    st.stop()

# =====================
# ROUTE SUMMARY (Below Shipment Details)
# =====================
st.markdown("---")
st.markdown("### 📊 Route Summary")

# Calculate YoY metrics
yoy = calculate_yoy_metrics(filtered_data, window_days=90)
periods = yoy['periods']

st.caption(f"📅 Comparing: {periods['current_start'].strftime('%b %d')} - {periods['current_end'].strftime('%b %d, %Y')} vs same period last year")

# Summary metrics with YoY
col1, col2, col3, col4 = st.columns(4)

with col1:
    volume_delta = yoy.get('volume_change_pct')
    delta_str = f"{volume_delta:+.0f}% YoY" if volume_delta is not None else None
    delta_color = "normal" if volume_delta and volume_delta > 0 else ("off" if volume_delta and abs(volume_delta) < 0.01 else "inverse")
    st.metric(
        "Shipments (90d)",
        f"{yoy['current_count']:,}",
        delta=delta_str,
        delta_color=delta_color
    )

with col2:
    on_time_delta = yoy.get('on_time_rate_delta')
    current_otr = yoy.get('current_on_time_rate', 0)
    delta_str = f"{on_time_delta:+.1f}% YoY" if on_time_delta is not None else None
    delta_color = "normal" if on_time_delta and on_time_delta > 0 else ("off" if on_time_delta and abs(on_time_delta) < 0.01 else "inverse")
    st.metric(
        "On-Time Rate",
        f"{current_otr:.0f}%",
        delta=delta_str,
        delta_color=delta_color
    )

with col3:
    transit_delta = yoy.get('avg_transit_delta')
    current_transit = yoy.get('current_avg_transit', 0)
    delta_str = f"{transit_delta:+.1f}d YoY" if transit_delta is not None else None
    delta_color = "inverse" if transit_delta and transit_delta < 0 else ("off" if transit_delta and abs(transit_delta) < 0.01 else "normal")
    st.metric(
        "Avg Transit",
        f"{current_transit:.1f} days",
        delta=delta_str,
        delta_color=delta_color
    )

with col4:
    late_delta = yoy.get('late_rate_delta')
    current_late = yoy.get('current_late_rate', 0)
    delta_str = f"{late_delta:+.1f}% YoY" if late_delta is not None else None
    delta_color = "inverse" if late_delta and late_delta < 0 else ("off" if late_delta and abs(late_delta) < 0.01 else "normal")
    st.metric(
        "Late Rate",
        f"{current_late:.0f}%",
        delta=delta_str,
        delta_color=delta_color
    )

# =====================
# GENERATE RECOMMENDATIONS
# =====================
st.markdown("---")
st.markdown("### 🚀 Carrier Recommendations")

if st.button("Generate Recommendations", type="primary", use_container_width=True):
    ship_datetime = datetime.combine(ship_date, ship_time)
    goal_hours = goal_days * 24
    
    with st.spinner("Analyzing carriers and computing SHAP-based key factors..."):
        # Load model and feature columns for SHAP computation
        try:
            model = load_model()
            feature_columns = None
            try:
                features_df = load_processed_features()
                feature_columns = features_df.columns.tolist()
            except:
                pass
        except:
            model = None
            feature_columns = None
        
        # Use last year data for recommendations
        recent_filtered = filter_last_n_years(filtered_data, years=1)
        if len(recent_filtered) < 10:
            recent_filtered = filtered_data
        
        # Group by carrier and mode
        carrier_stats = recent_filtered.groupby(['carrier_pseudo', 'carrier_mode']).agg({
            'actual_transit_days': ['mean', 'std', 'count'],
            'otd_designation': lambda x: (x == 'On Time').mean()
        }).reset_index()
        
        carrier_stats.columns = ['carrier', 'mode', 'avg_days', 'std_days', 'sample_size', 'on_time_rate']
        carrier_stats['avg_hours'] = carrier_stats['avg_days'] * 24
        carrier_stats['std_hours'] = carrier_stats['std_days'] * 24
        
        # Filter to carriers with enough samples
        carrier_stats = carrier_stats[carrier_stats['sample_size'] >= 3]
        
        if len(carrier_stats) == 0:
            st.warning("Not enough data for reliable recommendations.")
        else:
            # Build recommendations
            recommendations = []
            for _, row in carrier_stats.iterrows():
                eta_hours = row['avg_hours']
                std_hours = row['std_hours'] if pd.notna(row['std_hours']) else eta_hours * 0.2
                
                # Check if meets goal (on-time)
                goal_upper = goal_hours * 1.1  # 10% tolerance
                if eta_hours > goal_upper:
                    continue  # Skip late options
                
                eta_datetime = ship_datetime + timedelta(hours=eta_hours)
                
                # Confidence based on sample size and variance
                sample_confidence = min(row['sample_size'] / 30, 1.0)
                variance_confidence = max(0, 1 - (std_hours / (eta_hours + 1)))
                confidence = (sample_confidence * 0.5 + variance_confidence * 0.5)
                
                # Determine status
                if eta_hours < goal_hours * 0.9:
                    status = 'early'
                elif eta_hours <= goal_upper:
                    status = 'on_time'
                else:
                    status = 'late'
                
                # Compute SHAP-based key factor
                key_factor = get_key_factor_for_recommendation(
                    row['carrier'], row['mode'], recent_filtered, model, feature_columns
                )
                
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
                    'is_best': False,
                    'key_factor': key_factor
                })
            
            # Sort by confidence (highest first) among on-time options
            recommendations = sorted(recommendations, key=lambda x: x['confidence'], reverse=True)
            
            # Mark best option
            if recommendations:
                recommendations[0]['is_best'] = True
            
            st.session_state.recommendations = recommendations
            st.session_state.ship_datetime = ship_datetime
            st.session_state.goal_days = goal_days

# =====================
# DISPLAY RECOMMENDATIONS
# =====================
if 'recommendations' in st.session_state and st.session_state.recommendations:
    recommendations = st.session_state.recommendations
    ship_datetime = st.session_state.ship_datetime
    goal_days = st.session_state.goal_days
    
    if len(recommendations) == 0:
        st.warning(f"No carriers meet your goal of {goal_days} days transit time. Try adjusting your filters or goal.")
    else:
        st.markdown(f"### 📦 On-Time Carrier Options (Goal: {goal_days} days)")
        st.caption(f"Ship: {ship_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        best = recommendations[0] if recommendations else None
        
        with col1:
            st.metric("🏆 Best Confidence", f"{best['confidence']*100:.0f}%")
        with col2:
            st.metric("📊 Options", len(recommendations))
        with col3:
            avg_hours = sum(r['eta_hours'] for r in recommendations) / len(recommendations)
            st.metric("📈 Avg ETA", format_duration(avg_hours))
        with col4:
            st.metric("✅ All On-Time", f"{len(recommendations)}/{len(recommendations)}")
        
        st.markdown("---")
        
        # Carrier cards
        for i, rec in enumerate(recommendations):
            is_best = rec['is_best']
            status = rec['status']
            
            with st.container():
                cols = st.columns([0.5, 2.5, 2, 1.5, 2, 1.5, 1.5])
                
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
                
                # Key Factor (SHAP-based)
                with cols[6]:
                    st.markdown(f"<span class='key-factor-badge'>{rec.get('key_factor', 'Carrier')}</span>", unsafe_allow_html=True)
                
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
            'Key Factor': r.get('key_factor', 'Carrier'),
            'Is Best': r['is_best']
        } for r in recommendations])
        
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results",
            data=csv,
            file_name=f"carrier_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

# =====================
# HISTORICAL RECORDS
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
        display_cols = [
            'actual_ship', 'actual_delivery', 'origin_state', 'dest_state',
            'carrier_pseudo', 'carrier_mode', 'actual_transit_days', 
            'otd_designation', 'customer_distance'
        ]
        available_cols = [c for c in display_cols if c in history_data.columns]
        
        history_df = history_data[available_cols].copy()
        history_df = history_df.sort_values('actual_ship', ascending=False)
        
        # Compute SHAP-based key factors
        with st.spinner("Computing key factors..."):
            try:
                # Use filtered_data as context for comparison
                key_factors = batch_compute_key_factors(
                    history_df, context_data=filtered_data
                )
                history_df['Key Factor'] = key_factors
            except Exception as e:
                # Fallback on error
                history_df['Key Factor'] = 'Carrier'
        
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
        
        # Reorder columns
        cols_order = ['Ship Date', 'Delivery Date', 'Origin', 'Dest', 'Carrier', 'Mode', 
                      'Transit Days', 'Key Factor', 'Status', 'Miles']
        cols_order = [c for c in cols_order if c in history_df.columns]
        history_df = history_df[cols_order]
        
        # Display
        max_rows = 500
        st.dataframe(
            history_df.head(max_rows),
            use_container_width=True,
            height=400
        )
        
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
