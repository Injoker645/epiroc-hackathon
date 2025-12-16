"""
ETA Simulator Page
What-If analysis for carrier selection and ETA prediction.
Uses dynamic filters: Origin ZIP3 → Destination ZIP3 → Distance Bucket
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

col1, col2, col3 = st.columns(3)

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

# Filter distance buckets based on origin + dest
if selected_origin != 'All':
    filtered_for_distance = filtered_for_dest.copy()
else:
    filtered_for_distance = raw_data.copy()

if selected_dest != 'All':
    filtered_for_distance = filtered_for_distance[filtered_for_distance['dest_zip_3d'] == selected_dest]

with col3:
    distance_bucket_options = sorted(filtered_for_distance['distance_bucket'].dropna().unique().tolist(), 
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
else:
    st.info(f"🌐 All routes selected | **{len(filtered_data):,} historical shipments**")

# Ship date/time and goal days
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
            # Group by carrier and mode
            carrier_stats = filtered_data.groupby(['carrier_pseudo', 'carrier_mode']).agg({
                'transit_hours': ['mean', 'std', 'count'],
                'otd_designation': lambda x: (x == 'On Time').mean()
            }).reset_index()
            
            carrier_stats.columns = ['carrier', 'mode', 'avg_hours', 'std_hours', 'sample_size', 'on_time_rate']
            
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
                st.session_state.route_desc = ' → '.join(route_desc) if route_desc else 'All Routes'

# =====================
# DISPLAY RESULTS
# =====================
if 'recommendations' in st.session_state and st.session_state.recommendations:
    recommendations = st.session_state.recommendations
    ship_datetime = st.session_state.ship_datetime
    route_desc = st.session_state.get('route_desc', 'Selected Route')
    
    st.markdown("---")
    st.markdown(f"### 📦 Carrier Options for {route_desc}")
    st.caption(f"Ship: {ship_datetime.strftime('%A, %B %d, %Y at %I:%M %p')}")
    
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
