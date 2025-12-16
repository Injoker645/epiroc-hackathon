"""
Last-Mile ETA Optimizer Dashboard
Home page for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import load_raw_data, compute_lane_statistics
from utils.yoy_utils import calculate_yoy_metrics, filter_last_n_years

# Page configuration
st.set_page_config(
    page_title="ETA Optimizer",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #34a853;
        --warning-color: #fbbc04;
        --danger-color: #ea4335;
    }
    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    .yoy-up { color: #34a853; font-weight: 600; }
    .yoy-down { color: #ea4335; font-weight: 600; }
    .yoy-neutral { color: #8b949e; }
    /* Feature cards: light/dark mode adaptive */
    @media (prefers-color-scheme: dark) {
        .feature-card {
            background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #30363d;
            margin-bottom: 1rem;
            transition: transform 0.2s, border-color 0.2s;
            color: #e6e6e6;
        }
        .feature-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
    }
    @media (prefers-color-scheme: light) {
        .feature-card {
            background: linear-gradient(145deg, #f7f7fa 0%, #e6eaf5 100%);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #d0d6e0;
            margin-bottom: 1rem;
            transition: transform 0.2s, border-color 0.2s;
            color: #222;
        }
        .feature-card:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def main():
    """Main home page."""
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Last-Mile ETA Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict delivery times, compare carriers, and optimize your supply chain</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading data...'):
        try:
            raw_data = load_raw_data()
            lane_stats = compute_lane_statistics(raw_data)
            data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Make sure the dataset is in the correct location: `Dataset/last-mile-data.csv`")
            data_loaded = False
            raw_data = None
            lane_stats = None
    
    if data_loaded and raw_data is not None:
        # Sidebar filters
        st.sidebar.markdown("### 🔍 Filters")
        
        # State filters
        origin_states = ['All'] + sorted(raw_data['origin_state'].dropna().unique().tolist())
        selected_origin = st.sidebar.selectbox("Origin State", origin_states)
        
        if selected_origin != 'All':
            filtered_for_dest = raw_data[raw_data['origin_state'] == selected_origin]
        else:
            filtered_for_dest = raw_data
        
        dest_states = ['All'] + sorted(filtered_for_dest['dest_state'].dropna().unique().tolist())
        selected_dest = st.sidebar.selectbox("Destination State", dest_states)
        
        # Apply filters
        filtered_data = raw_data.copy()
        if selected_origin != 'All':
            filtered_data = filtered_data[filtered_data['origin_state'] == selected_origin]
        if selected_dest != 'All':
            filtered_data = filtered_data[filtered_data['dest_state'] == selected_dest]
        
        # Carrier filter
        carriers = ['All'] + sorted(filtered_data['carrier_pseudo'].dropna().unique().tolist())
        selected_carrier = st.sidebar.selectbox("Carrier", carriers)
        
        if selected_carrier != 'All':
            filtered_data = filtered_data[filtered_data['carrier_pseudo'] == selected_carrier]
        
        # Show filter summary
        filter_desc = []
        if selected_origin != 'All':
            filter_desc.append(f"Origin: {selected_origin}")
        if selected_dest != 'All':
            filter_desc.append(f"Dest: {selected_dest}")
        if selected_carrier != 'All':
            filter_desc.append(f"Carrier: {selected_carrier[:10]}...")
        
        if filter_desc:
            st.info(f"🔍 Filters: {' | '.join(filter_desc)} | {len(filtered_data):,} shipments")
        
        # Calculate YoY metrics (uses data's max date as reference, not today)
        yoy = calculate_yoy_metrics(filtered_data, window_days=90)
        
        # YoY Period info
        st.markdown("---")
        periods = yoy['periods']
        st.caption(f"📅 Comparing: Last 90 days ({periods['current_start'].strftime('%b %d')} - {periods['current_end'].strftime('%b %d, %Y')}) vs Same period last year ({periods['prior_start'].strftime('%b %d')} - {periods['prior_end'].strftime('%b %d, %Y')})")
        
        # Key Metrics with YoY
        st.markdown("### 📊 Key Performance Metrics (YoY Comparison)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            volume_delta = yoy.get('volume_change_pct')
            # Shipments up = green
            st.metric(
                "Shipments (90d)",
                f"{yoy['current_count']:,}",
                delta=f"{volume_delta:+.1f}% YoY" if volume_delta is not None else None,
                delta_color="normal"  # Green when positive
            )
        
        with col2:
            on_time_delta = yoy.get('on_time_rate_delta')
            current_otr = yoy.get('current_on_time_rate', 0)
            # On-time rate up = green
            st.metric(
                "On-Time Rate",
                f"{current_otr:.1f}%",
                delta=f"{on_time_delta:+.1f}% YoY" if on_time_delta is not None else None,
                delta_color="normal"  # Green when positive
            )
        
        with col3:
            late_delta = yoy.get('late_rate_delta')
            current_late = yoy.get('current_late_rate', 0)
            # Late rate down = green (inverse)
            st.metric(
                "Late Rate",
                f"{current_late:.1f}%",
                delta=f"{late_delta:+.1f}% YoY" if late_delta is not None else None,
                delta_color="inverse"  # Green when negative
            )
        
        with col4:
            transit_delta = yoy.get('avg_transit_delta')
            current_transit = yoy.get('current_avg_transit', 0)
            # Transit down = green (inverse)
            st.metric(
                "Avg Transit Days",
                f"{current_transit:.1f}",
                delta=f"{transit_delta:+.1f}d YoY" if transit_delta is not None else None,
                delta_color="inverse"  # Green when negative (faster is better)
            )
        
        # Second row of metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unique_routes = filtered_data['lane_state_pair'].nunique() if 'lane_state_pair' in filtered_data.columns else 0
            st.metric("Unique Routes", f"{unique_routes:,}")
        
        with col2:
            unique_carriers = filtered_data['carrier_pseudo'].nunique()
            st.metric("Active Carriers", f"{unique_carriers:,}")
        
        with col3:
            std_delta = yoy.get('transit_std_delta')
            current_std = yoy.get('current_transit_std', 0)
            # Variability down = green (inverse)
            st.metric(
                "Transit Variability",
                f"±{current_std:.1f} days",
                delta=f"{std_delta:+.1f}d YoY" if std_delta is not None else None,
                delta_color="inverse"  # Green when negative (less variable is better)
            )
        
        with col4:
            total_all_time = len(filtered_data)
            st.metric("Total (All Time)", f"{total_all_time:,}")
        
        # Charts
        # st.markdown("---")
        # col1, col2 = st.columns(2)
        # Use last year's data for charts
        # recent_data = filter_last_n_years(filtered_data, years=1)
        # with col1:
        #     st.markdown("#### OTD Distribution (Last 12 Months)")
        #     if len(recent_data) > 0:
        #         otd_counts = recent_data['otd_designation'].value_counts()
        #         st.bar_chart(otd_counts)
        #     else:
        #         st.info("No data for selected filters")
        # with col2:
        #     st.markdown("#### Transit Days Distribution (Last 12 Months)")
        #     if len(recent_data) > 0:
        #         transit_hist = recent_data['actual_transit_days'].value_counts().sort_index().head(15)
        #         st.bar_chart(transit_hist)
        #     else:
        #         st.info("No data for selected filters")

        # Top/Bottom Routes with YoY
        st.markdown("---")
        
        # Recompute lane stats for filtered data
        if len(filtered_data) > 0:
            filtered_lane_stats = filtered_data.groupby(['lane_state_pair', 'origin_state', 'dest_state']).agg({
                'actual_transit_days': ['mean', 'count'],
                'otd_designation': lambda x: (x == 'On Time').mean()
            }).reset_index()
            filtered_lane_stats.columns = ['lane_state_pair', 'origin_state', 'dest_state', 
                                           'avg_transit_days', 'total_shipments', 'on_time_rate']
        else:
            filtered_lane_stats = pd.DataFrame()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top Performing Routes (Last 12 Months)")
            if len(filtered_lane_stats) > 0:
                filtered_lane_stats['delay_rate'] = 1 - filtered_lane_stats['on_time_rate']
                top_routes = filtered_lane_stats.nsmallest(10, 'delay_rate')[
                    ['lane_state_pair', 'delay_rate', 'total_shipments', 'avg_transit_days']
                ].copy()
                top_routes['delay_rate'] = (top_routes['delay_rate'] * 100).round(1).astype(str) + '%'
                top_routes['avg_transit_days'] = top_routes['avg_transit_days'].round(1)
                top_routes.columns = ['Route', 'Delay %', 'Shipments', 'Avg Days']
                # 跳转链接到Manager Page（Streamlit多页面格式）
                def make_route_link(route):
                    if '_' in route:
                        origin, dest = route.split('_')
                        # Streamlit多页面跳转格式
                        return f"[**{route}**](/Manager_Page?origin_state={origin}&dest_state={dest})"
                    return route
                top_routes['Route'] = top_routes['Route'].apply(make_route_link)
                st.markdown(top_routes.to_markdown(index=False), unsafe_allow_html=True)
            else:
                st.info("No routes for selected filters")
        
        with col2:
            st.markdown("#### ⚠️ Routes Needing Attention")
            if len(filtered_lane_stats) > 0:
                filtered_lane_stats['delay_rate'] = 1 - filtered_lane_stats['on_time_rate']
                # 只考虑发货量>=100的路线
                bottom_routes = filtered_lane_stats[filtered_lane_stats['total_shipments'] >= 100].nlargest(10, 'delay_rate')[
                    ['lane_state_pair', 'delay_rate', 'total_shipments', 'avg_transit_days']
                ].copy()
                if len(bottom_routes) > 0:
                    bottom_routes['delay_rate'] = (bottom_routes['delay_rate'] * 100).round(1).astype(str) + '%'
                    bottom_routes['avg_transit_days'] = bottom_routes['avg_transit_days'].round(1)
                    bottom_routes.columns = ['Route', 'Delay %', 'Shipments', 'Avg Days']
                    def make_route_link(route):
                        if '_' in route:
                            origin, dest = route.split('_')
                            return f"[**{route}**](/Manager_Page?origin_state={origin}&dest_state={dest})"
                        return route
                    bottom_routes['Route'] = bottom_routes['Route'].apply(make_route_link)
                    st.markdown(bottom_routes.to_markdown(index=False), unsafe_allow_html=True)
                else:
                    st.info("All routes performing well!")
            else:
                st.info("No routes for selected filters")
    
    # Feature navigation cards
    st.markdown("---")
    st.markdown("### 🧭 Explore Features")
    st.markdown("*Use the sidebar to navigate to different pages*")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🗺️ Lane Explorer</h3>
            <p>Interactive map visualization of shipping routes by state. 
            Filter by performance, view statistics, and identify patterns.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 ETA Simulator</h3>
            <p>What-If analysis with dynamic filters. Select route, compare carriers, 
            and get predicted ETAs with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Model Explainer</h3>
            <p>Understand what drives ETA predictions. View feature importance 
            and SHAP analysis for model transparency.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("Built for the Epiroc Last-Mile Delivery Optimization Hackathon")


if __name__ == "__main__":
    main()
