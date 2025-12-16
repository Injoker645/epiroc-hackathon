"""
Last-Mile ETA Optimizer Dashboard
Main entry point for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.data_loader import (
    load_model, load_raw_data, compute_lane_statistics,
    get_lane_carrier_combos, get_unique_lanes
)

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
    /* Main theme colors */
    :root {
        --primary-color: #1a73e8;
        --secondary-color: #34a853;
        --warning-color: #fbbc04;
        --danger-color: #ea4335;
        --background-dark: #0e1117;
        --card-background: #1e2329;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e2329 0%, #262c33 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .status-ontime {
        background-color: rgba(52, 168, 83, 0.2);
        color: #34a853;
        border: 1px solid #34a853;
    }
    
    .status-late {
        background-color: rgba(234, 67, 53, 0.2);
        color: #ea4335;
        border: 1px solid #ea4335;
    }
    
    .status-early {
        background-color: rgba(26, 115, 232, 0.2);
        color: #1a73e8;
        border: 1px solid #1a73e8;
    }
    
    /* Carrier card */
    .carrier-card {
        background: #1e2329;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #30363d;
        transition: all 0.2s ease;
    }
    
    .carrier-card:hover {
        border-left-color: #667eea;
        background: #262c33;
    }
    
    .carrier-card.best {
        border-left-color: #ffd700;
        background: linear-gradient(90deg, rgba(255, 215, 0, 0.1) 0%, #1e2329 100%);
    }
    
    /* Progress bar */
    .confidence-bar {
        height: 8px;
        background: #30363d;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application."""
    
    # Header
    st.markdown('<h1 class="main-header">🚚 Last-Mile ETA Optimizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict delivery times, compare carriers, and optimize your supply chain</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.selected_lane = None
    
    # Load data
    with st.spinner('Loading data...'):
        try:
            raw_data = load_raw_data()
            lane_stats = compute_lane_statistics(raw_data)
            st.session_state.data_loaded = True
            st.session_state.raw_data = raw_data
            st.session_state.lane_stats = lane_stats
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Make sure the dataset is in the correct location: Dataset/last-mile-data.csv")
            return
    
    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Dashboard", "ETA Simulator", "Lane Explorer", "Model Explainer"],
        label_visibility="collapsed"
    )
    
    # Display selected page
    if page == "Dashboard":
        show_dashboard(raw_data, lane_stats)
    elif page == "ETA Simulator":
        show_simulator(raw_data, lane_stats)
    elif page == "Lane Explorer":
        show_lane_explorer(raw_data, lane_stats)
    elif page == "Model Explainer":
        show_explainer(raw_data)


def show_dashboard(raw_data, lane_stats):
    """Show main dashboard with key metrics."""
    
    st.markdown("### Key Performance Metrics")
    
    # Calculate metrics
    total_shipments = len(raw_data)
    on_time_rate = (raw_data['otd_designation'] == 'On Time').mean() * 100
    late_rate = (raw_data['otd_designation'] == 'Late').mean() * 100
    avg_transit = raw_data['actual_transit_days'].mean()
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Shipments",
            value=f"{total_shipments:,}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="On-Time Rate",
            value=f"{on_time_rate:.1f}%",
            delta="Target: 80%"
        )
    
    with col3:
        st.metric(
            label="Late Rate",
            value=f"{late_rate:.1f}%",
            delta=f"-{late_rate:.1f}%" if late_rate < 20 else f"+{late_rate-20:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="Avg Transit Days",
            value=f"{avg_transit:.1f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### OTD Distribution")
        otd_counts = raw_data['otd_designation'].value_counts()
        st.bar_chart(otd_counts)
    
    with col2:
        st.markdown("#### Transit Days Distribution")
        transit_hist = raw_data['actual_transit_days'].value_counts().sort_index().head(15)
        st.bar_chart(transit_hist)
    
    # Top/Bottom Lanes
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top Performing Lanes")
        top_lanes = lane_stats.nlargest(10, 'on_time_rate')[
            ['lane_zip3_pair', 'on_time_rate', 'total_shipments', 'avg_transit_hours']
        ]
        top_lanes['on_time_rate'] = (top_lanes['on_time_rate'] * 100).round(1).astype(str) + '%'
        top_lanes['avg_transit_hours'] = top_lanes['avg_transit_hours'].round(1)
        top_lanes.columns = ['Lane', 'On-Time %', 'Shipments', 'Avg Hours']
        st.dataframe(top_lanes, hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("#### Lanes Needing Attention")
        bottom_lanes = lane_stats[lane_stats['total_shipments'] >= 10].nsmallest(10, 'on_time_rate')[
            ['lane_zip3_pair', 'on_time_rate', 'total_shipments', 'avg_transit_hours']
        ]
        bottom_lanes['on_time_rate'] = (bottom_lanes['on_time_rate'] * 100).round(1).astype(str) + '%'
        bottom_lanes['avg_transit_hours'] = bottom_lanes['avg_transit_hours'].round(1)
        bottom_lanes.columns = ['Lane', 'On-Time %', 'Shipments', 'Avg Hours']
        st.dataframe(bottom_lanes, hide_index=True, use_container_width=True)


def show_simulator(raw_data, lane_stats):
    """Show ETA simulator interface."""
    
    st.markdown("### What-If ETA Simulator")
    st.markdown("Select a lane and shipping details to see predicted ETAs from all carriers.")
    
    # Get unique lanes
    lanes = get_unique_lanes(raw_data)
    lane_options = {f"{l[1]} ({l[0][:8]}...)": l[0] for l in lanes}
    
    # Input form
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_lane_display = st.selectbox(
            "Select Lane",
            options=list(lane_options.keys()),
            help="Choose a shipping lane (origin → destination)"
        )
        selected_lane_id = lane_options[selected_lane_display]
    
    with col2:
        ship_date = st.date_input(
            "Ship Date",
            value=datetime.now().date(),
            help="Expected shipping date"
        )
        ship_time = st.time_input(
            "Ship Time",
            value=datetime.now().replace(hour=10, minute=0).time()
        )
    
    # Goal days (optional)
    goal_days = st.number_input(
        "Goal Transit Days (optional)",
        min_value=0.0,
        max_value=30.0,
        value=0.0,
        step=0.5,
        help="Enter target transit days to see early/on-time/late status"
    )
    goal_days = goal_days if goal_days > 0 else None
    
    # Generate predictions button
    if st.button("Generate Carrier Recommendations", type="primary"):
        ship_datetime = datetime.combine(ship_date, ship_time)
        
        with st.spinner("Analyzing carriers..."):
            # Get carrier options for this lane
            from utils.prediction_utils import generate_carrier_recommendations
            
            recommendations = generate_carrier_recommendations(
                lane_id=selected_lane_id,
                ship_datetime=ship_datetime,
                model=None,  # Using historical averages for now
                raw_data=raw_data,
                feature_columns=None,
                goal_days=goal_days
            )
            
            if not recommendations:
                st.warning("No carrier data available for this lane.")
                return
            
            st.session_state.recommendations = recommendations
            st.session_state.ship_datetime = ship_datetime
    
    # Display recommendations
    if 'recommendations' in st.session_state:
        recommendations = st.session_state.recommendations
        ship_datetime = st.session_state.ship_datetime
        
        st.markdown("---")
        st.markdown(f"### Carrier Options for {selected_lane_display}")
        st.markdown(f"*Ship: {ship_datetime.strftime('%B %d, %Y at %I:%M %p')}*")
        
        for i, rec in enumerate(recommendations):
            # Determine styling
            is_best = rec['is_best']
            status = rec['status']
            
            # Create card
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    prefix = "🏆 " if is_best else ""
                    st.markdown(f"**{prefix}{rec['carrier'][:12]}... - {rec['mode']}**")
                    if is_best:
                        st.caption("Best Option")
                
                with col2:
                    st.markdown(f"**ETA: {rec['eta_formatted']}**")
                    st.caption(rec['eta_datetime'].strftime('%b %d, %I:%M %p'))
                
                with col3:
                    # Status badge
                    if status == 'on_time':
                        st.success("✅ On-Time")
                    elif status == 'early':
                        st.info("🟢 Early")
                    elif status == 'late':
                        st.error("🔴 Late")
                    else:
                        st.warning("⚪ Unknown")
                
                with col4:
                    # Confidence
                    conf_pct = int(rec['confidence'] * 100)
                    st.progress(rec['confidence'])
                    st.caption(f"{conf_pct}% confidence ({rec['sample_size']} shipments)")
                
                st.markdown("---")


def show_lane_explorer(raw_data, lane_stats):
    """Show lane explorer with map."""
    
    st.markdown("### Lane Explorer")
    st.markdown("Explore shipping lanes and their performance metrics.")
    
    try:
        from streamlit_folium import st_folium
        from utils.map_utils import create_lane_map, add_legend_to_map
        
        # Create map
        with st.spinner("Generating map..."):
            m = create_lane_map(lane_stats)
            add_legend_to_map(m)
        
        # Display map
        st_data = st_folium(m, width=None, height=500)
        
    except ImportError:
        st.warning("Map visualization requires streamlit-folium. Install with: pip install streamlit-folium")
        st.info("Showing lane data in table format instead.")
    
    # Lane filter and table
    st.markdown("---")
    st.markdown("#### Lane Statistics")
    
    # Filter
    min_shipments = st.slider("Minimum shipments", 0, 100, 10)
    filtered_lanes = lane_stats[lane_stats['total_shipments'] >= min_shipments]
    
    # Display table
    display_cols = ['lane_zip3_pair', 'total_shipments', 'avg_transit_hours', 
                    'on_time_rate', 'avg_distance']
    display_df = filtered_lanes[display_cols].copy()
    display_df['on_time_rate'] = (display_df['on_time_rate'] * 100).round(1)
    display_df['avg_transit_hours'] = display_df['avg_transit_hours'].round(1)
    display_df['avg_distance'] = display_df['avg_distance'].round(0)
    display_df.columns = ['Lane', 'Shipments', 'Avg Hours', 'On-Time %', 'Avg Miles']
    
    st.dataframe(
        display_df.sort_values('Shipments', ascending=False),
        hide_index=True,
        use_container_width=True
    )


def show_explainer(raw_data):
    """Show model explainability interface."""
    
    st.markdown("### Model Explainability")
    st.markdown("Understand what factors drive ETA predictions.")
    
    # Try to load model
    try:
        model = load_model()
        
        # Show feature importance from XGBoost
        st.markdown("#### Feature Importance (XGBoost)")
        
        try:
            from utils.data_loader import load_processed_features
            features_df = load_processed_features()
            
            # Get feature importance
            importance = pd.DataFrame({
                'Feature': features_df.columns[:len(model.feature_importances_)],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Show top 20
            st.bar_chart(importance.head(20).set_index('Feature'))
            
            st.markdown("#### Top 20 Most Important Features")
            top_20 = importance.head(20).copy()
            top_20['Importance'] = top_20['Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(top_20, hide_index=True, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not load features: {e}")
            st.info("Run the ETL notebook first to generate processed features.")
        
        # SHAP section
        st.markdown("---")
        st.markdown("#### SHAP Analysis")
        
        if st.button("Compute SHAP Values"):
            with st.spinner("Computing SHAP values (this may take a few minutes)..."):
                try:
                    from utils.shap_utils import create_shap_explainer, compute_shap_values
                    from utils.data_loader import get_background_data
                    
                    # Get background data
                    X_background = get_background_data(500)
                    
                    # Compute SHAP values
                    shap_values, explainer = compute_shap_values(
                        model, 
                        X_background[:100],
                        X_background[:50]
                    )
                    
                    st.success("SHAP values computed successfully!")
                    
                    # Show summary
                    import shap
                    import matplotlib.pyplot as plt
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.summary_plot(shap_values, X_background[:50], show=False)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"SHAP computation failed: {e}")
                    st.info("Using XGBoost feature importance instead (shown above).")
        
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.info("Run the Model notebook first to train and save the model.")


if __name__ == "__main__":
    main()

