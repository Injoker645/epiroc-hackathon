"""
Model Explainer Page
Feature importance visualization using pre-computed permutation importance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Model Explainer",
    page_icon="🔍",
    layout="wide"
)

st.markdown("# 🔍 Model Explainer")
st.markdown("Understand what drives ETA predictions with feature importance analysis.")

# Try to load feature importance results
try:
    from utils.shap_utils import (
        load_feature_importance,
        get_importance_dataframe,
        get_top_features,
        create_importance_bar_chart,
        get_category_importance,
        format_importance_explanation,
        get_feature_importance_for_prediction
    )
    from utils.data_loader import load_model, load_processed_features
    
    # Load pre-computed importance
    importance_results = load_feature_importance()
    
    if importance_results is not None:
        st.success(f"✅ Loaded pre-computed feature importance ({importance_results['n_samples']} samples, {importance_results['n_repeats']} repeats)")
        importance_df = get_importance_dataframe(importance_results)
    else:
        st.warning("No pre-computed feature importance found. Using XGBoost native importance.")
        importance_df = None
    
    # Also try to load model
    try:
        model = load_model()
        model_loaded = True
    except:
        model_loaded = False
        
except Exception as e:
    st.error(f"Could not load utilities: {e}")
    st.info("Please run the Model notebook first to train and save the model.")
    st.stop()

# Main content
st.markdown("---")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📊 Feature Importance", "📈 Category Analysis", "💡 Insights"])

with tab1:
    st.markdown("## Feature Importance")
    st.markdown("Features ranked by their impact on prediction accuracy (permutation importance).")
    
    if importance_df is not None:
        # Top N selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            top_n = st.slider("Show Top N Features", 10, 50, 25)
        with col2:
            show_std = st.checkbox("Show uncertainty", value=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
        
        top_features = importance_df.head(top_n)
        
        # Color by importance
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
        
        if show_std:
            bars = ax.barh(range(len(top_features)), top_features['importance_mean'].values,
                          xerr=top_features['importance_std'].values, 
                          color=colors, alpha=0.8, capsize=3, ecolor='gray')
        else:
            bars = ax.barh(range(len(top_features)), top_features['importance_mean'].values,
                          color=colors, alpha=0.8)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (MAE increase when feature is shuffled)', fontsize=11)
        ax.set_title(f'Top {top_n} Features by Permutation Importance', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance_mean'].values)):
            if val > 0.001:
                ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{val:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Feature importance table
        with st.expander("📋 View Full Feature Importance Table"):
            display_df = importance_df.copy()
            display_df['Rank'] = range(1, len(display_df) + 1)
            display_df['Importance'] = display_df['importance_mean'].apply(lambda x: f"{x:.6f}")
            display_df['Std Dev'] = display_df['importance_std'].apply(lambda x: f"±{x:.6f}")
            display_df = display_df[['Rank', 'feature', 'Importance', 'Std Dev']]
            display_df.columns = ['Rank', 'Feature', 'Importance', 'Uncertainty']
            st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)
    
    elif model_loaded:
        # Fallback to XGBoost native importance
        st.info("Using XGBoost native feature importance (gain-based)")
        
        features_df = load_processed_features()
        n_features = min(len(model.feature_importances_), len(features_df.columns))
        
        xgb_importance = pd.DataFrame({
            'feature': features_df.columns[:n_features],
            'importance': model.feature_importances_[:n_features]
        }).sort_values('importance', ascending=False)
        
        top_n = st.slider("Show Top N Features", 10, 50, 25)
        
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.35)))
        top_features = xgb_importance.head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.invert_yaxis()
        ax.set_xlabel('XGBoost Feature Importance (Gain)')
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.error("No importance data available. Please run the Model notebook first.")

with tab2:
    st.markdown("## Importance by Category")
    st.markdown("Understanding which types of features matter most.")
    
    if importance_df is not None:
        category_df = get_category_importance(importance_df)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
            
            # Filter out zero values
            nonzero = category_df[category_df['Total Importance'] > 0]
            
            if len(nonzero) > 0:
                wedges, texts, autotexts = ax.pie(
                    nonzero['Total Importance'], 
                    labels=nonzero['Category'],
                    autopct='%1.1f%%',
                    colors=colors[:len(nonzero)],
                    startangle=90,
                    explode=[0.02] * len(nonzero)
                )
                ax.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            plt.close()
        
        with col2:
            # Bar chart
            fig, ax = plt.subplots(figsize=(8, 6))
            
            bars = ax.bar(category_df['Category'], category_df['Total Importance'], 
                         color=colors[:len(category_df)], alpha=0.8)
            ax.set_ylabel('Total Importance')
            ax.set_title('Category Importance Comparison', fontsize=14, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars, category_df['Total Importance']):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Category breakdown table
        st.markdown("### Category Details")
        for _, row in category_df.iterrows():
            with st.expander(f"**{row['Category']}** (Total: {row['Total Importance']:.4f})"):
                cat_features = importance_df[importance_df['feature'].apply(
                    lambda x: any(p in x.lower() for p in {
                        'Carrier': ['carrier_', 'carrier_mode_'],
                        'Lane': ['lane_', 'origin_', 'dest_'],
                        'Distance': ['distance', 'customer_distance'],
                        'Temporal': ['ship_', 'is_', 'season_', 'month', 'quarter', 'day', 'hour', 'week'],
                        'Service': ['service', 'posted']
                    }.get(row['Category'], []))
                )].head(10)
                
                if not cat_features.empty:
                    st.dataframe(
                        cat_features[['feature', 'importance_mean', 'importance_std']].rename(
                            columns={'feature': 'Feature', 'importance_mean': 'Importance', 'importance_std': 'Std Dev'}
                        ),
                        hide_index=True,
                        use_container_width=True
                    )
    else:
        st.info("Category analysis requires pre-computed permutation importance. Please run the Model notebook.")

with tab3:
    st.markdown("## Key Insights & Interpretation")
    
    st.markdown("""
    ### Understanding Permutation Importance
    
    Permutation importance measures how much the model's accuracy **decreases** when a feature's values 
    are randomly shuffled. Higher values mean the feature is more important.
    
    - **High importance** (>0.1): Feature is crucial for predictions
    - **Medium importance** (0.01-0.1): Feature contributes meaningfully  
    - **Low importance** (<0.01): Feature has minimal impact
    """)
    
    if importance_df is not None:
        st.markdown("### Your Top 5 Most Important Features")
        
        top5 = importance_df.head(5)
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            importance = row['importance_mean']
            if importance > 0.1:
                emoji = "🔴"
                level = "Critical"
            elif importance > 0.01:
                emoji = "🟡"
                level = "Important"
            else:
                emoji = "🟢"
                level = "Moderate"
            
            st.markdown(f"""
            {emoji} **{i}. {row['feature']}** ({level})
            - Importance: {importance:.4f} (±{row['importance_std']:.4f})
            """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Common Patterns to Look For
    
    **Carrier Features** 🚚
    - Which carriers consistently deliver faster/slower
    - Historical reliability (`carrier_on_time_rate`)
    - Stated vs actual service days
    
    **Temporal Features** 📅
    - Weekend vs weekday effects
    - Holiday impacts
    - Seasonal patterns (Q4 busy season)
    
    **Distance Features** 📏
    - Longer distance typically means longer ETA
    - But relationship may not be perfectly linear
    
    **Lane Features** 🛣️
    - Some routes are inherently problematic
    - Lane-specific historical performance
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Actionable Recommendations
    
    1. **Carrier Selection**: Prioritize carriers with strong historical performance on your lanes
    2. **Timing**: Consider shipping earlier in the week when possible
    3. **Lane Analysis**: Identify and investigate high-variance lanes
    4. **Buffer Planning**: Add extra buffer time for features with high uncertainty (±std dev)
    """)
