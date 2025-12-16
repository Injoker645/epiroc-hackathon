"""
Feature importance utilities for model explainability.
Uses pre-computed permutation importance for reliability with XGBoost 2.x.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Get the project directory
APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent


def load_feature_importance(importance_path=None):
    """
    Load pre-computed feature importance results.
    
    Parameters:
    -----------
    importance_path : str, optional
        Path to the saved importance pickle file
        
    Returns:
    --------
    dict or None
        Dictionary with 'permutation_importance', 'feature_names', etc.
        Returns None if file doesn't exist
    """
    if importance_path is None:
        importance_path = PROJECT_DIR / 'outputs' / 'models' / 'feature_importance_results.pkl'
    
    if not os.path.exists(importance_path):
        return None
    
    with open(importance_path, 'rb') as f:
        return pickle.load(f)


def get_importance_dataframe(importance_results=None):
    """
    Get feature importance as a sorted DataFrame.
    
    Parameters:
    -----------
    importance_results : dict, optional
        Pre-loaded importance results. If None, loads from default path.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: feature, importance_mean, importance_std
    """
    if importance_results is None:
        importance_results = load_feature_importance()
    
    if importance_results is None:
        return None
    
    perm_importance = importance_results['permutation_importance']
    feature_names = importance_results['feature_names']
    
    return pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)


def get_top_features(n=20, importance_results=None):
    """
    Get the top N most important features.
    
    Parameters:
    -----------
    n : int
        Number of top features to return
    importance_results : dict, optional
        Pre-loaded importance results
        
    Returns:
    --------
    list
        List of top feature names
    """
    df = get_importance_dataframe(importance_results)
    if df is None:
        return []
    return df.head(n)['feature'].tolist()


def get_feature_importance_for_prediction(feature_values, feature_names, importance_results=None, top_n=10):
    """
    Get importance-weighted feature contributions for a specific prediction.
    
    This provides SHAP-like explanations using permutation importance weights.
    
    Parameters:
    -----------
    feature_values : array-like
        Feature values for the prediction
    feature_names : list
        Names of features
    importance_results : dict, optional
        Pre-loaded importance results
    top_n : int
        Number of top features to show
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature contributions
    """
    importance_df = get_importance_dataframe(importance_results)
    if importance_df is None:
        return None
    
    # Create lookup for importance
    importance_lookup = dict(zip(importance_df['feature'], importance_df['importance_mean']))
    
    # Build contribution data
    contributions = []
    for i, feat in enumerate(feature_names):
        if i < len(feature_values):
            importance = importance_lookup.get(feat, 0)
            contributions.append({
                'feature': feat,
                'value': feature_values[i],
                'importance': importance
            })
    
    df = pd.DataFrame(contributions)
    return df.nlargest(top_n, 'importance')


def create_importance_bar_chart(importance_df, top_n=20, figsize=(10, 8)):
    """
    Create a horizontal bar chart of feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with importance values
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    top_features = importance_df.head(top_n)
    
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_features['importance_mean']]
    bars = ax.barh(range(len(top_features)), top_features['importance_mean'],
                   xerr=top_features['importance_std'], color=colors, alpha=0.8, capsize=3)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Feature Importance (MAE increase when permuted)')
    ax.set_title(f'Top {top_n} Features by Importance', fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def create_waterfall_chart(feature_contributions, base_value, final_value, figsize=(10, 6)):
    """
    Create a waterfall chart showing feature contributions.
    
    Parameters:
    -----------
    feature_contributions : pd.DataFrame
        DataFrame with feature names, values, and contributions
    base_value : float
        Base prediction value
    final_value : float
        Final prediction value
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Start with base value
    current = base_value
    positions = []
    widths = []
    labels = ['Base']
    colors = ['#3498db']
    
    positions.append(0)
    widths.append(base_value)
    
    # Add feature contributions
    for i, (_, row) in enumerate(feature_contributions.iterrows()):
        # This is a simplified waterfall - showing importance-weighted impacts
        contribution = row['importance'] * (1 if row['value'] > 0 else -1)
        positions.append(current)
        widths.append(contribution)
        labels.append(f"{row['feature'][:30]}...")
        colors.append('#2ecc71' if contribution > 0 else '#e74c3c')
        current += contribution
    
    # Final value
    positions.append(0)
    widths.append(final_value)
    labels.append('Prediction')
    colors.append('#9b59b6')
    
    y_pos = range(len(labels))
    ax.barh(y_pos, widths, left=[0]*len(widths), color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Hours')
    ax.set_title('Prediction Breakdown', fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    return fig


def format_importance_explanation(feature_contributions, prediction_hours):
    """
    Format feature importance as human-readable text.
    
    Parameters:
    -----------
    feature_contributions : pd.DataFrame
        DataFrame with feature contributions
    prediction_hours : float
        The predicted hours
        
    Returns:
    --------
    str
        Formatted explanation text
    """
    lines = [
        f"**Predicted ETA:** {prediction_hours:.1f} hours ({prediction_hours/24:.1f} days)",
        "",
        "**Key Factors Influencing This Prediction:**",
        ""
    ]
    
    for i, (_, row) in enumerate(feature_contributions.iterrows(), 1):
        feat_name = row['feature'].replace('_', ' ').title()
        importance = row['importance']
        value = row['value']
        
        if importance > 0.1:
            impact = "High impact"
        elif importance > 0.01:
            impact = "Moderate impact"
        else:
            impact = "Low impact"
        
        lines.append(f"{i}. **{feat_name}**: {value:.2f} ({impact})")
    
    return '\n'.join(lines)


def get_category_importance(importance_df):
    """
    Aggregate feature importance by category.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance
        
    Returns:
    --------
    pd.DataFrame
        Aggregated importance by category
    """
    categories = {
        'Carrier': ['carrier_', 'carrier_mode_'],
        'Lane': ['lane_', 'origin_', 'dest_'],
        'Distance': ['distance', 'customer_distance'],
        'Temporal': ['ship_', 'is_', 'season_', 'month', 'quarter', 'day', 'hour', 'week'],
        'Service': ['service', 'posted']
    }
    
    category_importance = {}
    for category, patterns in categories.items():
        mask = importance_df['feature'].apply(
            lambda x: any(p in x.lower() for p in patterns)
        )
        category_importance[category] = importance_df[mask]['importance_mean'].sum()
    
    return pd.DataFrame([
        {'Category': k, 'Total Importance': v}
        for k, v in sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
    ])
