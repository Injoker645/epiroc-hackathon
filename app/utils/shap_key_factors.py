"""
SHAP-based key factor identification for shipments.
Maps SHAP values to human-readable categories.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import os

# Get project directory
APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent


def map_feature_to_category(feature_name):
    """
    Map a feature name to its category.
    
    Parameters:
    -----------
    feature_name : str
        Feature name (may be one-hot encoded)
        
    Returns:
    --------
    str
        Category: 'Carrier', 'Lane', 'Distance', 'Temporal', 'Route', or 'Other'
    """
    feature_lower = feature_name.lower()
    
    # Carrier features
    if 'carrier' in feature_lower:
        return 'Carrier'
    
    # Lane features
    if any(x in feature_lower for x in ['lane_', 'origin_', 'dest_']):
        return 'Lane'
    
    # Distance features
    if any(x in feature_lower for x in ['distance', 'customer_distance']):
        return 'Distance'
    
    # Route features
    if any(x in feature_lower for x in ['route_', 'state_route', 'granular_route']):
        return 'Route'
    
    # Temporal features
    if any(x in feature_lower for x in ['ship_', 'season', 'holiday', 'weekend', 'month', 'hour', 'day_of_week', 'quarter']):
        return 'Temporal'
    
    # Service features
    if any(x in feature_lower for x in ['service', 'posted', 'goal_transit']):
        return 'Service'
    
    return 'Other'


def get_key_factor_from_shap_values(shap_values, feature_names):
    """
    Get the key factor category from SHAP values.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for one prediction
    feature_names : list
        Feature names corresponding to SHAP values
        
    Returns:
    --------
    str
        Key factor category (e.g., 'Carrier', 'Lane', 'Distance')
    """
    if shap_values is None or len(shap_values) == 0:
        return 'Carrier'  # Default
    
    # Get absolute SHAP values
    abs_shap = np.abs(shap_values)
    
    # Find top contributing feature
    top_idx = np.argmax(abs_shap)
    top_feature = feature_names[top_idx] if top_idx < len(feature_names) else None
    
    if top_feature:
        category = map_feature_to_category(top_feature)
        return category
    
    # Fallback: aggregate by category
    category_shap = {}
    for i, feat_name in enumerate(feature_names):
        if i < len(shap_values):
            category = map_feature_to_category(feat_name)
            if category not in category_shap:
                category_shap[category] = 0
            category_shap[category] += abs(shap_values[i])
    
    if category_shap:
        return max(category_shap, key=category_shap.get)
    
    return 'Carrier'  # Default


def compute_shap_for_historical_row(row_data, context_data=None):
    """
    Compute key factor for a single historical row using data-driven heuristics.
    
    Analyzes the row's characteristics compared to context to identify the key factor.
    
    Parameters:
    -----------
    row_data : pd.Series
        Raw data row
    context_data : pd.DataFrame, optional
        Context data for comparison (e.g., filtered dataset)
        
    Returns:
    --------
    str
        Key factor category
    """
    try:
        # Get context averages if available
        if context_data is not None and len(context_data) > 0:
            avg_distance = context_data['customer_distance'].mean() if 'customer_distance' in context_data.columns else 0
            avg_transit = context_data['actual_transit_days'].mean() if 'actual_transit_days' in context_data.columns else 0
        else:
            avg_distance = 500  # Default
            avg_transit = 3  # Default
        
        # Get row values
        row_distance = row_data.get('customer_distance', avg_distance)
        row_transit = row_data.get('actual_transit_days', avg_transit)
        row_mode = str(row_data.get('carrier_mode', ''))
        
        # Calculate deviations
        distance_dev = abs(row_distance - avg_distance) / (avg_distance + 1)
        transit_dev = abs(row_transit - avg_transit) / (avg_transit + 1)
        
        # Distance is key if significantly different (>30% deviation)
        if distance_dev > 0.3:
            if row_distance > avg_distance * 1.3:
                return 'Distance'  # Long distance
            elif row_distance < avg_distance * 0.7:
                return 'Distance'  # Short distance
        
        # Carrier mode is key if LTL (typically slower) or if transit is unusual
        if row_mode == 'LTL' and row_transit > avg_transit * 1.2:
            return 'Carrier'
        
        # Temporal if weekend/holiday
        if 'actual_ship' in row_data.index:
            try:
                ship_date = pd.to_datetime(row_data['actual_ship'])
                if ship_date.weekday() >= 5:  # Weekend
                    return 'Temporal'
            except:
                pass
        
        # Lane/Route if specific route
        if 'lane_state_pair' in row_data.index:
            return 'Lane'
        
        # Default to carrier
        return 'Carrier'
        
    except Exception as e:
        return 'Carrier'  # Fallback on error


def batch_compute_key_factors(df, model=None, feature_columns=None, X_background=None, sample_size=500, context_data=None):
    """
    Compute key factors for a batch of rows efficiently.
    
    Uses data-driven heuristics based on row characteristics compared to context.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Historical data rows
    model : xgb.XGBRegressor, optional
        Trained model (not used currently, kept for API compatibility)
    feature_columns : list, optional
        Feature column names (not used currently)
    X_background : np.ndarray, optional
        Background data (not used currently)
    sample_size : int
        Maximum number of rows to process (for performance)
    context_data : pd.DataFrame, optional
        Context data for comparison (e.g., filtered dataset)
        
    Returns:
    --------
    pd.Series
        Series of key factor categories
    """
    if len(df) == 0:
        return pd.Series(dtype=str, index=df.index)
    
    # Use context data if provided, otherwise use df itself
    if context_data is None:
        context_data = df
    
    # Process all rows (heuristics are fast)
    key_factors = []
    for _, row in df.iterrows():
        factor = compute_shap_for_historical_row(row, context_data)
        key_factors.append(factor)
    
    return pd.Series(key_factors, index=df.index)


def get_key_factor_for_recommendation(carrier, mode, filtered_data, model, feature_columns):
    """
    Get key factor for a carrier recommendation.
    
    Parameters:
    -----------
    carrier : str
        Carrier name
    mode : str
        Carrier mode
    filtered_data : pd.DataFrame
        Filtered historical data
    model : xgb.XGBRegressor
        Trained model
    feature_columns : list
        Feature column names
        
    Returns:
    --------
    str
        Key factor category
    """
    # Get carrier-specific data
    carrier_data = filtered_data[
        (filtered_data['carrier_pseudo'] == carrier) &
        (filtered_data['carrier_mode'] == mode)
    ]
    
    if len(carrier_data) == 0:
        return 'Carrier'
    
    # Analyze what makes this carrier different
    avg_distance = carrier_data['customer_distance'].mean()
    overall_avg_distance = filtered_data['customer_distance'].mean()
    
    avg_transit = carrier_data['actual_transit_days'].mean()
    overall_avg_transit = filtered_data['actual_transit_days'].mean()
    
    # Distance is key if significantly different
    if abs(avg_distance - overall_avg_distance) / overall_avg_distance > 0.2:
        return 'Distance'
    
    # Mode is key if LTL (typically slower)
    if mode == 'LTL':
        return 'Carrier'
    
    # Lane is key if route-specific
    if 'lane_state_pair' in carrier_data.columns:
        unique_routes = carrier_data['lane_state_pair'].nunique()
        if unique_routes == 1:
            return 'Lane'
    
    # Default to carrier
    return 'Carrier'

