"""
Data loading utilities for the ETA Dashboard.
Handles model loading, data caching, and lane statistics computation.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import streamlit as st
from pathlib import Path


# Get the base directory (app folder)
APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent


def sanitize_xgboost_model(model):
    """
    Fix XGBoost model for SHAP compatibility.
    
    XGBoost 2.x can store parameters in formats that cause issues with SHAP.
    This function saves/reloads the model to normalize parameter formats.
    
    Parameters:
    -----------
    model : xgb.XGBRegressor
        The loaded XGBoost model
        
    Returns:
    --------
    xgb.XGBRegressor
        Model with normalized parameters
    """
    import tempfile
    
    try:
        # Save to JSON format (normalizes parameters)
        tmp_path = tempfile.mktemp(suffix='.json')
        model.save_model(tmp_path)
        
        # Read and fix the JSON if needed
        with open(tmp_path, 'r') as f:
            model_json = json.load(f)
        
        # Fix base_score if it's in bracketed scientific notation
        if 'learner' in model_json:
            lmp = model_json['learner'].get('learner_model_param', {})
            if 'base_score' in lmp:
                bs = lmp['base_score']
                if isinstance(bs, str) and '[' in bs:
                    fixed_bs = float(bs.strip('[]'))
                    model_json['learner']['learner_model_param']['base_score'] = str(fixed_bs)
        
        # Save fixed JSON and reload
        with open(tmp_path, 'w') as f:
            json.dump(model_json, f)
        
        fixed_model = xgb.XGBRegressor()
        fixed_model.load_model(tmp_path)
        
        os.unlink(tmp_path)
        return fixed_model
        
    except Exception as e:
        # If sanitization fails, return original model
        return model


@st.cache_resource
def load_model(model_path=None):
    """
    Load the trained XGBoost model with SHAP-compatible parameter sanitization.
    
    Parameters:
    -----------
    model_path : str, optional
        Path to the model file. Defaults to outputs/models/xgboost_model.pkl
        
    Returns:
    --------
    xgb.XGBRegressor
        The trained model (sanitized for SHAP compatibility)
    """
    if model_path is None:
        model_path = PROJECT_DIR / 'outputs' / 'models' / 'xgboost_model.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Sanitize for SHAP compatibility
    model = sanitize_xgboost_model(model)
    
    return model


@st.cache_data
def load_raw_data(data_path=None):
    """
    Load the raw delivery data.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Raw delivery data
    """
    if data_path is None:
        data_path = PROJECT_DIR / 'Dataset' / 'last-mile-data.csv'
    
    df = pd.read_csv(data_path)
    df['actual_ship'] = pd.to_datetime(df['actual_ship'])
    df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
    
    # Clean data
    df = df[df['actual_transit_days'] >= 0].copy()
    df = df[df['customer_distance'] > 0].copy()
    
    # Calculate transit hours
    df['transit_hours'] = (df['actual_delivery'] - df['actual_ship']).dt.total_seconds() / 3600
    
    return df


@st.cache_data
def load_processed_features(features_path=None):
    """
    Load the processed features used for model training.
    
    Parameters:
    -----------
    features_path : str, optional
        Path to the processed features CSV
        
    Returns:
    --------
    pd.DataFrame
        Processed features
    """
    if features_path is None:
        features_path = PROJECT_DIR / 'outputs' / 'data' / 'processed_features.csv'
    
    return pd.read_csv(features_path)


@st.cache_data
def compute_lane_statistics(df=None):
    """
    Compute statistics for each lane.
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Raw data. If None, loads from default path.
        
    Returns:
    --------
    pd.DataFrame
        Lane-level statistics
    """
    if df is None:
        df = load_raw_data()
    
    # Group by lane
    lane_stats = df.groupby(['lane_id', 'lane_zip3_pair', 'origin_zip_3d', 'dest_zip_3d']).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'customer_distance': 'mean',
        'otd_designation': lambda x: (x == 'On Time').mean(),
        'actual_transit_days': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    lane_stats.columns = [
        'lane_id', 'lane_zip3_pair', 'origin_zip_3d', 'dest_zip_3d',
        'avg_transit_hours', 'std_transit_hours', 'total_shipments',
        'avg_distance', 'on_time_rate',
        'avg_transit_days', 'min_transit_days', 'max_transit_days'
    ]
    
    # Fill NaN std with 0 for lanes with single shipment
    lane_stats['std_transit_hours'] = lane_stats['std_transit_hours'].fillna(0)
    
    # Calculate variance score (lower is more reliable)
    lane_stats['variance_score'] = lane_stats['std_transit_hours'] / (lane_stats['avg_transit_hours'] + 1)
    
    # Normalize volume for visualization
    max_volume = lane_stats['total_shipments'].max()
    lane_stats['volume_normalized'] = lane_stats['total_shipments'] / max_volume
    
    return lane_stats


@st.cache_data
def get_lane_carrier_combos(df=None):
    """
    Get all unique carrier/mode combinations for each lane.
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Raw data
        
    Returns:
    --------
    pd.DataFrame
        Lane-carrier-mode combinations with performance stats
    """
    if df is None:
        df = load_raw_data()
    
    combos = df.groupby(['lane_id', 'carrier_pseudo', 'carrier_mode']).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'otd_designation': lambda x: (x == 'On Time').mean(),
        'carrier_posted_service_days': 'mean'
    }).reset_index()
    
    combos.columns = [
        'lane_id', 'carrier', 'mode',
        'avg_hours', 'std_hours', 'shipment_count',
        'on_time_rate', 'avg_service_days'
    ]
    
    combos['std_hours'] = combos['std_hours'].fillna(0)
    
    return combos


@st.cache_data
def get_unique_lanes(df=None):
    """
    Get list of unique lanes for dropdown selection.
    
    Returns:
    --------
    list
        List of (lane_id, lane_zip3_pair) tuples
    """
    if df is None:
        df = load_raw_data()
    
    lanes = df[['lane_id', 'lane_zip3_pair']].drop_duplicates()
    return lanes.sort_values('lane_zip3_pair').values.tolist()


@st.cache_data
def get_carriers_for_lane(lane_id, df=None):
    """
    Get carriers that serve a specific lane.
    
    Parameters:
    -----------
    lane_id : str
        The lane ID
        
    Returns:
    --------
    pd.DataFrame
        Carriers with their performance on this lane
    """
    combos = get_lane_carrier_combos(df)
    return combos[combos['lane_id'] == lane_id].sort_values('avg_hours')


def get_feature_columns(features_df=None):
    """
    Get the list of feature columns used by the model.
    
    Returns:
    --------
    list
        Feature column names
    """
    if features_df is None:
        features_df = load_processed_features()
    
    return features_df.columns.tolist()


@st.cache_data
def get_background_data(n_samples=1000):
    """
    Get background data for SHAP explainer.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to use as background
        
    Returns:
    --------
    np.ndarray
        Background data array
    """
    features_df = load_processed_features()
    
    if len(features_df) > n_samples:
        # Random sample
        return features_df.sample(n=n_samples, random_state=42).values
    
    return features_df.values


def save_lane_statistics(lane_stats, output_path=None):
    """
    Save lane statistics to CSV.
    
    Parameters:
    -----------
    lane_stats : pd.DataFrame
        Lane statistics DataFrame
    output_path : str, optional
        Output path
    """
    if output_path is None:
        output_path = PROJECT_DIR / 'outputs' / 'data' / 'lane_statistics.csv'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lane_stats.to_csv(output_path, index=False)
    print(f"Lane statistics saved to {output_path}")


# Initialize lane statistics on first load
def initialize_data():
    """
    Initialize all required data on app startup.
    Called once when the app loads.
    """
    try:
        # Load raw data
        df = load_raw_data()
        
        # Compute and cache lane statistics
        lane_stats = compute_lane_statistics(df)
        
        # Load model
        model = load_model()
        
        return {
            'raw_data': df,
            'lane_stats': lane_stats,
            'model': model,
            'status': 'success'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

