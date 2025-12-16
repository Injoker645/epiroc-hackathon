"""
Data loading utilities for the ETA Dashboard.
Handles model loading, data caching, state lookups, and lane statistics computation.
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

# ============================================================================
# ZIP CODE TO STATE MAPPING
# ============================================================================

# Cache for ZIP3 to state lookups
_ZIP3_STATE_CACHE = {}
_PGEOCODE_NOMI = None


def _get_pgeocode_nomi():
    """Get or create the pgeocode Nominatim instance."""
    global _PGEOCODE_NOMI
    if _PGEOCODE_NOMI is None:
        try:
            import pgeocode
            _PGEOCODE_NOMI = pgeocode.Nominatim('us')
        except ImportError:
            st.warning("pgeocode not installed. Run: pip install pgeocode")
            return None
    return _PGEOCODE_NOMI


def zip3_to_state(zip3_prefix):
    """
    Convert a 3-digit ZIP code prefix to a state code using pgeocode.
    
    Parameters:
    -----------
    zip3_prefix : str or int
        3-digit ZIP code prefix (e.g., '100', '441', 100, 441)
        
    Returns:
    --------
    str
        State abbreviation (e.g., 'NY', 'OH') or 'XX' if not found
    """
    global _ZIP3_STATE_CACHE
    
    # Normalize input
    zip3 = str(zip3_prefix).zfill(3)[:3]
    
    # Check cache first
    if zip3 in _ZIP3_STATE_CACHE:
        return _ZIP3_STATE_CACHE[zip3]
    
    # Try pgeocode lookup
    nomi = _get_pgeocode_nomi()
    if nomi is not None:
        try:
            df = nomi._data
            filtered = df[df['postal_code'].astype(str).str.zfill(5).str.startswith(zip3)]
            
            if len(filtered) > 0 and 'state_code' in filtered.columns:
                state = filtered['state_code'].mode().iloc[0]
                _ZIP3_STATE_CACHE[zip3] = state
                return state
        except Exception:
            pass
    
    # Fallback: return 'XX' for unknown
    _ZIP3_STATE_CACHE[zip3] = 'XX'
    return 'XX'


def batch_zip3_to_state(zip3_series):
    """
    Convert a pandas Series of ZIP3 codes to state codes efficiently.
    
    Parameters:
    -----------
    zip3_series : pd.Series
        Series of 3-digit ZIP code prefixes
        
    Returns:
    --------
    pd.Series
        Series of state abbreviations
    """
    # Get unique ZIP3s
    unique_zip3s = zip3_series.dropna().astype(str).str.replace('xx', '').str.zfill(3).str[:3].unique()
    
    # Batch lookup
    nomi = _get_pgeocode_nomi()
    if nomi is not None:
        df = nomi._data
        
        for zip3 in unique_zip3s:
            if zip3 not in _ZIP3_STATE_CACHE:
                try:
                    filtered = df[df['postal_code'].astype(str).str.zfill(5).str.startswith(zip3)]
                    if len(filtered) > 0 and 'state_code' in filtered.columns:
                        state = filtered['state_code'].mode().iloc[0]
                        _ZIP3_STATE_CACHE[zip3] = state
                    else:
                        _ZIP3_STATE_CACHE[zip3] = 'XX'
                except Exception:
                    _ZIP3_STATE_CACHE[zip3] = 'XX'
    
    # Apply mapping (handle 'XXXxx' format by stripping 'xx')
    return zip3_series.astype(str).str.replace('xx', '').str.zfill(3).str[:3].map(_ZIP3_STATE_CACHE).fillna('XX')


# ============================================================================
# MODEL LOADING
# ============================================================================

def sanitize_xgboost_model(model):
    """
    Fix XGBoost model for SHAP compatibility.
    
    XGBoost 2.x can store parameters in formats that cause issues with SHAP.
    This function saves/reloads the model to normalize parameter formats.
    """
    import tempfile
    
    try:
        tmp_path = tempfile.mktemp(suffix='.json')
        model.save_model(tmp_path)
        
        with open(tmp_path, 'r') as f:
            model_json = json.load(f)
        
        # Fix base_score if in bracketed scientific notation
        if 'learner' in model_json:
            lmp = model_json['learner'].get('learner_model_param', {})
            if 'base_score' in lmp:
                bs = lmp['base_score']
                if isinstance(bs, str) and '[' in bs:
                    fixed_bs = float(bs.strip('[]'))
                    model_json['learner']['learner_model_param']['base_score'] = str(fixed_bs)
        
        with open(tmp_path, 'w') as f:
            json.dump(model_json, f)
        
        fixed_model = xgb.XGBRegressor()
        fixed_model.load_model(tmp_path)
        
        os.unlink(tmp_path)
        return fixed_model
        
    except Exception:
        return model


@st.cache_resource
def load_model(model_path=None):
    """Load the trained XGBoost model with SHAP-compatible parameter sanitization."""
    if model_path is None:
        model_path = PROJECT_DIR / 'outputs' / 'models' / 'xgboost_model.pkl'
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    model = sanitize_xgboost_model(model)
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_raw_data(data_path=None):
    """
    Load the raw delivery data with state code enrichment.
    
    Returns:
    --------
    pd.DataFrame
        Raw delivery data with origin_state, dest_state, lane_state_pair columns
    """
    if data_path is None:
        data_path = PROJECT_DIR / 'Dataset' / 'last-mile-data.csv'
    
    # Read with string dtypes for ID columns
    dtype_overrides = {
        'lane_id': str,
        'load_id_pseudo': str,
        'carrier_pseudo': str,
        'origin_zip': str,
        'dest_zip': str,
        'origin_zip_3d': str,
        'dest_zip_3d': str,
    }
    
    df = pd.read_csv(data_path, dtype=dtype_overrides)
    df['actual_ship'] = pd.to_datetime(df['actual_ship'])
    df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
    
    # Clean data
    df = df[df['actual_transit_days'] >= 0].copy()
    df = df[df['customer_distance'] > 0].copy()
    
    # Ensure ZIP3 columns are properly formatted
    if 'origin_zip_3d' in df.columns:
        df['origin_zip_3d'] = df['origin_zip_3d'].astype(str).str.replace('xx', '').str.zfill(3).str[:3]
    elif 'origin_zip' in df.columns:
        df['origin_zip_3d'] = df['origin_zip'].astype(str).str.zfill(5).str[:3]
    
    if 'dest_zip_3d' in df.columns:
        df['dest_zip_3d'] = df['dest_zip_3d'].astype(str).str.replace('xx', '').str.zfill(3).str[:3]
    elif 'dest_zip' in df.columns:
        df['dest_zip_3d'] = df['dest_zip'].astype(str).str.zfill(5).str[:3]
    
    # Convert ZIP3 to state codes
    if 'origin_zip_3d' in df.columns:
        df['origin_state'] = batch_zip3_to_state(df['origin_zip_3d'])
    if 'dest_zip_3d' in df.columns:
        df['dest_state'] = batch_zip3_to_state(df['dest_zip_3d'])
    
    # Create lane_state_pair (e.g., 'OH_PA')
    if 'origin_state' in df.columns and 'dest_state' in df.columns:
        df['lane_state_pair'] = df['origin_state'] + '_' + df['dest_state']
    
    # Create lane_state_pair_distance_bucket (most granular route)
    if 'lane_state_pair' in df.columns and 'distance_bucket' in df.columns:
        df['lane_state_pair_distance_bucket'] = df['lane_state_pair'] + '_' + df['distance_bucket'].astype(str)
    
    # Calculate transit hours
    df['transit_hours'] = (df['actual_delivery'] - df['actual_ship']).dt.total_seconds() / 3600
    
    return df


@st.cache_data
def load_processed_features(features_path=None):
    """Load the processed features used for model training."""
    if features_path is None:
        features_path = PROJECT_DIR / 'outputs' / 'data' / 'processed_features.csv'
    
    return pd.read_csv(features_path)


# ============================================================================
# LANE STATISTICS
# ============================================================================

@st.cache_data
def compute_lane_statistics(df=None):
    """
    Compute statistics for each state-based lane.
    
    Returns lane_state_pair level statistics (e.g., OH_PA).
    """
    if df is None:
        df = load_raw_data()
    
    # Group by state pair route
    lane_stats = df.groupby(['lane_state_pair', 'origin_state', 'dest_state']).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'customer_distance': 'mean',
        'otd_designation': lambda x: (x == 'On Time').mean(),
        'actual_transit_days': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    lane_stats.columns = [
        'lane_state_pair', 'origin_state', 'dest_state',
        'avg_transit_hours', 'std_transit_hours', 'total_shipments',
        'avg_distance', 'on_time_rate',
        'avg_transit_days', 'min_transit_days', 'max_transit_days'
    ]
    
    # Fill NaN std with 0
    lane_stats['std_transit_hours'] = lane_stats['std_transit_hours'].fillna(0)
    
    # Calculate variance score
    lane_stats['variance_score'] = lane_stats['std_transit_hours'] / (lane_stats['avg_transit_hours'] + 1)
    
    # Normalize volume
    max_volume = lane_stats['total_shipments'].max()
    lane_stats['volume_normalized'] = lane_stats['total_shipments'] / max_volume
    
    return lane_stats


@st.cache_data
def compute_granular_route_statistics(df=None):
    """
    Compute statistics for granular routes (state pair + distance bucket).
    """
    if df is None:
        df = load_raw_data()
    
    if 'lane_state_pair_distance_bucket' not in df.columns:
        return None
    
    route_stats = df.groupby([
        'lane_state_pair_distance_bucket', 'lane_state_pair', 
        'origin_state', 'dest_state', 'distance_bucket'
    ]).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'customer_distance': 'mean',
        'otd_designation': lambda x: (x == 'On Time').mean(),
    }).reset_index()
    
    route_stats.columns = [
        'granular_route', 'lane_state_pair', 'origin_state', 'dest_state', 'distance_bucket',
        'avg_transit_hours', 'std_transit_hours', 'total_shipments',
        'avg_distance', 'on_time_rate'
    ]
    
    route_stats['std_transit_hours'] = route_stats['std_transit_hours'].fillna(0)
    
    return route_stats


# ============================================================================
# CARRIER COMBINATIONS
# ============================================================================

@st.cache_data
def get_lane_carrier_combos(df=None):
    """Get all unique carrier/mode combinations for each state-based lane."""
    if df is None:
        df = load_raw_data()
    
    combos = df.groupby(['lane_state_pair', 'carrier_pseudo', 'carrier_mode']).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'otd_designation': lambda x: (x == 'On Time').mean(),
        'carrier_posted_service_days': 'mean'
    }).reset_index()
    
    combos.columns = [
        'lane_state_pair', 'carrier', 'mode',
        'avg_hours', 'std_hours', 'shipment_count',
        'on_time_rate', 'avg_service_days'
    ]
    
    combos['std_hours'] = combos['std_hours'].fillna(0)
    
    return combos


@st.cache_data
def get_granular_route_carrier_combos(df=None):
    """Get carrier/mode combinations for granular routes (state pair + distance)."""
    if df is None:
        df = load_raw_data()
    
    if 'lane_state_pair_distance_bucket' not in df.columns:
        return None
    
    combos = df.groupby([
        'lane_state_pair_distance_bucket', 'lane_state_pair', 
        'origin_state', 'dest_state', 'distance_bucket',
        'carrier_pseudo', 'carrier_mode'
    ]).agg({
        'transit_hours': ['mean', 'std', 'count'],
        'otd_designation': lambda x: (x == 'On Time').mean(),
        'carrier_posted_service_days': 'mean'
    }).reset_index()
    
    combos.columns = [
        'granular_route', 'lane_state_pair', 'origin_state', 'dest_state', 'distance_bucket',
        'carrier', 'mode',
        'avg_hours', 'std_hours', 'shipment_count',
        'on_time_rate', 'avg_service_days'
    ]
    
    combos['std_hours'] = combos['std_hours'].fillna(0)
    
    return combos


# ============================================================================
# DROPDOWN OPTIONS
# ============================================================================

@st.cache_data
def get_unique_state_pairs(df=None):
    """Get list of unique state-based lane pairs."""
    if df is None:
        df = load_raw_data()
    
    pairs = df[['lane_state_pair', 'origin_state', 'dest_state']].drop_duplicates()
    return pairs.sort_values('lane_state_pair')


@st.cache_data
def get_unique_origin_states(df=None):
    """Get list of unique origin states."""
    if df is None:
        df = load_raw_data()
    return sorted(df['origin_state'].dropna().unique().tolist())


@st.cache_data
def get_unique_dest_states(df=None):
    """Get list of unique destination states."""
    if df is None:
        df = load_raw_data()
    return sorted(df['dest_state'].dropna().unique().tolist())


@st.cache_data
def get_distance_buckets(df=None):
    """Get list of unique distance buckets."""
    if df is None:
        df = load_raw_data()
    return sorted(df['distance_bucket'].dropna().unique().tolist())


@st.cache_data
def get_carriers_for_route(lane_state_pair, distance_bucket=None, df=None):
    """
    Get carriers that serve a specific route.
    
    Parameters:
    -----------
    lane_state_pair : str
        The state pair route (e.g., 'OH_PA')
    distance_bucket : str, optional
        Distance bucket filter
        
    Returns:
    --------
    pd.DataFrame
        Carriers with their performance on this route
    """
    if distance_bucket:
        combos = get_granular_route_carrier_combos(df)
        if combos is None:
            return pd.DataFrame()
        return combos[
            (combos['lane_state_pair'] == lane_state_pair) & 
            (combos['distance_bucket'] == distance_bucket)
        ].sort_values('avg_hours')
    else:
        combos = get_lane_carrier_combos(df)
        return combos[combos['lane_state_pair'] == lane_state_pair].sort_values('avg_hours')


# ============================================================================
# FEATURE UTILITIES
# ============================================================================

def get_feature_columns(features_df=None):
    """Get the list of feature columns used by the model."""
    if features_df is None:
        features_df = load_processed_features()
    return features_df.columns.tolist()


@st.cache_data
def get_background_data(n_samples=1000):
    """Get background data for SHAP explainer."""
    features_df = load_processed_features()
    
    if len(features_df) > n_samples:
        return features_df.sample(n=n_samples, random_state=42).values
    
    return features_df.values


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_data():
    """Initialize all required data on app startup."""
    try:
        df = load_raw_data()
        lane_stats = compute_lane_statistics(df)
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
