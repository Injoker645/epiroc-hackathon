"""
ETL Utilities for Epiroc Last-Mile Delivery Optimization

This module provides reusable functions for data loading, cleaning, feature engineering,
and model preparation for ETA prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def sanitize_numeric_columns(df):
    """
    Ensure all numeric columns are properly parsed as floats.
    
    Handles scientific notation strings like '[9.8E1]' that might
    cause issues with downstream libraries.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with sanitized numeric columns
    """
    df = df.copy()
    
    for col in df.columns:
        # Check if column has object dtype (strings)
        if df[col].dtype == 'object':
            # Try to convert to numeric, handling bracketed scientific notation
            try:
                # Check if any values look like '[1.23E4]' format
                sample = df[col].dropna().iloc[:100] if len(df[col].dropna()) > 0 else []
                has_brackets = any(str(v).startswith('[') and str(v).endswith(']') for v in sample)
                
                if has_brackets:
                    # Strip brackets and convert
                    df[col] = df[col].apply(lambda x: float(str(x).strip('[]')) if pd.notna(x) and str(x).startswith('[') else x)
                
                # Try numeric conversion
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                
                # If more than 90% converted successfully, use numeric
                if numeric_vals.notna().sum() / len(df[col]) > 0.9:
                    df[col] = numeric_vals
            except:
                pass  # Keep original if conversion fails
    
    return df


def load_data(filepath):
    """
    Load CSV data with proper date parsing and numeric sanitization.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with date columns parsed and numeric columns sanitized
    """
    df = pd.read_csv(filepath)
    
    # Convert date columns
    df['actual_ship'] = pd.to_datetime(df['actual_ship'])
    df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])
    
    # Sanitize any numeric columns that might have odd formats
    df = sanitize_numeric_columns(df)
    
    return df


def clean_data(df):
    """
    Remove error rows from the dataset.
    
    Removes:
    - Rows with negative actual_transit_days
    - Rows with zero customer_distance
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    initial_count = len(df)
    df = df.copy()
    
    # Remove negative transit days
    df = df[df['actual_transit_days'] >= 0].copy()
    
    # Remove zero distance
    df = df[df['customer_distance'] > 0].copy()
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"Cleaned data: Removed {removed:,} rows ({removed/initial_count*100:.2f}%)")
    
    return df


def round_timestamps_to_hour(df):
    """
    Round actual_ship and actual_delivery timestamps to nearest hour.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with rounded timestamps
    """
    df = df.copy()
    
    # Round to nearest hour
    df['actual_ship'] = df['actual_ship'].dt.round('H')
    df['actual_delivery'] = df['actual_delivery'].dt.round('H')
    
    return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_temporal_features(df):
    """
    Create temporal and calendar features.
    
    Features created:
    - US holidays (is_holiday, holiday_name)
    - Season (spring, summer, fall, winter)
    - is_weekend, is_month_end, is_quarter_end
    - days_until_holiday, days_since_holiday
    - hour, day_of_week, day_of_month, week_of_year
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with actual_ship column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with temporal features added
    """
    df = df.copy()
    
    # Get US holidays
    us_holidays = holidays.UnitedStates(years=range(2020, 2026))
    
    # Extract date components from actual_ship
    df['ship_hour'] = df['actual_ship'].dt.hour
    df['ship_day_of_week'] = df['actual_ship'].dt.dayofweek
    df['ship_day_of_month'] = df['actual_ship'].dt.day
    df['ship_week_of_year'] = df['actual_ship'].dt.isocalendar().week
    df['ship_month'] = df['actual_ship'].dt.month
    df['ship_year'] = df['actual_ship'].dt.year
    
    # Weekend flag
    df['is_weekend'] = (df['ship_day_of_week'] >= 5).astype(int)
    
    # Month end and quarter end
    df['is_month_end'] = df['actual_ship'].dt.is_month_end.astype(int)
    df['is_quarter_end'] = df['actual_ship'].dt.is_quarter_end.astype(int)
    
    # Season (based on month)
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['ship_month'].apply(get_season)
    
    # Holiday features
    df['is_holiday'] = df['actual_ship'].dt.date.isin(us_holidays.keys()).astype(int)
    df['holiday_name'] = df['actual_ship'].dt.date.map(us_holidays).fillna('None')
    
    # Days until/since next/previous holiday
    def days_to_holiday(date):
        date_obj = date.date() if isinstance(date, pd.Timestamp) else date
        future_holidays = [h for h in us_holidays.keys() if h > date_obj]
        if future_holidays:
            return (min(future_holidays) - date_obj).days
        return 365  # Default if no future holiday
    
    def days_from_holiday(date):
        date_obj = date.date() if isinstance(date, pd.Timestamp) else date
        past_holidays = [h for h in us_holidays.keys() if h < date_obj]
        if past_holidays:
            return (date_obj - max(past_holidays)).days
        return 365  # Default if no past holiday
    
    df['days_until_holiday'] = df['actual_ship'].apply(days_to_holiday)
    df['days_since_holiday'] = df['actual_ship'].apply(days_from_holiday)
    
    return df


def create_carrier_features(df):
    """
    Create carrier performance history features using ONLY historical data.
    
    Features created (using expanding window - only past data):
    - carrier_total_shipments (historical count up to current record)
    - carrier_avg_transit_days (historical average up to current record)
    - carrier_on_time_rate (historical rate up to current record)
    - carrier_late_rate (historical rate up to current record)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'actual_ship' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with carrier features added
    """
    df = df.copy()
    
    # Sort by date to ensure proper time-based features
    df = df.sort_values('actual_ship').reset_index(drop=True)
    
    print("  Calculating carrier features with temporal safety (no data leakage)...")
    
    # Use expanding window grouped by carrier (much faster than row-by-row)
    # Need to ensure index alignment - use transform to maintain original index
    carrier_groups = df.groupby('carrier_pseudo', group_keys=False)
    
    # Historical count (cumcount already excludes current row in its logic)
    # But we need to shift it to get count of previous rows
    df['carrier_total_shipments'] = carrier_groups.cumcount()  # 0, 1, 2, ... for each carrier
    
    # Historical average transit days (using transform to maintain index alignment)
    # Calculate expanding mean, then shift to exclude current row
    expanding_mean = carrier_groups['actual_transit_days'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    df['carrier_avg_transit_days'] = expanding_mean
    
    # Historical on-time rate
    if 'otd_designation' in df.columns:
        # Create binary columns for OTD
        df['_is_on_time'] = (df['otd_designation'] == 'On Time').astype(int)
        df['_is_late'] = (df['otd_designation'] == 'Late').astype(int)
        
        # Expanding mean with shift to exclude current row (using transform)
        df['carrier_on_time_rate'] = carrier_groups['_is_on_time'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        
        df['carrier_late_rate'] = carrier_groups['_is_late'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        
        # Drop temporary columns
        df = df.drop(columns=['_is_on_time', '_is_late'])
    else:
        df['carrier_on_time_rate'] = np.nan
        df['carrier_late_rate'] = np.nan
    
    # Fill NaN values (first occurrence of each carrier)
    df['carrier_total_shipments'] = df['carrier_total_shipments'].fillna(0)
    df['carrier_avg_transit_days'] = df['carrier_avg_transit_days'].fillna(df['actual_transit_days'].mean())
    df['carrier_on_time_rate'] = df['carrier_on_time_rate'].fillna(0)
    df['carrier_late_rate'] = df['carrier_late_rate'].fillna(0)
    
    return df


def create_lane_features(df):
    """
    Create lane-specific features using ONLY historical data.
    
    Features created (using expanding window - only past data):
    - lane_total_shipments (historical count up to current record)
    - lane_avg_transit_days (historical average up to current record)
    - lane_avg_distance (historical average up to current record)
    - lane_on_time_rate (historical rate up to current record)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with 'actual_ship' column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with lane features added
    """
    df = df.copy()
    
    # Sort by date
    df = df.sort_values('actual_ship').reset_index(drop=True)
    
    print("  Calculating lane features with temporal safety (no data leakage)...")
    
    # Use expanding window grouped by lane (much faster than row-by-row)
    # Use transform to maintain index alignment
    lane_groups = df.groupby('lane_id', group_keys=False)
    
    # Historical count (cumcount gives count of previous rows)
    df['lane_total_shipments'] = lane_groups.cumcount()
    
    # Historical averages (using transform to maintain index alignment)
    df['lane_avg_transit_days'] = lane_groups['actual_transit_days'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    
    df['lane_avg_distance'] = lane_groups['customer_distance'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    
    # Historical on-time rate
    if 'otd_designation' in df.columns:
        df['_is_on_time'] = (df['otd_designation'] == 'On Time').astype(int)
        
        df['lane_on_time_rate'] = lane_groups['_is_on_time'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        
        # Drop temporary column
        df = df.drop(columns=['_is_on_time'])
    else:
        df['lane_on_time_rate'] = np.nan
    
    # Fill NaN values (first occurrence of each lane)
    df['lane_total_shipments'] = df['lane_total_shipments'].fillna(0)
    df['lane_avg_transit_days'] = df['lane_avg_transit_days'].fillna(df['actual_transit_days'].mean())
    df['lane_avg_distance'] = df['lane_avg_distance'].fillna(df['customer_distance'].mean())
    df['lane_on_time_rate'] = df['lane_on_time_rate'].fillna(0)
    
    return df


def create_distance_features(df):
    """
    Create distance-based features.
    
    Features created:
    - distance_squared (non-linear feature)
    - distance_log (log transformation)
    - distance_sqrt (square root transformation)
    - route_key (origin_zip_3d + dest_zip_3d + distance_bucket for ETA simulator)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with distance features added
    """
    df = df.copy()
    
    # Non-linear transformations
    df['distance_squared'] = df['customer_distance'] ** 2
    df['distance_log'] = np.log1p(df['customer_distance'])  # log1p to handle zeros
    df['distance_sqrt'] = np.sqrt(df['customer_distance'])
    
    # Distance bucket encoding (already exists, but ensure it's categorical)
    if 'distance_bucket' in df.columns:
        df['distance_bucket'] = df['distance_bucket'].astype('category')
    
    # Create route_key for ETA simulator (origin_zip3 + dest_zip3 + distance_bucket)
    # This allows filtering by: 1) Origin zip3, 2) Destination zip3, 3) Distance bucket
    if all(col in df.columns for col in ['origin_zip_3d', 'dest_zip_3d', 'distance_bucket']):
        df['route_key'] = (
            df['origin_zip_3d'].astype(str) + '_' + 
            df['dest_zip_3d'].astype(str) + '_' + 
            df['distance_bucket'].astype(str)
        )
        
        # Also create a lane_zip3_pair without distance for broader matching
        df['lane_zip3_pair'] = (
            df['origin_zip_3d'].astype(str) + '_' + 
            df['dest_zip_3d'].astype(str)
        )
    
    return df


def create_route_features(df):
    """
    Create route-level features (based on route_key = origin_zip3 + dest_zip3 + distance_bucket).
    
    This is designed for the ETA simulator where users select:
    - Origin ZIP3
    - Destination ZIP3
    - Distance bucket
    
    Features created (using expanding window - only past data):
    - route_total_shipments (historical count)
    - route_avg_transit_days (historical average)
    - route_on_time_rate (historical rate)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with route_key column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with route features added
    """
    df = df.copy()
    
    if 'route_key' not in df.columns:
        print("  Warning: route_key not found, skipping route features")
        return df
    
    # Sort by date
    df = df.sort_values('actual_ship').reset_index(drop=True)
    
    print("  Calculating route features with temporal safety (no data leakage)...")
    
    route_groups = df.groupby('route_key', group_keys=False)
    
    # Historical count
    df['route_total_shipments'] = route_groups.cumcount()
    
    # Historical averages
    df['route_avg_transit_days'] = route_groups['actual_transit_days'].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    
    # Historical on-time rate
    if 'otd_designation' in df.columns:
        df['_is_on_time'] = (df['otd_designation'] == 'On Time').astype(int)
        
        df['route_on_time_rate'] = route_groups['_is_on_time'].transform(
            lambda x: x.expanding(min_periods=1).mean().shift(1)
        )
        
        df = df.drop(columns=['_is_on_time'])
    else:
        df['route_on_time_rate'] = np.nan
    
    # Fill NaN values
    df['route_total_shipments'] = df['route_total_shipments'].fillna(0)
    df['route_avg_transit_days'] = df['route_avg_transit_days'].fillna(df['actual_transit_days'].mean())
    df['route_on_time_rate'] = df['route_on_time_rate'].fillna(0)
    
    return df


def prepare_features(df):
    """
    Master function that applies all feature engineering.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with all features engineered
    """
    df = df.copy()
    
    print("Feature Engineering:")
    print("=" * 50)
    
    print("Creating temporal features...")
    df = create_temporal_features(df)
    
    print("Creating carrier features...")
    df = create_carrier_features(df)
    
    print("Creating lane features...")
    df = create_lane_features(df)
    
    print("Creating distance features...")
    df = create_distance_features(df)
    
    print("Creating route features...")
    df = create_route_features(df)
    
    print(f"Feature engineering complete. Shape: {df.shape}")
    print("=" * 50)
    
    return df


# ============================================================================
# MODEL DATA PREPARATION
# ============================================================================

def prepare_model_data(df):
    """
    Prepare final dataset for modeling.
    
    Drops:
    - Leakage columns: actual_transit_days, otd_designation
    - Useless columns: load_id_pseudo
    
    Creates:
    - Target: actual_delivery (rounded to hour, as datetime)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with all features
        
    Returns:
    --------
    tuple
        (X, y, metadata) where:
        - X: feature dataframe
        - y: target series (actual_delivery as datetime)
        - metadata: dict with column info
    """
    df = df.copy()
    
    # Columns to drop (leakage and useless)
    drop_cols = ['actual_transit_days', 'otd_designation', 'load_id_pseudo']
    
    # Also drop actual_ship and actual_delivery from features (we'll use derived features)
    # But keep actual_delivery as target
    feature_drop_cols = drop_cols + ['actual_ship', 'actual_delivery']
    
    # Create target (actual_delivery rounded to hour - already rounded in round_timestamps_to_hour)
    y = df['actual_delivery'].copy()
    
    # Prepare features (drop target and leakage columns)
    X = df.drop(columns=feature_drop_cols, errors='ignore').copy()
    
    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # One-hot encode categorical variables
    for col in categorical_cols:
        if col in X.columns:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    
    # Handle missing values (fill with median for numeric, mode for categorical)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)
    
    # Metadata
    metadata = {
        'feature_count': len(X.columns),
        'sample_count': len(X),
        'categorical_cols': categorical_cols,
        'numeric_cols': X.select_dtypes(include=[np.number]).columns.tolist()
    }
    
    print(f"\nModel Data Preparation:")
    print("=" * 50)
    print(f"Features: {metadata['feature_count']}")
    print(f"Samples: {metadata['sample_count']}")
    print(f"Dropped columns: {drop_cols}")
    print("=" * 50)
    
    return X, y, metadata


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlations(X, y):
    """
    Calculate feature-to-feature and feature-to-target correlations.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature dataframe
    y : pd.Series
        Target series (datetime)
        
    Returns:
    --------
    dict
        Dictionary with:
        - 'feature_to_feature': correlation matrix of features
        - 'feature_to_target': correlations of features with target
    """
    # Convert target to numeric (hours since epoch or hours until delivery)
    # Use hours until delivery from actual_ship
    if isinstance(y, pd.Series) and pd.api.types.is_datetime64_any_dtype(y):
        # We need actual_ship to calculate hours until delivery
        # For now, use timestamp as numeric
        y_numeric = (y - pd.Timestamp('2022-01-01')).dt.total_seconds() / 3600
    else:
        y_numeric = y
    
    # Feature-to-feature correlations (vectorized)
    feature_corr = X.corr()
    
    # Feature-to-target correlations (vectorized - much faster)
    # Convert to numpy for faster computation
    X_numeric = X.select_dtypes(include=[np.number])
    
    if len(X_numeric.columns) == 0:
        # No numeric columns, return empty series
        target_corr = pd.Series(dtype=float)
    else:
        # Convert to numpy arrays explicitly
        X_numeric_values = np.asarray(X_numeric.values, dtype=np.float64)
        if isinstance(y_numeric, pd.Series):
            y_numeric_array = np.asarray(y_numeric.values, dtype=np.float64)
        else:
            y_numeric_array = np.asarray(y_numeric, dtype=np.float64)
        
        # Ensure same length
        min_len = min(len(X_numeric_values), len(y_numeric_array))
        X_numeric_values = X_numeric_values[:min_len]
        y_numeric_array = y_numeric_array[:min_len]
        
        # Vectorized correlation calculation (Pearson correlation)
        X_centered = X_numeric_values - np.mean(X_numeric_values, axis=0, keepdims=True)
        y_centered = y_numeric_array - np.mean(y_numeric_array)
        
        # Calculate standard deviations (ensure numpy arrays)
        # Calculate sum of squares first, then sqrt
        X_sum_sq = np.sum(X_centered ** 2, axis=0)
        y_sum_sq = np.sum(y_centered ** 2)
        
        # Ensure they are numpy arrays/scalars
        X_sum_sq = np.asarray(X_sum_sq, dtype=np.float64)
        if np.isscalar(y_sum_sq):
            y_sum_sq = float(y_sum_sq)
        else:
            y_sum_sq = np.asarray(y_sum_sq, dtype=np.float64)
        
        # Now calculate sqrt (ensure we're using numpy sqrt)
        X_std = np.sqrt(X_sum_sq)
        y_std = np.sqrt(y_sum_sq) if isinstance(y_sum_sq, (int, float, np.number)) else np.sqrt(y_sum_sq)
        
        # Ensure X_std is a numpy array
        X_std = np.asarray(X_std, dtype=np.float64)
        
        # Ensure y_std is a scalar float
        if not np.isscalar(y_std):
            y_std = float(y_std)
        
        # Avoid division by zero
        denominator = X_std * y_std
        denominator = np.where(denominator == 0, 1.0, denominator)  # Set zero denominators to 1
        
        numerator = np.dot(X_centered.T, y_centered)
        correlations = numerator / denominator
        
        # Handle NaN values (from constant columns)
        correlations = np.nan_to_num(correlations, nan=0.0)
        
        # Ensure correlations is a numpy array
        correlations = np.asarray(correlations, dtype=np.float64)
        
        # Create Series with feature names
        target_corr = pd.Series(correlations, index=X_numeric.columns).sort_values(ascending=False)
    
    return {
        'feature_to_feature': feature_corr,
        'feature_to_target': target_corr
    }


def plot_correlation_heatmap(corr_matrix, figsize=(12, 10), title="Correlation Heatmap"):
    """
    Plot correlation heatmap.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    figsize : tuple
        Figure size
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object for further customization
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

