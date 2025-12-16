"""
Prediction utilities for ETA estimation.
Includes confidence interval calculation and feature building.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import holidays


# US holidays for feature engineering
US_HOLIDAYS = holidays.US()


def predict_with_confidence(model, X, lane_stats=None, carrier_stats=None):
    """
    Make predictions with confidence intervals.
    
    Uses historical variance to estimate prediction uncertainty.
    
    Parameters:
    -----------
    model : trained model
        The XGBoost model
    X : np.ndarray or pd.DataFrame
        Features for prediction
    lane_stats : pd.DataFrame, optional
        Lane-level statistics for variance estimation
    carrier_stats : pd.DataFrame, optional
        Carrier-level statistics for variance estimation
        
    Returns:
    --------
    dict
        Contains 'prediction', 'lower', 'upper', 'confidence'
    """
    # Get point prediction
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    pred = model.predict(X_values)
    
    if len(pred.shape) == 0:
        pred = np.array([pred])
    
    # Calculate confidence based on available stats
    # Default: use 15% uncertainty
    uncertainty_pct = 0.15
    
    # TODO: If we have historical variance for the specific lane/carrier,
    # use that instead for more accurate uncertainty estimation
    
    lower = pred * (1 - uncertainty_pct)
    upper = pred * (1 + uncertainty_pct)
    
    # Confidence score: inverse of relative uncertainty
    # Higher confidence = narrower interval relative to prediction
    confidence = 1 - uncertainty_pct
    
    return {
        'prediction': pred[0] if len(pred) == 1 else pred,
        'lower': lower[0] if len(lower) == 1 else lower,
        'upper': upper[0] if len(upper) == 1 else upper,
        'confidence': confidence
    }


def classify_prediction_status(pred_hours, goal_days=None, threshold_pct=0.1):
    """
    Classify prediction as early, on_time, or late.
    
    Parameters:
    -----------
    pred_hours : float
        Predicted transit hours
    goal_days : float, optional
        Target transit days
    threshold_pct : float
        Tolerance percentage for on-time classification
        
    Returns:
    --------
    str
        Status: 'early', 'on_time', 'late', or 'unknown'
    """
    if goal_days is None:
        return 'unknown'
    
    goal_hours = goal_days * 24
    
    # Within threshold is on-time
    lower_bound = goal_hours * (1 - threshold_pct)
    upper_bound = goal_hours * (1 + threshold_pct)
    
    if pred_hours < lower_bound:
        return 'early'
    elif pred_hours <= upper_bound:
        return 'on_time'
    else:
        return 'late'


def get_status_color(status):
    """Get color for status badge."""
    colors = {
        'early': '#2ecc71',      # Green
        'on_time': '#27ae60',    # Darker green
        'late': '#e74c3c',       # Red
        'unknown': '#95a5a6',    # Gray
        'risk': '#f39c12'        # Orange/yellow
    }
    return colors.get(status, '#95a5a6')


def get_status_emoji(status):
    """Get emoji for status."""
    emojis = {
        'early': '🟢',
        'on_time': '✅',
        'late': '🔴',
        'unknown': '⚪',
        'risk': '🟡',
        'best': '🏆'
    }
    return emojis.get(status, '⚪')


def build_temporal_features(ship_datetime):
    """
    Build temporal features from ship datetime.
    
    Parameters:
    -----------
    ship_datetime : datetime
        The ship datetime
        
    Returns:
    --------
    dict
        Temporal features
    """
    features = {
        'ship_year': ship_datetime.year,
        'ship_month': ship_datetime.month,
        'ship_week': ship_datetime.isocalendar()[1],
        'ship_day_of_week': ship_datetime.weekday(),
        'ship_hour': ship_datetime.hour,
        'is_weekend': 1 if ship_datetime.weekday() >= 5 else 0,
        'is_holiday': 1 if ship_datetime.date() in US_HOLIDAYS else 0,
    }
    
    # Season (1-4)
    month = ship_datetime.month
    if month in [12, 1, 2]:
        features['season'] = 1  # Winter
    elif month in [3, 4, 5]:
        features['season'] = 2  # Spring
    elif month in [6, 7, 8]:
        features['season'] = 3  # Summer
    else:
        features['season'] = 4  # Fall
    
    return features


def build_prediction_features(lane_id, carrier, mode, ship_datetime, 
                               raw_data=None, feature_columns=None):
    """
    Build feature vector for prediction.
    
    This function constructs a feature vector matching the model's expected input.
    
    Parameters:
    -----------
    lane_id : str
        Lane identifier
    carrier : str
        Carrier identifier
    mode : str
        Carrier mode (LTL, TL Dry, TL Flatbed)
    ship_datetime : datetime
        Ship datetime
    raw_data : pd.DataFrame, optional
        Raw data for looking up lane/carrier info
    feature_columns : list, optional
        List of feature column names expected by model
        
    Returns:
    --------
    np.ndarray
        Feature vector ready for prediction
    """
    # Get temporal features
    temporal = build_temporal_features(ship_datetime)
    
    # Start with basic features
    features = {
        'ship_day_of_week': temporal['ship_day_of_week'],
        'ship_week': temporal['ship_week'],
        'ship_month': temporal['ship_month'],
        'ship_year': temporal['ship_year'],
        'is_weekend': temporal['is_weekend'],
        'is_holiday': temporal['is_holiday'],
        'season': temporal['season'],
    }
    
    # Add lane-specific features from raw data if available
    if raw_data is not None:
        lane_data = raw_data[raw_data['lane_id'] == lane_id]
        
        if len(lane_data) > 0:
            # Get average values for this lane
            features['customer_distance'] = lane_data['customer_distance'].mean()
            features['all_modes_goal_transit_days'] = lane_data['all_modes_goal_transit_days'].mean()
            
            # Get carrier-specific averages
            carrier_lane_data = lane_data[
                (lane_data['carrier_pseudo'] == carrier) & 
                (lane_data['carrier_mode'] == mode)
            ]
            
            if len(carrier_lane_data) > 0:
                features['carrier_posted_service_days'] = carrier_lane_data['carrier_posted_service_days'].mean()
                features['truckload_service_days'] = carrier_lane_data['truckload_service_days'].mean()
            else:
                features['carrier_posted_service_days'] = lane_data['carrier_posted_service_days'].mean()
                features['truckload_service_days'] = lane_data['truckload_service_days'].mean()
    
    # Return as DataFrame for easier handling
    return pd.DataFrame([features])


def calculate_eta_datetime(ship_datetime, pred_hours):
    """
    Calculate ETA datetime from ship datetime and predicted hours.
    
    Parameters:
    -----------
    ship_datetime : datetime
        Ship datetime
    pred_hours : float
        Predicted transit hours
        
    Returns:
    --------
    datetime
        Estimated arrival datetime
    """
    return ship_datetime + timedelta(hours=pred_hours)


def format_duration(hours):
    """
    Format duration in hours to human-readable string.
    
    Parameters:
    -----------
    hours : float
        Duration in hours
        
    Returns:
    --------
    str
        Formatted string (e.g., "2 days 5 hours")
    """
    days = int(hours // 24)
    remaining_hours = int(hours % 24)
    
    if days == 0:
        return f"{remaining_hours} hours"
    elif days == 1:
        return f"1 day {remaining_hours} hours"
    else:
        return f"{days} days {remaining_hours} hours"


def generate_carrier_recommendations(lane_state_pair=None, lane_id=None, ship_datetime=None, 
                                      model=None, raw_data=None, feature_columns=None, 
                                      goal_days=None):
    """
    Generate carrier recommendations for a route.
    
    Parameters:
    -----------
    lane_state_pair : str
        State-based route identifier (e.g., 'OH_PA')
    lane_id : str (deprecated)
        Legacy lane identifier, used if lane_state_pair not provided
    ship_datetime : datetime
        Ship datetime
    model : trained model
        XGBoost model
    raw_data : pd.DataFrame
        Raw delivery data
    feature_columns : list
        Feature columns expected by model
    goal_days : float, optional
        Target transit days
        
    Returns:
    --------
    list[dict]
        List of carrier recommendations sorted by ETA
    """
    if raw_data is None:
        return []
    
    # Filter data by route
    if lane_state_pair is not None and 'lane_state_pair' in raw_data.columns:
        route_data = raw_data[raw_data['lane_state_pair'] == lane_state_pair]
    elif lane_id is not None:
        route_data = raw_data[raw_data['lane_id'] == lane_id]
    else:
        return []
    
    if len(route_data) == 0:
        return []
    
    # Get unique carrier/mode combinations
    combos = route_data[['carrier_pseudo', 'carrier_mode']].drop_duplicates()
    
    results = []
    for _, row in combos.iterrows():
        carrier = row['carrier_pseudo']
        mode = row['carrier_mode']
        
        # Get historical performance for this carrier/mode on this route
        carrier_route_data = route_data[
            (route_data['carrier_pseudo'] == carrier) & 
            (route_data['carrier_mode'] == mode)
        ]
        
        # Use historical average as prediction
        avg_hours = carrier_route_data['transit_hours'].mean() if 'transit_hours' in carrier_route_data.columns else \
                    carrier_route_data['actual_transit_days'].mean() * 24
        std_hours = carrier_route_data['transit_hours'].std() if 'transit_hours' in carrier_route_data.columns else \
                    carrier_route_data['actual_transit_days'].std() * 24
        
        # Calculate confidence based on sample size and variance
        n_samples = len(carrier_route_data)
        variance_factor = min(std_hours / (avg_hours + 1), 1) if pd.notna(std_hours) else 0.5
        sample_factor = min(n_samples / 100, 1)  # More samples = higher confidence
        
        confidence = (1 - variance_factor) * 0.5 + sample_factor * 0.5
        confidence = max(0.3, min(0.95, confidence))  # Clamp between 30% and 95%
        
        # Determine status
        status = classify_prediction_status(avg_hours, goal_days)
        
        # Calculate on-time rate
        on_time_rate = (carrier_route_data['otd_designation'] == 'On Time').mean() \
                       if 'otd_designation' in carrier_route_data.columns else 0.5
        
        results.append({
            'carrier': carrier,
            'mode': mode,
            'eta_hours': avg_hours,
            'eta_datetime': calculate_eta_datetime(ship_datetime, avg_hours),
            'eta_formatted': format_duration(avg_hours),
            'confidence': confidence,
            'status': status,
            'on_time_rate': on_time_rate,
            'sample_size': n_samples,
            'std_hours': std_hours if pd.notna(std_hours) else 0,
            'is_best': False
        })
    
    # Sort by ETA
    results = sorted(results, key=lambda x: x['eta_hours'])
    
    # Mark best option (fastest with good reliability)
    if results:
        # Best = lowest ETA among those with >= 50% on-time rate
        reliable_results = [r for r in results if r['on_time_rate'] >= 0.5]
        if reliable_results:
            reliable_results[0]['is_best'] = True
        else:
            results[0]['is_best'] = True
    
    return results

