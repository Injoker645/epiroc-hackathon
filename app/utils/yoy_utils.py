"""
Year-over-Year (YoY) comparison utilities.
Calculates metrics comparing current period to same period last year.
Uses the data's max date as reference (not today) to handle historical datasets.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_data_date_range(df, date_column='actual_ship'):
    """
    Get the date range of the data.
    
    Returns:
    --------
    tuple
        (min_date, max_date)
    """
    df[date_column] = pd.to_datetime(df[date_column])
    return (df[date_column].min(), df[date_column].max())


def get_yoy_periods(reference_date=None, window_days=90, df=None, date_column='actual_ship'):
    """
    Get date ranges for YoY comparison.
    
    Parameters:
    -----------
    reference_date : datetime, optional
        Reference date (defaults to max date in data, or today if no data)
    window_days : int
        Number of days in the comparison window (default 90)
    df : pd.DataFrame, optional
        Data to get max date from
    date_column : str
        Date column name
        
    Returns:
    --------
    dict
        Contains 'current_start', 'current_end', 'prior_start', 'prior_end'
    """
    # Use data's max date if available, otherwise use today
    if reference_date is None:
        if df is not None and len(df) > 0:
            df[date_column] = pd.to_datetime(df[date_column])
            reference_date = df[date_column].max()
            # Convert to datetime if it's a Timestamp
            if hasattr(reference_date, 'to_pydatetime'):
                reference_date = reference_date.to_pydatetime()
        else:
            reference_date = datetime.now()
    
    current_end = reference_date
    current_start = current_end - timedelta(days=window_days)
    
    prior_end = current_end - timedelta(days=365)
    prior_start = prior_end - timedelta(days=window_days)
    
    return {
        'current_start': current_start,
        'current_end': current_end,
        'prior_start': prior_start,
        'prior_end': prior_end,
        'window_days': window_days
    }


def filter_by_period(df, date_column, start_date, end_date):
    """Filter dataframe to a date range."""
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Convert dates to datetime if needed
    if hasattr(start_date, 'to_pydatetime'):
        start_date = start_date.to_pydatetime()
    if hasattr(end_date, 'to_pydatetime'):
        end_date = end_date.to_pydatetime()
    
    return df[(df[date_column] >= start_date) & (df[date_column] <= end_date)]


def calculate_yoy_metrics(df, date_column='actual_ship', reference_date=None, window_days=90):
    """
    Calculate YoY comparison metrics.
    
    Uses the data's max date as reference point to handle historical datasets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data with shipment records
    date_column : str
        Column name containing dates
    reference_date : datetime, optional
        Reference date for comparison (defaults to max date in data)
    window_days : int
        Window size in days (default 90)
        
    Returns:
    --------
    dict
        YoY metrics with current, prior, and delta values
    """
    if df is None or len(df) == 0:
        return {
            'periods': get_yoy_periods(reference_date, window_days),
            'current_count': 0,
            'prior_count': 0,
        }
    
    # Get periods using data's max date
    periods = get_yoy_periods(reference_date, window_days, df, date_column)
    
    current_data = filter_by_period(df, date_column, periods['current_start'], periods['current_end'])
    prior_data = filter_by_period(df, date_column, periods['prior_start'], periods['prior_end'])
    
    metrics = {
        'periods': periods,
        'current_count': len(current_data),
        'prior_count': len(prior_data),
    }
    
    # Shipment volume change
    if metrics['prior_count'] > 0:
        metrics['volume_change_pct'] = ((metrics['current_count'] - metrics['prior_count']) / metrics['prior_count']) * 100
    else:
        metrics['volume_change_pct'] = None
    
    # On-time rate
    if 'otd_designation' in df.columns:
        current_on_time = (current_data['otd_designation'] == 'On Time').mean() * 100 if len(current_data) > 0 else 0
        prior_on_time = (prior_data['otd_designation'] == 'On Time').mean() * 100 if len(prior_data) > 0 else 0
        
        metrics['current_on_time_rate'] = current_on_time
        metrics['prior_on_time_rate'] = prior_on_time
        metrics['on_time_rate_delta'] = current_on_time - prior_on_time if len(current_data) > 0 and len(prior_data) > 0 else None
        
        # Late rate
        current_late = (current_data['otd_designation'] == 'Late').mean() * 100 if len(current_data) > 0 else 0
        prior_late = (prior_data['otd_designation'] == 'Late').mean() * 100 if len(prior_data) > 0 else 0
        
        metrics['current_late_rate'] = current_late
        metrics['prior_late_rate'] = prior_late
        metrics['late_rate_delta'] = current_late - prior_late if len(current_data) > 0 and len(prior_data) > 0 else None
    
    # Transit days
    if 'actual_transit_days' in df.columns:
        current_transit = current_data['actual_transit_days'].mean() if len(current_data) > 0 else 0
        prior_transit = prior_data['actual_transit_days'].mean() if len(prior_data) > 0 else 0
        
        metrics['current_avg_transit'] = current_transit
        metrics['prior_avg_transit'] = prior_transit
        metrics['avg_transit_delta'] = current_transit - prior_transit if len(current_data) > 0 and len(prior_data) > 0 else None
        
        # Variance/std dev
        current_std = current_data['actual_transit_days'].std() if len(current_data) > 1 else 0
        prior_std = prior_data['actual_transit_days'].std() if len(prior_data) > 1 else 0
        
        metrics['current_transit_std'] = current_std if pd.notna(current_std) else 0
        metrics['prior_transit_std'] = prior_std if pd.notna(prior_std) else 0
        metrics['transit_std_delta'] = (current_std - prior_std) if pd.notna(current_std) and pd.notna(prior_std) and len(current_data) > 1 and len(prior_data) > 1 else None
    
    return metrics


def format_delta(value, suffix='%', inverse=False, decimals=1):
    """
    Format a delta value with arrow and color hint.
    
    Parameters:
    -----------
    value : float
        The delta value
    suffix : str
        Suffix to add (e.g., '%', ' days')
    inverse : bool
        If True, negative is good (e.g., for late rate)
    decimals : int
        Decimal places
        
    Returns:
    --------
    tuple
        (formatted_string, delta_color)
        delta_color: "normal" (green), "inverse" (red), "off" (grey for 0)
    """
    if value is None or pd.isna(value):
        return ("N/A", "off")
    
    # Check if value is effectively zero
    if abs(value) < 0.01:  # Less than 0.01% or 0.01 days
        return ("~", "off")  # Grey indicator for no change
    
    formatted = f"{value:+.{decimals}f}{suffix}"
    
    # Determine if change is "good"
    if inverse:
        is_good = value < 0  # Decrease is good
    else:
        is_good = value > 0  # Increase is good
    
    delta_color = "normal" if is_good else "inverse"
    return (formatted, delta_color)


def get_default_date_range(df=None, date_column='actual_ship', years_back=1):
    """
    Get default date range for historical data (last N years from data's max date).
    
    Parameters:
    -----------
    df : pd.DataFrame, optional
        Data to get max date from
    date_column : str
        Date column name
    years_back : int
        Number of years to go back
        
    Returns:
    --------
    tuple
        (start_date, end_date)
    """
    if df is not None and len(df) > 0:
        df[date_column] = pd.to_datetime(df[date_column])
        end_date = df[date_column].max()
        if hasattr(end_date, 'to_pydatetime'):
            end_date = end_date.to_pydatetime()
    else:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=365 * years_back)
    
    return (start_date, end_date)


def filter_last_n_years(df, date_column='actual_ship', years=1, reference_date=None):
    """
    Filter dataframe to last N years from data's max date.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Date column name
    years : int
        Number of years to include
    reference_date : datetime, optional
        Reference date (defaults to data's max date)
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    if df is None or len(df) == 0:
        return df
    
    start_date, end_date = get_default_date_range(df, date_column, years)
    return filter_by_period(df, date_column, start_date, end_date)
