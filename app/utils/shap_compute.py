"""
SHAP computation utilities for model explainability.
Computes and caches SHAP values for individual predictions and key factors.
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get project directory
APP_DIR = Path(__file__).parent.parent
PROJECT_DIR = APP_DIR.parent

# Cache for SHAP explainer
_SHAP_EXPLAINER = None
_SHAP_MODEL = None
_SHAP_FEATURE_NAMES = None


def get_shap_explainer(model=None, X_background=None, force_reload=False):
    """
    Get or create SHAP explainer (cached).
    
    Parameters:
    -----------
    model : xgb.XGBRegressor, optional
        The trained model
    X_background : np.ndarray, optional
        Background data for explainer
    force_reload : bool
        Force reload even if cached
        
    Returns:
    --------
    shap.Explainer or None
        SHAP explainer instance
    """
    global _SHAP_EXPLAINER, _SHAP_MODEL, _SHAP_FEATURE_NAMES
    
    if _SHAP_EXPLAINER is not None and not force_reload:
        return _SHAP_EXPLAINER
    
    if model is None:
        # Try to load model
        model_path = PROJECT_DIR / 'outputs' / 'models' / 'xgboost_model.pkl'
        if not model_path.exists():
            return None
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    try:
        import shap
        
        # Try multiple SHAP initialization methods
        if X_background is None:
            # Load background data
            features_path = PROJECT_DIR / 'outputs' / 'data' / 'processed_features.csv'
            if features_path.exists():
                features_df = pd.read_csv(features_path)
                X_background = features_df.sample(min(1000, len(features_df)), random_state=42).values
            else:
                return None
        
        # Try TreeExplainer first (fastest for XGBoost)
        try:
            explainer = shap.TreeExplainer(model, X_background[:100])
            _SHAP_EXPLAINER = explainer
            _SHAP_MODEL = model
            _SHAP_FEATURE_NAMES = None  # Will be set when we have feature names
            return explainer
        except Exception:
            pass
        
        # Fallback to Explainer (newer API)
        try:
            explainer = shap.Explainer(model, X_background[:100])
            _SHAP_EXPLAINER = explainer
            _SHAP_MODEL = model
            return explainer
        except Exception:
            pass
        
        # Last resort: KernelExplainer (slow but works)
        try:
            def model_predict(X):
                return model.predict(X)
            explainer = shap.KernelExplainer(model_predict, X_background[:50])
            _SHAP_EXPLAINER = explainer
            _SHAP_MODEL = model
            return explainer
        except Exception:
            return None
            
    except ImportError:
        return None


def compute_shap_for_row(row_features, feature_names=None, model=None, X_background=None):
    """
    Compute SHAP values for a single row.
    
    Parameters:
    -----------
    row_features : np.ndarray or pd.Series
        Feature values for one row
    feature_names : list, optional
        Names of features
    model : xgb.XGBRegressor, optional
        The trained model
    X_background : np.ndarray, optional
        Background data
        
    Returns:
    --------
    dict
        Contains 'shap_values', 'base_value', 'prediction', 'top_feature'
    """
    explainer = get_shap_explainer(model, X_background)
    if explainer is None:
        return None
    
    # Ensure row_features is 2D
    if row_features.ndim == 1:
        row_features = row_features.reshape(1, -1)
    
    try:
        import shap
        shap_values = explainer(row_features)
        
        # Handle different SHAP output formats
        if hasattr(shap_values, 'values'):
            values = shap_values.values[0]
            base = shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0
        elif isinstance(shap_values, np.ndarray):
            values = shap_values[0] if shap_values.ndim > 1 else shap_values
            base = 0
        else:
            values = shap_values[0]
            base = 0
        
        prediction = base + values.sum()
        
        # Find top contributing feature
        abs_values = np.abs(values)
        top_idx = np.argmax(abs_values)
        
        result = {
            'shap_values': values,
            'base_value': base,
            'prediction': prediction,
            'top_feature_idx': top_idx,
            'top_feature_shap': values[top_idx]
        }
        
        if feature_names is not None and len(feature_names) > top_idx:
            result['top_feature_name'] = feature_names[top_idx]
        
        return result
        
    except Exception as e:
        return None


def get_key_factor_from_shap(row_data, model=None, feature_columns=None, X_background=None):
    """
    Get key factor explanation using SHAP for a data row.
    
    Parameters:
    -----------
    row_data : pd.Series or dict
        Row of data with actual values
    model : xgb.XGBRegressor, optional
        Trained model
    feature_columns : list, optional
        Feature column names
    X_background : np.ndarray, optional
        Background data
        
    Returns:
    --------
    str
        Human-readable key factor explanation
    """
    # This would need the processed features for the row
    # For now, return a placeholder that will be computed when we have full feature pipeline
    return "Computing..."


def compute_and_save_shap_values(df, model, feature_columns, output_path=None, sample_size=1000):
    """
    Compute SHAP values for a sample of data and save them.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to compute SHAP for
    model : xgb.XGBRegressor
        Trained model
    feature_columns : list
        Feature column names
    output_path : str, optional
        Path to save SHAP values
    sample_size : int
        Number of samples to compute SHAP for
        
    Returns:
    --------
    dict
        Dictionary with SHAP values and metadata
    """
    if output_path is None:
        output_path = PROJECT_DIR / 'outputs' / 'models' / 'shap_values_cache.pkl'
    
    # Sample data
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    explainer = get_shap_explainer(model)
    if explainer is None:
        return None
    
    try:
        import shap
        
        # Get feature values
        X_sample = df_sample[feature_columns].values
        
        # Compute SHAP values
        shap_values = explainer(X_sample)
        
        # Extract values
        if hasattr(shap_values, 'values'):
            values = shap_values.values
            base_values = shap_values.base_values if hasattr(shap_values, 'base_values') else None
        else:
            values = shap_values
            base_values = None
        
        result = {
            'shap_values': values,
            'base_values': base_values,
            'feature_names': feature_columns,
            'sample_indices': df_sample.index.tolist(),
            'predictions': model.predict(X_sample)
        }
        
        # Save to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        
        return result
        
    except Exception as e:
        return None


def load_shap_cache(cache_path=None):
    """Load cached SHAP values."""
    if cache_path is None:
        cache_path = PROJECT_DIR / 'outputs' / 'models' / 'shap_values_cache.pkl'
    
    if not os.path.exists(cache_path):
        return None
    
    with open(cache_path, 'rb') as f:
        return pickle.load(f)

