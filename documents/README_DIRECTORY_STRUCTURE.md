# Directory Structure

## Current Organization

```
Epiroc/
├── notebooks/              # (Optional) For organizing notebooks
├── outputs/
│   ├── data/              # Processed data files
│   │   ├── processed_features.csv
│   │   ├── target.csv
│   │   └── metadata.json
│   ├── models/            # Trained models
│   │   └── xgboost_model.pkl
│   └── graphs/            # Visualizations
│       ├── feature_target_correlations.png
│       ├── feature_feature_correlations.png
│       ├── predictions_vs_actual.png
│       ├── feature_importance.png
│       ├── shap_summary_plot.png
│       └── shap_feature_importance.png
├── Dataset/               # Original dataset
├── EDA_Notebook.ipynb     # Exploratory Data Analysis
├── ETL_Notebook.ipynb     # Data pipeline
├── Model_Notebook.ipynb   # Model training and evaluation
└── etl_utils.py          # Reusable ETL functions
```

## Improvements Made

1. **Vectorized Correlation Calculation**: Much faster correlation computation using NumPy vectorization
2. **GPU Support**: XGBoost automatically detects and uses GPU if available
3. **SHAP Compatibility**: Added fallback mechanisms for SHAP compatibility issues
4. **Organized Outputs**: All outputs saved to organized folders

## Usage

1. Run `EDA_Notebook.ipynb` for data exploration
2. Run `ETL_Notebook.ipynb` to process data (saves to `outputs/data/`)
3. Run `Model_Notebook.ipynb` to train model (saves to `outputs/models/` and `outputs/graphs/`)

