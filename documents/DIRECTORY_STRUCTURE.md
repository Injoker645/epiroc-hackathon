# Directory Structure

## Current Organization

```
Epiroc/
├── notebooks/              # All Jupyter notebooks
│   ├── EDA_Notebook.ipynb
│   ├── ETL_Notebook.ipynb
│   ├── Model_Notebook.ipynb
│   └── eda_analysis.py
│
├── documents/              # Documentation and PDFs
│   ├── *.md files
│   ├── *.pdf files
│   └── README files
│
├── Dataset/                # All data files
│   ├── last-mile-data.csv
│   ├── last-mile-column-descriptions.pdf
│   ├── sla.pdf
│   └── processed_features.csv (if exists)
│
├── outputs/                # Generated outputs
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
│
└── etl_utils.py          # Reusable ETL functions (root level for imports)
```

## Key Improvements

### 1. **No Data Leakage** ✅
- Carrier and lane features now use **expanding windows** with `.shift(1)` to exclude current row
- Only historical data (before current timestamp) is used
- Features calculated row-by-row using only past data

### 2. **Organized Structure** ✅
- All notebooks in `notebooks/` folder
- All documentation in `documents/` folder
- All data in `Dataset/` folder
- All outputs in `outputs/` folder

### 3. **Path Updates** ✅
- All notebooks updated to use relative paths (`../`)
- Works from `notebooks/` directory
- `etl_utils.py` remains in root for easy imports

## Usage

1. Navigate to `notebooks/` folder
2. Run notebooks in order:
   - `EDA_Notebook.ipynb` - Data exploration
   - `ETL_Notebook.ipynb` - Data processing (saves to `../outputs/data/`)
   - `Model_Notebook.ipynb` - Model training (saves to `../outputs/models/` and `../outputs/graphs/`)

## Data Leakage Prevention

The feature engineering functions now ensure temporal safety:

- **Carrier features**: Use `expanding().mean().shift(1)` to only use past carrier performance
- **Lane features**: Use `expanding().mean().shift(1)` to only use past lane performance
- **Count features**: Use `cumcount()` which automatically excludes current row

This ensures that when predicting delivery time for a shipment, we only use information that would have been available at the time of shipment.

