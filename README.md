# 🚚 What's Your ETA?

**Last-Mile Delivery Optimization** — Epiroc AI & Data Hackathon

> Transform last-mile delivery from a source of uncertainty into a key driver of customer satisfaction.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🎯 Overview

This project tackles the critical challenge of **last-mile delivery optimization** by building accurate ETA prediction models and providing actionable insights for improving delivery reliability.

**Key Features:**
- 🔮 **ETA Prediction**: XGBoost model predicts delivery times with high accuracy
- 🗺️ **Lane Explorer**: Interactive map visualization of delivery routes with state-level filtering
- ⚡ **What-If Simulator**: Dynamic filters (Origin ZIP3 → Dest ZIP3 → Distance) to compare carriers
- 📊 **Feature Importance**: Permutation-based importance analysis for model explainability
- 🎯 **Business Insights**: Identify optimization opportunities by carrier, lane, and timing

## 📁 Project Structure

```
Epiroc/
├── app/                        # Streamlit Dashboard Application
│   ├── streamlit_app.py            # Main entry point
│   ├── pages/                      # Dashboard pages
│   │   ├── 1_Lane_Explorer.py          # Map visualization
│   │   ├── 2_ETA_Simulator.py          # What-If predictions (dynamic filters)
│   │   └── 3_Model_Explainer.py        # Feature importance
│   ├── utils/                      # Utility modules
│   │   ├── data_loader.py              # Data loading & caching
│   │   ├── prediction_utils.py         # Prediction functions
│   │   ├── map_utils.py                # Map generation
│   │   └── shap_utils.py               # Feature importance utilities
│   ├── data/                       # App reference data
│   │   ├── zip_coordinates.csv         # ZIP3 → lat/lon/state mapping
│   │   └── generate_zip_coords.py      # Script to regenerate ZIP data
│   └── requirements.txt            # App dependencies
│
├── notebooks/                  # Jupyter Notebooks (run in order)
│   ├── EDA_Notebook.ipynb          # 1. Exploratory Data Analysis
│   ├── ETL_Notebook.ipynb          # 2. Data Pipeline & Feature Engineering
│   └── Model_Notebook.ipynb        # 3. Model Training & Evaluation
│
├── src/                        # Source Code Modules
│   └── etl_utils.py                # Reusable ETL functions
│
├── documents/                  # Documentation & References
│
├── Dataset/                    # Data Files (⚠️ gitignored)
│   └── last-mile-data.csv          # Place raw data here
│
├── outputs/                    # Generated Outputs (⚠️ gitignored)
│   ├── data/                       # Processed features & targets
│   ├── models/                     # Trained model & importance results
│   └── graphs/                     # Visualization outputs
│
├── requirements.txt            # Root dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/epiroc-eta-optimizer.git
cd epiroc-eta-optimizer

# Install all dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place `last-mile-data.csv` in the `Dataset/` folder.

### 3. Run the Pipeline

```bash
# Step 1: Explore the data (optional)
jupyter notebook notebooks/EDA_Notebook.ipynb

# Step 2: Process data and engineer features
jupyter notebook notebooks/ETL_Notebook.ipynb

# Step 3: Train the model and compute feature importance
jupyter notebook notebooks/Model_Notebook.ipynb
```

### 4. Launch the Dashboard

```bash
cd app
streamlit run streamlit_app.py
```

Open http://localhost:8501 in your browser.

## 📊 Dashboard Features

### 1. Lane Explorer 🗺️
- Interactive US map showing all shipping lanes
- Color-coded by historical on-time performance
- Filter by state, view lane statistics
- Click lanes to see detailed metrics

### 2. ETA Simulator ⚡
**Dynamic cascading filters by state:**
```
📍 Origin State  →  📍 Dest State  →  📏 Distance Bucket
      [OH]       →       [PA]      →     [250-500mi]
```

- Select any combination of origin state, destination state, and distance
- See all viable carrier options ranked by predicted ETA
- Status indicators: 🟢 On-time | 🟡 Risk | 🔴 Late
- 🏆 Best option highlighted
- Confidence scores based on sample size and variance
- View route details including ZIP3 codes within each state

### 3. Model Explainer 📊
- **Permutation Importance**: See which features matter most
- **Category Analysis**: Importance breakdown by Carrier/Lane/Distance/Temporal
- **Insights**: Actionable recommendations based on feature analysis

## 📓 Notebooks

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| **EDA_Notebook** | Data exploration & quality analysis | Visualizations, anomaly detection |
| **ETL_Notebook** | Feature engineering pipeline | `processed_features.csv`, `target.csv` |
| **Model_Notebook** | Model training & evaluation | `xgboost_model.pkl`, importance results |

### Feature Engineering Highlights

- **Temporal**: US holidays, seasons, day of week, month/quarter end
- **Carrier**: Historical count, avg transit days, on-time rate (per carrier)
- **Lane**: Historical statistics per origin-destination pair
- **Route**: Stats by `route_key` (origin_zip3 + dest_zip3 + distance_bucket)
- **Distance**: Linear, log, sqrt transformations

**⚠️ No Data Leakage**: All historical features use expanding windows with `.shift(1)` to exclude current row.

## 🔧 Technical Details

### Target Variable
```python
actual_transit_hours = (actual_delivery - actual_ship).total_seconds() / 3600
```

### Key Features
| Category | Features |
|----------|----------|
| Temporal | `is_holiday`, `season`, `ship_day_of_week`, `is_month_end` |
| Carrier | `carrier_total_shipments`, `carrier_avg_transit_days`, `carrier_on_time_rate` |
| Lane | `lane_total_shipments`, `lane_avg_transit_days`, `lane_on_time_rate` |
| State Route | `lane_state_pair`, `state_route_avg_transit_days`, `state_route_on_time_rate` |
| Granular Route | `lane_state_pair_distance_bucket`, `granular_route_avg_transit_days` |
| Distance | `customer_distance`, `distance_log`, `distance_sqrt` |

### State-Based Routing
Routes are identified by state pairs (e.g., `OH_PA` = Ohio → Pennsylvania):
- **`lane_state_pair`**: Broad route (state to state)
- **`lane_state_pair_distance_bucket`**: Granular route (state pair + distance band)

State codes are derived from 3-digit ZIP prefixes using the `pgeocode` library.

### Model
- **Algorithm**: XGBoost Regressor
- **Explainability**: Permutation Importance (saved to `feature_importance_results.pkl`)
- **Metrics**: MAE, RMSE, R², OTD classification accuracy

### ZIP Code Mapping
Uses `pgeocode` library for accurate US ZIP3 → State/Coordinates mapping:
```python
# Regenerate ZIP coordinates (if needed)
cd app/data
python generate_zip_coords.py
```

## 🛠️ Dependencies

**Core:**
- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, holidays
- tqdm (progress bars)

**Dashboard:**
- streamlit, plotly
- folium, streamlit-folium
- pgeocode (ZIP code lookup)

See `requirements.txt` for complete list.

## 🎓 Challenge Objectives Addressed

| Objective | Implementation |
|-----------|----------------|
| ✅ Lead Time Prediction | XGBoost model with 3000+ engineered features |
| ✅ Anomaly Identification | EDA notebook highlights data quality issues |
| ✅ Intuitive UX | Interactive Streamlit dashboard with maps |
| ✅ Business Impact Analysis | Feature importance shows optimization opportunities |

## 📈 Sample Results

After running the pipeline, you'll see metrics like:
- **MAE**: ~X hours
- **R²**: ~Y%
- **On-Time Classification Accuracy**: ~Z%

*(Actual values depend on your data)*

## 🔄 Regenerating Outputs

```bash
# Delete cached outputs to force regeneration
rm outputs/models/feature_importance_results.pkl  # Recompute importance
rm outputs/models/xgboost_model.pkl               # Retrain model

# Then re-run Model_Notebook.ipynb
```

## 📄 License

Created for the **Epiroc Last-Mile Delivery Optimization AI & Data Hackathon**.

---

**Built with ❤️ for better deliveries**
