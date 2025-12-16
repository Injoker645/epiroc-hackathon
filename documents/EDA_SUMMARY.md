# Epiroc Last-Mile Delivery Optimization - EDA Summary

## Competition Overview

**Challenge**: Build UX and analytics for the last mile delivery process

**Key Objectives**:
- Develop solutions for more reliable and transparent customer experience
- Deliver more accurate Estimated Time of Arrivals (ETAs)
- Lead time prediction
- Anomaly identification
- Intuitive UX for data exploration to guide human root-cause investigations
- Business impact analysis to guide prioritization of optimization initiatives

**Goal**: Transform the last-mile from a source of uncertainty into a key driver of customer satisfaction

---

## Dataset Overview

### Basic Statistics
- **Total Records**: 72,966 delivery records
- **Time Period**: January 3, 2022 to August 29, 2025 (3.5+ years)
- **Total Columns**: 20 features
- **Date Range**: ~3.5 years of historical data

### Key Metrics
- **On-Time Rate**: 63.88%
- **Late Rate**: 19.18% (13,998 deliveries)
- **Early Rate**: 16.94% (12,359 deliveries)
- **Average Actual Transit Days**: 2.91 days
- **Average Goal Transit Days**: 2.79 days
- **Average Difference**: 0.13 days (slightly over goal on average)

---

## Data Quality

### Missing Values
- `truckload_service_days`: 95.01% missing (expected - only for Truckload mode)
- `carrier_posted_service_days`: 4.99% missing (expected - only for LTL mode)

### Data Anomalies
- **Negative transit days**: Some records show negative actual_transit_days (min: -270 days) - likely data quality issues
- **Extreme delays**: Maximum delay of 67 days beyond goal
- **Date range includes future dates**: Data extends to August 2025 (likely includes forecasted or planned shipments)

---

## Feature Analysis

### 1. Carrier Mode Distribution
- **LTL (Less Than Truckload)**: 95.01% (69,323 records) - Dominant mode
- **Truckload**: 3.30% (2,410 records)
- **TL Flatbed**: 1.15% (838 records)
- **TL Dry**: 0.54% (395 records)

### 2. On-Time Delivery (OTD) Performance by Carrier Mode

| Carrier Mode | On Time | Late | Delivered Early |
|-------------|---------|------|----------------|
| **LTL** | 65.31% | 19.86% | 14.83% |
| **Truckload** | 39.09% | 7.14% | 53.78% |
| **TL Flatbed** | 34.61% | 5.73% | 59.67% |
| **TL Dry** | 26.33% | 3.04% | 70.63% |

**Key Insights**:
- LTL has the highest on-time rate but also the highest late rate
- Truckload modes (TL Dry, TL Flatbed) have very low late rates but often deliver early
- Early delivery might be as much of a concern as late delivery (inventory/customer readiness)

### 3. Distance Analysis
- **Average Distance**: 1,021 miles
- **Distance Range**: 0 to 3,917 miles
- **Distance Buckets**:
  - 1k-2k miles: 34.8% (most common)
  - 500-1k miles: 28.8%
  - 250-500 miles: 15.0%
  - 2k+ miles: 11.0%
  - 100-250 miles: 6.8%
  - 0-100 miles: 3.6%

### 4. Late Delivery Analysis
- **Total Late Deliveries**: 13,998 (19.18%)
- **Average Delay**: 1.87 days beyond goal
- **Maximum Delay**: 67 days
- **Late Deliveries by Distance**:
  - 500-1k miles: 33.8% of late deliveries
  - 1k-2k miles: 27.3% of late deliveries
  - 250-500 miles: 20.8% of late deliveries

**Insight**: Medium-distance shipments (250-2k miles) are most prone to delays

### 5. Time-Based Patterns

**Year Distribution**:
- 2022: 20,518 records (28.1%)
- 2023: 20,921 records (28.7%)
- 2024: 19,712 records (27.0%)
- 2025: 11,815 records (16.2%)

**Day of Week**:
- Monday-Thursday: Most shipments (Monday: 13,972; Tuesday: 14,464; Wednesday: 14,772; Thursday: 14,860; Friday: 14,885)
- Saturday: 11 shipments
- Sunday: 2 shipments

**Insight**: Business days dominate, with very few weekend shipments

### 6. Lane Analysis
- **Total Unique Lanes**: 970 unique origin-destination pairs
- **Top Lanes** (by frequency):
  - 750xx→857xx: 1,275 shipments
  - 750xx→898xx: 1,241 shipments
  - 544xx→441xx: 1,199 shipments
  - 750xx→331xx: 1,031 shipments

**Insight**: Some lanes have high volume and could benefit from specialized optimization

### 7. Carrier Analysis
- **Total Unique Carriers**: 117 carriers
- **Top Carrier**: `0e32a59c0c8e` handles 62,539 shipments (85.7% of all shipments)
- **Top 10 carriers** handle the vast majority of shipments

**Insight**: High concentration - top carrier performance significantly impacts overall metrics

### 8. Service Days Analysis

**LTL (Carrier Posted Service Days)**:
- Mean: 2.82 days
- Range: 1-11 days
- Median: 3 days

**Truckload (Truckload Service Days)**:
- Mean: 2.23 days
- Range: 0-8 days
- Median: 2 days

---

## Key Insights for Solution Development

### 1. Prediction Challenges
- Need to predict transit days accurately (currently 19.18% late rate)
- Account for carrier mode differences (LTL vs Truckload have different patterns)
- Consider distance buckets and lane-specific patterns
- Factor in temporal patterns (day of week, month, year)

### 2. Anomaly Detection Opportunities
- Negative transit days (data quality issues)
- Extreme delays (67 days max)
- Early deliveries (might indicate inefficiency or customer readiness issues)
- Carrier-specific anomalies

### 3. Root Cause Investigation Areas
- **LTL performance**: High volume but also high late rate (19.86%)
- **Distance-based patterns**: Medium distances (250-2k miles) show highest late rates
- **Lane-specific issues**: Some high-volume lanes may have recurring problems
- **Carrier performance**: Top carrier handles 85.7% of shipments - their performance is critical

### 4. Business Impact Prioritization
- Focus on LTL optimization (95% of volume)
- Target medium-distance shipments (250-2k miles) for delay reduction
- Investigate high-volume lanes for recurring issues
- Monitor top carrier performance closely

### 5. UX Requirements
- Visualize delivery performance by carrier mode, distance, lane, and time
- Enable filtering and drilling down for root cause analysis
- Highlight anomalies and outliers
- Show business impact metrics (cost, customer satisfaction implications)
- Provide predictive insights for ETA accuracy

---

## Recommended Next Steps

1. **Data Cleaning**: Address negative transit days and extreme outliers
2. **Feature Engineering**: 
   - Create time-based features (seasonality, holidays)
   - Lane-specific features
   - Carrier performance history
3. **Model Development**:
   - Transit time prediction models
   - Anomaly detection models
   - Risk scoring for deliveries
4. **Visualization Dashboard**:
   - Performance metrics overview
   - Interactive filtering and exploration
   - Anomaly alerts
   - Predictive insights


