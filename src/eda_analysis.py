import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("=" * 80)
print("EPIROC LAST-MILE DELIVERY OPTIMIZATION - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv('Dataset/last-mile-data.csv')

# Basic information
print("\n1. DATASET OVERVIEW")
print("-" * 80)
print(f"Total records: {len(df):,}")
print(f"Total columns: {len(df.columns)}")
print(f"Date range: {df['actual_ship'].min()} to {df['actual_ship'].max()}")

# Convert date columns
df['actual_ship'] = pd.to_datetime(df['actual_ship'])
df['actual_delivery'] = pd.to_datetime(df['actual_delivery'])

# Missing values
print("\n2. MISSING VALUES")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("No missing values!")

# Column information
print("\n3. COLUMN INFORMATION")
print("-" * 80)
print("\nColumns and their data types:")
for col in df.columns:
    dtype = df[col].dtype
    unique = df[col].nunique()
    print(f"  {col:35s} | {str(dtype):15s} | Unique values: {unique:,}")

# Carrier mode analysis
print("\n4. CARRIER MODE DISTRIBUTION")
print("-" * 80)
print(df['carrier_mode'].value_counts())
print(f"\nPercentage distribution:")
print(df['carrier_mode'].value_counts(normalize=True) * 100)

# OTD (On-Time Delivery) designation
print("\n5. ON-TIME DELIVERY (OTD) STATUS")
print("-" * 80)
print(df['otd_designation'].value_counts())
print(f"\nPercentage distribution:")
print(df['otd_designation'].value_counts(normalize=True) * 100)

# OTD by carrier mode
print("\n6. OTD STATUS BY CARRIER MODE")
print("-" * 80)
otd_by_mode = pd.crosstab(df['carrier_mode'], df['otd_designation'], normalize='index') * 100
print(otd_by_mode.round(2))

# Transit days analysis
print("\n7. TRANSIT DAYS ANALYSIS")
print("-" * 80)
print(f"Actual Transit Days:")
print(df['actual_transit_days'].describe())
print(f"\nGoal Transit Days:")
print(df['all_modes_goal_transit_days'].describe())
print(f"\nDifference (Actual - Goal):")
df['transit_days_diff'] = df['actual_transit_days'] - df['all_modes_goal_transit_days']
print(df['transit_days_diff'].describe())

# Distance analysis
print("\n8. DISTANCE ANALYSIS")
print("-" * 80)
print(f"Customer Distance (miles):")
print(df['customer_distance'].describe())
print(f"\nDistance Buckets:")
print(df['distance_bucket'].value_counts().sort_index())

# Time-based analysis
print("\n9. TIME-BASED ANALYSIS")
print("-" * 80)
print(f"Ship Year distribution:")
print(df['ship_year'].value_counts().sort_index())
print(f"\nShip Month distribution:")
print(df['ship_month'].value_counts().sort_index())
print(f"\nShip Day of Week (0=Monday, 6=Sunday):")
print(df['ship_dow'].value_counts().sort_index())

# Lane analysis
print("\n10. LANE ANALYSIS")
print("-" * 80)
print(f"Total unique lanes: {df['lane_id'].nunique():,}")
print(f"Total unique lane pairs: {df['lane_zip3_pair'].nunique():,}")
print(f"\nTop 10 most frequent lanes:")
print(df['lane_zip3_pair'].value_counts().head(10))

# Carrier analysis
print("\n11. CARRIER ANALYSIS")
print("-" * 80)
print(f"Total unique carriers: {df['carrier_pseudo'].nunique():,}")
print(f"\nTop 10 carriers by volume:")
print(df['carrier_pseudo'].value_counts().head(10))

# Service days analysis
print("\n12. SERVICE DAYS ANALYSIS")
print("-" * 80)
print("Carrier Posted Service Days (LTL only):")
ltl_data = df[df['carrier_mode'] == 'LTL']
if len(ltl_data) > 0:
    print(ltl_data['carrier_posted_service_days'].describe())
    
print("\nTruckload Service Days (Truckload only):")
tl_data = df[df['carrier_mode'] == 'Truckload']
if len(tl_data) > 0:
    print(tl_data['truckload_service_days'].describe())

# Late delivery analysis
print("\n13. LATE DELIVERY ANALYSIS")
print("-" * 80)
late_deliveries = df[df['otd_designation'] == 'Late']
print(f"Total late deliveries: {len(late_deliveries):,} ({len(late_deliveries)/len(df)*100:.2f}%)")
if len(late_deliveries) > 0:
    print(f"\nAverage delay for late deliveries: {late_deliveries['transit_days_diff'].mean():.2f} days")
    print(f"Max delay: {late_deliveries['transit_days_diff'].max()} days")
    print(f"\nLate deliveries by carrier mode:")
    print(late_deliveries['carrier_mode'].value_counts())
    print(f"\nLate deliveries by distance bucket:")
    print(late_deliveries['distance_bucket'].value_counts())

# Early delivery analysis
print("\n14. EARLY DELIVERY ANALYSIS")
print("-" * 80)
early_deliveries = df[df['otd_designation'] == 'Delivered Early']
print(f"Total early deliveries: {len(early_deliveries):,} ({len(early_deliveries)/len(df)*100:.2f}%)")
if len(early_deliveries) > 0:
    print(f"\nAverage early delivery: {early_deliveries['transit_days_diff'].abs().mean():.2f} days early")

# Summary statistics
print("\n15. KEY METRICS SUMMARY")
print("-" * 80)
on_time_rate = len(df[df['otd_designation'] == 'On Time']) / len(df) * 100
late_rate = len(df[df['otd_designation'] == 'Late']) / len(df) * 100
early_rate = len(df[df['otd_designation'] == 'Delivered Early']) / len(df) * 100

print(f"On-Time Rate: {on_time_rate:.2f}%")
print(f"Late Rate: {late_rate:.2f}%")
print(f"Early Rate: {early_rate:.2f}%")
print(f"\nAverage actual transit days: {df['actual_transit_days'].mean():.2f}")
print(f"Average goal transit days: {df['all_modes_goal_transit_days'].mean():.2f}")
print(f"Average difference: {df['transit_days_diff'].mean():.2f} days")

print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)


