"""
Generate 3-digit zip code coordinates and state codes for the US.
Uses the pgeocode library for accurate data.

Install: pip install pgeocode
"""

import pandas as pd
from collections import defaultdict


def generate_zip_coordinates_csv(output_path='zip_coordinates.csv'):
    """
    Generate the zip coordinates CSV file using pgeocode library.
    
    Creates a CSV with columns:
    - zip_3d: 3-digit prefix with 'xx' (e.g., '100xx')
    - zip_prefix: 3-digit prefix (e.g., '100')
    - state: State abbreviation (e.g., 'NY')
    - latitude: Average latitude for the prefix
    - longitude: Average longitude for the prefix
    """
    try:
        import pgeocode
    except ImportError:
        print("Error: pgeocode not installed. Run: pip install pgeocode")
        return None
    
    print("Loading US ZIP code database...")
    nomi = pgeocode.Nominatim('us')
    
    # Get the full dataset
    df = nomi._data
    print(f"Loaded {len(df)} ZIP codes")
    
    # Create ZIP3 prefix column
    df = df.copy()
    df['zip3'] = df['postal_code'].astype(str).str.zfill(5).str[:3]
    
    # Group by ZIP3 and aggregate
    print("Aggregating by 3-digit prefix...")
    
    zip3_data = df.groupby('zip3').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'state_code': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common state
    }).reset_index()
    
    # Build output
    rows = []
    for _, row in zip3_data.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['state_code']):
            rows.append({
                'zip_3d': f"{row['zip3']}xx",
                'zip_prefix': row['zip3'],
                'state': row['state_code'],
                'latitude': round(row['latitude'], 4),
                'longitude': round(row['longitude'], 4)
            })
    
    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values('zip_prefix').reset_index(drop=True)
    result_df.to_csv(output_path, index=False)
    
    print(f"\nGenerated {len(result_df)} zip code entries to {output_path}")
    print(f"States covered: {result_df['state'].nunique()}")
    print(f"\nState distribution (top 10):")
    print(result_df['state'].value_counts().head(10))
    
    return result_df


def get_state_for_zip3(zip3_prefix):
    """
    Get state code for a 3-digit ZIP prefix using pgeocode.
    
    Parameters:
    -----------
    zip3_prefix : str
        3-digit ZIP prefix (e.g., '100', '900')
        
    Returns:
    --------
    str or None
        State abbreviation (e.g., 'NY', 'CA') or None if not found
    """
    try:
        import pgeocode
        nomi = pgeocode.Nominatim('us')
        df = nomi._data
        
        zip3 = str(zip3_prefix).zfill(3)
        filtered = df[df['postal_code'].astype(str).str.zfill(5).str.startswith(zip3)]
        
        if len(filtered) > 0 and 'state_code' in filtered.columns:
            # Return most common state
            return filtered['state_code'].mode().iloc[0]
        
        return None
        
    except ImportError:
        print("Warning: pgeocode not installed")
        return None


def get_coords_for_zip3(zip3_prefix):
    """
    Get average coordinates for a 3-digit ZIP prefix.
    
    Parameters:
    -----------
    zip3_prefix : str
        3-digit ZIP prefix (e.g., '100', '900')
        
    Returns:
    --------
    tuple or None
        (latitude, longitude) or None if not found
    """
    try:
        import pgeocode
        nomi = pgeocode.Nominatim('us')
        df = nomi._data
        
        zip3 = str(zip3_prefix).zfill(3)
        filtered = df[df['postal_code'].astype(str).str.zfill(5).str.startswith(zip3)]
        
        if len(filtered) > 0:
            avg_lat = filtered['latitude'].mean()
            avg_lon = filtered['longitude'].mean()
            if pd.notna(avg_lat) and pd.notna(avg_lon):
                return (round(avg_lat, 4), round(avg_lon, 4))
        
        return None
        
    except ImportError:
        print("Warning: pgeocode not installed")
        return None


if __name__ == '__main__':
    generate_zip_coordinates_csv()
