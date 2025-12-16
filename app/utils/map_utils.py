"""
Map utilities for lane visualization using Folium.
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
from pathlib import Path


# Load zip coordinates
def load_zip_coordinates():
    """Load the zip code coordinates mapping."""
    coords_path = Path(__file__).parent.parent / 'data' / 'zip_coordinates.csv'
    
    if coords_path.exists():
        df = pd.read_csv(coords_path)
        # Create dictionary mapping zip_3d to (lat, lon)
        return {
            row['zip_3d']: (row['latitude'], row['longitude'])
            for _, row in df.iterrows()
        }
    else:
        return {}


# Global zip coordinates cache
ZIP_COORDS = None


def get_zip_coordinates():
    """Get zip coordinates (cached)."""
    global ZIP_COORDS
    if ZIP_COORDS is None:
        ZIP_COORDS = load_zip_coordinates()
    return ZIP_COORDS


def get_coords_for_zip(zip_3d):
    """
    Get coordinates for a 3-digit zip code.
    
    Parameters:
    -----------
    zip_3d : str
        3-digit zip code (e.g., '441xx')
        
    Returns:
    --------
    tuple or None
        (latitude, longitude) or None if not found
    """
    coords = get_zip_coordinates()
    return coords.get(zip_3d)


def get_lane_color(on_time_rate):
    """
    Get color based on on-time rate.
    
    Parameters:
    -----------
    on_time_rate : float
        On-time delivery rate (0-1)
        
    Returns:
    --------
    str
        Hex color code
    """
    if pd.isna(on_time_rate):
        return '#95a5a6'  # Gray for unknown
    elif on_time_rate >= 0.8:
        return '#27ae60'  # Green - excellent
    elif on_time_rate >= 0.6:
        return '#2ecc71'  # Light green - good
    elif on_time_rate >= 0.4:
        return '#f39c12'  # Orange - moderate risk
    else:
        return '#e74c3c'  # Red - high risk


def create_lane_map(lane_stats_df, selected_lane_id=None, height=500):
    """
    Create interactive US map with lane routes.
    
    Parameters:
    -----------
    lane_stats_df : pd.DataFrame
        Lane statistics with columns:
        - lane_id, lane_zip3_pair
        - origin_zip_3d, dest_zip_3d
        - on_time_rate, volume_normalized
    selected_lane_id : str, optional
        Currently selected lane to highlight
    height : int
        Map height in pixels
        
    Returns:
    --------
    folium.Map
        Interactive map object
    """
    # Center on US
    m = folium.Map(
        location=[39.8, -98.5],
        zoom_start=4,
        tiles='cartodbpositron'
    )
    
    coords = get_zip_coordinates()
    
    # Add lanes as polylines
    for _, lane in lane_stats_df.iterrows():
        origin_coords = coords.get(lane['origin_zip_3d'])
        dest_coords = coords.get(lane['dest_zip_3d'])
        
        if origin_coords is None or dest_coords is None:
            continue
        
        # Determine styling
        is_selected = (selected_lane_id is not None and 
                       lane['lane_id'] == selected_lane_id)
        
        color = get_lane_color(lane.get('on_time_rate', 0.5))
        weight = 4 if is_selected else 2 + lane.get('volume_normalized', 0.5) * 2
        opacity = 1.0 if is_selected else 0.6
        
        # Create popup content
        popup_html = f"""
        <div style="width: 200px;">
            <b>{lane['lane_zip3_pair']}</b><br>
            <hr style="margin: 5px 0;">
            Shipments: {lane.get('total_shipments', 'N/A'):,}<br>
            Avg Transit: {lane.get('avg_transit_hours', 0):.1f} hrs<br>
            On-Time Rate: {lane.get('on_time_rate', 0)*100:.1f}%<br>
            Distance: {lane.get('avg_distance', 0):.0f} mi
        </div>
        """
        
        # Add polyline
        folium.PolyLine(
            locations=[origin_coords, dest_coords],
            color='#3498db' if is_selected else color,
            weight=weight,
            opacity=opacity,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=lane['lane_zip3_pair']
        ).add_to(m)
        
        # Add markers for selected lane
        if is_selected:
            # Origin marker
            folium.CircleMarker(
                location=origin_coords,
                radius=8,
                color='#2980b9',
                fill=True,
                fill_color='#3498db',
                fill_opacity=0.8,
                popup=f"Origin: {lane['origin_zip_3d']}"
            ).add_to(m)
            
            # Destination marker
            folium.CircleMarker(
                location=dest_coords,
                radius=8,
                color='#c0392b',
                fill=True,
                fill_color='#e74c3c',
                fill_opacity=0.8,
                popup=f"Destination: {lane['dest_zip_3d']}"
            ).add_to(m)
    
    return m


def create_single_lane_map(origin_zip, dest_zip, lane_info=None, height=300):
    """
    Create a focused map showing a single lane.
    
    Parameters:
    -----------
    origin_zip : str
        Origin 3-digit zip code
    dest_zip : str
        Destination 3-digit zip code
    lane_info : dict, optional
        Additional lane information for display
    height : int
        Map height in pixels
        
    Returns:
    --------
    folium.Map
        Map focused on the lane
    """
    coords = get_zip_coordinates()
    
    origin_coords = coords.get(origin_zip)
    dest_coords = coords.get(dest_zip)
    
    if origin_coords is None or dest_coords is None:
        # Create default US map if coords not found
        m = folium.Map(location=[39.8, -98.5], zoom_start=4, tiles='cartodbpositron')
        return m
    
    # Calculate center point
    center_lat = (origin_coords[0] + dest_coords[0]) / 2
    center_lon = (origin_coords[1] + dest_coords[1]) / 2
    
    # Calculate appropriate zoom level based on distance
    lat_diff = abs(origin_coords[0] - dest_coords[0])
    lon_diff = abs(origin_coords[1] - dest_coords[1])
    max_diff = max(lat_diff, lon_diff)
    
    if max_diff < 2:
        zoom = 8
    elif max_diff < 5:
        zoom = 6
    elif max_diff < 10:
        zoom = 5
    else:
        zoom = 4
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles='cartodbpositron'
    )
    
    # Add lane line
    folium.PolyLine(
        locations=[origin_coords, dest_coords],
        color='#3498db',
        weight=4,
        opacity=0.8
    ).add_to(m)
    
    # Add origin marker
    folium.Marker(
        location=origin_coords,
        popup=f"Origin: {origin_zip}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add destination marker
    folium.Marker(
        location=dest_coords,
        popup=f"Destination: {dest_zip}",
        icon=folium.Icon(color='red', icon='flag-checkered', prefix='fa')
    ).add_to(m)
    
    return m


def create_heatmap_layer(lane_stats_df, metric='total_shipments'):
    """
    Create a heatmap layer for destinations.
    
    Parameters:
    -----------
    lane_stats_df : pd.DataFrame
        Lane statistics
    metric : str
        Metric to use for heatmap intensity
        
    Returns:
    --------
    folium.plugins.HeatMap
        Heatmap layer
    """
    coords = get_zip_coordinates()
    
    heat_data = []
    for _, lane in lane_stats_df.iterrows():
        dest_coords = coords.get(lane['dest_zip_3d'])
        if dest_coords:
            intensity = lane.get(metric, 1)
            heat_data.append([dest_coords[0], dest_coords[1], intensity])
    
    return plugins.HeatMap(heat_data, radius=15, blur=10)


def add_legend_to_map(m):
    """
    Add a legend to the map explaining color coding.
    
    Parameters:
    -----------
    m : folium.Map
        The map to add legend to
    """
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px;
                border: 2px solid grey; z-index: 1000;
                background-color: white; padding: 10px;
                font-size: 12px; border-radius: 5px;">
        <b>Lane Performance</b><br>
        <i style="background: #27ae60; width: 15px; height: 15px; 
           display: inline-block; margin-right: 5px;"></i> >80% On-Time<br>
        <i style="background: #2ecc71; width: 15px; height: 15px; 
           display: inline-block; margin-right: 5px;"></i> 60-80% On-Time<br>
        <i style="background: #f39c12; width: 15px; height: 15px; 
           display: inline-block; margin-right: 5px;"></i> 40-60% On-Time<br>
        <i style="background: #e74c3c; width: 15px; height: 15px; 
           display: inline-block; margin-right: 5px;"></i> <40% On-Time
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

