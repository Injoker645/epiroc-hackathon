"""
Map utilities for route visualization using Folium.
Supports state-based route visualization.
"""

import folium
from folium import plugins
import pandas as pd
import numpy as np
from pathlib import Path


# Global caches
_ZIP_COORDS = None
_STATE_COORDS = None


def load_zip_coordinates():
    """Load the zip code coordinates mapping."""
    coords_path = Path(__file__).parent.parent / 'data' / 'zip_coordinates.csv'
    
    if coords_path.exists():
        df = pd.read_csv(coords_path)
        # Create dictionary mapping zip_3d to (lat, lon, state)
        zip_coords = {}
        for _, row in df.iterrows():
            zip_prefix = row['zip_prefix']
            zip_3d = row['zip_3d'] if 'zip_3d' in row else f"{zip_prefix}xx"
            zip_coords[zip_prefix] = (row['latitude'], row['longitude'], row.get('state', 'XX'))
            zip_coords[zip_3d] = (row['latitude'], row['longitude'], row.get('state', 'XX'))
        return zip_coords
    return {}


def load_state_coordinates():
    """Compute state center coordinates from ZIP data."""
    coords_path = Path(__file__).parent.parent / 'data' / 'zip_coordinates.csv'
    
    if coords_path.exists():
        df = pd.read_csv(coords_path)
        # Compute average lat/lon per state
        state_coords = df.groupby('state').agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).to_dict('index')
        return {state: (data['latitude'], data['longitude']) for state, data in state_coords.items()}
    return {}


def get_zip_coordinates():
    """Get zip coordinates (cached)."""
    global _ZIP_COORDS
    if _ZIP_COORDS is None:
        _ZIP_COORDS = load_zip_coordinates()
    return _ZIP_COORDS


def get_state_coordinates():
    """Get state coordinates (cached)."""
    global _STATE_COORDS
    if _STATE_COORDS is None:
        _STATE_COORDS = load_state_coordinates()
    return _STATE_COORDS


def get_coords_for_zip(zip_3d):
    """Get coordinates for a 3-digit zip code."""
    coords = get_zip_coordinates()
    # Try the full zip_3d format first, then just the prefix
    result = coords.get(zip_3d)
    if result is None:
        prefix = str(zip_3d).replace('xx', '')[:3]
        result = coords.get(prefix)
    if result:
        return (result[0], result[1])  # Return just lat, lon
    return None


def get_coords_for_state(state_code):
    """Get center coordinates for a state."""
    coords = get_state_coordinates()
    return coords.get(state_code)


def get_lane_color(on_time_rate):
    """Get color based on on-time rate."""
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


def create_lane_map(lane_stats_df, selected_route_id=None, height=500):
    """
    Create interactive US map with state-based routes.
    
    Parameters:
    -----------
    lane_stats_df : pd.DataFrame
        Lane statistics with columns:
        - lane_state_pair, origin_state, dest_state
        - on_time_rate, volume_normalized, total_shipments
    selected_route_id : str, optional
        Currently selected route (lane_state_pair) to highlight
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
    
    state_coords = get_state_coordinates()
    
    if len(lane_stats_df) == 0:
        return m
    
    # Add routes as polylines
    for _, route in lane_stats_df.iterrows():
        origin_state = route.get('origin_state')
        dest_state = route.get('dest_state')
        
        origin_coords = state_coords.get(origin_state)
        dest_coords = state_coords.get(dest_state)
        
        if origin_coords is None or dest_coords is None:
            continue
        
        # Determine styling
        route_id = route.get('lane_state_pair', f"{origin_state}_{dest_state}")
        is_selected = (selected_route_id is not None and route_id == selected_route_id)
        
        color = get_lane_color(route.get('on_time_rate', 0.5))
        weight = 5 if is_selected else 2 + route.get('volume_normalized', 0.5) * 2
        opacity = 1.0 if is_selected else 0.6
        
        # Create popup content
        popup_html = f"""
        <div style="width: 200px;">
            <b>{origin_state} → {dest_state}</b><br>
            <hr style="margin: 5px 0;">
            Shipments: {route.get('total_shipments', 'N/A'):,}<br>
            Avg Transit: {route.get('avg_transit_hours', 0):.1f} hrs<br>
            On-Time Rate: {route.get('on_time_rate', 0)*100:.1f}%<br>
            Distance: {route.get('avg_distance', 0):.0f} mi
        </div>
        """
        
        # Add polyline
        folium.PolyLine(
            locations=[origin_coords, dest_coords],
            color='#3498db' if is_selected else color,
            weight=weight,
            opacity=opacity,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{origin_state} → {dest_state}"
        ).add_to(m)
        
        # Add markers for selected route
        if is_selected:
            # Origin marker
            folium.CircleMarker(
                location=origin_coords,
                radius=10,
                color='#2980b9',
                fill=True,
                fill_color='#3498db',
                fill_opacity=0.8,
                popup=f"Origin: {origin_state}"
            ).add_to(m)
            
            # Destination marker
            folium.CircleMarker(
                location=dest_coords,
                radius=10,
                color='#c0392b',
                fill=True,
                fill_color='#e74c3c',
                fill_opacity=0.8,
                popup=f"Destination: {dest_state}"
            ).add_to(m)
    
    return m


def create_single_route_map(origin_state, dest_state, route_info=None, height=300):
    """
    Create a focused map showing a single state-to-state route.
    
    Parameters:
    -----------
    origin_state : str
        Origin state code (e.g., 'OH')
    dest_state : str
        Destination state code (e.g., 'PA')
    route_info : dict, optional
        Additional route information for display
    height : int
        Map height in pixels
        
    Returns:
    --------
    folium.Map
        Map focused on the route
    """
    state_coords = get_state_coordinates()
    
    origin_coords = state_coords.get(origin_state)
    dest_coords = state_coords.get(dest_state)
    
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
        zoom = 7
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
    
    # Add route line
    folium.PolyLine(
        locations=[origin_coords, dest_coords],
        color='#3498db',
        weight=4,
        opacity=0.8
    ).add_to(m)
    
    # Add origin marker
    folium.Marker(
        location=origin_coords,
        popup=f"Origin: {origin_state}",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)
    
    # Add destination marker
    folium.Marker(
        location=dest_coords,
        popup=f"Destination: {dest_state}",
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
    state_coords = get_state_coordinates()
    
    heat_data = []
    for _, route in lane_stats_df.iterrows():
        dest_state = route.get('dest_state')
        dest_coords = state_coords.get(dest_state)
        if dest_coords:
            intensity = route.get(metric, 1)
            heat_data.append([dest_coords[0], dest_coords[1], intensity])
    
    return plugins.HeatMap(heat_data, radius=20, blur=15)


def add_legend_to_map(m):
    """Add a legend to the map explaining color coding."""
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 160px;
                border: 2px solid grey; z-index: 1000;
                background-color: white; padding: 10px;
                font-size: 12px; border-radius: 5px;">
        <b>Route Performance</b><br>
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
