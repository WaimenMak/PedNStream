"""
Network dashboard visualization. Visualize the results of the simulation.
"""


import folium
from branca.colormap import LinearColormap
import json
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
import argparse
import os

class NetworkDashboard:
    def __init__(self, data_path, pos, zoom_start=14):
        """
        Initialize the dashboard
        
        Args:
            link_data_path: Path to link_data.json
            pos: Dictionary of node positions {node_id: (x, y)}
            zoom_start: Initial zoom level
        """
        self.pos = pos
        self.zoom_start = zoom_start
        
        # Calculate center from node positions
        lats = [pos[1] for pos in self.pos.values()]
        lons = [pos[0] for pos in self.pos.values()]
        self.center = [
            (max(lats) + min(lats)) / 2,
            (max(lons) + min(lons)) / 2
        ]
        
        link_data_path = os.path.join(data_path, "link_data.json")
        network_params_path = os.path.join(data_path, "network_params.json")
        # Load link data
        with open(link_data_path, 'r') as f:
            self.link_data = json.load(f)
        
        with open(network_params_path, 'r') as f:
            self.network_params = json.load(f)
            
        # Get time steps range
        self.max_time = len(next(iter(self.link_data.values()))['density'])
        
        # Initialize the base map
        self.base_map = None
        
    def create_base_map(self):
        """Create the base map with fixed elements"""
        # Calculate bounds
        lats = [pos[1] for pos in self.pos.values()]
        lons = [pos[0] for pos in self.pos.values()]
        
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        lat_padding = lat_range * 0.01  # Increased padding for better visibility
        lon_padding = lon_range * 0.01
        
        # Calculate bounds
        min_bounds = [min(lats) - lat_padding, min(lons) - lon_padding]
        max_bounds = [max(lats) + lat_padding, max(lons) + lon_padding]
        
        # Create map with bounds
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom_start,
            max_bounds=True,  # Restrict panning to max bounds
            min_zoom=16,      # Restrict maximum zoom out
            max_zoom=18,      # Restrict maximum zoom in
            bounds=[min_bounds, max_bounds]  # Set initial view bounds
        )
        
        # Add nodes (these don't change)
        for node_id, pos in self.pos.items():
            is_origin = int(node_id) in self.network_params['origin_nodes']
            is_destination = int(node_id) in self.network_params['destination_nodes']

            if is_origin and is_destination:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='flag', prefix='fa', color='red'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            elif is_origin:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='map-marker', prefix='fa'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            elif is_destination:
                folium.Marker(
                    location=[pos[1], pos[0]],
                    icon=folium.Icon(icon='flag', prefix='fa'),
                    popup=f"Node: {node_id}"
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=[pos[1], pos[0]],
                    radius=5,
                    color='blue',
                    fill=True,
                    fillColor='lightblue',
                    fillOpacity=0.7,
                    popup=f"Node: {node_id}"
                ).add_to(m)
            
        return m
    
    def update_links(self, m, time_step, variable):
        """Update only the links on the map"""
        # Set value range based on variable
        if variable == 'density':
            vmin, vmax = 0, 8
            colors = ['green', 'yellow', 'red']  # Low to high
        elif variable == 'speed':
            vmin, vmax = 0, 1.5
            colors = ['red', 'yellow', 'green']  # Low to high (inverted)
        else:  # num_pedestrians
            vmin, vmax = 0, 100
            colors = ['green', 'yellow', 'red']  # Low to high
        
        # Create colormap
        colormap = LinearColormap(
            colors=colors,
            vmin=vmin,
            vmax=vmax,
            caption=variable.capitalize()
        )
        
        # Create feature group for links
        links_group = folium.FeatureGroup(name="links")
        
        # Keep track of processed links to handle bidirectional links
        processed_pairs = set()
        
        # Draw links
        for link_id, link_info in self.link_data.items():
            u, v = link_id.split('-')
            reverse_id = f"{v}-{u}"
            
            # Check if this is a bidirectional link
            is_bidirectional = reverse_id in self.link_data
            
            # Skip if we've already processed this link pair
            link_pair = tuple(sorted([u, v]))
            if link_pair in processed_pairs:
                continue
            
            value = link_info[variable][time_step]
            
            # If bidirectional, get the value for the reverse direction
            if is_bidirectional:
                reverse_value = self.link_data[reverse_id][variable][time_step]
                if variable == 'speed':
                    # For speed, use the maximum of both directions
                    value = max(value, reverse_value)
                else:
                    # For density and num_pedestrians, sum the values
                    value = value + reverse_value
                processed_pairs.add(link_pair)
            
            start = self.pos[u]
            end = self.pos[v]
            coords = [(start[1], start[0]), (end[1], end[0])]
            
            # Normalize width to be between 1 and 3 for better visualization
            # width = max(1, min(3, value / vmax * 3))
            if variable == 'num_pedestrians':
                width = min(10, value * 0.5)
            elif variable == 'speed':
                width = max(1, min(10, value * 10))
            else: # density
                width = min(10, value * 8)
            
            # Draw the link
            folium.PolyLine(
                coords,
                color=colormap(value),
                weight=width,
                opacity=0.8,
                popup=f"{value:.1f}" + (" (bi)" if is_bidirectional else "")
            ).add_to(links_group)
        
        # Add links group and colormap to map
        links_group.add_to(m)
        colormap.add_to(m)
        
        return m
    
    def run_dashboard(self):
        """Run the Streamlit dashboard"""
        st.title("Pedestrian Network Traffic Evolution")
        
        # Control panel
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Variable selection dropdown
            variable = st.selectbox(
                "Select Variable",
                ['density', 'speed', 'num_pedestrians'],
                format_func=lambda x: x.capitalize()
            )
        
        with col1:
            # Time step slider
            time_step = st.slider(
                "Time Step",
                0, self.max_time-1,
                value=0
            )
        
        # Create a new map instead of copying
        m = self.create_base_map()
        m = self.update_links(m, time_step, variable)
        
        # Display map
        st_folium(m, height=600, width=None)

def run_visualization(link_data_path, pos):
    """Helper function to run the dashboard"""
    dashboard = NetworkDashboard(link_data_path, pos)
    dashboard.run_dashboard()

if __name__ == "__main__":
    # Command line: streamlit run network_Dashboard.py -- --name delft --pos data/delft/node_positions.json
    parser = argparse.ArgumentParser(description='Network Dashboard Visualization')
    parser.add_argument('--name', type=str, 
                       default="delft_directions",
                       help='Name of the simulation')
    parser.add_argument('--pos', type=str,
                       default="node_positions.json",
                       help='Path to node_positions.json file')
    
    args = parser.parse_args()
    path_to_pos = os.path.join(".", args.pos)
    # Load node positions
    with open(path_to_pos, 'r') as f:
        pos = {str(k): tuple(v) for k, v in json.load(f).items()}
    
    path_to_data = os.path.join(".", "outputs", args.name)
    # Run visualization with parsed arguments
    run_visualization(path_to_data, pos)