import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from src.LTM.link import Separator

class NetworkVisualizer:
    def __init__(self, network=None, simulation_dir=None, pos=None):
        """
        Initialize visualizer with either a network instance or simulation data
        :param network: Direct network object (optional)
        :param simulation_dir: Directory containing saved simulation data (optional)
        """
        if network is not None:
            self.network = network
            self.node_data = network.nodes
            self.link_data = network.links
            self.from_saved = False
        elif simulation_dir is not None:
            self.load_simulation_data(simulation_dir)
            self.from_saved = True
            self.network = None
        else:
            raise ValueError("Either network object or simulation_dir must be provided")
        
        # Initialize fixed position for nodes
        self.pos = pos
        self.G = nx.DiGraph()
        # Initialize the graph structure
        for node_id in self.node_data:
            self.G.add_node(node_id, size=0)
        if self.network is not None:
            for link_id, link_info in self.link_data.items():
                start_node, end_node = link_id
                self.G.add_edge(start_node, end_node)
        else:
            for link_id, link_info in self.link_data.items():
                start_node, end_node = link_id.split('-')
                self.G.add_edge(start_node, end_node)
            
        # If no position provided, calculate it once and store it
        if self.pos is None:
            self.pos = nx.spring_layout(self.G, k=1, iterations=50, seed=42)

    def load_simulation_data(self, simulation_dir):
        """Load saved simulation data"""
        # Load network parameters
        with open(os.path.join(simulation_dir, 'network_params.json'), 'r') as f:
            self.network_params = json.load(f)
        # Load od paths
        self.od_paths = self.network_params['od_paths']
        # self.od_paths = {}
        # for od_pair, paths in self.network_params['od_paths'].items():
        #     self.od_paths[tuple(map(int, od_pair.split('-')))] = paths
        
        # Load link data
        with open(os.path.join(simulation_dir, 'link_data.json'), 'r') as f:
            self.link_data = json.load(f)
            
        # Load node data
        with open(os.path.join(simulation_dir, 'node_data.json'), 'r') as f:
            self.node_data = json.load(f)
        
        # Load time series if available
        time_series_path = os.path.join(simulation_dir, 'time_series.csv')
        if os.path.exists(time_series_path):
            self.time_series = pd.read_csv(time_series_path)

    def _visualize_network_nx(self, time_step, edge_property='density', with_colorbar=False, set_title=True, figsize=(10, 8)):
        """
        Visualize network state at a specific time step using networkx, for the small network
        :param time_step: Time step to visualize
        :param edge_property: Property to visualize ('density', 'flow', or 'speed')
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        # fix the size of the figure
                # Calculate fixed axis limits once
        x_coords = [coord[0] for coord in self.pos.values()]
        y_coords = [coord[1] for coord in self.pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding (e.g., 10% of the range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        G = nx.DiGraph()
        
        if self.from_saved:
            # Add nodes from saved data
            for node_id, node_info in self.node_data.items():
                if hasattr(self, 'time_series'):
                    total_flow = self.time_series[
                        (self.time_series['time_step'] == time_step) &
                        (self.time_series['link_id'].isin([f"{link}" for link in node_info['incoming_links']]))
                    ]['inflow'].sum()
                else:
                    total_flow = 0
                G.add_node(node_id, size=total_flow)
            
            # Add edges from saved data
            for link_id, link_info in self.link_data.items():
                u, v = link_id.split('-')
                if edge_property == 'density':
                    value = link_info['density'][time_step]
                elif edge_property == 'flow':
                    value = link_info['link_flow'][time_step]
                elif edge_property == 'speed':
                    value = link_info['speed'][time_step]
                elif edge_property == 'num_pedestrians':
                    value = link_info['num_pedestrians'][time_step]
                G.add_edge(u, v, value=value)
        else:
            # Original logic for direct network object
            for node in self.network.nodes:
                total_flow = sum(link.cumulative_inflow[time_step] 
                               for link in node.incoming_links)
                G.add_node(node.node_id, size=total_flow)
            
            for (u, v), link in self.network.links.items():
                if edge_property == 'density':
                    value = link.density[time_step]
                elif edge_property == 'flow':
                    value = link.link_flow[time_step]
                elif edge_property == 'speed':
                    value = link.speed[time_step]
                elif edge_property == 'num_pedestrians':
                    value = link.num_pedestrians[time_step]
                G.add_edge(u, v, value=value)
        
        # Initialize the position if not already set
        if self.pos is None:
            # Set random seed for reproducible layout
            seed = 42  # You can change this seed value
            self.pos = nx.spring_layout(G, k=1, iterations=50, seed=seed)
        
        # Draw nodes
        node_sizes = [G.nodes[node]['size'] * 50 + 1000 for node in G.nodes()]
        # Color nodes: red for origins, pink for destinations, lightblue for others
        if self.from_saved:
            node_colors = ['red' if int(node) in self.network_params['origin_nodes'] 
                          else 'pink' if int(node) in self.network_params['destination_nodes'] 
                          else 'lightblue' for node in G.nodes()]
        else:
            node_colors = ['red' if int(node) in self.network.origin_nodes 
                          else 'pink' if int(node) in self.network.destination_nodes
                          else 'lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(G, self.pos, node_size=node_sizes,
                             node_color=node_colors,
                             ax=ax)
        
        # Add node labels
        nx.draw_networkx_labels(G, self.pos, 
                              font_size=22,
                              font_weight='bold',
                              ax=ax)
        
        # Draw edges
        edges = list(G.edges())
        for u, v in edges:
            width = G[u][v]['value'] * 5
            arrowsize = G[u][v]['value'] * 20 + 0
            
            # Set value range based on property type
            if edge_property == 'density':
                vmin, vmax = 0, 8  # density typically ranges from 0 to 1
            elif edge_property == 'flow':
                vmin, vmax = 0, 3  # adjust these values based on your flow range
            elif edge_property == 'speed':
                vmin, vmax = 0, 3  # adjust these values based on your speed range
            elif edge_property == 'num_pedestrians':
                vmin, vmax = 0, 100  # adjust these values based on your pedestrians range
            
            # Determine connection style for drawing arcs
            rad = 0
            is_separator = False
            if self.from_saved:
                link_key = f"{u}-{v}"
                if self.link_data.get(link_key, {}).get('is_separator'):
                    is_separator = True
            elif self.network:
                # Need to handle potential key errors if u,v are strings
                try:
                    is_separator = isinstance(self.network.links[(int(u), int(v))], Separator)
                except (KeyError, ValueError):
                    pass # Not a separator or key format issue

            if is_separator:
                if self.from_saved:
                    link_info = self.link_data[f"{u}-{v}"]
                    sep_width = np.array(link_info['separator_width'])[time_step]
                    total_width = link_info['parameters']['width']
                else: # live network
                    link = self.network.links[(int(u), int(v))]
                    sep_width = link.separator_width_data[time_step]
                    total_width = link.width
                
                ratio = sep_width / total_width
                rad = ratio * 0.8  # Scale factor for curvature
            elif (v, u) in edges: # Default arc for bidirectional non-separator links
                rad = 0.2
            
            connectionstyle = f"arc3,rad={rad}"

            nx.draw_networkx_edges(G, self.pos, 
                                 edgelist=[(u, v)],
                                 edge_color=[G[u][v]['value']],
                                 edge_cmap=plt.cm.RdYlGn_r,
                                 width=width,
                                 edge_vmin=vmin,
                                 edge_vmax=vmax,
                                 arrowsize=arrowsize,
                                 ax=ax,
                                 connectionstyle=connectionstyle)
        
        # Add title
        if set_title:
            ax.set_title(f'Network State at Time Step {time_step}', 
                        fontdict={'fontsize': 20, 'fontweight': 'bold'})
        
        # Update colorbar with same value range
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                  norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        if with_colorbar:
            cbar = plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
            cbar.ax.tick_params(labelsize=12)  # Enlarge tick labels
            cbar.set_label(edge_property.capitalize(), size=20)  # Enlarge colorbar label
        
        # Turn off axis
        ax.set_axis_off()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Adjust layout to prevent cutting off
        plt.tight_layout()
        
        # Draw gate apertures
        self._draw_gate_apertures(ax, time_step=time_step)
        
        plt.show()
        
        return fig, ax

    def visualize_network_state(self, time_step, edge_property='density', use_folium=False, with_colorbar=True, set_title=True, figsize=(10, 8)):
        """
        Visualize network state at a specific time step using either networkx or folium
        """
        if not use_folium:
            # Original networkx visualization code
            return self._visualize_network_nx(time_step, edge_property, with_colorbar, set_title, figsize)

        # Folium visualization
        import folium
        from branca.colormap import LinearColormap

        # Calculate center from node positions
        lats = [pos[1] for pos in self.pos.values()]
        lons = [pos[0] for pos in self.pos.values()]
        center = [
            (max(lats) + min(lats)) / 2,
            (max(lons) + min(lons)) / 2
        ]
        
        # Calculate bounds for restricting the view
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        lat_padding = lat_range * 0.1
        lon_padding = lon_range * 0.1
        
        min_bounds = [min(lats) - lat_padding, min(lons) - lon_padding]
        max_bounds = [max(lats) + lat_padding, max(lons) + lon_padding]
        
        # Create map centered on the network with bounds
        m = folium.Map(
            location=center,
            zoom_start=14,
            max_bounds=True,
            min_zoom=16,
            max_zoom=18,
            bounds=[min_bounds, max_bounds]
        )

        # Set value range based on property type
        if edge_property == 'density':
            vmin, vmax = 0, 8
        elif edge_property == 'flow':
            vmin, vmax = 0, 3
        elif edge_property == 'speed':
            vmin, vmax = 0, 3
        elif edge_property == 'num_pedestrians':
            vmin, vmax = 0, 100

        # Create color map matching the original RdYlGn_r colormap
        colormap = LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=vmin,
            vmax=vmax,
            caption=edge_property.capitalize()
        )

        # Get edge values
        edge_values = {}
        if self.from_saved:
            # Keep track of processed links to handle bidirectional links
            processed_pairs = set()
            
            for link_id, link_info in self.link_data.items():
                u, v = map(int, link_id.split('-'))
                reverse_id = f"{v}-{u}"
                
                # Skip if we've already processed this link pair
                link_pair = tuple(sorted([u, v]))
                if link_pair in processed_pairs:
                    continue
                
                # Get value for current direction
                if edge_property == 'density':
                    value = link_info['density'][time_step]
                elif edge_property == 'flow':
                    value = link_info['link_flow'][time_step]
                elif edge_property == 'speed':
                    value = link_info['speed'][time_step]
                elif edge_property == 'num_pedestrians':
                    value = link_info['num_pedestrians'][time_step]
                
                # If bidirectional, consider reverse direction
                if reverse_id in self.link_data:
                    reverse_info = self.link_data[reverse_id]
                    if edge_property == 'density':
                        reverse_value = reverse_info['density'][time_step]
                    elif edge_property == 'flow':
                        reverse_value = reverse_info['link_flow'][time_step]
                    elif edge_property == 'speed':
                        reverse_value = reverse_info['speed'][time_step]
                    elif edge_property == 'num_pedestrians':
                        reverse_value = reverse_info['num_pedestrians'][time_step]
                    
                    # Use the maximum value of both directions
                    # value = max(value, reverse_value)
                    value = value + reverse_value
                
                edge_values[(u, v)] = value
                processed_pairs.add(link_pair)
        else:
            for (u, v), link in self.network.links.items():
                value = getattr(link, edge_property)[time_step]
                edge_values[(u, v)] = value

        # Add edges to map
        for (u, v), value in edge_values.items():
            if hasattr(self, 'edges_gdf'):
                # If we have GeoDataFrame with actual street geometries
                try:
                    geom = self.edges_gdf.loc[(u, v), 'geometry']
                    coords = [(y, x) for x, y in geom.coords]
                except KeyError:
                    # If edge not found, use node positions
                    start = self.pos[str(u)]
                    end = self.pos[str(v)]
                    coords = [(start[1], start[0]), (end[1], end[0])]
            else:
                # Use node positions from networkx layout
                start = self.pos[str(u)]
                end = self.pos[str(v)]
                coords = [(start[1], start[0]), (end[1], end[0])]

            # Calculate width based on value (similar to original)
            # width = 2 + value * 3
            width = min(10, value * 0.5)

            folium.PolyLine(
                coords,
                color=colormap(value),
                weight=width,
                opacity=0.8,
                popup=f"Link: {u}->{v}<br>{edge_property}: {value:.2f}"
            ).add_to(m)

        # Add nodes to map
        for node_id in self.G.nodes():
            pos = self.pos[node_id]
            size = self.G.nodes[node_id].get('size', 0) * 50 + 300
            radius = np.sqrt(size) / 10

            is_origin = int(node_id) in self.network_params['origin_nodes']
            is_destination = int(node_id) in self.network_params['destination_nodes']

            if is_origin:
                folium.Marker(
                    location=[pos[1], pos[0]],
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
                    radius=radius,
                    color='blue',
                    fill=True,
                    fillColor='lightblue',
                    fillOpacity=0.7,
                    popup=f"Node: {node_id}"
                ).add_to(m)


        # Add colormap to map
        if with_colorbar:
            colormap.add_to(m)
        
        return m

    def save_visualization(self, time_step, filename, edge_property='density'):
        """Save the visualization to an HTML file"""
        m = self.visualize_network_state(time_step, edge_property, use_folium=True)
        m.save(filename)

    def animate_network(self, start_time=0, end_time=None, interval=50, figsize=(10, 8), 
                       edge_property='density', tag=False, vis_actions=False):
        """
        Create an animation of the network evolution
        :param edge_property: Property to visualize ('density', 'flow', or 'speed')
        :param tag: Boolean to control whether to show value labels on links
        :param vis_actions: Boolean to control whether to visualize separators and gaters
        """
        if end_time is None:
            if self.from_saved:
                end_time = self.network_params['simulation_steps']
            else:
                end_time = self.network.simulation_steps
        
        # Create initial figure
        fig, ax = plt.subplots(figsize=figsize)
        # G = nx.DiGraph()

        # Initialize the position if not already set
        if self.pos is None:
            # Create temporary graph for initial layout
            # G = nx.DiGraph()
            # if self.from_saved:
            #     for node_id in self.node_data:
            #         G.add_node(node_id)
            #     for link_id in self.link_data:
            #         u, v = link_id.split('-')
            #         G.add_edge(u, v)
            # else:
            #     for node in self.network.nodes:
            #         G.add_node(node.node_id)
            #     for (u, v) in self.network.links:
            #         G.add_edge(u, v)
            
            seed = 42  # You can change this seed value
            # self.pos = nx.spring_layout(G, k=1, iterations=50, seed=seed)
            self.pos = nx.spring_layout(self.G, k=1, iterations=50, seed=seed)
            # print(self.pos)
        
        # Calculate fixed axis limits once
        x_coords = [coord[0] for coord in self.pos.values()]
        y_coords = [coord[1] for coord in self.pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add some padding (e.g., 10% of the range)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding
        
        def update(frame):
            fig.clear()
            ax = fig.add_subplot(111)
            
            # Update edge values
            edge_labels = {}  # Dictionary to store edge labels
            if self.from_saved:
                for link_id, link_info in self.link_data.items():
                    u, v = link_id.split('-')
                    if edge_property == 'density':
                        value = link_info['density'][frame]
                    elif edge_property == 'flow':
                        value = link_info['link_flow'][frame]
                    elif edge_property == 'speed':
                        value = link_info['speed'][frame]
                    self.G[u][v]['value'] = value
                    if tag:  # Only create labels if tag is True
                        edge_labels[(u, v)] = f'{value:.2f}'
            else:
                for (u, v), link in self.network.links.items():
                    if edge_property == 'density':
                        value = link.density[frame]
                    elif edge_property == 'flow':
                        value = link.link_flow[frame]
                    elif edge_property == 'speed':
                        value = link.speed[frame]
                    self.G[u][v]['value'] = value
                    if tag:  # Only create labels if tag is True
                        edge_labels[(u, v)] = f'{value:.2f}'
            
            # Draw nodes
            node_sizes = [self.G.nodes[node]['size'] * 100 + 100 for node in self.G.nodes()]
            if self.from_saved:
                node_colors = ['red' if int(node) in self.network_params['origin_nodes'] 
                else 'pink' if int(node) in self.network_params['destination_nodes'] 
                else 'lightblue' for node in self.G.nodes()]
            else:
                node_colors = ['red' if int(node) in self.network.origin_nodes 
                else 'pink' if int(node) in self.network.destination_nodes
                else 'lightblue' for node in self.G.nodes()]
            # print(self.pos)
            # self.pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            nx.draw_networkx_nodes(self.G, pos=self.pos,
                                 node_size=node_sizes,
                                 node_color=node_colors,
                                 ax=ax)
            
            # Draw edges
            edges = list(self.G.edges())
            edge_values = np.array(list(nx.get_edge_attributes(self.G, 'value').values()))
            edges_widths = edge_values * 5
            edges_arrowsizes = edge_values * 20
            # edge_arrowsizes = []
            # edge_colors = []
            if edge_property == 'density':
                vmin, vmax = 0, 8  # density typically ranges from 0 to 1
            elif edge_property == 'flow':
                vmin, vmax = 0, 3  # adjust these values based on your flow range
            elif edge_property == 'speed':
                vmin, vmax = 0, 3  # adjust these values based on your speed range
            
            # Draw edges individually to customize arc curvature for separators
            for u, v in self.G.edges():
                edge_value = self.G[u][v].get('value', 0)
                edge_width = edge_value * 5
                arrow_size = edge_value * 20

                # Determine connection style for drawing arcs
                rad = 0
                is_separator = False
                if self.from_saved:
                    link_key = f"{u}-{v}"
                    if self.link_data.get(link_key, {}).get('is_separator'):
                        is_separator = True
                elif self.network:
                    try:
                        is_separator = isinstance(self.network.links[(int(u), int(v))], Separator)
                    except (KeyError, ValueError):
                        pass

                # Only apply special visualization if vis_actions is True
                if is_separator and vis_actions:
                    if self.from_saved:
                        link_info = self.link_data[f"{u}-{v}"]
                        sep_width = np.array(link_info['separator_width'])[frame]
                        total_width = link_info['parameters']['width']
                    else: # live network
                        link = self.network.links[(int(u), int(v))]
                        sep_width = link.separator_width_data[frame]
                        total_width = link.width
                    
                    rad = sep_width * 0.8 / total_width
                elif (v, u) in self.G.edges() and not vis_actions:
                    # Standard bidirectional curve only when not visualizing actions
                    rad = 0.2
                elif (v, u) in self.G.edges() and vis_actions and not is_separator:
                    # Slight curve for regular bidirectional edges when visualizing actions
                    rad = 0.2

                connectionstyle = f"arc3,rad={rad}"

                nx.draw_networkx_edges(self.G, self.pos,
                                     edgelist=[(u, v)],
                                     edge_color=[edge_value],
                                     edge_cmap=plt.cm.RdYlGn_r,
                                     width=edge_width,
                                     edge_vmin=vmin,
                                     edge_vmax=vmax,
                                     arrowsize=arrow_size,
                                     ax=ax,
                                     connectionstyle=connectionstyle)
                
                # Draw separator indicator line only when visualizing actions
                if is_separator and vis_actions:
                    u_pos = np.array(self.pos[u])
                    v_pos = np.array(self.pos[v])
                    ax.plot([u_pos[0], v_pos[0]], [u_pos[1], v_pos[1]],
                            color='black', linewidth=1.5, alpha=0.6, 
                            linestyle='-', zorder=1)  # Draw behind the curved edge

            # Only draw edge labels if tag is True
            if tag:
                # Separate edge labels for forward and reverse directions
                forward_labels = {}
                reverse_labels = {}
                
                # Sort edges into forward and reverse based on node indices
                for (u, v), label in edge_labels.items():
                    if int(u) < int(v):  # Forward direction
                        forward_labels[(u, v)] = label
                    else:  # Reverse direction
                        reverse_labels[(u, v)] = label
                
                # Filter forward labels to only show when value > 0
                filtered_forward_labels = {}
                if forward_labels:
                    for edge, label in forward_labels.items():
                        # Extract numeric value from label (assuming format like "2.34" or similar)
                        try:
                            value = float(label)
                            if value > 0:
                                filtered_forward_labels[edge] = label
                        except (ValueError, TypeError):
                            # If can't parse as number, show the label anyway
                            filtered_forward_labels[edge] = label
                
                # Draw forward edge labels above the edge
                if filtered_forward_labels:
                    nx.draw_networkx_edge_labels(
                        self.G, self.pos,
                        edge_labels=filtered_forward_labels,
                        bbox=dict(facecolor='none', edgecolor='none', alpha=1),
                        font_size=12,
                        label_pos=0.3,
                        rotate=False,
                        verticalalignment='bottom'  # Place above the edge
                    )
                
                # Filter reverse labels to only show when value > 0
                filtered_reverse_labels = {}
                if reverse_labels:
                    for edge, label in reverse_labels.items():
                        # Extract numeric value from label (assuming format like "2.34" or similar)
                        try:
                            value = float(label)
                            if value > 0:
                                filtered_reverse_labels[edge] = label
                        except (ValueError, TypeError):
                            # If can't parse as number, show the label anyway
                            filtered_reverse_labels[edge] = label
                
                # Draw reverse edge labels below the edge
                if filtered_reverse_labels:
                    nx.draw_networkx_edge_labels(
                        self.G, self.pos,
                        edge_labels=filtered_reverse_labels,
                        bbox=dict(facecolor='none', edgecolor='none', alpha=1),
                        font_size=12,
                        label_pos=0.5,
                        rotate=False,
                        verticalalignment='top'  # Place below the edge
                    )
            
            # Draw node labels
            nx.draw_networkx_labels(self.G, self.pos,font_size=20, ax=ax)
            
            # Draw gate apertures only when visualizing actions
            if vis_actions:
                self._draw_gate_apertures(ax, time_step=frame)
            
            # Update colorbar
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                      norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label(edge_property.capitalize(), size=14)
            
            # Set fixed axis limits
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            
            # Set title and turn off axis
            ax.set_title(f'Time Step: {frame}')
            ax.set_axis_off()
            
            # Clear the graph for next frame
            # self.G.clear()
            
            # Adjust layout
            plt.tight_layout()
            
            return ax
        
        ani = animation.FuncAnimation(fig, update,
                                    frames=range(start_time, end_time),
                                    interval=interval,
                                    repeat=True,
                                    blit=False)

        
        return ani

    def plot_od_paths(self, figsize=(10, 8), show_legend=True):
        """Plot the OD paths"""
        if self.from_saved:
            od_paths = self.od_paths
        elif self.network is not None:
            od_paths = self.network.path_finder.od_paths
        else:
            raise ValueError("No OD paths found")
        
        # Build a drawing graph from the live network
        graph = nx.DiGraph()
        if self.from_saved:
            for node_id in self.node_data:
                graph.add_node(node_id)
            for link_id in self.link_data:
                u, v = link_id.split('-')
                graph.add_edge(u, v)
        else:
            for node_id in self.network.nodes:
                graph.add_node(node_id)
            for (u, v) in self.network.links:
                graph.add_edge(u, v)

        # Resolve positions
        pos = self.pos
        if pos is None and hasattr(self.network, 'pos') and self.network.pos is not None:
            pos = self.network.pos
        if pos is None:
            pos = nx.spring_layout(graph, k=1, iterations=50, seed=42)

        # Normalize pos keys to be ints if possible (handles cases where keys are str)
        # normalized_pos = {}
        # for k, v in pos.items():
        #     try:
        #         ik = int(k)
        #     except (TypeError, ValueError):
        #         ik = k
        #     normalized_pos[ik] = v

        # Compute axis bounds with padding
        # x_coords = [coord[0] for coord in normalized_pos.values()]
        # y_coords = [coord[1] for coord in normalized_pos.values()]
        x_coords = [coord[0] for coord in pos.values()]
        y_coords = [coord[1] for coord in pos.values()]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        # Figure and axes
        fig, ax = plt.subplots(figsize=figsize)

        # Draw base nodes (origins red, destinations pink, others lightblue)
        origin_nodes = set(str(n) for n in getattr(self.network, 'origin_nodes', []))
        destination_nodes = set(str(n) for n in getattr(self.network, 'destination_nodes', []))
        node_colors = [
            'red' if node in origin_nodes else (
                'pink' if node in destination_nodes else 'lightblue'
            )
            for node in graph.nodes()
        ]
        node_sizes = [400 for _ in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos, node_size=node_sizes, node_color=node_colors, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=22, font_weight='bold', ax=ax)

        # Draw base edges in light gray
        all_edges = list(graph.edges())
        bidirectional = [(u, v) for (u, v) in all_edges if (v, u) in graph.edges()]
        # unidirectional = [(u, v) for (u, v) in all_edges if (v, u) not in graph.edges()]
        if bidirectional:
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=bidirectional,
                arrows=True,
                arrowsize=8,
                edge_color='lightgray',
                width=1.5,
                alpha=0.7,
                ax=ax,
                connectionstyle="arc3,rad=0.2"
            )
        # if unidirectional:
        #     nx.draw_networkx_edges(
        #         graph, pos,
        #         edgelist=unidirectional,
        #         arrows=True,
        #         arrowsize=8,
        #         edge_color='lightgray',
        #         width=1.5,
        #         alpha=0.7,
        #         ax=ax
        #     )

        # Prepare colors for OD pairs (one color per OD)
        cmap = plt.get_cmap('tab20')
        od_pairs = list(od_paths.keys())
        color_map = {od: cmap(i % 20) for i, od in enumerate(od_pairs)}

        # Draw OD paths overlay
        legend_handles = []
        from matplotlib.lines import Line2D
        for od_pair, paths in od_paths.items():
            color = color_map[od_pair]
            # Create legend entry for this OD pair
            legend_handles.append(Line2D([0], [0], color=color, lw=4, label=f"{od_pair[0]} → {od_pair[1]}"))

            for path in paths or []:
                # Convert any str node ids to int when possible to match normalized_pos keys
                # processed_path = []
                # for n in path:
                #     try:
                #         processed_path.append(int(n))
                #     except (TypeError, ValueError):
                #         processed_path.append(n)
                processed_path = path

                # Draw each segment with arrows and slight arc for bidirectional pairs
                for u, v in zip(processed_path[:-1], processed_path[1:]):
                    if isinstance(u, int):
                        u = str(u)
        
                    if isinstance(v, int):
                        v = str(v)

                    has_reverse = (v, u) in graph.edges()
                    nx.draw_networkx_edges(
                        graph, pos,
                        edgelist=[(u, v)],
                        arrows=True,
                        arrowsize=12,
                        edge_color=color,
                        width=4.0,
                        alpha=0.8,
                        ax=ax,
                        connectionstyle=("arc3,rad=0.2" if has_reverse else None)
                    )

        # Legend
        if legend_handles and show_legend:
            ax.legend(handles=legend_handles, title="OD pairs", loc='upper right')

        # Final touches
        ax.set_axis_off()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title('OD Paths', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        plt.tight_layout()
        plt.show()

        return fig, ax

    def plot_link_evolution(self, link_ids=None):
        """Plot the evolution of density, flow, and speed for selected links"""
        if self.from_saved:
            if link_ids is None:
                link_ids = list(self.link_data.keys())[:3]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            for link_id in link_ids:
                link_info = self.link_data[link_id]
                time = range(len(link_info['density']))
                
                # Plot density
                ax1.plot(time, link_info['density'], label=f'Link {link_id}')
                ax1.set_ylabel('Density')
                ax1.legend()
                
                # Plot flows
                ax2.plot(time, link_info['inflow'], '--', label=f'Link {link_id} (in)')
                ax2.plot(time, link_info['outflow'], '-', label=f'Link {link_id} (out)')
                ax2.set_ylabel('Flow')
                ax2.legend()
                
                # Plot speed
                ax3.plot(time, link_info['speed'], label=f'Link {link_id}')
                ax3.set_ylabel('Speed')
                ax3.set_xlabel('Time Step')
                ax3.legend()
        else:
            # Original logic for direct network object
            if link_ids is None:
                link_ids = list(self.network.links.keys())[:3]
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
            
            for link_id in link_ids:
                link = self.network.links[link_id]
                time = range(self.network.simulation_steps)
                
                ax1.plot(time, link.density, label=f'Link {link_id}')
                ax1.set_ylabel('Density')
                ax1.legend()
                
                ax2.plot(time, link.inflow, '--', label=f'Link {link_id} (in)')
                ax2.plot(time, link.outflow, '-', label=f'Link {link_id} (out)')
                ax2.set_ylabel('Flow')
                ax2.legend()
                
                ax3.plot(time, link.speed, label=f'Link {link_id}')
                ax3.set_ylabel('Speed')
                ax3.set_xlabel('Time Step')
                ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def _draw_gate_apertures(self, ax, time_step=None):
        """
        Draw gate apertures (lines at node-edge junctions representing gate width)
        Works with both live network objects and saved simulation data
        :param ax: Matplotlib axis to draw on
        :param time_step: Current time step (for reference, not used in aperture calculation)
        """
        # Determine which edges have gate width data
        edges_with_gates = {}
        
        if self.from_saved:
            # For saved data: collect edges that have back_gate_width
            for link_id, link_info in self.link_data.items():
                if 'back_gate_width' in link_info:
                    u, v = link_id.split('-')
                    edges_with_gates[(u, v)] = np.array(link_info['back_gate_width'])[time_step]
        else:
            # For live network: collect edges from gater nodes
            if self.network is None:
                return
            
            for (u, v), link in self.network.links.items():
                # Get gate width from back_gate_width property
                edges_with_gates[(str(u), str(v))] = link.back_gate_width_data[time_step]
        
        # Draw apertures for all edges with gate width data
        for (u, v), gate_width in edges_with_gates.items():
            # Get node positions
            u_pos = np.array(self.pos[u])
            v_pos = np.array(self.pos[v])
            
            # Calculate direction vector (from u to v)
            direction = v_pos - u_pos
            direction_length = np.linalg.norm(direction)
            if direction_length == 0:
                continue
            
            # Normalize direction and get perpendicular vector
            direction_norm = direction / direction_length
            perpendicular = np.array([-direction_norm[1], direction_norm[0]])
            
            # Connection point: slightly away from the node (about 3% of edge distance)
            connection_point = u_pos + 0.08 * direction
            
            # Scale factor for gate width visualization (adjust for readability)
            scale_factor = 0.07  # Adjust this to make apertures more/less visible
            
            # Calculate aperture endpoints: perpendicular line through connection point
            half_width = gate_width * scale_factor / 2
            aperture_start = connection_point - perpendicular * half_width
            aperture_end = connection_point + perpendicular * half_width
            
            # Draw aperture line as a dashed line
            ax.plot(
                [aperture_start[0], aperture_end[0]],
                [aperture_start[1], aperture_end[1]],
                color='darkblue',
                linewidth=2.5,
                alpha=0.7,
                zorder=2,  # Draw on top of edges
                linestyle='--'  # Dashed line
            )


def progress_callback(current_frame, total_frames):
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = tqdm(total=total_frames, desc='Saving animation')
    progress_callback.pbar.update(1)
    if current_frame == total_frames - 1:
        progress_callback.pbar.close()

if __name__ == "__main__":
    import matplotlib
    # matplotlib.use('TkAgg')
    # Example usage
    # simulation_dir = "/Users/mmai/Devs/Crowd-Control/outputs/delft" # Replace with actual timestamp
    # visualizer = NetworkVisualizer(simulation_dir=simulation_dir)
    # ani = visualizer.animate_network(start_time=0, interval=100, edge_property='speed')
    # m = visualizer.visualize_network_state(time_step=100, edge_property='num_pedestrians')
    # m.save('../../network_state_t100.html')
    # m = visualizer.visualize_network_state(time_step=499, edge_property='density')
    # plt.show()

    output_dir = os.path.join("..", "..", "outputs")
    visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "forky_queues"))
    # visualizer = NetworkVisualizer(simulation_dir=os.path.join(output_dir, "delft_directions"), pos=pos)
    time_step = 200
    fig, ax = visualizer.visualize_network_state(time_step=time_step, edge_property='density', with_colorbar=False)