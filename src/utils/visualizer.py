import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

class NetworkVisualizer:
    def __init__(self, network=None, simulation_dir=None, pos=None):
        """
        Initialize visualizer with either a network instance or simulation data
        :param network: Direct network object (optional)
        :param simulation_dir: Directory containing saved simulation data (optional)
        """
        if network is not None:
            self.network = network
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

    def _visualize_network_nx(self, time_step, edge_property='density'):
        """
        Visualize network state at a specific time step using networkx
        :param time_step: Time step to visualize
        :param edge_property: Property to visualize ('density', 'flow', or 'speed')
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
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
        node_sizes = [G.nodes[node]['size'] * 50 + 100 for node in G.nodes()]
        nx.draw_networkx_nodes(G, self.pos, node_size=node_sizes,
                             node_color='lightblue',
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
            
            if (v, u) in edges:
                nx.draw_networkx_edges(G, self.pos, 
                                    edgelist=[(u, v)],
                                    edge_color=[G[u][v]['value']],
                                    edge_cmap=plt.cm.RdYlGn_r,
                                    width=width,
                                    edge_vmin=vmin,
                                    edge_vmax=vmax,
                                    arrowsize=arrowsize,
                                    ax=ax,
                                    connectionstyle=f"arc3,rad=0.2")
            else:
                nx.draw_networkx_edges(G, self.pos, 
                                     edgelist=[(u, v)],
                                     edge_color=[G[u][v]['value']],
                                     edge_cmap=plt.cm.RdYlGn_r,
                                     width=width,
                                     edge_vmin=vmin,
                                     edge_vmax=vmax,
                                     arrowsize=arrowsize,
                                     ax=ax)
        
        # Add title
        ax.set_title(f'Network State at Time Step {time_step}', 
                    fontdict={'fontsize': 20, 'fontweight': 'bold'})
        
        # Update colorbar with same value range
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                  norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
        cbar.ax.tick_params(labelsize=12)  # Enlarge tick labels
        cbar.set_label(edge_property.capitalize(), size=14)  # Enlarge colorbar label
        
        # Turn off axis
        ax.set_axis_off()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Adjust layout to prevent cutting off
        plt.tight_layout()
        plt.show()
        
        return fig, ax

    def visualize_network_state(self, time_step, edge_property='density', use_folium=False):
        """
        Visualize network state at a specific time step using either networkx or folium
        """
        if not use_folium:
            # Original networkx visualization code
            return self._visualize_network_nx(time_step, edge_property)

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
            # Calculate node size (similar to original)
            size = self.G.nodes[node_id].get('size', 0) * 50 + 300 # there is not size in the node for now
            radius = np.sqrt(size) / 10  # Convert size to reasonable radius

            if node_id in ['0', '8']:
                folium.CircleMarker(
                    location=[pos[1], pos[0]],
                    radius=np.sqrt(size) / 5,
                    color='red',
                    fill=True,
                    fillColor='lightred',
                    fillOpacity=0.7,
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
        colormap.add_to(m)
        
        return m

    def save_visualization(self, time_step, filename, edge_property='density'):
        """Save the visualization to an HTML file"""
        m = self.visualize_network_state(time_step, edge_property, use_folium=True)
        m.save(filename)

    def animate_network(self, start_time=0, end_time=None, interval=50, figsize=(10, 8), edge_property='density'):
        """
        Create an animation of the network evolution
        :param edge_property: Property to visualize ('density', 'flow', or 'speed')
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
            
            # Update edge values without changing positions
            if self.from_saved:
                # print("aaa")    
                # Add edges from saved data
                for link_id, link_info in self.link_data.items():
                    u, v = link_id.split('-')
                    if edge_property == 'density':
                        value = link_info['density'][frame]
                    elif edge_property == 'flow':
                        value = link_info['link_flow'][frame]
                    elif edge_property == 'speed':
                        value = link_info['speed'][frame]
                    # G.add_edge(u, v, value=value)
                    #change value of edge
                    self.G[u][v]['value'] = value
            else:
                # Original logic for direct network object
                # for node in self.network.nodes:
                #     total_flow = sum(link.cumulative_inflow[frame] 
                #                    for link in node.incoming_links)
                #     G.add_node(node.node_id, size=total_flow)
                
                for (u, v), link in self.network.links.items():
                    if edge_property == 'density':
                        value = link.density[frame]
                    elif edge_property == 'flow':
                        value = link.link_flow[frame]
                    elif edge_property == 'speed':
                        value = link.speed[frame]
                    # G.add_edge(u, v, value=value)
                    #change value of edge
                    self.G[u][v]['value'] = value
            
            # Use stored positions for drawing
            node_sizes = [self.G.nodes[node]['size'] * 100 + 100 for node in self.G.nodes()]
            # print(self.pos)
            # self.pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
            nx.draw_networkx_nodes(self.G, pos=self.pos,  # Use stored positions
                                 node_size=node_sizes,
                                 node_color='lightblue',
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
            
            nx.draw_networkx_edges(self.G, self.pos, 
                                 edgelist=edges,
                                 edge_color=edge_values.tolist(),
                                 edge_cmap=plt.cm.RdYlGn_r,
                                 width=edges_widths.tolist(),
                                 edge_vmin=vmin,
                                 edge_vmax=vmax,
                                 arrowsize=edges_arrowsizes.tolist(),
                                 ax=ax,
                                 connectionstyle=f"arc3,rad=0.2")
            
            # Draw labels
            nx.draw_networkx_labels(self.G, self.pos, ax=ax)
            
            # Update colorbar with same value range
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                      norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
            cbar.ax.tick_params(labelsize=12)  # Enlarge tick labels
            cbar.set_label(edge_property.capitalize(), size=14)  # Enlarge colorbar label
            
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


def progress_callback(current_frame, total_frames):
    if not hasattr(progress_callback, 'pbar'):
        progress_callback.pbar = tqdm(total=total_frames, desc='Saving animation')
    progress_callback.pbar.update(1)
    if current_frame == total_frames - 1:
        progress_callback.pbar.close()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    # Example usage
    simulation_dir = "/Users/mmai/Devs/Crowd-Control/outputs/delft" # Replace with actual timestamp
    visualizer = NetworkVisualizer(simulation_dir=simulation_dir)
    # ani = visualizer.animate_network(start_time=0, interval=100, edge_property='speed')
    m = visualizer.visualize_network_state(time_step=100, edge_property='num_pedestrians')
    m.save('../../network_state_t100.html')
    # m = visualizer.visualize_network_state(time_step=499, edge_property='density')
    # plt.show()