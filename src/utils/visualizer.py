import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json
import pandas as pd
import os

class NetworkVisualizer:
    def __init__(self, network=None, simulation_dir=None):
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
        else:
            raise ValueError("Either network object or simulation_dir must be provided")
        
        # Initialize fixed position for nodes
        self.pos = None

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

    def visualize_network_state(self, time_step, edge_property='density'):
        """
        Visualize network state at a specific time step
        Works with both direct network object and saved data
        :param time_step: Time step to visualize
        :param edge_property: Property to visualize ('density', 'flow', or 'speed')
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
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
                G.add_edge(u, v, value=value)
        
        # Draw the network
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_sizes = [G.nodes[node]['size'] * 100 + 500 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                             node_color='lightblue',
                             ax=ax)  # Specify the axis
        
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
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax)  # Specify the axis
        
        # Add title
        ax.set_title(f'Network State at Time Step {time_step}')
        
        # Update colorbar with same value range
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                  norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
        
        # Turn off axis
        ax.set_axis_off()
        
        # Adjust layout to prevent cutting off
        plt.tight_layout()
        
        return fig, ax

    def animate_network(self, start_time=0, end_time=None, interval=50, edge_property='density'):
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
        fig, ax = plt.subplots(figsize=(10, 8))
        G = nx.DiGraph()
        
        # Initialize the position if not already set
        if self.pos is None:
            # Create temporary graph for initial layout
            temp_G = nx.DiGraph()
            if self.from_saved:
                for node_id in self.node_data:
                    temp_G.add_node(node_id)
                for link_id in self.link_data:
                    u, v = link_id.split('-')
                    temp_G.add_edge(u, v)
            else:
                for node in self.network.nodes:
                    temp_G.add_node(node.node_id)
                for (u, v) in self.network.links:
                    temp_G.add_edge(u, v)
            self.pos = nx.spring_layout(temp_G, k=1, iterations=50)
        
        def update(frame):
            fig.clear()  # Clear the entire figure
            ax = fig.add_subplot(111)  # Create new axis
            
            # Create network for current frame
            if self.from_saved:
                # Add nodes from saved data
                for node_id, node_info in self.node_data.items():
                    if hasattr(self, 'time_series'):
                        total_flow = self.time_series[
                            (self.time_series['time_step'] == frame) &
                            (self.time_series['link_id'].isin([f"{link}" for link in node_info['incoming_links']]))
                        ]['inflow'].sum()
                    else:
                        total_flow = 0
                    G.add_node(node_id, size=total_flow)
                
                # Add edges from saved data
                for link_id, link_info in self.link_data.items():
                    u, v = link_id.split('-')
                    if edge_property == 'density':
                        value = link_info['density'][frame]
                    elif edge_property == 'flow':
                        value = link_info['link_flow'][frame]
                    elif edge_property == 'speed':
                        value = link_info['speed'][frame]
                    G.add_edge(u, v, value=value)
            else:
                # Original logic for direct network object
                for node in self.network.nodes:
                    total_flow = sum(link.cumulative_inflow[frame] 
                                   for link in node.incoming_links)
                    G.add_node(node.node_id, size=total_flow)
                
                for (u, v), link in self.network.links.items():
                    if edge_property == 'density':
                        value = link.density[frame]
                    elif edge_property == 'flow':
                        value = link.link_flow[frame]
                    elif edge_property == 'speed':
                        value = link.speed[frame]
                    G.add_edge(u, v, value=value)
            
            # Draw nodes
            node_sizes = [G.nodes[node]['size'] * 100 + 100 for node in G.nodes()]
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
            
            # Draw labels
            nx.draw_networkx_labels(G, self.pos, ax=ax)
            
            # Update colorbar with same value range
            sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                      norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=edge_property.capitalize())
            
            # Set title and turn off axis
            ax.set_title(f'Time Step: {frame}')
            ax.set_axis_off()
            
            # Clear the graph for next frame
            G.clear()
            
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

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    # Example usage
    simulation_dir = "/Users/mmai/Devs/Crowd-Control/outputs/test" # Replace with actual timestamp
    visualizer = NetworkVisualizer(simulation_dir=simulation_dir)
    ani = visualizer.animate_network(start_time=0, end_time=200, interval=100, edge_property='flow')
    plt.show()