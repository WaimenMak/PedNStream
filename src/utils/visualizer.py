import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class NetworkVisualizer:
    def __init__(self, network):
        """
        Initialize visualizer with a network instance
        """
        self.network = network
        
    def visualize_network_state(self, time_step):
        """
        Visualize network state at a specific time step with link colors based on density
        and node sizes based on cumulative flows
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for node in self.network.nodes:
            total_flow = sum(link.cumulative_inflow[time_step] for link in node.incoming_links)
            G.add_node(node.node_id, size=total_flow)
        
        for (u, v), link in self.network.links.items():
            normalized_density = link.density[time_step] / link.k_jam
            G.add_edge(u, v, density=normalized_density, 
                      flow=link.inflow[time_step])
        
        # Draw the network
        pos = nx.spring_layout(G)
        
        # Draw nodes
        node_sizes = [G.nodes[node]['size'] * 100 + 500 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue')
        
        # Draw edges
        edges = G.edges()
        densities = [G[u][v]['density'] for (u, v) in edges]
        nx.draw_networkx_edges(G, pos, edge_color=densities, 
                              edge_cmap=plt.cm.RdYlGn_r,
                              width=2, 
                              edge_vmin=0, 
                              edge_vmax=1,
                              arrowsize=20)
        
        nx.draw_networkx_labels(G, pos)
        
        plt.title(f'Network State at Time Step {time_step}')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r), 
                    label='Normalized Density')
        plt.axis('off')
        plt.show()

    def animate_network(self, start_time=0, end_time=None, interval=50):
        """
        Create an animation of the network evolution
        """
        if end_time is None:
            end_time = self.network.simulation_steps
        
        fig = plt.figure(figsize=(10, 8))
        
        def update(frame):
            plt.clf()
            self.visualize_network_state(frame)
            plt.title(f'Time Step: {frame}')
        
        ani = animation.FuncAnimation(fig, update, 
                                    frames=range(start_time, end_time),
                                    interval=interval)
        plt.show()
        return ani

    def plot_link_evolution(self, link_ids=None):
        """
        Plot the evolution of density, flow, and speed for selected links
        """
        if link_ids is None:
            link_ids = list(self.network.links.keys())[:3]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        for link_id in link_ids:
            link = self.network.links[link_id]
            time = range(self.network.simulation_steps)
            
            # Plot density
            ax1.plot(time, link.density, label=f'Link {link_id}')
            ax1.set_ylabel('Density')
            ax1.legend()
            
            # Plot flows
            ax2.plot(time, link.inflow, '--', label=f'Link {link_id} (in)')
            ax2.plot(time, link.outflow, '-', label=f'Link {link_id} (out)')
            ax2.set_ylabel('Flow')
            ax2.legend()
            
            # Plot speed
            ax3.plot(time, link.speed, label=f'Link {link_id}')
            ax3.set_ylabel('Speed')
            ax3.set_xlabel('Time Step')
            ax3.legend()
        
        plt.tight_layout()
        plt.show() 