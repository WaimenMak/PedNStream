import networkx as nx
import numpy as np

class PathFinder:
    """Handles path finding and path-related operations"""
    def __init__(self, links):
        self.od_paths = {}
        self.graph = self.create_graph(links)
        self.nodes_in_paths = set()

    def create_graph(self, links):
        """Convert network to NetworkX graph"""
        G = nx.DiGraph()
        for (start, end), link in links.items():
            G.add_edge(start, end, weight=link.length)
        return G

    def find_od_paths(self, graph, od_pairs, k_paths=3):
        """Find k shortest paths and track which nodes are used"""
        self.nodes_in_paths = set()  # Reset nodes set
        
        for origin, dest in od_pairs:
            try:
                paths = list(nx.shortest_simple_paths(
                    graph, origin, dest, weight='weight'))[:k_paths]
                self.od_paths[(origin, dest)] = paths
                
                # Add all nodes from paths to the set
                for path in paths:
                    self.nodes_in_paths.update(path)
                    
            except nx.NetworkXNoPath:
                print(f"No path found between {origin} and {dest}")
                self.od_paths[(origin, dest)] = []

    def get_path_attributes(self, path):
        """Calculate path attributes"""
        length = 0
        free_flow_time = 0
        
        for i in range(len(path)-1):
            link = self.graph.links[(path[i], path[i+1])]
            length += link.length
            free_flow_time += link.length / link.free_flow_speed
            
        return {'length': length, 'free_flow_time': free_flow_time} 
    
    def get_od_pairs_for_turn(self, upstream_node: int, current_node: int, downstream_node: int):
        """
        Find all OD pairs whose paths use the specified turn.
        
        Args:
            upstream_node: ID of the upstream node
            current_node: ID of the current node
            downstream_node: ID of the downstream node
            
        Returns:
            list of (origin, destination) tuples that use this turn
        """
        od_pairs_using_turn = []
        
        # Check each OD pair
        for (origin, dest), paths in self.od_paths.items():
            # Check each path for this OD pair
            for path in paths:
                # Look for the turn sequence in the path
                for i in range(len(path)-2):
                    if (path[i] == upstream_node and 
                        path[i+1] == current_node and 
                        path[i+2] == downstream_node):
                        od_pairs_using_turn.append((origin, dest))
                        break  # Found turn in this path, move to next OD pair
                        
        return od_pairs_using_turn

    def calculate_turning_fractions(self, node_id: int, time_step: int, od_manager):
        """
        Calculate turning fractions for a node based on OD flows.
        """
        # Get upstream and downstream links
        upstream_nodes = [link.start_node.node_id for link in node.incoming_links]
        downstream_nodes = [link.end_node.node_id for link in node.outgoing_links]
        
        # Initialize turning fractions matrix
        n_upstream = len(upstream_nodes)
        n_downstream = len(downstream_nodes)
        turning_fractions = np.zeros((n_upstream, n_downstream))
        
        # For each possible turn
        for i, up_node in enumerate(upstream_nodes):
            total_flow = 0
            flows = np.zeros(n_downstream)
            
            # Calculate flow for each downstream option
            for j, down_node in enumerate(downstream_nodes):
                # Get all OD pairs using this turn
                od_pairs = self.get_od_pairs_for_turn(up_node, node_id, down_node)
                
                # Sum up actual flows from all OD pairs using this turn
                for origin, dest in od_pairs:
                    flows[j] += od_manager.get_od_flow(origin, dest, time_step)
                    
                total_flow += flows[j]
            
            # Normalize to get turning fractions
            if total_flow > 0:
                turning_fractions[i] = flows / total_flow
            
        return turning_fractions.flatten()

    def calculate_node_turning_fractions(self, nodes, time_step: int, od_manager):
        """
        Calculate turning fractions only for nodes that appear in OD paths.
        
        Args:
            nodes: List of all network nodes
            time_step: Current time step
            od_manager: ODManager instance
            
        Returns:
            dict: {node_id: turning_fractions_array}
        """
        turning_fractions = {}
        
        # Only process nodes that appear in paths
        for node_id, node in nodes.items():
            if node.node_id in self.nodes_in_paths:
                # Skip nodes with only one incoming or outgoing link
                if len(node.incoming_links) <= 1 or len(node.outgoing_links) <= 1:
                    continue
                    
                fractions = self.calculate_turning_fractions(node, time_step, od_manager)
                turning_fractions[node.node_id] = fractions
                
        return turning_fractions

class RouteChoice:
    """Handles route choice and route-related operations"""
    def __init__(self, path_finder):
        self.path_finder = path_finder
