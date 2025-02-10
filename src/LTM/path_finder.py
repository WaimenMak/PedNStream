import networkx as nx

class PathFinder:
    """Handles path finding and path-related operations"""
    def __init__(self, network):
        self.network = network
        self.od_paths = {}
        self.graph = self._create_graph()

    def _create_graph(self):
        """Convert network to NetworkX graph"""
        G = nx.DiGraph()
        for (start, end), link in self.network.links.items():
            G.add_edge(start, end, weight=link.length)
        return G

    def find_od_paths(self, od_pairs, k_paths=3):
        """Find k shortest paths for given OD pairs"""
        for origin, dest in od_pairs:
            try:
                paths = list(nx.shortest_simple_paths(
                    self.graph, origin, dest, weight='weight'))[:k_paths]
                self.od_paths[(origin, dest)] = paths
            except nx.NetworkXNoPath:
                print(f"No path found between {origin} and {dest}")
                self.od_paths[(origin, dest)] = []

    def get_path_attributes(self, path):
        """Calculate path attributes"""
        length = 0
        free_flow_time = 0
        
        for i in range(len(path)-1):
            link = self.network.links[(path[i], path[i+1])]
            length += link.length
            free_flow_time += link.length / link.free_flow_speed
            
        return {'length': length, 'free_flow_time': free_flow_time} 