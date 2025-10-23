import networkx as nx
import numpy as np
from collections import defaultdict
from heapq import heappush, heappop
from src.LTM.node import Node

def k_shortest_paths(graph, origin, dest, k):
    """
    Find k shortest paths using Yen's algorithm with priority queue.
    Slower than the enumerate_all_simple_paths function.
    """
    # Initialize
    A = []  # List of shortest paths found
    B = []  # Priority queue of candidate paths
    candidate_paths = {}  # Store candidate paths by ID
    next_path_id = 0  # Simple counter for path IDs
    found_paths = set()  # Keep track of paths we've already found
    
    # Find the shortest path using Dijkstra
    try:
        shortest_path = nx.shortest_path(graph, origin, dest, weight='weight')
        shortest_dist = nx.shortest_path_length(graph, origin, dest, weight='weight')
        path_tuple = tuple(shortest_path)  # Convert to tuple for hashing
        found_paths.add(path_tuple)
        candidate_paths[next_path_id] = (shortest_dist, shortest_path)
        A.append((shortest_dist, shortest_path))
        next_path_id += 1
    except nx.NetworkXNoPath:
        return []

    # ===== KEY PART: FINDING K DIFFERENT PATHS =====
    # for k_path in range(1, k):
    while len(A) < k:
        if not A:
            break
            
        prev_path = A[-1][1]
        
        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            spur_node_next = prev_path[i+1] # set the distance between deviation node and the next node to inf

            root_path = prev_path[:i+1]
            edges_removed = []
            nodes_removed = []
            
            # Remove nodes in root_path to avoid loops
            for node in root_path[:-1]:  # Exclude spur_node
                if node != spur_node and graph.has_node(node):
                    # Save all edges connected to this node before removing it
                    for neighbor in list(graph.neighbors(node)):
                        if graph.has_edge(node, neighbor):
                            edge_data = graph[node][neighbor].copy()  # Copy edge attributes
                            edges_removed.append((node, neighbor, edge_data))
                    
                    # Also save incoming edges (for directed graphs)
                    for neighbor in list(graph.predecessors(node)):
                        if graph.has_edge(neighbor, node):
                            edge_data_inv = graph[neighbor][node].copy()
                            edges_removed.append((neighbor, node, edge_data_inv))
                    
                    nodes_removed.append(node)
                    graph.remove_node(node)
            
            # Handle the direct edge between spur_node and next node
            # u, v = prev_path[i], prev_path[i+1]
            if graph.has_edge(spur_node, spur_node_next):
                original_weight = graph[spur_node][spur_node_next].get('weight', 1)
                graph[spur_node][spur_node_next]['weight'] = np.inf  # Set to infi to avoid this edge
                
            try:
                spur_path = nx.shortest_path(graph, spur_node, dest, weight='weight')
                total_path = root_path[:-1] + spur_path

                # Restore removed nodes and their edges
                for node in nodes_removed:
                    graph.add_node(node)

                # Restore all removed edges
                for u, v, edge_data in edges_removed:
                    graph.add_edge(u, v, **edge_data)

                if tuple(total_path) in found_paths:
                    continue
                total_dist = sum(graph[total_path[i]][total_path[i+1]].get('weight', 1)
                            for i in range(len(total_path)-1))
                
                # Store the candidate path with next available ID
                candidate_paths[next_path_id] = (total_dist, total_path)
                heappush(B, (total_dist, next_path_id))
                next_path_id += 1
            except nx.NetworkXNoPath:
                pass
            finally:
                # Restore the edge weight
                if graph.has_edge(spur_node, spur_node_next):
                    graph[spur_node][spur_node_next]['weight'] = original_weight

        if B:
            # Get the shortest candidate from priority queue
            _, candidate_id = heappop(B)
            candidate = candidate_paths[candidate_id]
            path_tuple = tuple(candidate[1])
            if path_tuple not in found_paths:
                found_paths.add(path_tuple)
                A.append(candidate)
            # Clean up the stored candidate
            # del candidate_paths[candidate_id]
        # else:
        #     break
    
    return [path for _, path in A]

def enumerate_shortest_simple_paths(graph, origin, dest, max_paths=None):
    """
    Enumerate all simple paths from origin to dest.

    Args:
        graph (nx.DiGraph): Directed graph.
        origin (int): Source node id.
        dest (int): Target node id.
        cutoff (int, optional): Maximum path length (number of nodes) to consider.
        max_paths (int, optional): Maximum number of paths to return (early stop).

    Returns:
        list[list[int]]: List of paths (each path is a list of node ids).

    Notes:
        - Enumerating all simple paths can be exponential; use cutoff/max_paths to bound.
    """
    try:
        paths_iter = nx.shortest_simple_paths(graph, origin, dest, weight='weight')
    except Exception:
        print(f"No path found between {origin} and {dest}")
        return []

    paths = []
    for path in paths_iter:
        paths.append(path)
        if max_paths is not None and len(paths) >= max_paths:
            print(f"Early stopping: Found {len(paths)} paths")
            break
    return paths

class PathFinder:
    """Handles path finding and path-related operations"""
    def __init__(self, links, params=None, controller_nodes=None, controller_links=None):
        self.od_paths = {}  # {(origin, dest): [path1, path2, ...]}
        self.graph = self._create_graph(links)
        self.links = links
        self.nodes_in_paths = set()
        self.node_turn_probs = {}  # {node_id: {(o,d): {(up_node, down_node): probability}}}
        self.node_to_od_pairs = {}  # {node_id: set((o1,d1), (o2,d2), ...)}, to get the relevant od pairs for the node
        self._initialized = False  # Add initialization flag

        # Get parameters from config or use defaults
        path_params = params.get('path_finder', {}) if params else {}
        self.theta = path_params.get('theta', 0.1)  # like the temperature in the logit model
        self.alpha = path_params.get('alpha', 1.0)  # distance weight
        self.beta = path_params.get('beta', 0.05)   # congestion weight
        self.omega = path_params.get('omega', 0.05)   # capacity weight
        self.k_paths = path_params.get('k_paths', 3)
        
        # Controller configuration
        self.controller_nodes = controller_nodes
        self.controller_links = controller_links
        self.controllers_enabled = True if controller_nodes or controller_links else False
        
        # Detour exploration settings (hardcoded for now)
        self.detour_exploration_mode = 'penalize'  # 'penalize' or 'remove' - penalize makes prefix edges expensive
        self.detour_penalty_factor = 2  # Weight multiplier for penalized edges
        self.max_detour_paths = 3  # Maximum alternative paths to try per neighbor

    def _create_graph(self, links, time_step=0):
        """Convert network to NetworkX graph"""
        G = nx.DiGraph()
        for (start, end), link in links.items():
            G.add_edge(start, end, weight=link.length, num_pedestrians=link.num_pedestrians[time_step])
        return G
    
    def is_controller_node(self, node_id):
        """
        Args:
            node_id: Node ID to check
        Returns:
            bool: True if node is a controller
        """
        if not self.controllers_enabled:
            return False
        
        if node_id not in self.controller_nodes:
            return False
        
        return True

    def find_od_paths(self, od_pairs, nodes):
        """Find k shortest paths and track which nodes and their OD pairs"""
        # self.nodes_in_paths = set()
        # self.node_to_od_pairs = {}
        
        for origin, dest in od_pairs:
            try:
                # paths = k_shortest_paths(self.graph, origin, dest, k=self.k_paths)
                paths = enumerate_shortest_simple_paths(self.graph, origin, dest, max_paths=self.k_paths)
                self.od_paths[(origin, dest)] = paths
                
                # Record which nodes are used in this OD pair
                for path in paths:
                    for node in path:
                        self.nodes_in_paths.add(node)
                        if node not in self.node_to_od_pairs:
                            self.node_to_od_pairs[node] = set()
                        self.node_to_od_pairs[node].add((origin, dest))
                    
            except nx.NetworkXNoPath:
                print(f"No path found between {origin} and {dest}")
                self.od_paths[(origin, dest)] = []

        # expand the paths at controller nodes
        if not self._initialized and self.controllers_enabled:
            for node in self.controller_nodes:
                for od_pair in self.node_to_od_pairs[node]:
                    self.expand_controller_paths(nodes[node], od_pair)
                    print(f"Controller node {node}: Added {len(self.od_paths[od_pair])} detour path(s) for OD {od_pair}")
        self.check_if_paths_are_different(self.od_paths)
        # Calculate and store turn probabilities for all nodes in paths
        self.calculate_all_turn_probs(nodes=nodes)
    
    @staticmethod
    def check_if_paths_are_different(od_paths):
        """Check if the paths of a od pair are all different"""
        # check if the paths of a od pair are all different
        for od_pair, paths in od_paths.items():
            def _norm_node(n):
                try:
                    return int(n)
                except Exception:
                    return str(n)
            normalized = [tuple(_norm_node(n) for n in p) for p in (paths or [])]
            unique = set(normalized)
            if len(unique) != len(normalized):
                dup_count = len(normalized) - len(unique)
                print(f"Warning: duplicate paths detected for OD {od_pair}: {dup_count} duplicate(s)")
                od_paths[od_pair] = [list(p) for p in unique]
                print(f"Unique paths for OD {od_pair}: {od_paths[od_pair]}")

    def calculate_all_turn_probs(self, nodes):
        """Calculate and store turn probabilities for all nodes in paths"""
        # self.node_turn_probs = {}
        for node_id in self.nodes_in_paths:
            if nodes[node_id].source_num > 2: # only process intersection nodes
                self.calculate_turn_probabilities(nodes[node_id])
            # self.calculate_turn_probabilities(nodes[node_id])
            # nodes[node_id].turns = list(turns)
            # nodes[node_id].turns = dict.fromkeys(turns.keys(), 0)
            # nodes[node_id].turns_in_ods = turns
        self._initialized = True

    def get_path_attributes(self, path):
        """Calculate path attributes"""
        length = 0
        free_flow_time = 0
        
        for i in range(len(path)-1):
            link = self.graph.edges[(path[i], path[i+1])]
            length += link.length
            free_flow_time += link.length / link.free_flow_speed
            
        return {'length': length, 'free_flow_time': free_flow_time} 
    


    def calculate_path_distance(self, path, start_idx=0):
        """
        Calculate path distance from a given point to the destination.
        
        Args:
            path: List of node IDs representing the path
            start_idx: Index in path from where to start calculating distance
            
        Returns:
            float: Distance from start_idx to destination
        """
        distance = 0
        for i in range(start_idx, len(path)-1):
            link = self.graph.edges[(path[i], path[i+1])]
            if link:
                distance += link["weight"]
        return distance



    def expand_controller_paths(self, current_node: Node, od_pair):
        """
        Expand paths at controller nodes by adding detours through non-path neighbors.
        
        Args:
            current_node: The controller node
            od_pair: (origin, destination) tuple
            
        Returns:
            list: New paths added for this OD pair
        """
        current_node_id = current_node.node_id
        origin, dest = od_pair
        paths = self.od_paths[od_pair]
        new_paths = []
        
        # Get all outgoing neighbors of current node
        all_outgoing_neighbors = set()
        for link in current_node.outgoing_links:
            if link.end_node is not None:
                all_outgoing_neighbors.add(link.end_node.node_id)
        
        # Process each existing path that contains current node
        for path in paths:
            try:
                node_idx = path.index(current_node_id)
                
                # Skip if this is the destination node (no downstream to expand)
                if current_node_id == dest:
                    continue
                
                # Get the upstream node
                if current_node_id == origin:
                    up_node = -1
                else:
                    up_node = path[node_idx - 1] if node_idx > 0 else -1
                
                # Get the on-path downstream node
                on_path_down = path[node_idx + 1] if node_idx < len(path) - 1 else None
                
                # Check each outgoing neighbor that's NOT on the current path
                for neighbor in all_outgoing_neighbors:
                    if neighbor == on_path_down or neighbor == up_node:
                        continue  # Skip the already-on-path neighbor
                    
                    # Check if neighbor is already in the prefix (would create immediate loop)
                    prefix_nodes = set(path[:node_idx])  # Nodes before current_node (not including it)
                    if neighbor in prefix_nodes:
                        continue  # Skip - neighbor already visited earlier in path
                    
                    # Try to find multiple paths from neighbor to destination
                    # We'll try several paths in case the shortest creates a loop
                    try:
                        # Create a modified graph that encourages exploration away from existing OD paths
                        modified_graph = self.graph.copy()
                        
                        # Collect ALL edges used in ANY existing path for this OD pair
                        # and calculate their distance to destination for dynamic penalties
                        all_od_edges = {}  # {(u, v): distance_to_dest}
                        for p in self.od_paths[od_pair]:
                            for i in range(len(p) - 1):
                                edge = (p[i], p[i+1])
                                if edge not in all_od_edges:
                                    # Calculate remaining distance from the end of this edge to destination
                                    try:
                                        dist_to_dest = nx.shortest_path_length(
                                            self.graph, p[i+1], dest, weight='weight'
                                        )
                                        all_od_edges[edge] = dist_to_dest
                                    except nx.NetworkXNoPath:
                                        all_od_edges[edge] = 0  # Edge already at destination
                        
                        if self.detour_exploration_mode == 'remove':
                            # Remove already-used edges entirely - forces completely different routes
                            edges_to_remove = []
                            for (u, v) in all_od_edges.keys():
                                if modified_graph.has_edge(u, v):
                                    edges_to_remove.append((u, v))
                            modified_graph.remove_edges_from(edges_to_remove)
                        else:
                            # Penalize already-used edges with distance-based dynamic factor
                            # Edges farther from destination get higher penalty (more exploration early)
                            # Edges closer to destination get lower penalty (less exploration near end)
                            if all_od_edges:
                                max_dist = max(all_od_edges.values()) if all_od_edges.values() else 1
                                
                                for (u, v), dist_to_dest in all_od_edges.items():
                                    if modified_graph.has_edge(u, v):
                                        # Dynamic penalty: scales from base_penalty to base_penalty * detour_penalty_factor
                                        # based on normalized distance to destination
                                        if max_dist > 0:
                                            normalized_dist = dist_to_dest / max_dist
                                            # Penalty ranges from 1.0 (at dest) to detour_penalty_factor (far from dest)
                                            dynamic_penalty = 1.0 + (self.detour_penalty_factor - 1.0) * normalized_dist
                                        else:
                                            dynamic_penalty = self.detour_penalty_factor
                                        
                                        original_weight = modified_graph[u][v].get('weight', 1)
                                        modified_graph[u][v]['weight'] = original_weight * dynamic_penalty
                        
                        # Get multiple simple paths from neighbor to destination using modified graph
                        # This encourages finding truly alternative routes
                        detour_paths = enumerate_shortest_simple_paths(
                            modified_graph, neighbor, dest, max_paths=self.max_detour_paths
                        )
                        
                        if not detour_paths:
                            continue  # No path from neighbor to destination
                        
                        prefix_and_current = set(path[:node_idx + 1])  # All nodes up to and including current
                        
                        # Try each detour path and add those that don't create loops
                        for detour_suffix in detour_paths:
                            # Check if detour would revisit any nodes from the prefix (creating a loop)
                            # detour_suffix = [neighbor, ..., dest]
                            # We need to check if any node in [..., dest] part is already in prefix + current_node
                            detour_nodes_after_neighbor = set(detour_suffix[1:])  # All nodes after neighbor
                            
                            if detour_nodes_after_neighbor & prefix_and_current:  # Use the set operation to check if there are any shared nodes
                                continue  # Skip this detour path - would create a loop
                            
                            # Build concatenated path: prefix + [current_node] + detour_suffix
                            # (detour_suffix includes neighbor, so we don't add it separately)
                            new_path = path[:node_idx + 1] + detour_suffix
                            
                            # Check for duplicates, if the new path is already in the od_paths (convert to tuple for hashing)
                            new_path_tuple = tuple(new_path)
                            existing_paths_tuples = set(tuple(p) for p in self.od_paths[od_pair])
                            
                            if new_path_tuple not in existing_paths_tuples:
                                new_paths.append(new_path)
                            
                    except Exception:
                        # Neighbor cannot reach destination or other error, skip
                        continue
                        
            except ValueError:
                # Current node not in this path
                continue
        
        # Add new paths to od_paths and update bookkeeping
        if new_paths:
            self.od_paths[od_pair].extend(new_paths)
            
            # Update nodes_in_paths and node_to_od_pairs for all nodes in new paths
            for new_path in new_paths:
                for node in new_path:
                    self.nodes_in_paths.add(node)
                    if node not in self.node_to_od_pairs:
                        self.node_to_od_pairs[node] = set()
                    self.node_to_od_pairs[node].add(od_pair)
        
        return new_paths

    def calculate_turn_probabilities(self, current_node):
        """Calculate turn probabilities including special cases for origin/destination nodes"""
        # node = self.graph.nodes[current_node]
        current_node_id = current_node.node_id
        # if current_node_id == 3 or current_node_id == 4:
        #     pass
        # get the relevant od pairs for this node
        relevant_od_pairs = self.node_to_od_pairs.get(current_node_id, set())
        # turns_od_dict = {}
        for od_pair in relevant_od_pairs:
            # if not self._initialized:
            paths = self.od_paths[od_pair]
            od_turn_distances = {}  # {(up_node, down_node): shortest_remaining_distance}
            origin, dest = od_pair
            
            for path in paths:
                try:
                    node_idx = path.index(current_node_id)
                    
                    if current_node_id == origin:
                        down_node = path[node_idx + 1]
                        turn = (-1, down_node)
                    elif current_node_id == dest:
                        # Destination node: no need for turn probabilities
                        up_node = path[node_idx - 1]
                        turn = (up_node, -1)

                    elif node_idx < len(path) - 1:
                        up_node = path[node_idx - 1]
                        down_node = path[node_idx + 1]
                        turn = (up_node, down_node)
                    
                    # if not self._initialized:
                    remaining_dist = self.calculate_path_distance(path, start_idx=node_idx)
                        
                    # Keep only the shortest remaining distance for this turn
                    if turn not in od_turn_distances or remaining_dist < od_turn_distances[turn]:
                        od_turn_distances[turn] = remaining_dist
                        # turns_od_dict[turn] = turns_od_dict.get(turn, []) + [od_pair] #no need recalculate
                        # if ods_in_turns is e
                        if not self._initialized:
                            current_node.ods_in_turns[turn] = current_node.ods_in_turns.get(turn, []) + [od_pair]
                        
                except ValueError:
                    # Current node not in this path
                    continue
            
            if od_turn_distances:
                # Initialize or get existing turn_probs from node
                if not hasattr(current_node, 'node_turn_probs'):
                    current_node.node_turn_probs = {}
                
                # Initialize turns_by_upstream in node if not exists
                if not hasattr(current_node, 'turns_distances'):
                    current_node.turns_distances = {} # {(o,d): {up_node: {down_node: distance}}}

                # Initialize up_od_probs if not exists
                if not hasattr(current_node, 'up_od_probs'):
                    current_node.up_od_probs = defaultdict(lambda: defaultdict(int))
                
                current_node.turns_distances[od_pair] = {}
                # Update distances in existing structure or create new
                for turn, distance in od_turn_distances.items():
                    up_node = turn[0]
                    down_node = turn[1]
                    if up_node not in current_node.turns_distances[od_pair]:
                        # current_node.turns_distances[up_node] = {}
                        current_node.turns_distances[od_pair][up_node] = {}
                    # current_node.turns_by_upstream[up_node][turn] = distance
                    # current_node.turns_distances[up_node][down_node] = distance
                    current_node.turns_distances[od_pair][up_node][down_node] = distance
                    # current_node.up_od_probs[up_node][od_pair] += 1
                    current_node.up_od_probs[up_node][od_pair] = 0 # just assign od_pair to the upstream node

                # Calculate probabilities for each upstream node separately
                if od_pair not in current_node.node_turn_probs:
                    current_node.node_turn_probs[od_pair] = {}

            # calculate the turn probabilities based on the distances and num_pedestrians of the downstream nodes
            #TODO: calibrate the parameters
            # theta = 0.1  # like the temperature in the logit model
            # alpha = 1.0  # distance weight
            # beta = 0.01   # congestion weight
            self.update_node_turn_probs(current_node, od_pair, time_step=0)   # this is the first time we calculate the turn probabilities, so we can use time_step=0
            # for up_node, down_nodes in current_node.turns_distances[od_pair].items():
            #     if down_nodes:
            #         turns = list((up_node, down_node) for down_node in down_nodes)
            #         distances = list(down_nodes.values())
            #         num_pedestrians = [0 if down_node == -1 else self.graph.edges[current_node_id, down_node].get('num_pedestrians', 0) for down_node in down_nodes]
            #         utilities = self.alpha * np.array(distances) + self.beta * np.array(num_pedestrians)
            #         exp_utilities = np.exp(-self.theta * utilities)
            #         probs = exp_utilities / np.sum(exp_utilities)
            #         current_node.node_turn_probs[od_pair].update(dict(zip(turns, probs)))

        # current_node.turns_in_ods = turns_od_dict
        # store the turns for the node
        # return turns_od_dict

    def update_node_turn_probs(self, node, od_pair, time_step):
        """Update the turn probabilities for the node, P(down|up,od)"""
        for up_node, down_nodes in node.turns_distances[od_pair].items():
            if down_nodes:
                turns = list((up_node, down_node) for down_node in down_nodes)
                distances = list(down_nodes.values())
                # num_pedestrians = []
                densities = []
                capacities = []
                for down_node in down_nodes:
                    try:
                        link = self.links[(node.node_id, down_node)]
                        # num_pedestrians.append(self.links[(node.node_id, down_node)].num_pedestrians[time_step-1]) # use the previous time step, current time step is not available is not updated yet
                        densities.append(link.get_density(time_step-1))
                        capacity = link.receiving_flow[time_step-2] # -2 steps is the most recent, the capacity of -1 step is -1 by default
                        capacities.append(capacity if capacity >= 0 else link.back_gate_width * link.free_flow_speed * link.k_critical * link.unit_time)
                    except KeyError:
                        densities.append(0)
                        capacities.append(100) # set a high capacity for origin/destination nodes
                # utilities = self.alpha * np.array(distances) + self.beta * np.array(num_pedestrians)
                norm_densities = np.maximum(np.array(densities) - 2, 0) / (10 - 2) # 2: k_critical, 10: max density
                utilities = (self.alpha * np.array(distances)/(np.sum(distances)+1e-6)
                             + self.beta * norm_densities
                             - self.omega * np.array(capacities)/(np.sum(capacities)+1e-6))
                exp_utilities = np.exp(-self.theta * utilities)
                probs = exp_utilities / np.sum(exp_utilities)
                node.node_turn_probs[od_pair].update(dict(zip(turns, probs)))
        
        return node.node_turn_probs
    
    def update_turning_fractions(self, node, time_step: int, od_manager):
        """Calculate turning fractions using stored turn probabilities: node_turn_probs,
           Return turning_fractions: np.array
        """
        # if node.node_id == 3 or node.node_id == 4:
        #     pass
        turning_fractions = np.zeros(node.edge_num)
        # Update P(od|up) for each upstream node
        for up_node, od_pairs in node.up_od_probs.items():
            total_flow = 0
            # First pass: calculate total flow
            for od_pair in od_pairs:
                flow = od_manager.get_od_flow(od_pair[0], od_pair[1], time_step)
                od_pairs[od_pair] = flow
                total_flow += flow
            
            # Second pass: normalize to get probabilities
            if total_flow > 0:
                for od_pair in od_pairs:
                    od_pairs[od_pair] /= total_flow
            else:
                # If no flow, set equal probabilities
                n_pairs = len(od_pairs)
                for od_pair in od_pairs:
                    od_pairs[od_pair] = 1.0 / n_pairs if n_pairs > 0 else 0
        # Use stored turn probabilities
        # od_turn_probs = self.node_turn_probs.get(node.node_id, {})
        # od_turn_probs = node.node_turn_probs
        
        # Create mapping from node IDs to link indices
        # up_node_to_idx = {link.start_node.node_id if link.start_node is not None else -1: i
        #                  for i, link in enumerate(node.incoming_links)}
        # down_node_to_idx = {link.end_node.node_id if link.end_node is not None else -1: i
        #                    for i, link in enumerate(node.outgoing_links)}
        upstream_nodes = [link.start_node.node_id if link.start_node is not None else -1 for link in node.incoming_links]
        downstream_nodes = [link.end_node.node_id if link.end_node is not None else -1 for link in node.outgoing_links]

        # # calculate od probability
        # flow_list = [od_manager.get_od_flow(origin, dest, time_step) for origin, dest in node.node_turn_probs.keys()]
        # total_flow = sum(flow_list)
        # od_probs = {od_pair: flow / total_flow for od_pair, flow in zip(node.node_turn_probs.keys(), flow_list)}

        # Calculate od probs for each turn
        # for turn in node.turns_in_ods.keys():
        #     od_probs = {}
        #     for od_pair in node.turns_in_ods[turn]:
        #         od_probs[od_pair] = od_manager.get_od_flow(od_pair[0], od_pair[1], time_step)
        #     total_flow = sum(od_probs.values())
        #     if total_flow > 0:
        #         od_probs = {od_pair: flow / total_flow for od_pair, flow in od_probs.items()}
        #     else:
        #         od_probs = {od_pair: 0 for od_pair in od_probs.keys()}
        #     node.turns_in_ods[turn] = od_probs

        # for od_pair in od_turn_probs.keys():
        #     for turn in od_turn_probs[od_pair].keys():
        #         # origin, dest = od_pair
        #         # Get the probability of choosing this OD pair
        #         # od_prob = node.turns_in_ods[turn].get(od_pair, 0)
        #         od_prob = od_probs[od_pair]
        #         # Get the turning probability for this OD pair
        #         turn_prob = od_turn_probs[od_pair].get(turn, 0)
        #         # Accumulate the turning probability
        #         node.turns[turn] += od_prob * turn_prob


        # assign the turns probs to turning fractions
        # idx = 0
        # for up in upstream_nodes:
        #     for down in downstream_nodes:
        #         if up == down: continue
        #         try:
        #             turning_fractions[idx] = node.turns[(up, down)]
        #         except:
        #             continue
        #         idx += 1
        # Calculate final turning fractions
        idx = 0
        for up in upstream_nodes:
            for down in downstream_nodes:
                if up == down: continue
                turn = (up, down)
                prob_sum = 0
                
                # Get all OD pairs for this turn
                od_pairs = node.ods_in_turns.get(turn, [])
                for od_pair in od_pairs:
                    # P(down|up,od) from node_turn_probs
                    self.update_node_turn_probs(node, od_pair, time_step=time_step) # with updating
                    turn_prob = node.node_turn_probs[od_pair].get(turn, 0)
                    # P(od|up) from up_od_probs
                    od_prob = node.up_od_probs[up].get(od_pair, 0)
                    prob_sum += turn_prob * od_prob
                
                turning_fractions[idx] = prob_sum
                idx += 1

        
        return turning_fractions

    @staticmethod
    def check_fractions(node):
        """
        Check if the turning fractions are valid
        """
        fract = node.turning_fractions.reshape(node.dest_num, node.source_num - 1)
        # check if the sum of each column equals to 1, if not assign equal probabilities
        for i in range(node.dest_num):
            if np.abs(sum(fract[i]) - 1) > 1e-3:
                fract[i] = 1/(node.source_num - 1)*np.ones(node.source_num - 1)
        node.turning_fractions = fract.flatten()
        return node.turning_fractions

    def calculate_node_turning_fractions(self, time_step: int, od_manager, node):
        """
        Calculate turning fractions only for nodes that appear in OD paths.
        
        Args:
            nodes: List of all network nodes
            time_step: Current time step
            od_manager: ODManager instance
            
        Returns:
            np.array: Turning fractions for the node
        """
        # turning_fractions = {}
        # Only process nodes that appear in paths
        if node.node_id in self.nodes_in_paths:
            # if node.node_id == 4:
            #     print(1)
            if node.source_num > 2:  # only process intersection nodes
                fractions = self.update_turning_fractions(node, time_step, od_manager)
                node.turning_fractions = fractions
                self.check_fractions(node)
            

# class RouteChoice:
#     """Handles route choice and route-related operations"""
#     def __init__(self, path_finder):
#         self.path_finder = path_finder
