import math

import networkx as nx
import numpy as np
from collections import defaultdict
from heapq import heappush, heappop
from pednstream.ltm.node import Node


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
        shortest_path = nx.shortest_path(graph, origin, dest, weight="weight")
        shortest_dist = nx.shortest_path_length(graph, origin, dest, weight="weight")
        path_tuple = tuple(shortest_path)  # Convert to tuple for hashing
        found_paths.add(path_tuple)
        candidate_paths[next_path_id] = (shortest_dist, shortest_path)
        A.append((shortest_dist, shortest_path))
        next_path_id += 1
    except nx.NetworkXNoPath:
        return []

    # ===== KEY PART: FINDING K DIFFERENT PATHS =====
    while len(A) < k:
        if not A:
            break

        prev_path = A[-1][1]

        for i in range(len(prev_path) - 1):
            spur_node = prev_path[i]
            spur_node_next = prev_path[
                i + 1
            ]  # set the distance between deviation node and the next node to inf

            root_path = prev_path[: i + 1]
            edges_removed = []
            nodes_removed = []

            # Remove nodes in root_path to avoid loops
            for node in root_path[:-1]:  # Exclude spur_node
                if node != spur_node and graph.has_node(node):
                    # Save all edges connected to this node before removing it
                    for neighbor in list(graph.neighbors(node)):
                        if graph.has_edge(node, neighbor):
                            edge_data = graph[node][
                                neighbor
                            ].copy()  # Copy edge attributes
                            edges_removed.append((node, neighbor, edge_data))

                    # Also save incoming edges (for directed graphs)
                    for neighbor in list(graph.predecessors(node)):
                        if graph.has_edge(neighbor, node):
                            edge_data_inv = graph[neighbor][node].copy()
                            edges_removed.append((neighbor, node, edge_data_inv))

                    nodes_removed.append(node)
                    graph.remove_node(node)

            # Handle the direct edge between spur_node and next node
            if graph.has_edge(spur_node, spur_node_next):
                original_weight = graph[spur_node][spur_node_next].get("weight", 1)
                graph[spur_node][spur_node_next]["weight"] = (
                    np.inf
                )  # Set to infi to avoid this edge

            try:
                spur_path = nx.shortest_path(graph, spur_node, dest, weight="weight")
                total_path = root_path[:-1] + spur_path

                # Restore removed nodes and their edges
                for node in nodes_removed:
                    graph.add_node(node)

                # Restore all removed edges
                for u, v, edge_data in edges_removed:
                    graph.add_edge(u, v, **edge_data)

                if tuple(total_path) in found_paths:
                    continue
                total_dist = sum(
                    graph[total_path[i]][total_path[i + 1]].get("weight", 1)
                    for i in range(len(total_path) - 1)
                )

                # Store the candidate path with next available ID
                candidate_paths[next_path_id] = (total_dist, total_path)
                heappush(B, (total_dist, next_path_id))
                next_path_id += 1
            except nx.NetworkXNoPath:
                pass
            finally:
                # Restore the edge weight
                if graph.has_edge(spur_node, spur_node_next):
                    graph[spur_node][spur_node_next]["weight"] = original_weight

        if B:
            # Get the shortest candidate from priority queue
            _, candidate_id = heappop(B)
            candidate = candidate_paths[candidate_id]
            path_tuple = tuple(candidate[1])
            if path_tuple not in found_paths:
                found_paths.add(path_tuple)
                A.append(candidate)

    return [path for _, path in A]


def enumerate_shortest_simple_paths(graph, origin, dest, max_paths=None):
    """
    Enumerate all simple paths from origin to dest. This method is used when generating the paths between O-D pairs, and expanding paths at controller nodes.

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
        paths_iter = nx.shortest_simple_paths(graph, origin, dest, weight="weight")
    except Exception:
        return []

    paths = []
    for path in paths_iter:
        paths.append(path)
        if max_paths is not None and len(paths) >= max_paths:
            # print(f"Early stopping: Found {len(paths)} paths")
            break
    return paths


class PathFinder:
    """Handles path finding and path-related operations"""

    def __init__(
        self,
        links,
        params=None,
        controller_nodes=None,
        controller_links=None,
        logger=None,
    ):
        self.od_paths = {}  # {(origin, dest): [path1, path2, ...]}
        self.graph = self._create_graph(links)
        self.links = links
        self.nodes_in_paths = set()
        self.node_turn_probs = {}  # {node_id: {(o,d): {(up_node, down_node): probability}}}
        self.node_to_od_pairs = {}  # {node_id: set((o1,d1), (o2,d2), ...)}, to get the relevant od pairs for the node
        self._initialized = False  # Add initialization flag
        self._link_cache_step = -1  # Last timestep for which link cache was built
        self._link_density_cache = {}  # {(node_id, down_node): normalized_density}
        self._link_capacity_cache = {}  # {(node_id, down_node): capacity}
        self.logger = logger

        # Get parameters from config or use defaults
        path_params = params.get("path_finder", {}) if params else {}
        self.temp = path_params.get(
            "temp", 0.1
        )  # like the temperature in the logit model
        self.alpha = path_params.get("alpha", 1.0)  # distance weight
        self.beta = path_params.get("beta", 0.05)  # congestion weight
        self.omega = path_params.get("omega", 0.05)  # capacity weight
        self.std_dev = path_params.get(
            "std_dev", 0
        )  # standard deviation of the normal distribution
        self.epsilon = np.random.normal(
            0, self.std_dev
        )  # random variable in the utility function follow the normal distribution
        self.k_paths = path_params.get("k_paths", 3)
        self.controller_k_paths = path_params.get("controller_k_paths", 100) # default to 100 paths
        self.verbose = path_params.get("verbose", True)  # Control path finding logging

        # Controller configuration
        self.controller_nodes = controller_nodes
        self.controller_links = controller_links
        self.controllers_enabled = (
            True if controller_nodes or controller_links else False
        )

        # Detour exploration settings (hardcoded for now)
        self.detour_exploration_mode = (
            "penalize"  # 'penalize' or 'remove' - penalize makes prefix edges expensive
        )
        self.detour_penalty_factor = 2  # Weight multiplier for penalized edges
        self.max_detour_paths = 3  # Maximum alternative paths to try per neighbor

    def _create_graph(self, links, time_step=0):
        """Convert network to NetworkX graph"""
        G = nx.DiGraph()
        for (start, end), link in links.items():
            G.add_edge(
                start,
                end,
                weight=link.length,
                num_pedestrians=link.num_pedestrians[time_step],
            )
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

        for origin, dest in od_pairs:
            try:
                # paths = k_shortest_paths(self.graph, origin, dest, k=self.k_paths)
                paths = enumerate_shortest_simple_paths(
                    self.graph, origin, dest, max_paths=self.k_paths
                )
                self.od_paths[(origin, dest)] = paths

                # Record which nodes are used in this OD pair
                for path in paths:
                    for node in path:
                        self.nodes_in_paths.add(node)
                        if node not in self.node_to_od_pairs:
                            self.node_to_od_pairs[node] = set()
                        self.node_to_od_pairs[node].add((origin, dest))

            except nx.NetworkXNoPath:
                if self.logger and self.verbose:
                    self.logger.info(f"No path found between {origin} and {dest}")
                self.od_paths[(origin, dest)] = []

        # expand the paths at controller nodes
        if not self._initialized and self.controllers_enabled:
            for node in self.controller_nodes:
                total_paths_added = 0
                for od_pair in self.node_to_od_pairs[node]:
                    remaining_budget = self.controller_k_paths - total_paths_added
                    if remaining_budget <= 0:
                        break  # Reached limit for this controller node
                    num_paths_before = len(self.od_paths[od_pair])
                    self.expand_controller_paths(nodes[node], od_pair, max_paths=remaining_budget)
                    num_paths_after = len(self.od_paths[od_pair])
                    paths_added = num_paths_after - num_paths_before
                    total_paths_added += paths_added
                    if self.logger and self.verbose:
                        self.logger.info(
                            f"Controller node {node}: Added {paths_added} detour path(s) for OD {od_pair} (total: {total_paths_added}/{self.controller_k_paths})"
                        )
        self.check_if_paths_are_different(self.od_paths, self.logger, self.verbose)
        # Calculate and store turn probabilities for all nodes in paths
        self.calculate_all_turn_probs(nodes=nodes)

    @staticmethod
    def check_if_paths_are_different(od_paths, logger=None, verbose=False):
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
                if logger and verbose:
                    logger.info(
                        f"Warning: duplicate paths detected for OD {od_pair}: {dup_count} duplicate(s)"
                    )
                od_paths[od_pair] = [list(p) for p in unique]
                if logger and verbose:
                    logger.info(f"Unique paths for OD {od_pair}: {od_paths[od_pair]}")

    def calculate_all_turn_probs(self, nodes):
        """Calculate and store turn probabilities for all nodes in paths"""
        # self.node_turn_probs = {}
        for node_id in self.nodes_in_paths:
            if nodes[node_id].source_num > 2:  # only process intersection nodes
                self.calculate_turn_probabilities(nodes[node_id])
        self._initialized = True

    def get_path_attributes(self, path):
        """Calculate path attributes"""
        length = 0
        free_flow_time = 0

        for i in range(len(path) - 1):
            link = self.graph.edges[(path[i], path[i + 1])]
            length += link.length
            free_flow_time += link.length / link.free_flow_speed

        return {"length": length, "free_flow_time": free_flow_time}

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
        for i in range(start_idx, len(path) - 1):
            link = self.graph.edges[(path[i], path[i + 1])]
            if link:
                distance += link["weight"]
        return distance

    def expand_controller_paths(self, current_node: Node, od_pair, max_paths=None):
        """
        Expand paths at controller nodes by adding detours through non-path neighbors. If path
        already exist in original routes skip it.

        Args:
            current_node: The controller node
            od_pair: (origin, destination) tuple
            max_paths: Maximum number of new paths to add (defaults to controller_k_paths)

        Returns:
            list: New paths added for this OD pair
        """
        current_node_id = current_node.node_id
        origin, dest = od_pair
        paths = self.od_paths[od_pair]
        new_paths = []

        if max_paths is None:
            max_paths = self.controller_k_paths

        if max_paths <= 0:
            return new_paths

        # Get all outgoing neighbors of current node
        all_outgoing_neighbors = set()
        for link in current_node.outgoing_links:
            if link.end_node is not None:
                all_outgoing_neighbors.add(link.end_node.node_id)

        # Create a modified graph that encourages exploration away from existing OD paths
        # This is computed once per OD pair as it depends only on existing OD paths
        modified_graph = self.graph.copy()

        # Collect ALL edges used in ANY existing path for this OD pair
        # and calculate their distance to destination for dynamic penalties
        all_od_edges = {}  # {(u, v): distance_to_dest}
        for p in paths:
            for i in range(len(p) - 1):
                edge = (p[i], p[i + 1])
                if edge not in all_od_edges:
                    # Calculate remaining distance from the end of this edge to destination
                    try:
                        dist_to_dest = nx.shortest_path_length(
                            self.graph, p[i + 1], dest, weight="weight"
                        )
                        all_od_edges[edge] = dist_to_dest
                    except nx.NetworkXNoPath:
                        all_od_edges[edge] = 0  # Edge already at destination

        if self.detour_exploration_mode == "remove":
            # Remove already-used edges entirely - forces completely different routes
            edges_to_remove = []
            for u, v in all_od_edges.keys():
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
                            dynamic_penalty = (
                                1.0
                                + (self.detour_penalty_factor - 1.0) * normalized_dist
                            )
                        else:
                            dynamic_penalty = self.detour_penalty_factor

                        original_weight = modified_graph[u][v].get("weight", 1)
                        modified_graph[u][v]["weight"] = (
                            original_weight * dynamic_penalty
                        )

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
                    prefix_nodes = set(
                        path[:node_idx]
                    )  # Nodes before current_node (not including it)
                    if neighbor in prefix_nodes:
                        continue  # Skip - neighbor already visited earlier in path

                    # Try to find multiple paths from neighbor to destination
                    # We'll try several paths in case the shortest creates a loop
                    try:
                        # Get multiple simple paths from neighbor to destination using modified graph
                        # This encourages finding truly alternative routes
                        detour_paths = enumerate_shortest_simple_paths(
                            modified_graph,
                            neighbor,
                            dest,
                            max_paths=self.max_detour_paths,
                        )

                        if not detour_paths:
                            continue  # No path from neighbor to destination

                        prefix_and_current = set(
                            path[: node_idx + 1]
                        )  # All nodes up to and including current

                        # Try each detour path and add those that don't create loops
                        for detour_suffix in detour_paths:
                            # Check if detour would revisit any nodes from the prefix (creating a loop)
                            # detour_suffix = [neighbor, ..., dest]
                            # We need to check if any node in [..., dest] part is already in prefix + current_node
                            detour_nodes_after_neighbor = set(
                                detour_suffix[1:]
                            )  # All nodes after neighbor

                            if (
                                detour_nodes_after_neighbor & prefix_and_current
                            ):  # Use the set operation to check if there are any shared nodes
                                continue  # Skip this detour path - would create a loop

                            # Build concatenated path: prefix + [current_node] + detour_suffix
                            new_path = path[: node_idx + 1] + detour_suffix

                            # Check for duplicates, if the new path is already in the od_paths (convert to tuple for hashing)
                            new_path_tuple = tuple(new_path)
                            existing_paths_tuples = set(
                                tuple(p) for p in self.od_paths[od_pair]
                            )

                            if new_path_tuple not in existing_paths_tuples:
                                new_paths.append(new_path)
                                if len(new_paths) >= max_paths:
                                    break  # Reached limit for this OD pair

                    except Exception:
                        # Neighbor cannot reach destination or other error, skip
                        continue

                    if len(new_paths) >= max_paths:
                        break  # Reached limit, stop exploring neighbors

            except ValueError:
                # Current node not in this path
                continue

            if len(new_paths) >= max_paths:
                break  # Reached limit, stop processing paths

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
        """Calculate turn probabilities including special cases for origin/destination nodes, this function is called only once at the initialization of the simulation"""
        # node = self.graph.nodes[current_node]
        current_node_id = current_node.node_id
        relevant_od_pairs = self.node_to_od_pairs.get(current_node_id, set())
        for od_pair in relevant_od_pairs:
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
                    remaining_dist = self.calculate_path_distance(
                        path, start_idx=node_idx
                    )

                    # Keep only the shortest remaining distance for this turn
                    if (
                        turn not in od_turn_distances
                        or remaining_dist < od_turn_distances[turn]
                    ):
                        od_turn_distances[turn] = remaining_dist
                        # turns_od_dict[turn] = turns_od_dict.get(turn, []) + [od_pair] #no need recalculate
                        # if ods_in_turns is e
                        if not self._initialized:
                            # Use set to automatically handle duplicates with O(1) insertion
                            if turn not in current_node.ods_in_turns:
                                current_node.ods_in_turns[turn] = set()
                            current_node.ods_in_turns[turn].add(od_pair)

                except ValueError:
                    # Current node not in this path
                    continue

            if od_turn_distances:
                # Attributes are pre-initialized in Node.__init__; no hasattr needed.
                current_node.turns_distances[od_pair] = {}
                # Update distances in existing structure or create new
                for turn, distance in od_turn_distances.items():
                    up_node = turn[0]
                    down_node = turn[1]
                    if up_node not in current_node.turns_distances[od_pair]:
                        current_node.turns_distances[od_pair][up_node] = {}
                    current_node.turns_distances[od_pair][up_node][down_node] = distance
                    current_node.up_od_probs[up_node][od_pair] = (
                        0  # just assign od_pair to the upstream node
                    )

                # Calculate probabilities for each upstream node separately
                if od_pair not in current_node.node_turn_probs:
                    current_node.node_turn_probs[od_pair] = {}

            # calculate the turn probabilities based on the distances and num_pedestrians of the downstream nodes
            # TODO: calibrate the parameters
            current_node.update_node_turn_probs(
                od_pair,
                time_step=0,
                alpha=self.alpha,
                beta=self.beta,
                omega=self.omega,
                temp=self.temp,
                std_dev=self.std_dev,
            )  # this is the first time we calculate the turn probabilities, so we can use time_step=0

    def _cache_link_attributes(self, time_step):
        """Cache link density and capacity data for the current timestep.

        Called once per timestep before processing any nodes, so that
        update_node_turn_probs can look up pre-computed values instead of
        redundantly querying link objects across OD pairs.
        """
        if self._link_cache_step == time_step:
            return  # already cached for this step

        density_ts = max(0, time_step - 1)
        capacity_ts = max(0, time_step - 2)
        density_cache = self._link_density_cache
        capacity_cache = self._link_capacity_cache
        density_cache.clear()
        capacity_cache.clear()

        for (start, end), link in self.links.items():
            raw_density = link.get_density(density_ts)
            norm_density = max(raw_density - link.k_critical, 0.0) / (
                link.k_jam - link.k_critical
            )
            density_cache[(start, end)] = norm_density

            cap = link.receiving_flow[capacity_ts]
            if cap < 0:
                cap = (
                    link.back_gate_width
                    * link.free_flow_speed
                    * link.k_critical
                    * link.unit_time
                )
            capacity_cache[(start, end)] = cap

        self._link_cache_step = time_step

    def update_node_turn_probs(self, node, od_pair, time_step):
        """Update the turn probabilities for the node, P(down|up,od).

        Optimized to use:
        - Cached link attributes (density/capacity) to avoid redundant lookups
        - math.exp instead of np.exp for tiny arrays (2-3 elements)
        - Direct dict assignment instead of dict(zip(...)) allocation
        """
        # current_turn_probs_step is pre-initialized in Node.__init__
        # Reuse the probabilities within the same timestep so one choice set
        # is generated from one consistent utility/noise realization.
        if node.current_turn_probs_step.get(od_pair) == time_step:
            return node.node_turn_probs

        node_id = node.node_id
        density_cache = self._link_density_cache
        capacity_cache = self._link_capacity_cache
        alpha = self.alpha
        beta = self.beta
        omega = self.omega
        neg_temp = -self.temp
        std_dev = self.std_dev
        od_probs = node.node_turn_probs[od_pair]

        for up_node, down_nodes in node.turns_distances[od_pair].items():
            if not down_nodes:
                continue

            n = len(down_nodes)
            # Build lists of turns, distances, densities, capacities using
            # plain Python — avoids np.array overhead for tiny n (typically 2-3)
            turns = []
            distances = []
            densities = []
            capacities = []
            for down_node, dist in down_nodes.items():
                turns.append((up_node, down_node))
                distances.append(dist)
                key = (node_id, down_node)
                if key in density_cache:
                    densities.append(density_cache[key])
                    capacities.append(capacity_cache[key])
                else:
                    # origin/destination virtual nodes (down_node == -1)
                    densities.append(0.0)
                    capacities.append(100.0)

            # Compute utilities using plain Python math for small n
            sum_dist = sum(distances) + 1e-6
            sum_cap = sum(capacities) + 1e-6

            # Pre-generate noise if needed
            if std_dev != 0:
                noise = np.random.normal(0, std_dev, n)
            else:
                noise = None

            # Compute exp(-temp * utility) for each turn
            exp_utils = [0.0] * n
            for i in range(n):
                u = (
                    alpha * distances[i] / sum_dist
                    + beta * densities[i]
                    - omega * capacities[i] / sum_cap
                )
                if noise is not None:
                    u += noise[i]
                exp_utils[i] = math.exp(neg_temp * u)

            # Normalize to probabilities
            total = sum(exp_utils)
            if total > 0:
                inv_total = 1.0 / total
                for i in range(n):
                    od_probs[turns[i]] = exp_utils[i] * inv_total
            else:
                eq_prob = 1.0 / n
                for i in range(n):
                    od_probs[turns[i]] = eq_prob

        node.current_turn_probs_step[od_pair] = time_step
        return node.node_turn_probs

    def update_turning_fractions(self, node, time_step: int, od_manager):
        """Calculate turning fractions using stored turn probabilities: node_turn_probs,
        Return turning_fractions: np.array

        Optimized to:
        - Cache link attributes once per timestep before processing
        - Batch-update all OD pairs' turn probs before assembling fractions
        - Use pre-fetched OD flow values
        """
        # Ensure link caches are fresh for this timestep
        self._cache_link_attributes(time_step)

        turning_fractions = np.zeros(node.edge_num)
        od_flows_cache = self._od_manager_cache  # pre-fetched in calculate_node_turning_fractions

        # Update P(od|up) for each upstream node using pre-fetched flows
        for up_node, od_pairs in node.up_od_probs.items():
            total_flow = 0.0
            # First pass: calculate total flow
            for od_pair in od_pairs:
                flow = od_flows_cache.get(od_pair, 0.0)
                od_pairs[od_pair] = flow
                total_flow += flow

            # Second pass: normalize to get probabilities
            if total_flow > 0:
                inv_total = 1.0 / total_flow
                for od_pair in od_pairs:
                    od_pairs[od_pair] *= inv_total
            else:
                # If no flow, set equal probabilities
                n_pairs = len(od_pairs)
                eq_prob = 1.0 / n_pairs if n_pairs > 0 else 0.0
                for od_pair in od_pairs:
                    od_pairs[od_pair] = eq_prob

        # Batch-update turn probabilities for all OD pairs at this node first,
        # so each OD pair is computed exactly once (not once per turn reference).
        node_turn_probs = node.node_turn_probs
        for od_pair in node_turn_probs:
            self.update_node_turn_probs(node, od_pair, time_step=time_step)

        # Calculate final turning fractions
        upstream_nodes = [
            link.start_node.node_id if link.start_node is not None else -1
            for link in node.incoming_links
        ]
        downstream_nodes = [
            link.end_node.node_id if link.end_node is not None else -1
            for link in node.outgoing_links
        ]

        up_od_probs = node.up_od_probs
        ods_in_turns = node.ods_in_turns

        idx = 0
        for up in upstream_nodes:
            up_probs = up_od_probs.get(up)
            for down in downstream_nodes:
                if up == down:
                    continue
                turn = (up, down)
                prob_sum = 0.0

                # Get all OD pairs for this turn
                turn_od_pairs = ods_in_turns.get(turn)
                if turn_od_pairs and up_probs:
                    for od_pair in turn_od_pairs:
                        # P(down|up,od) from node_turn_probs
                        turn_prob = node_turn_probs[od_pair].get(turn, 0)
                        # P(od|up) from up_od_probs
                        od_prob = up_probs.get(od_pair, 0)
                        prob_sum += turn_prob * od_prob

                turning_fractions[idx] = prob_sum
                idx += 1

        return turning_fractions
        """Delegate turn probability updates to the node."""
        return node.update_node_turn_probs(
            od_pair,
            time_step,
            alpha=self.alpha,
            beta=self.beta,
            omega=self.omega,
            temp=self.temp,
            std_dev=self.std_dev,
        )

    def update_turning_fractions(self, node, time_step: int, od_manager):
        """Delegate turning fraction updates to the node."""
        return node.update_turning_fractions(
            time_step,
            od_manager,
            alpha=self.alpha,
            beta=self.beta,
            omega=self.omega,
            temp=self.temp,
            std_dev=self.std_dev,
        )

    @staticmethod
    def check_fractions(node):
        """Delegate turning fraction validation to the node."""
        return node.check_fractions()

    def _ensure_od_flow_cache(self, time_step: int, od_manager):
        """Pre-fetch all OD flows for the current timestep to avoid per-call overhead.

        This cache is shared across all nodes processed in the same timestep.
        """
        if not hasattr(self, '_od_cache_step') or self._od_cache_step != time_step:
            self._od_manager_cache = {}
            for od_pair in self.od_paths:
                self._od_manager_cache[od_pair] = od_manager.get_od_flow(
                    od_pair[0], od_pair[1], time_step
                )
            self._od_cache_step = time_step

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
        # TODO: consider if this can  extracted from here. Maybe not.
        # Only process nodes that appear in paths
        if node.node_id in self.nodes_in_paths:
            if node.source_num > 2:  # only process intersection nodes
                # Pre-fetch OD flows once per timestep (shared across all nodes)
                self._ensure_od_flow_cache(time_step, od_manager)
                fractions = self.update_turning_fractions(node, time_step, od_manager)
                node.turning_fractions = fractions # updtes value on the node. 
                self.check_fractions(node)
