# -*- coding: utf-8 -*-
# @Time    : 14/10/2025 15:30
# @Author  : mmai
# @FileName: discovery
# @Software: PyCharm

"""
Agent discovery and ID mapping for PedNet multi-agent environment.

Discovers separator and gater agents from network topology and maintains
mappings between agent IDs and network components.
"""

import numpy as np
from typing import Dict, List, Set, Tuple, Any
from src.LTM.link import Separator, Link
from src.LTM.node import Node


class AgentDiscovery:
    """
    Maps predefined controllers to network components.
    
    Creates two types of agents based on configuration:
    - Separators: control bidirectional lane allocation via separator_width
    - Gaters: control node-level gate widths via front_gate_width
    """
    
    def __init__(self, network, controller_config: Dict[str, Any]):
        """
        Initialize agent mapping from predefined controller configuration.
        
        Args:
            network: Network instance from LTM simulation
            controller_config: Dictionary specifying controller placement:
                {
                    "separators": [(node1, node2), (node3, node4), ...],  # Bidirectional links to control
                    "gaters": [node1, node2, node3, ...]  # Nodes with gate control
                }
        """
        self.network = network
        self.controller_config = controller_config
        
        # Agent mappings
        self.separator_agents = {}  # agent_id -> {"forward": link, "reverse": link, "total_width": float}
        self.gater_agents = {}      # agent_id -> {"node": node, "out_links": [link1, link2, ...]}
        self.agent_to_type = {}     # agent_id -> "sep" or "gate"
        
        # Create agents from predefined configuration
        self._create_predefined_separators()
        self._create_predefined_gaters()
        
        # Compute max outdegree for gater action space padding
        self.max_outdegree = self._compute_max_outdegree()
    
    def _create_predefined_separators(self):
        """Create separator agents from predefined configuration."""
        separator_pairs = self.controller_config.get("separators", [])
        
        for node_pair in separator_pairs:
            if len(node_pair) != 2:
                raise ValueError(f"Separator pair must have exactly 2 nodes: {node_pair}")
            
            node1, node2 = node_pair
            # Create undirected pair for consistent naming
            pair = tuple(sorted([node1, node2]))
            agent_id = f"sep:{pair[0]}_{pair[1]}"
            
            # Find forward and reverse links
            forward_link = self.network.links.get((pair[0], pair[1]))
            reverse_link = self.network.links.get((pair[1], pair[0]))
            
            if not forward_link or not reverse_link:
                raise ValueError(f"Missing bidirectional links for separator {pair}")
            
            # Verify links are Separators (or convert regular Links to Separators if needed)
            if not isinstance(forward_link, Separator):
                raise ValueError(f"Link {pair[0]}->{pair[1]} is not a Separator. Use Separator links for lane control.")
            
            self.separator_agents[agent_id] = {
                "forward": forward_link,
                "reverse": reverse_link,
                "total_width": forward_link._width  # Total corridor width
            }
            self.agent_to_type[agent_id] = "sep"
    
    def _create_predefined_gaters(self):
        """Create gater agents from predefined node configuration."""
        gater_nodes = self.controller_config.get("gaters", [])
        
        for node_id in gater_nodes:
            if node_id not in self.network.nodes:
                raise ValueError(f"Gater node {node_id} not found in network")
            
            node = self.network.nodes[node_id]
            
            # Find real (non-virtual) outgoing links
            real_out_links = []
            for link in node.outgoing_links:
                # Skip virtual links (origin/destination nodes)
                if hasattr(node, 'virtual_outgoing_link') and link == node.virtual_outgoing_link:
                    continue
                real_out_links.append(link)
            
            if not real_out_links:
                raise ValueError(f"Gater node {node_id} has no real outgoing links to control")
            
            agent_id = f"gat:{node_id}"
            self.gater_agents[agent_id] = {
                "node": node,
                "out_links": real_out_links
            }
            self.agent_to_type[agent_id] = "gate"
    
    def _compute_max_outdegree(self) -> int:
        """Compute maximum outdegree across all gater agents for action padding."""
        if not self.gater_agents:
            return 0
        return max(len(agent_data["out_links"]) for agent_data in self.gater_agents.values())
    
    def get_all_agent_ids(self) -> List[str]:
        """Return list of all discovered agent IDs."""
        return list(self.separator_agents.keys()) + list(self.gater_agents.keys())
    
    def get_separator_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return separator agent mappings."""
        return self.separator_agents.copy()
    
    def get_gater_agents(self) -> Dict[str, Dict[str, Any]]:
        """Return gater agent mappings."""
        return self.gater_agents.copy()
    
    def get_agent_type(self, agent_id: str) -> str:
        """Return agent type ('sep' or 'gate') for given agent ID."""
        if agent_id not in self.agent_to_type:
            raise ValueError(f"Unknown agent ID: {agent_id}")
        return self.agent_to_type[agent_id]
    
    def get_separator_links(self, agent_id: str) -> Tuple[Link, Link]:
        """Return (forward, reverse) links for separator agent."""
        if agent_id not in self.separator_agents:
            raise ValueError(f"Unknown separator agent: {agent_id}")
        agent_data = self.separator_agents[agent_id]
        return agent_data["forward"], agent_data["reverse"]
    
    def get_separator_total_width(self, agent_id: str) -> float:
        """Return total corridor width for separator agent."""
        if agent_id not in self.separator_agents:
            raise ValueError(f"Unknown separator agent: {agent_id}")
        return self.separator_agents[agent_id]["total_width"]
    
    def get_gater_node(self, agent_id: str) -> Node:
        """Return node for gater agent."""
        if agent_id not in self.gater_agents:
            raise ValueError(f"Unknown gater agent: {agent_id}")
        return self.gater_agents[agent_id]["node"]
    
    def get_gater_outgoing_links(self, agent_id: str) -> List[Link]:
        """Return outgoing links for gater agent."""
        if agent_id not in self.gater_agents:
            raise ValueError(f"Unknown gater agent: {agent_id}")
        return self.gater_agents[agent_id]["out_links"]
    
    def get_gater_action_mask(self, agent_id: str) -> np.ndarray:
        """Return action mask for gater agent (1 for valid actions, 0 for padding)."""
        if agent_id not in self.gater_agents:
            raise ValueError(f"Unknown gater agent: {agent_id}")
        
        num_out_links = len(self.gater_agents[agent_id]["out_links"])
        mask = np.zeros(self.max_outdegree, dtype=np.float32)
        mask[:num_out_links] = 1.0
        return mask
    
    def get_max_outdegree(self) -> int:
        """Return maximum outdegree across all gater agents."""
        return self.max_outdegree
