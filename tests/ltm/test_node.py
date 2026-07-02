"""Unit test of the Node module in the LTM package"""

import pytest
from pednstream.ltm.node import Node, NodeConfig

@pytest.fixture
def node_config():
    id = 1
    gate_width = None
    turning_fractions = None
    demand  = None  
    M  = 1e6            
    w = 1e-2          
    is_controller = False   

    return NodeConfig(
        id=id,
        gate_width=gate_width,
        turning_fractions=turning_fractions,
        demand=demand,  
        M=M,
        w=w,
        is_controller=is_controller
    )



class TestNode:
    """Tests for the Node class"""

    def test_node_type_property(self, node_config):
        """Test initial node_type property of the Node class"""
        node = Node(node_config)
        assert node.node_type == ""  # Initially, node_type should be empty


    def test_node_type_setter(self, node_config):
        """Test the node_type setter of the Node class"""
        node = Node(node_config)
        adjacency_matrix = [[0, 1], [1, 0]]
        origin_nodes = [0]
        destination_nodes = [1]

        node.node_type = (adjacency_matrix, origin_nodes, destination_nodes)
        assert node.node_type == "onetoone"  # Based on the provided adjacency matrix and nodes

    def test_node_type_setter_invalid_value(self, node_config):
        """Test that ValueError is raised when setting node_type with an invalid value"""
        node = Node(node_config)
        with pytest.raises(ValueError):
            node.node_type = "invalid_value"  # Not a tuple of (adjacency_matrix, origin_nodes, destination_nodes)      


    def test_node_type_setter_regular_node(self, node_config):
        """Test the node_type setter for a regular node"""
        node = Node(node_config)
        adjacency_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        origin_nodes = [1]
        destination_nodes = [2]

        node.node_type = (adjacency_matrix, origin_nodes, destination_nodes)
        assert node.node_type == "regular"  # Based on the provided adjacency matrix and nodes