"""Unit test for network_env_generator.py"""

import pytest
from pednstream.utils.network_env_generator import NetworkEnvGenerator


@pytest.fixture
def data_directory(shared_datadir):
    return shared_datadir / "delft"


@pytest.fixture
def network_environment(data_directory):
    return NetworkEnvGenerator(data_directory)


class TestNetworkEnvGenerator:
    def test_initialization(self, data_directory):
        """Tests instance creation"""
        network_environment = NetworkEnvGenerator(data_directory)

        assert isinstance(network_environment, NetworkEnvGenerator)
        assert network_environment.network is None
        assert network_environment.network_data is None
        assert network_environment.config == {}
        assert network_environment._original_config is None

    def test_normalize_path(self, network_environment):
        """Test relative path are converted to absolute paths and normilized"""

        result = network_environment._normalize_path("./data/../path")
        assert result.is_absolute()
        assert result == result.resolve()

    def test_load_network_data_original_config(
        self, data_directory, network_environment
    ):
        """Tests load_network_data creates deep copy of configuration parameters
        a copy"""

        network_environment.load_network_data(data_directory)
        assert network_environment.config == network_environment._original_config

    def test_load_network_data_adj_matrix(self, data_directory, network_environment):
        """Tests adjancency matrix is loaded from a file"""

        import numpy as np

        file_matrix = np.load(data_directory / "adj_matrix.npy")  # independent

        loaded_matrix = network_environment.load_network_data(data_directory)
        assert np.array_equal(file_matrix, loaded_matrix["adjacency_matrix"])

    def test_load_network_data_edge_distances(
        self, data_directory, network_environment
    ):
        """Tests edge distances are loaded from a file"""

        import pickle

        with open(data_directory / "edge_distances.pkl", "rb") as f:
            file_edges = pickle.load(f)

        loaded_edges = network_environment.load_network_data(data_directory)
        assert file_edges == loaded_edges["edge_distances"]

    def test_load_network_data_node_positions(
        self, data_directory, network_environment
    ):
        """Tests node positions are loaded from a file"""

        import json

        with open(data_directory / "node_positions.json") as f:
            file_positions = {str(node): pos for node, pos in json.load(f).items()}

        loaded_positions = network_environment.load_network_data(data_directory)
        assert loaded_positions["node_positions"] == file_positions

    def test_load_network_data_output_structure(
        self, data_directory, network_environment
    ):
        """Tests ouput is returned in the expected structure"""

        import numpy as np
        from typing import Dict

        loaded_data = network_environment.load_network_data(data_directory)

        assert set(loaded_data.keys()) == {
            "adjacency_matrix",
            "edge_distances",
            "node_positions",
        }

        # Check data types
        assert isinstance(loaded_data["adjacency_matrix"], np.ndarray)
        assert isinstance(loaded_data["edge_distances"], Dict)
        assert isinstance(loaded_data["node_positions"], Dict)
