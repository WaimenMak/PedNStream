"""Unit test for network_env_generator.py"""

import pytest
from pednstream.utils.network_env_generator import NetworkEnvGenerator


@pytest.fixture
def data_directory(shared_datadir):
    return shared_datadir / "delft"


@pytest.fixture
def other_data_directory(shared_datadir):
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
        """Test relative path are converted to absolute paths and normalized"""

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
        assert network_environment.config is not network_environment._original_config

    def test_load_network_data_adj_matrix(self, data_directory, network_environment):
        """Tests adjacency matrix is loaded from a file"""

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
        """Tests output is returned in the expected structure"""

        import numpy as np

        loaded_data = network_environment.load_network_data(data_directory)

        assert set(loaded_data.keys()) == {
            "adjacency_matrix",
            "edge_distances",
            "node_positions",
        }

        # Check data types
        assert isinstance(loaded_data["adjacency_matrix"], np.ndarray)
        assert isinstance(loaded_data["edge_distances"], dict)
        assert isinstance(loaded_data["node_positions"], dict)

    def test_create_network_default_dir(self, network_environment):
        """Test a Network is created using the default data directory"""

        from pednstream.ltm.network import Network

        assert network_environment.network is None  # must be None before operation
        network_environment.create_network()
        assert network_environment.network is not None
        assert isinstance(network_environment.network, Network)

    def test_create_network_overwrite_dir(
        self, network_environment, other_data_directory, mocker
    ):
        """Test a Network is created using using a different data directory"""
        from pednstream.ltm.network import Network

        mock_load = mocker.patch.object(
            network_environment,
            "load_network_data",
            wraps=network_environment.load_network_data,
        )

        assert network_environment.network is None
        network_environment.create_network(data_path=other_data_directory)
        mock_load.assert_called_once_with(other_data_directory)
        assert network_environment.network is not None
        assert isinstance(network_environment.network, Network)

    def test_build_link_params(self, network_environment):
        """Test link parameters are build and return in the correct format"""
        edge_distances = {(0, 1): 100.0, (2, 3): 200.0}
        default_link_params = {"width": 2.0, "free_flow_speed": 1.34}
        existing_link_configs = {}

        result = network_environment._build_link_params(
            edge_distances, default_link_params, existing_link_configs
        )

        # Forward and reverse links created for each edge
        assert "0_1" in result
        assert "1_0" in result
        assert "2_3" in result
        assert "3_2" in result
        assert len(result) == 4

        # Length set from edge distance
        assert result["0_1"]["length"] == 100.0
        assert result["1_0"]["length"] == 100.0
        assert result["2_3"]["length"] == 200.0

        # Default params applied
        assert result["0_1"]["width"] == 2.0
        assert result["0_1"]["free_flow_speed"] == 1.34

    def test_build_link_params_existing_override(self, network_environment):
        """Test existing link configs override defaults."""
        edge_distances = {(0, 1): 50.0}
        default_link_params = {"width": 2.0, "free_flow_speed": 1.34}
        existing_link_configs = {"0_1": {"width": 5.0, "k_jam": 3.0}}

        result = network_environment._build_link_params(
            edge_distances, default_link_params, existing_link_configs
        )

        # Existing config overrides default for forward link
        assert result["0_1"]["width"] == 5.0
        assert result["0_1"]["k_jam"] == 3.0
        assert result["0_1"]["free_flow_speed"] == 1.34
        assert result["0_1"]["length"] == 50.0

        # Reverse link gets a copy of the forward link's merged params
        assert result["1_0"]["width"] == 5.0
        assert result["1_0"]["k_jam"] == 3.0
        assert result["1_0"]["length"] == 50.0

    def test_build_link_params_no_reverse_if_existing(self, network_environment):
        """Test reverse link is not auto-created when it has its own existing config."""
        edge_distances = {(0, 1): 75.0}
        default_link_params = {"width": 2.0}
        existing_link_configs = {"1_0": {"width": 9.0}}

        result = network_environment._build_link_params(
            edge_distances, default_link_params, existing_link_configs
        )

        # Forward link created normally
        assert result["0_1"]["length"] == 75.0
        # Reverse link NOT auto-created because it's in existing_link_configs
        assert "1_0" not in result

    def test_build_link_params_reverse_not_duplicated(self, network_environment):
        """Test reverse link is not overwritten when both directions appear in edge_distances."""
        edge_distances = {(0, 1): 60.0, (1, 0): 80.0}
        default_link_params = {"width": 2.0}
        existing_link_configs = {}

        result = network_environment._build_link_params(
            edge_distances, default_link_params, existing_link_configs
        )

        # Both forward links created from their own edge entry
        assert result["0_1"]["length"] == 60.0
        assert result["1_0"]["length"] == 80.0
