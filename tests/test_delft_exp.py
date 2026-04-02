"""
This is a test that uses the street network of Delft, Netherlands.
This test may take a minute or more to complete.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pednstream.utils.network_env_generator import NetworkEnvGenerator
from pednstream.utils.output_handler import OutputHandler
import pytest


@pytest.fixture
def env_generator():
    return NetworkEnvGenerator()


@pytest.fixture
def network_env(env_generator, shared_datadir):
    """A valid data directory contains:
    - .yaml config file,
    - edge distances,
    - adjacency matrix,
    - node positions
    """
    return env_generator.create_network(shared_datadir / "delft")


class TestDeflExp:
    """Test the Simulaition using datasets"""

    def test_run_simulation(self, network_env, env_generator, tmp_path):
        "Test a simulation run and output files are written"

        for t in range(1, env_generator.config["params"]["simulation_steps"]):
            network_env.network_loading(t)

        output_dir = tmp_path / "outputs"
        output_handler = OutputHandler(
            base_dir=str(output_dir), simulation_dir="delft_paths"
        )
        output_handler.save_network_state(network_env)

        sim_output_dir = os.path.join(output_dir, "delft_paths")
        assert os.path.isfile(os.path.join(sim_output_dir, "link_data.json")), (
            "link_data.json not found"
        )
        assert os.path.isfile(os.path.join(sim_output_dir, "network_params.json")), (
            "network_params.json not found"
        )
        assert os.path.isfile(os.path.join(sim_output_dir, "node_data.json")), (
            "node_data.json not found"
        )
