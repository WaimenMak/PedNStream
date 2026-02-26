"""
Test basic functionality of PedNStream
"""

# Add project root to Python path
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import matplotlib

matplotlib.use("macosx")
import numpy as np
import matplotlib.pyplot as plt
import pytest
from handlers.output_handler import OutputHandler

# Now you can import using the project structure
from pednstream.utils.visualizer import NetworkVisualizer
from pednstream.ltm.network import Network
from pathlib import Path


@pytest.fixture
def adj():
    adj = np.array(
        [
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    return adj


@pytest.fixture
def params():

    return {
        "unit_time": 10,
        "simulation_steps": 600,
        "default_link": {
            "length": 100,  # make it to 50 to see spillback
            "width": 2,
            "free_flow_speed": 1.1,
            "k_critical": 2,
            "k_jam": 6,
            "fd_type": "yperman",  # type of fundamental diagram
            "bi_factor": 1,  # factor for bi-directional FD
            "controller_type": "gate",  # type of controller
        },
        # 'controllers': {
        #     'enabled': True,
        #     'links': ["0-1","1-2","2-3","3-4","4-5"],
        # },
        "demand": {
            "origin_0": {
                "peak_lambda": 25,
                "base_lambda": 5,
            },
            "origin_5": {
                "peak_lambda": 25,
                "base_lambda": 5,
            },
        },
    }


@pytest.fixture
def network_env(params, adj):

    return Network(adj, params, origin_nodes=[5, 0])


class TestLongCorridor:
    def test_run_simulation(self, params, network_env):
        """Test simulation case for long corridor"""
        for t in range(1, params["simulation_steps"]):
            network_env.network_loading(t)

        # Construct paths relative to the project root
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / "outputs"

        sim_name = "long_corridor"
        # Use the constructed paths
        output_handler = OutputHandler(
            base_dir=str(output_dir), simulation_dir=f"{sim_name}"
        )
        output_handler.save_network_state(network_env)

        # Create the visualization
        visualizer = NetworkVisualizer(
            simulation_dir=os.path.join(output_dir, f"{sim_name}")
        )
        anim = visualizer.animate_network(
            start_time=0,
            end_time=params["simulation_steps"],
            interval=100,
            edge_property="density",
            tag=False,
            vis_actions=True,
        )

        plt.show()
