"""OutputHandler and NetworkVisualizer take a Simulation (live) or a saved simulation directory."""

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from pednstream.ltm.simulation import Simulation  # noqa: E402
from pednstream.utils.config import (  # noqa: E402
    DemandParams,
    PathFinderParams,
    ScenarioParams,
    build_configs,
)
from pednstream.utils.output_handler import OutputHandler  # noqa: E402
from pednstream.utils.visualizer import NetworkVisualizer  # noqa: E402

STEPS = 160  # walking the 4 x 30 m corridor takes ~100 steps


@pytest.fixture(scope="module")
def simulation():
    """Corridor 0-1-2-3-4: separator on 1-2, gate controller at node 3, two-way demand."""
    adjacency = np.zeros((5, 5))
    for i, j in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        adjacency[i, j] = adjacency[j, i] = 1
    params = ScenarioParams(
        origin_nodes=(0, 4),
        destination_nodes=(4, 0),
        simulation_steps=STEPS,
        unit_time=1,
        default_link={"length": 30, "width": 3, "free_flow_speed": 1.2, "k_critical": 2, "k_jam": 6},
        link_overrides={},
        demand={
            0: DemandParams(pattern="constant", base_lambda=2),
            4: DemandParams(pattern="constant", base_lambda=1),
        },
        controller_nodes=frozenset({3}),
        controller_links=frozenset({(1, 2)}),
        seed=1,
        path_finder=PathFinderParams(k_paths=2),
        adjacency_matrix=adjacency,
        od_flows={},
    )
    sim = Simulation.from_configs(build_configs(params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.run()
    return sim


@pytest.fixture(scope="module")
def saved_dir(simulation, tmp_path_factory):
    handler = OutputHandler(base_dir=tmp_path_factory.mktemp("outputs"), simulation_dir="corridor")
    handler.save_network_state(simulation)
    handler.save_time_series(simulation)
    return handler.simulation_dir


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_controller_nodes_reach_the_simulation(simulation):
    assert simulation.controller_gaters == [3]


def test_save_network_state(simulation, saved_dir):
    params = json.loads((saved_dir / "network_params.json").read_text())
    assert params["simulation_steps"] == STEPS
    assert params["unit_time"] == 1
    assert params["origin_nodes"] == [0, 4] and params["destination_nodes"] == [4, 0]
    assert set(params["od_paths"]) == {"0-4", "4-0"}

    links = json.loads((saved_dir / "link_data.json").read_text())
    with_gates = {k for k, v in links.items() if "back_gate_width" in v}
    assert with_gates == {"2-3", "3-2", "3-4", "4-3"}  # links touching gate controller 3
    assert {k for k, v in links.items() if v.get("is_separator")} == {"1-2", "2-1"}

    nodes = json.loads((saved_dir / "node_data.json").read_text())
    assert nodes["0"]["total_demand"] > 0 and nodes["4"]["total_outflow"] > 0
    assert (saved_dir / "time_series.csv").exists()


@pytest.mark.parametrize("source", ["live", "saved"])
def test_visualizer(simulation, saved_dir, source):
    if source == "live":
        visualizer = NetworkVisualizer(simulation=simulation)
    else:
        visualizer = NetworkVisualizer(simulation_dir=str(saved_dir))
    visualizer.visualize_network_state(40)
    visualizer.visualize_network_state(40, style="paper")
    visualizer.plot_od_paths()
    visualizer.animate_network(start_time=0, end_time=3)
    if source == "live":
        visualizer.plot_link_evolution([(0, 1), (1, 2)])


def test_visualizer_requires_a_source():
    with pytest.raises(ValueError):
        NetworkVisualizer()
