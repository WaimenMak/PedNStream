"""Scenarios for the end-to-end characterization test (tests/ltm/test_simulation_characterization.py).

Plain data only (no pednstream imports): the same definitions drive the reference
implementation (origin/main, v0.0.2-alpha) in scripts/generate_simulation_golden.py and
the current code in the test.

Each scenario is either:
- an inline network: ``edges`` (undirected, i < j), ``num_nodes`` and YAML-like sections
  (``simulation``, ``default_link``, ``demand``, ``controller_links``), or
- a data directory: ``data_dir`` with sim_params.yaml, adj_matrix.npy and
  edge_distances.pkl (link lengths), plus optional ``simulation`` overrides.

``events`` change the network at runtime: {time_step: {"separator_width": {(i, j): width}}},
applied before that step is simulated.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = "origin/main"

# Time series recorded for every physical link, compared exactly.
LINK_ARRAYS = (
    "inflow", "outflow", "cumulative_inflow", "cumulative_outflow",
    "sending_flow", "receiving_flow", "num_pedestrians", "density", "speed",
    "travel_time", "avg_travel_time",
)

SCENARIOS = {
    # Corridor 0 - 1 - 2 - 3 - 4 with a separator on 1-2 that is narrowed at runtime;
    # pedestrians walk both ways, so the reverse-link interaction is exercised.
    "corridor_separator": {
        "num_nodes": 5,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4)],
        "origin_nodes": [0, 4],
        "destination_nodes": [4, 0],
        "simulation": {
            "simulation_steps": 300, "unit_time": 1, "seed": 42,
            "assign_flows_type": "classic", "path_finder": {"k_paths": 2},
        },
        "default_link": {
            "length": 30, "width": 3, "free_flow_speed": 1.2, "k_critical": 2,
            "k_jam": 6, "gamma": 0.01, "speed_noise_std": 0.05,
        },
        "demand": {
            "origin_0": {"pattern": "gaussian_peaks", "peak_lambda": 4, "base_lambda": 1},
            "origin_4": {"pattern": "constant", "base_lambda": 2},
        },
        "controller_links": [(1, 2)],
        "events": {120: {"separator_width": {(1, 2): 1.0}}},
        "run_steps": 300,
    },
    # Delft network (298 nodes, 818 links, 12 origins and destinations), classic assignment.
    "delft_classic": {
        "data_dir": "tests/data/delft",
        "simulation": {},
        "run_steps": 100,
    },
    # Same network with the linear-programming (optimal) node assignment.
    "delft_optimal": {
        "data_dir": "tests/data/delft",
        "simulation": {"assign_flows_type": "optimal"},
        "run_steps": 60,  # first arrival at a destination is at step 33
    },
}


def record(network, origin_nodes, run_steps) -> dict:
    """Collect every compared time series of a network after ``run_steps`` steps.

    Works with both implementations: they share link attribute names, network.links
    keyed by (i, j), and node.virtual_incoming_link / virtual_outgoing_link.
    """
    import numpy as np

    out = {}
    end = run_steps + 1
    for (i, j), link in network.links.items():
        for attr in LINK_ARRAYS:
            out[f"link.{i}_{j}.{attr}"] = np.asarray(getattr(link, attr)[:end], dtype=np.float64)
    for node_id, node in network.nodes.items():
        v_in = getattr(node, "virtual_incoming_link", None)
        v_out = getattr(node, "virtual_outgoing_link", None)
        if v_in is not None:
            out[f"node.{node_id}.entered"] = np.asarray(v_in.cumulative_outflow[:end], dtype=np.float64)
        if v_out is not None:
            out[f"node.{node_id}.left"] = np.asarray(v_out.cumulative_inflow[:end], dtype=np.float64)
    for origin in origin_nodes:
        # the reference demand has simulation_steps entries, the current one + 1
        out[f"demand.{origin}"] = np.asarray(network.nodes[origin].demand[:run_steps], dtype=np.float64)
    return out


def apply_events(network, scenario, time_step) -> None:
    for (i, j), width in scenario.get("events", {}).get(time_step, {}).get("separator_width", {}).items():
        network.links[(i, j)].separator_width = width
