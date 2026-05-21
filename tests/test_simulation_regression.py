"""Regression test for deterministic Delft simulation metrics."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pednstream.utils.network_env_generator import NetworkEnvGenerator


@pytest.fixture
def env_generator():
    return NetworkEnvGenerator()


def _compute_metrics(network, step: int) -> dict:
    origin_demand_sum = float(
        sum(np.sum(network.nodes[o].demand[: step + 1]) for o in network.origin_nodes)
    )

    return {
        "step": step,
        "num_nodes": len(network.nodes),
        "num_links": len(network.links),
        "origin_nodes_count": len(network.origin_nodes),
        "destination_nodes_count": len(network.destination_nodes),
        "origin_demand_sum_upto_step": origin_demand_sum,
        "total_cumulative_inflow_at_step": float(
            sum(link.cumulative_inflow[step] for link in network.links.values())
        ),
        "total_cumulative_outflow_at_step": float(
            sum(link.cumulative_outflow[step] for link in network.links.values())
        ),
        "total_num_pedestrians_at_step": float(
            sum(link.num_pedestrians[step] for link in network.links.values())
        ),
        "mean_density_at_step": float(
            np.mean([link.density[step] for link in network.links.values()])
        ),
        "mean_speed_at_step": float(
            np.mean([link.speed[step] for link in network.links.values()])
        ),
    }


def test_delft_regression_metrics_match_baseline(env_generator, shared_datadir):
    """Run a short deterministic simulation and compare key metrics to baseline."""
    network = env_generator.create_network(shared_datadir / "delft")
    baseline_path = (
        Path(__file__).resolve().parent
        / "data"
        / "delft"
        / "regression_baseline_step20.json"
    )
    expected = json.loads(baseline_path.read_text())

    step = int(expected["step"])
    for t in range(1, step + 1):
        network.network_loading(t)

    actual = _compute_metrics(network, step)

    exact_keys = {
        "step",
        "num_nodes",
        "num_links",
        "origin_nodes_count",
        "destination_nodes_count",
        "origin_demand_sum_upto_step",
        "total_cumulative_inflow_at_step",
        "total_cumulative_outflow_at_step",
        "total_num_pedestrians_at_step",
    }
    for key in exact_keys:
        assert actual[key] == expected[key], f"Mismatch for {key}"

    assert np.isclose(
        actual["mean_density_at_step"], expected["mean_density_at_step"], rtol=1e-9
    ), "Mismatch for mean_density_at_step"
    assert np.isclose(
        actual["mean_speed_at_step"], expected["mean_speed_at_step"], rtol=1e-9
    ), "Mismatch for mean_speed_at_step"


def test_delft_regression_metrics_match_baseline_optimal(env_generator, shared_datadir):
    """Run a short deterministic simulation with optimal assignment and compare key metrics to baseline."""
    # Modify sim_params.yaml to use optimal assignment
    yaml_path = shared_datadir / "delft" / "sim_params.yaml"
    content = yaml_path.read_text()
    content = content.replace('assign_flows_type: "classic"', 'assign_flows_type: "optimal"')
    yaml_path.write_text(content)

    network = env_generator.create_network(shared_datadir / "delft")
    baseline_path = (
        Path(__file__).resolve().parent
        / "data"
        / "delft"
        / "regression_baseline_optimal_step20.json"
    )
    expected = json.loads(baseline_path.read_text())

    step = int(expected["step"])
    for t in range(1, step + 1):
        network.network_loading(t)

    actual = _compute_metrics(network, step)

    exact_keys = {
        "step",
        "num_nodes",
        "num_links",
        "origin_nodes_count",
        "destination_nodes_count",
        "origin_demand_sum_upto_step",
        "total_cumulative_inflow_at_step",
        "total_cumulative_outflow_at_step",
        "total_num_pedestrians_at_step",
    }
    for key in exact_keys:
        assert actual[key] == expected[key], f"Mismatch for {key}"

    assert np.isclose(
        actual["mean_density_at_step"], expected["mean_density_at_step"], rtol=1e-9
    ), "Mismatch for mean_density_at_step"
    assert np.isclose(
        actual["mean_speed_at_step"], expected["mean_speed_at_step"], rtol=1e-9
    ), "Mismatch for mean_speed_at_step"

