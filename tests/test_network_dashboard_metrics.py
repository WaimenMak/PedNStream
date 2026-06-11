import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "adv_network_dashboard.py"
SPEC = importlib.util.spec_from_file_location("network_dashboard", MODULE_PATH)
network_dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = network_dashboard
SPEC.loader.exec_module(network_dashboard)


def _link(values, length=10):
    return {
        "density": list(values),
        "link_flow": list(values),
        "speed": list(values),
        "travel_time": list(values),
        "parameters": {"length": length},
    }


@pytest.fixture
def link_data():
    return {
        "0-1": _link([1, 2, 3], length=5),
        "1-0": _link([3, 4, 5], length=5),
        "1-2": _link([10, 20, 30], length=7),
        "2-1": _link([30, 40, 50], length=7),
        "2-3": _link([100, 200, 300], length=11),
    }


def test_path_metric_forward_reverse_mean_and_max(link_data):
    path = ["0", "1", "2"]

    forward = network_dashboard.compute_path_metric(link_data, path, "link_flow", "forward")
    reverse = network_dashboard.compute_path_metric(link_data, path, "link_flow", "reverse")
    mean = network_dashboard.compute_path_metric(link_data, path, "link_flow", "mean")
    max_values = network_dashboard.compute_path_metric(link_data, path, "link_flow", "max")

    np.testing.assert_allclose(forward, [5.5, 11.0, 16.5])
    np.testing.assert_allclose(reverse, [16.5, 22.0, 27.5])
    np.testing.assert_allclose(mean, [11.0, 16.5, 22.0])
    np.testing.assert_allclose(max_values, [16.5, 22.0, 27.5])


def test_map_density_mean_mode_combines_bidirectional_values(link_data):
    values = network_dashboard.combine_map_series(link_data, "0", "1", "density", "mean")

    np.testing.assert_allclose(values, [4, 6, 8])


def test_network_metric_uses_unique_pairs_for_mean_mode(link_data):
    values = network_dashboard.compute_network_metric(link_data, "link_flow", "mean")

    np.testing.assert_allclose(values, [40.66666667, 77.66666667, 114.66666667])


def test_moving_average_alignment():
    values = network_dashboard.moving_average(np.array([1, 2, 3, 4], dtype=float), 2)
    time_axis = network_dashboard.smoothed_time_axis(len(values), unit_time=10, window_size=2)

    np.testing.assert_allclose(values, [1.5, 2.5, 3.5])
    np.testing.assert_allclose(time_axis, [5.0, 15.0, 25.0])


def test_missing_reverse_link_falls_back_to_forward(link_data):
    values = network_dashboard.compute_path_metric(link_data, ["2", "3"], "speed", "reverse")

    np.testing.assert_allclose(values, [100, 200, 300])


def test_invalid_manual_path_raises():
    with pytest.raises(ValueError, match="at least two"):
        network_dashboard.parse_node_path("7")


def test_path_matrix_returns_lengths_and_link_rows(link_data):
    matrix, distance_edges = network_dashboard.compute_path_matrix(
        link_data, ["0", "1", "2"], "travel_time", "mean"
    )

    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(distance_edges, [0, 5, 12])
