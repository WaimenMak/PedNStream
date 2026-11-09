"""Unit tests for pednstream.utils.config."""

import copy
import textwrap

import numpy as np
import pytest

from pednstream.exceptions import InvalidConfigError, RequiredConfigError
from pednstream.utils.config import (
    DemandParams,
    PathFinderParams,
    ScenarioParams,
    build_configs,
    read_scenario,
    validate_config,
)

# Small network: 0 - 1 - 2 - 3, plus 1 - 4
ADJ = np.zeros((5, 5), dtype=int)
for i, j in [(0, 1), (1, 2), (2, 3), (1, 4)]:
    ADJ[i, j] = ADJ[j, i] = 1

DEFAULT_LINK = {
    "length": 50,
    "width": 2,
    "free_flow_speed": 1.1,
    "k_critical": 2,
    "k_jam": 6,
}

BASE_CONFIG = {
    "network": {"origin_nodes": [0, 2], "destination_nodes": [3]},
    "simulation": {"simulation_steps": 10, "unit_time": 1.0},
    "default_link": DEFAULT_LINK,
}

SCENARIO_YAML = """
network:
  origin_nodes: [0, 2]
  destination_nodes: [3]
simulation:
  simulation_steps: 10
  unit_time: 1.0
  seed: 42
  path_finder:
    k_paths: 3
default_link:
  length: 50
  width: 2
  free_flow_speed: 1.1
  k_critical: 2
  k_jam: 6
links:
  "1_2":
    length: 120
demand:
  origin_0:
    peak_lambda: 10
controllers:
  nodes: [1]
  links: ["2_3"]
od_flows_csv: my_od.csv
"""

OD_CSV = "origin,destination,weight\n0,3,10.5\n2,3,0\n9,3,1\n"


def make_params(**overrides) -> ScenarioParams:
    """ScenarioParams for the 5-node test network."""
    fields = dict(
        origin_nodes=(0, 2),
        destination_nodes=(3,),
        simulation_steps=10,
        unit_time=1.0,
        default_link=DEFAULT_LINK,
        link_overrides={},
        demand={},
        controller_nodes=frozenset(),
        controller_links=frozenset(),
        seed=42,
        path_finder=PathFinderParams(k_paths=3),
        adjacency_matrix=ADJ,
        od_flows={},
    )
    fields.update(overrides)
    return ScenarioParams(**fields)


def test_validate_config_accepts_valid_config():
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["od_flows"] = {"0_3": 10, "2_3": 5}
    cfg["demand"] = {"origin_0": {"peak_lambda": 10}}
    cfg["links"] = {"1_2": {"length": 120}}

    validate_config(cfg, adjacency_matrix=ADJ)


def _set(path, value):
    """Return a mutator that sets cfg[path[0]][path[1]]... = value."""
    def mutate(cfg):
        target = cfg
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    return mutate


def _drop(path):
    def mutate(cfg):
        target = cfg
        for key in path[:-1]:
            target = target[key]
        target.pop(path[-1])
    return mutate


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        pytest.param(_drop(["network"]), RequiredConfigError, "section in configuration: network", id="missing-section"),
        pytest.param(_drop(["default_link", "k_jam"]), RequiredConfigError, "k_jam in section default_link", id="missing-field"),
        pytest.param(_set(["od_flows"], {"9_3": 1}), InvalidConfigError, "origin 9", id="od-unknown-origin"),
        pytest.param(_set(["od_flows"], {"0_9": 1}), InvalidConfigError, "destination 9", id="od-unknown-destination"),
        pytest.param(_set(["od_flows"], {"0-3": 1}), InvalidConfigError, "Invalid od_flows key", id="od-malformed-key"),
        pytest.param(
            lambda c: (_set(["network", "destination_nodes"], [])(c), _set(["od_flows"], {"0_3": 1})(c)),
            InvalidConfigError, "destination_nodes", id="od-without-destinations",
        ),
        pytest.param(
            lambda c: (_set(["od_flows"], {"0_3": 1})(c), _set(["od_flows_csv"], "od.csv")(c)),
            InvalidConfigError, "pick one", id="od-and-csv",
        ),
        pytest.param(
            lambda c: (_drop(["network", "destination_nodes"])(c), _set(["od_flows_csv"], "od.csv")(c)),
            InvalidConfigError, "destination_nodes", id="csv-without-destinations",
        ),
        pytest.param(_set(["demand"], {"node_0": {}}), InvalidConfigError, "Invalid demand key", id="demand-bad-prefix"),
        pytest.param(_set(["demand"], {"origin_x": {}}), InvalidConfigError, "Invalid demand key", id="demand-non-int"),
        pytest.param(_set(["demand"], {"origin_9": {}}), InvalidConfigError, "origin 9", id="demand-unknown-origin"),
        pytest.param(_set(["links"], {"0_2": {}}), InvalidConfigError, "no edge exists", id="link-missing-edge"),
        pytest.param(_set(["links"], {"0_99": {}}), InvalidConfigError, "out of bounds", id="link-out-of-bounds"),
    ],
)
def test_validate_config_rejects_invalid_config(mutate, exc, match):
    cfg = copy.deepcopy(BASE_CONFIG)
    mutate(cfg)

    with pytest.raises(exc, match=match):
        validate_config(cfg, adjacency_matrix=ADJ)


@pytest.mark.parametrize("adjacency_source", ["npy_file", "inline_yaml"])
def test_read_scenario_normalizes_config(tmp_path, adjacency_source):
    yaml_text = SCENARIO_YAML
    if adjacency_source == "npy_file":
        np.save(tmp_path / "adj_matrix.npy", ADJ)
    else:
        yaml_text = yaml_text.replace(
            "  destination_nodes: [3]\n",
            f"  destination_nodes: [3]\n  adjacency_matrix: {ADJ.tolist()}\n",
        )
    (tmp_path / "my_od.csv").write_text(OD_CSV)
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text(textwrap.dedent(yaml_text))

    params = read_scenario(cfg_path)

    np.testing.assert_array_equal(params.adjacency_matrix, ADJ)
    assert params.origin_nodes == (0, 2)
    assert params.destination_nodes == (3,)
    assert params.simulation_steps == 10
    assert params.seed == 42
    assert params.assign_flows_type == "classic"
    assert params.path_finder.k_paths == 3
    assert params.link_overrides == {(1, 2): {"length": 120}}
    assert isinstance(params.demand[0], DemandParams)
    assert params.demand[0].peak_lambda == 10
    assert params.controller_nodes == frozenset({1})
    assert params.controller_links == frozenset({(2, 3)})
    # Row with unknown origin 9 is filtered out, zero weight is kept
    assert params.od_flows == {(0, 3): 10.5, (2, 3): 0.0}


@pytest.mark.parametrize(
    "network_extra, match",
    [
        pytest.param("  adjacency_matrix: [[0, 1], [1, 0]]\n", "pick one", id="inline-and-file-adjacency"),
        pytest.param("", "no edge exists", id="link-not-in-network"),
    ],
)
def test_read_scenario_rejects_inconsistent_config(tmp_path, network_extra, match):
    np.save(tmp_path / "adj_matrix.npy", ADJ)
    cfg_path = tmp_path / "sim.yaml"
    cfg_path.write_text(
        "network:\n"
        "  origin_nodes: [0]\n"
        + network_extra
        + textwrap.dedent("""
        simulation:
          simulation_steps: 10
          unit_time: 1.0
        default_link:
          length: 50
          width: 2
          free_flow_speed: 1.1
          k_critical: 2
          k_jam: 6
        links:
          "0_2":
            length: 120
        """)
    )

    with pytest.raises(InvalidConfigError, match=match):
        read_scenario(cfg_path)


@pytest.mark.parametrize(
    "csv_text, exc",
    [
        pytest.param(None, FileNotFoundError, id="missing-file"),
        pytest.param("from,to,value\n0,3,1\n", InvalidConfigError, id="missing-columns"),
    ],
)
def test_od_flows_from_file_rejects_bad_input(tmp_path, csv_text, exc):
    csv_path = tmp_path / "od_flows.csv"
    if csv_text is not None:
        csv_path.write_text(csv_text)

    with pytest.raises(exc):
        make_params().od_flows_from_file(csv_path)


def test_build_configs_creates_links_and_nodes():
    params = make_params(
        link_overrides={(1, 2): {"length": 120, "front_gate_width": 0.5}},
        controller_links=frozenset({(2, 3)}),
        demand={0: DemandParams(peak_lambda=10)},
    )

    network_config, simulation_config = build_configs(params)

    links = {link.link_id: link for link in network_config.links}
    assert set(links) == {
        (0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (1, 4), (4, 1),
    }
    # Overrides apply in both directions; gate widths swap on the reverse link
    assert links[(1, 2)].length == links[(2, 1)].length == 120
    assert links[(0, 1)].length == 50
    assert (links[(1, 2)].front_gate_width, links[(1, 2)].back_gate_width) == (0.5, 2)
    assert (links[(2, 1)].front_gate_width, links[(2, 1)].back_gate_width) == (2, 0.5)
    assert {lid for lid, link in links.items() if link.is_controller} == {(2, 3), (3, 2)}

    # Degree-1 and plain degree-2 nodes are one-to-one; OD degree-2 nodes and hubs are regular
    node_types = {node.id: node.type for node in network_config.nodes}
    assert node_types == {
        0: "onetoone", 1: "regular", 2: "regular", 3: "onetoone", 4: "onetoone",
    }
    assert network_config.origin_nodes == (0, 2)
    assert network_config.destination_nodes == (3,)

    assert simulation_config.simulation_steps == 10
    assert simulation_config.unit_time == 1.0
    assert simulation_config.seed == 42
    assert simulation_config.assign_flows_type == "classic"
    assert simulation_config.path_finder == {"k_paths": 3}
    assert simulation_config.demand_params is params.demand
    assert simulation_config.od_flows == {}


def test_build_configs_passes_od_flows():
    params = make_params(od_flows={(0, 3): 10.5, (2, 3): 0.0})

    _, simulation_config = build_configs(params)

    assert simulation_config.od_flows == {(0, 3): 10.5, (2, 3): 0.0}
