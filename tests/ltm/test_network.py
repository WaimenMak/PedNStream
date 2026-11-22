"""Unit tests for the Network class: topology creation is decoupled from the time horizon."""

import numpy as np
import pytest

from pednstream.ltm.network import Network
from pednstream.utils.config import PathFinderParams, ScenarioParams, build_configs

# Small network: 0 - 1 - 2 - 3, plus 1 - 4
ADJ = np.zeros((5, 5), dtype=int)
for i, j in [(0, 1), (1, 2), (2, 3), (1, 4)]:
    ADJ[i, j] = ADJ[j, i] = 1


@pytest.fixture
def network_config():
    params = ScenarioParams(
        origin_nodes=(0, 2),
        destination_nodes=(3,),
        simulation_steps=10,
        unit_time=1.0,
        default_link={"length": 50, "width": 2, "free_flow_speed": 1.1, "k_critical": 2, "k_jam": 6},
        link_overrides={},
        demand={},
        controller_nodes=frozenset(),
        controller_links=frozenset({(2, 3)}),
        path_finder=PathFinderParams(),
        adjacency_matrix=ADJ,
    )
    return build_configs(params)["network"]


@pytest.fixture
def network(network_config):
    network = Network(network_config)
    network.create()
    return network


def test_create_does_not_need_steps(network):
    assert set(network.nodes) == {0, 1, 2, 3, 4}
    assert len(network.links) == 8
    assert not network.is_allocated
    assert all(not link.is_allocated for link in network.links.values())
    assert network.links[(0, 1)].reverse_link is network.links[(1, 0)]


def test_simulation_steps_before_allocate_raises(network):
    with pytest.raises(RuntimeError, match="allocate"):
        network.simulation_steps


def test_virtual_links_only_on_od_and_dead_end_nodes(network):
    # Degree-1 nodes (0, 3, 4) and OD node 2 get virtual links; hub 1 does not.
    with_virtual = {nid for nid, n in network.nodes.items() if n.virtual_incoming_link is not None}
    assert with_virtual == {0, 2, 3, 4}
    node = network.nodes[2]
    assert node.virtual_incoming_link in node.incoming_links
    assert node.virtual_outgoing_link in node.outgoing_links


def test_allocate_sizes_and_resets_state(network):
    network.allocate(10)
    assert network.simulation_steps == 10
    links = list(network._all_links())
    assert all(len(link.inflow) == 11 for link in links)
    assert all(len(network.nodes[n].demand) == 11 for n in (0, 2, 3, 4))

    network.links[(0, 1)].inflow[2] = 7
    network.allocate(20)
    assert network.simulation_steps == 20
    assert all(len(link.inflow) == 21 for link in links)
    assert network.links[(0, 1)].inflow.sum() == 0


def test_set_demand(network):
    with pytest.raises(RuntimeError):
        network.set_demand(0, np.ones(11))

    network.allocate(10)
    network.set_demand(0, np.ones(10))  # DemandGenerator-style length is padded
    assert len(network.nodes[0].demand) == 11
    assert network.nodes[0].demand[-1] == 0

    with pytest.raises(ValueError):
        network.set_demand(0, np.ones(5))
    with pytest.raises(ValueError):
        network.set_demand(1, np.ones(11))  # node without virtual links


def test_allocate_before_create_raises(network_config):
    with pytest.raises(RuntimeError, match="create"):
        Network(network_config).allocate(10)
