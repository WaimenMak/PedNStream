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


def _hand_built_config(adjacency_matrix, link_pairs, origin_nodes, destination_nodes):
    from pednstream.ltm.link import LinkConfig
    from pednstream.ltm.network import NetworkConfig
    from pednstream.ltm.node import NodeConfig

    links = [
        LinkConfig(link_id=(i, j), start_node=i, end_node=j, unit_time=1.0, length=10.0,
                   width=2.0, free_flow_speed=1.0, k_critical=1.0, k_jam=5.0)
        for i, j in link_pairs
    ]
    nodes = [NodeConfig(id=n) for n in range(len(adjacency_matrix))]
    return NetworkConfig(
        adjacency_matrix=np.array(adjacency_matrix), links=links, nodes=nodes,
        origin_nodes=origin_nodes, destination_nodes=destination_nodes,
    )


def test_pass_through_node_has_no_virtual_links():
    # 0 ⇄ 1 ⇄ 2: node 1 is a pass-through (two neighbours), so no virtual links.
    config = _hand_built_config(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        origin_nodes=[0], destination_nodes=[2],
    )
    network = Network(config)
    network.create()
    with_virtual = {nid for nid, n in network.nodes.items() if n.virtual_incoming_link is not None}
    assert with_virtual == {0, 2}


def test_non_od_dead_end_gets_virtual_links():
    # 0 ⇄ 1 ⇄ 2: node 0 is a dead end even though it is not an origin/destination.
    config = _hand_built_config(
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [(0, 1), (1, 0), (1, 2), (2, 1)],
        origin_nodes=[2], destination_nodes=[2],
    )
    network = Network(config)
    network.create()
    assert network.nodes[0].virtual_incoming_link is not None
    assert network.nodes[1].virtual_incoming_link is None


def test_one_way_network_is_rejected():
    # 0 → 1 → 2 has no reverse links, which the node model cannot simulate.
    config = _hand_built_config(
        [[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        [(0, 1), (1, 2)],
        origin_nodes=[0], destination_nodes=[2],
    )
    with pytest.raises(ValueError, match="symmetric"):
        Network(config).create()


def test_discovery_order_follows_reference():
    # 0 - 1 - 2 - 3, plus 1 - 4: node 4 is discovered from node 1, before node 3.
    assert Network._discovery_order(ADJ) == [0, 1, 2, 4, 3]


def test_network_loading_requires_allocation(network):
    with pytest.raises(RuntimeError, match="allocate"):
        network.network_loading(1)
    network.allocate(10)
    with pytest.raises(ValueError):
        network.network_loading(11)


def _conserved(network, step):
    nodes = network.nodes.values()
    entered = sum(n.virtual_incoming_link.cumulative_outflow[step] for n in nodes if n.virtual_incoming_link)
    left = sum(n.virtual_outgoing_link.cumulative_inflow[step] for n in nodes if n.virtual_outgoing_link)
    on_links = sum(link.num_pedestrians[step] for link in network.links.values())
    return entered, left, on_links


def test_network_loading_without_path_finder(network):
    steps = 300  # a 50 m link takes ~45 steps at free flow
    np.random.seed(0)
    network.allocate(steps)
    network.set_demand(0, np.r_[np.full(30, 2.0), np.zeros(steps - 29)])
    for t in range(1, steps):
        network.network_loading(t)

    entered, left, on_links = _conserved(network, steps - 1)
    assert entered > 0 and left > 0
    assert entered - left == pytest.approx(on_links)
    for link in network.links.values():
        assert link.num_pedestrians.min() >= 0
        assert link.sending_flow[: steps - 1].min() >= 0  # every node assigns flows


def test_network_loading_with_path_finder(network):
    from pednstream.ltm import ODManager, PathFinder

    steps = 300  # a 50 m link takes ~45 steps at free flow
    np.random.seed(0)
    network.allocate(steps)
    for origin in (0, 2):
        network.set_demand(origin, np.r_[np.full(30, 2.0), np.zeros(steps - 29)])
    od_manager = ODManager(steps)
    od_manager.init_od_flows([0, 2], [3])
    path_finder = PathFinder(network.links, params={"path_finder": {}})
    path_finder.find_od_paths(od_pairs=od_manager.od_flows.keys(), nodes=network.nodes)
    for t in range(1, steps):
        network.network_loading(t, path_finder=path_finder, od_manager=od_manager)

    entered, left, on_links = _conserved(network, steps - 1)
    assert left > 0  # pedestrians reach destination 3
    assert entered - left == pytest.approx(on_links)
    assert network.nodes[3].virtual_outgoing_link.cumulative_inflow[steps - 1] == left
