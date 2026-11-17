"""OD-labelled flow tracking (simulation.od_tracking).

Uses a 7-node network where two origins share a corridor (2->4) on routes to
different destinations, which is where the weight-based P(od | upstream) leaks.
"""

from pathlib import Path

import numpy as np
import pytest

from pednstream.utils.network_env_generator import NetworkEnvGenerator

DATA = Path(__file__).parent / "data" / "od_tracking_station"
QUIET_B = {"origin_0": {"peak_lambda": 10}, "origin_1": {"base_lambda": 0, "peak_lambda": 0}}


def _run(od_flows, tracking, demand=None):
    gen = NetworkEnvGenerator()
    gen.network_data = gen.load_network_data(str(DATA))
    gen.config["params"]["od_tracking"] = tracking
    net = gen.create_network(
        str(DATA), od_flows=od_flows, demand_params_overrides=demand or {}, verbose=False
    )
    T = gen.config["params"]["simulation_steps"]
    for t in range(1, T):
        net.network_loading(t)
    return net, T


def _share_to(net, T, dest, other):
    arr = lambda d: sum(
        l.cumulative_outflow[T - 1]
        for l in net.nodes[d].incoming_links
        if not str(l.link_id).startswith("virtual")
    )
    a, b = arr(dest), arr(other)
    return a / (a + b)


def test_tracking_is_off_by_default():
    net, _ = _run({(0, 5): 1, (0, 6): 1, (1, 5): 1, (1, 6): 1}, tracking=False)
    assert net.od_tracker is None


def test_empty_origin_no_longer_steers_other_people():
    # Gate A sends everyone to 5. Gate B has no demand but weights 1:1.
    od = {(0, 5): 1, (0, 6): 0, (1, 5): 1, (1, 6): 1}
    net_off, T = _run(od, tracking=False, demand=QUIET_B)
    net_on, _ = _run(od, tracking=True, demand=QUIET_B)
    assert _share_to(net_off, T, 5, 6) < 0.8  # documents the leak without tracking
    assert _share_to(net_on, T, 5, 6) == pytest.approx(1.0)


@pytest.mark.parametrize("w5,w6", [(3, 1), (1, 3), (1, 1)])
def test_realised_split_matches_od_weights(w5, w6):
    od = {(0, 5): w5, (0, 6): w6, (1, 5): 1, (1, 6): 1}
    net, T = _run(od, tracking=True, demand=QUIET_B)
    assert _share_to(net, T, 5, 6) == pytest.approx(w5 / (w5 + w6), abs=0.01)


def test_labels_are_conserved():
    od = {(0, 5): 1, (0, 6): 0, (1, 5): 0, (1, 6): 1}  # crossing pattern, both gates busy
    net, T = _run(od, tracking=True)
    tr = net.od_tracker
    # labelled people on every link equal the link's own count
    for key, link in net.links.items():
        assert tr.counts[key].sum() == pytest.approx(link.num_pedestrians[T - 1], abs=1e-6)
    assert tr.unlabelled[T - 1] == 0
    # every OD pair: entered = arrived + still inside, and people only arrive at their own destination
    table = tr.od_table()
    for od_pair, row in table.items():
        assert row["entered"] == pytest.approx(row["arrived"] + row["inside"], abs=1e-6)
    for d in (5, 6):
        arrived_total = sum(
            l.cumulative_outflow[T - 1]
            for l in net.nodes[d].incoming_links
            if not str(l.link_id).startswith("virtual")
        )
        arrived_labelled = sum(row["arrived"] for (o, dd), row in table.items() if dd == d)
        assert arrived_labelled == pytest.approx(arrived_total, abs=1e-6)
    assert table[(0, 6)]["entered"] == 0 and table[(1, 5)]["entered"] == 0
