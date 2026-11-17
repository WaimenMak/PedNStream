"""OD-labelled flow tracking for the LTM ("multi-commodity" bookkeeping).

The node model keeps working on *total* flows (sending/receiving flow, capacity,
counter-flow). This module only labels those totals by OD pair:

* every physical link keeps ``counts[link][k]``: people on the link belonging to OD pair k;
* at each node the people that leave an incoming link carry that link's mix
  (fully-mixed assumption) and turn according to their *own* OD turn probabilities;
* origins label new people with the OD weights P(d | o); destinations remove them.

The tracked mix replaces the weight-based estimate of P(od | upstream link) in
``PathFinder.update_turning_fractions``. Without tracking that estimate uses the raw
OD weights of every pair whose routes pass the link, so (a) origins with no demand
still steer other people, and (b) realised OD totals drift from the weights.
"""

from __future__ import annotations

import numpy as np

from .node import OneToOneNode


class ODTracker:
    def __init__(self, network):
        self.network = network
        self.od_pairs = list(network.path_finder.od_paths.keys())
        self.index = {od: k for k, od in enumerate(self.od_pairs)}
        self.K = len(self.od_pairs)
        T = network.simulation_steps

        # people currently on each physical link, per OD pair
        self.counts = {key: np.zeros(self.K) for key in network.links}
        # cumulative people entered at origins / arrived at destinations, per OD pair
        self.entered = np.zeros((T + 1, self.K))
        self.arrived = np.zeros((T + 1, self.K))
        # flow that could not be labelled (should stay ~0; useful as a sanity check)
        self.unlabelled = np.zeros(T + 1)

        self._delta = {key: np.zeros(self.K) for key in network.links}
        self._entered_step = np.zeros(self.K)
        self._arrived_step = np.zeros(self.K)
        self._unlabelled_step = 0.0

        # OD pairs starting at each origin (for labelling new people)
        self._od_by_origin = {}
        for k, (o, _d) in enumerate(self.od_pairs):
            self._od_by_origin.setdefault(o, []).append(k)

    # ----------------------------------------------------------- compositions
    # All compositions are unnormalised: every place that uses one divides by a
    # sum afterwards, so the scale cancels.
    def link_counts(self, up_node: int, node_id: int):
        """People on link up_node -> node_id per OD pair (raw), or None if empty or virtual."""
        c = self.counts.get((up_node, node_id))
        if c is None or c.sum() <= 1e-9:
            return None
        return c

    def origin_weights(self, origin: int, time_step: int) -> np.ndarray:
        """Unnormalised P(od | origin) for new people, from the OD weights at this time step."""
        w = np.zeros(self.K)
        ks = self._od_by_origin.get(origin, [])
        if not ks:
            return w
        od_manager = self.network.od_manager
        vals = np.array([max(od_manager.get_od_flow(*self.od_pairs[k], time_step), 0.0) for k in ks])
        w[ks] = vals if vals.sum() > 0 else 1.0  # all weights 0: split equally
        return w

    def source_weights(self, node, up_node: int, time_step: int):
        """Unnormalised OD composition of the people leaving up_node into node."""
        if up_node == -1:
            return self.origin_weights(node.node_id, time_step)
        return self.link_counts(up_node, node.node_id)

    def up_od_probs(self, node, up_node: int, od_pairs) -> dict | None:
        """Tracked P(od | up) for the OD pairs listed at this node, or None to fall back."""
        c = self.link_counts(up_node, node.node_id)
        if c is None:
            return None
        probs = {od: c[self.index[od]] for od in od_pairs if od in self.index}
        total = sum(probs.values())
        if total <= 1e-12:
            return None
        return {od: v / total for od, v in probs.items()}

    # --------------------------------------------------------------- transfer
    def _turn_probs(self, node, up: int, down: int) -> np.ndarray | None:
        """P(down | up, od) for every OD pair, or None where the node has no route choice."""
        if not node.node_turn_probs:
            return None
        p = np.zeros(self.K)
        for od, turns in node.node_turn_probs.items():
            k = self.index.get(od)
            if k is not None:
                p[k] = turns.get((up, down), 0.0)
        return p

    def record_node(self, node, time_step: int):
        """Label the flows the node just moved (call right after node.assign_flows)."""
        edges = getattr(node, "edge_flows", None)
        if edges is None:
            return
        for k_in, j_out, g in edges:
            if g <= 0:
                continue
            in_link = node.incoming_links[k_in]
            out_link = node.outgoing_links[j_out]
            up = in_link.start_node.node_id if in_link.start_node is not None else -1
            down = out_link.end_node.node_id if out_link.end_node is not None else -1

            w = self.source_weights(node, up, time_step)
            if w is None or w.sum() <= 0:
                self._unlabelled_step += g
                continue

            # split g over OD pairs in proportion to composition * P(down | up, od)
            p = self._turn_probs(node, up, down)
            weights = w * p if p is not None else w
            if weights.sum() <= 1e-12:  # no tracked OD wants this turn: keep the composition
                weights = w
            moved = g * weights / weights.sum()

            if up == -1:
                self._entered_step += moved
            else:
                self._delta[(up, node.node_id)] -= moved
            if down == -1:
                self._arrived_step += moved
            else:
                self._delta[(node.node_id, down)] += moved

    def commit(self, time_step: int):
        """Apply this step's labelled moves (call after all nodes have assigned flows)."""
        for key, d in self._delta.items():
            if d.any():
                c = self.counts[key]
                c += d
                np.maximum(c, 0.0, out=c)  # guard tiny negative round-off
                d[:] = 0.0
        self.entered[time_step] = self.entered[time_step - 1] + self._entered_step
        self.arrived[time_step] = self.arrived[time_step - 1] + self._arrived_step
        self.unlabelled[time_step] = self.unlabelled[time_step - 1] + self._unlabelled_step
        self._entered_step[:] = 0.0
        self._arrived_step[:] = 0.0
        self._unlabelled_step = 0.0

    # ---------------------------------------------------------------- results
    def od_table(self, time_step: int | None = None) -> dict:
        """{od: {"entered", "arrived", "inside"}} at a time step (default: last)."""
        t = self.network.simulation_steps - 1 if time_step is None else time_step
        inside = sum(self.counts.values()) if time_step is None else None
        out = {}
        for k, od in enumerate(self.od_pairs):
            row = {"entered": float(self.entered[t, k]), "arrived": float(self.arrived[t, k])}
            if inside is not None:
                row["inside"] = float(inside[k])
            out[od] = row
        return out


def edge_flows_one_to_one(node: OneToOneNode):
    """(incoming index, outgoing index, flow) for a OneToOneNode after solve()."""
    q = node.q
    # q = [S0 -> R1, S1 -> R0, ...]: incoming 0 continues to outgoing 1 and vice versa
    return [(0, 1, float(q[0])), (1, 0, float(q[1]))]
