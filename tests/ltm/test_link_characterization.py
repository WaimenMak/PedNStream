"""Characterization test for the link flow model, against the reference model.

The reference is ``src/pednstream/ltm/link.py`` at commit 92fa2a3 (2026-04-18), the last
version of the flow model before the LinkConfig restructure. The golden file is produced
by running that reference code (see scripts/generate_link_golden.py) on the scenarios
below, and this test runs the *current* code on the same scenarios. Results must match
exactly, including the order of random draws.

Both models are driven by the same schedule. Only the call convention differs:
the reference methods take the previous step ``t - 1`` (as the node passed it), the
current ones take the current step ``t`` (see the adapters below).

Regenerate the golden file only when the reference itself changes:
    venv/bin/python scripts/generate_link_golden.py
"""

from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parents[1] / "data" / "golden" / "link_characterization.npz"
REFERENCE_COMMIT = "92fa2a3"
STEPS = 200
SEED = 0
GEOMETRY = dict(length=20.0, width=3.0, free_flow_speed=1.2, k_critical=2.0, k_jam=6.0)

# Compared time series. front_gate_width_data is left out on purpose: the reference never
# updated it (only back_gate_width_data), the current code also records the front gate
# for visualization. It is output only and does not feed back into the model.
ARRAYS = (
    "inflow", "outflow", "cumulative_inflow", "cumulative_outflow",
    "sending_flow", "receiving_flow", "num_pedestrians", "density", "speed",
    "link_flow", "travel_time", "avg_travel_time",
    "back_gate_width_data", "separator_width_data",
)


def _set_front_gate(t, forward, reverse):
    if t == 60:
        forward.front_gate_width = 0.8


def _set_separator(t, forward, reverse):
    if t == 60:
        forward.separator_width = 1.0  # reverse link gets width - 1.0


# name: (kind, front gate, back gate, activity probability, runtime event)
SCENARIOS = {
    "regular_symmetric": ("regular", None, None, 0.0, None),
    "regular_asymmetric": ("regular", 1.5, 3.0, 0.0, _set_front_gate),
    "regular_activity": ("regular", None, None, 0.1, None),
    "separator": ("separator", None, None, 0.0, _set_separator),
}


class CurrentModel:
    """Adapter for pednstream.ltm.link (methods take the current step t)."""

    @staticmethod
    def pair(kind, front, back, activity, steps):
        from pednstream.ltm.link import LinkConfig, Regular, Separator

        cls = Separator if kind == "separator" else Regular
        common = dict(unit_time=1, activity_probability=activity, **GEOMETRY)
        forward = cls(LinkConfig(link_id=(0, 1), start_node=0, end_node=1,
                                 front_gate_width=front, back_gate_width=back, **common))
        # Reverse link: gates swapped, as in config._build_link_configs
        reverse = cls(LinkConfig(link_id=(1, 0), start_node=1, end_node=0,
                                 front_gate_width=back, back_gate_width=front, **common))
        for link in (forward, reverse):
            link.allocate_state(steps)
        return forward, reverse

    @staticmethod
    def send(link, t):
        link.calculate_sending_flow(t)

    @staticmethod
    def receive(link, t):
        return link.calculate_receiving_flow(t, with_reverse=True)

    @staticmethod
    def update_density(link, t):
        link.update_density_flow(t)


def run_scenario(model, kind, front, back, activity, event, steps=STEPS, seed=SEED):
    """Run a pair of reverse links through a fixed demand schedule; return all time series."""
    forward, reverse = model.pair(kind, front, back, activity, steps)
    forward.reverse_link, reverse.reverse_link = reverse, forward
    links = {"fwd": (forward, 6), "rev": (reverse, 2)}  # name: (link, base demand)

    np.random.seed(seed)
    for t in range(1, steps + 1):
        if event is not None:
            event(t, forward, reverse)
        # Sending flows of all links first, then receiving flows (they read the reverse
        # link's sending flow), as the node model does within one step.
        for link, _ in links.values():
            model.send(link, t)
        for link, _ in links.values():
            link.receiving_flow[t - 1] = model.receive(link, t)

        for link, base in links.values():
            demand = base + (t % 7) if t < 0.6 * steps else 0
            link.update_cum_inflow(min(demand, link.receiving_flow[t - 1]), t)
            link.update_cum_outflow(link.sending_flow[t - 1], t)

        for link, _ in links.values():
            model.update_density(link, t)
        for link, _ in links.values():
            link.update_speeds(t)

    return {
        f"{name}.{attr}": np.asarray(getattr(link, attr)).copy()
        for name, (link, _) in links.items()
        for attr in ARRAYS
        if hasattr(link, attr)
    }


def all_results(model) -> dict[str, np.ndarray]:
    return {
        f"{scenario}.{key}": value
        for scenario, args in SCENARIOS.items()
        for key, value in run_scenario(model, *args).items()
    }


def _comparable(key: str, values: np.ndarray) -> np.ndarray:
    # The reference filled a Separator's back_gate_width_data before halving its gates, so
    # index 0 holds the full link width there. Recording only, never read by the model.
    if key.startswith("separator.") and key.endswith(".back_gate_width_data"):
        return values[1:]
    return values


def test_link_model_matches_reference():
    results = all_results(CurrentModel)
    golden = np.load(GOLDEN)
    assert sorted(golden.files) == sorted(results)
    for key in golden.files:
        np.testing.assert_array_equal(
            _comparable(key, results[key]), _comparable(key, golden[key]), err_msg=key
        )


def test_scenarios_exercise_the_model():
    """The schedule reaches the interesting regimes, so the golden file is not trivial."""
    golden = np.load(GOLDEN)
    for scenario in SCENARIOS:
        assert golden[f"{scenario}.fwd.sending_flow"].max() > 0, scenario
        avg_tt = golden[f"{scenario}.fwd.avg_travel_time"]
        assert not np.all(avg_tt == avg_tt[0]), f"{scenario}: moving-average window not reached"
    assert max(golden[f"{s}.fwd.density"].max() for s in SCENARIOS) > GEOMETRY["k_critical"]
    # runtime events took effect
    assert golden["separator.fwd.separator_width_data"][-1] == 1.0
    assert golden["separator.rev.separator_width_data"][-1] == GEOMETRY["width"] - 1.0
