"""Flow model of a physical link: sending and receiving flows.

Plain functions that take the link and compute its flows from its state, in the same
style as ``solver.py`` for nodes. ``Link.calculate_sending_flow`` and
``Link.calculate_receiving_flow`` delegate here.

Time-step convention: functions take the current step ``t``. Flows are computed from the
state at the previous step ``t - 1`` and stored at index ``t - 1``.

The order of random draws matters for reproducibility (seeded runs); keep it unchanged.
"""

import numpy as np

# Diffusion (platoon dispersion) outflow: number of past inflow steps that are dispersed.
DIFFUSION_LAYERS = 4

# Release mitigation: probability that a pedestrian able to leave actually leaves,
# growing from RELEASE_PROB_MIN (empty link) to RELEASE_PROB_MAX (jam density).
RELEASE_PROB_MIN = 0.7
RELEASE_PROB_MAX = 0.85
RELEASE_EXPONENT = 0.8

# Weight of the diffusion flow when a platoon arrives (rest: boundary flow).
PLATOON_WEIGHT = 0.8

# Smoothing with the previous step's flow.
SMOOTHING_WEIGHT = 0.8
SMOOTHING_PREVIOUS_WEIGHT = 0.2

# Share of the reverse link's pedestrians that occupy space in the receiving link.
REVERSE_OCCUPANCY_PROB = 0.9


def gate_capacity(link, gate_width: float) -> float:
    """Maximum number of pedestrians that can pass a gate in one time step."""
    return gate_width * link.k_critical * link.free_flow_speed * link.unit_time


def sending_flow(link, time_step: int) -> float:
    """Compute the sending flow of a link, store it at ``time_step - 1`` and return it.

    Steps: boundary flow (free-flow/congestion mix, capped by the front gate), release
    mitigation, pedestrians staying for an activity, smoothing with the previous step.

    Args:
        link: the physical link (Regular or Separator).
        time_step (int): current time step.

    Returns:
        float: sending flow.
    """
    previous_step = time_step - 1
    density = link.get_density(previous_step)
    tau = round(link.avg_travel_time[previous_step] / link.unit_time)

    # Initial stage: nobody can have crossed the link yet.
    if previous_step <= link.free_flow_tau:
        link.sending_flow[previous_step] = 0
        return link.sending_flow[previous_step]

    flow = _boundary_sending_flow(link, time_step, tau)
    original_flow = flow
    if flow > 0:
        flow = _mitigate_release(link, flow, density, time_step, tau)
    flow = _apply_activity_stay(link, flow)
    flow = _smooth(max(0, flow), link.sending_flow[previous_step - 1], cap=original_flow)
    if flow < 0:
        raise ValueError(
            "Negative sending flow detected, sending flow more than original flow"
        )

    link.sending_flow[previous_step] = flow
    return link.sending_flow[previous_step]


def receiving_flow(
    link,
    time_step: int,
    reverse_occupancy: float = 0,
    reverse_sending_flow: float = 0,
    with_reverse: bool = False,
) -> float:
    """Compute the receiving flow of a link at a time step.

    Args:
        link: the physical link (Regular or Separator).
        time_step (int): current time step.
        reverse_occupancy (float): pedestrians of the reverse link occupying space in this one.
        reverse_sending_flow (float): sending flow of the reverse link, subtracted when
            ``with_reverse`` is True.
        with_reverse (bool): whether to account for the reverse link's sending flow.

    Returns:
        float: receiving flow.
    """
    previous_step = time_step - 1
    # TODO: is using length the correct way to calculate receiving flow?
    tau_shockwave = round(link.length / (link.shockwave_speed * link.unit_time))
    storage = link.k_jam * link.area

    if time_step - tau_shockwave < 0:
        boundary = storage - reverse_occupancy
    else:
        boundary = (
            link.cumulative_outflow[time_step - tau_shockwave]
            + storage
            - reverse_occupancy
            - link.cumulative_inflow[previous_step]
        )

    flow = min(boundary, gate_capacity(link, link.back_gate_width))
    if flow < 0:
        print(f"Negative receiving flow detected, {flow} at {time_step}")
    flow = max(flow, 0)

    previous_flow = link.receiving_flow[previous_step - 1]
    if previous_flow >= 0:  # -1 means not computed yet
        flow = _smooth(flow, previous_flow, cap=flow)

    if with_reverse:
        return max(flow - reverse_sending_flow, 0)
    return flow


def _boundary_sending_flow(link, time_step: int, tau: int) -> float:
    """Mix of the free-flow and congestion boundaries, capped by the front gate."""
    previous_step = time_step - 1
    idx = min(max(0, time_step - tau), len(link.cumulative_inflow) - 1)

    congestion_factor = np.clip(
        (link.density[previous_step] - link.k_critical) / (link.k_jam - link.k_critical),
        0,
        1,
    )  # this one is theoratically more realistic for unidirectional condition
    boundary_congestion = link.num_pedestrians[previous_step]
    boundary_freeflow = max(
        0, link.cumulative_inflow[idx] - link.cumulative_outflow[previous_step]
    )
    boundary = (
        congestion_factor * boundary_congestion
        + (1 - congestion_factor) * boundary_freeflow
    )
    return min(boundary, gate_capacity(link, link.front_gate_width))


def _diffusion_outflow(link, time_step: int, tau: int) -> float:
    """Outflow of past inflows dispersed along the link (platoon dispersion)."""
    travel_time = link.avg_travel_time[time_step - 1]
    F = 1 / (1 + link.gamma * travel_time)
    idx = min(max(0, time_step - tau), len(link.inflow) - 1)

    flow = 0
    for i in range(DIFFUSION_LAYERS):
        flow += F * (1 - F) ** i * link.inflow[max(0, idx - i)]
    return max(np.ceil(flow), 0)


def _mitigate_release(link, flow: float, density: float, time_step: int, tau: int) -> float:
    """Avoid unrealistically high sending flows.

    In free flow, an arriving platoon (diffusion flow > 0) is released gradually; otherwise
    each pedestrian leaves with a density-dependent probability.
    """
    releasing_factor = np.clip(density / link.k_jam, 0, 1)
    releasing_prob = RELEASE_PROB_MIN + (RELEASE_PROB_MAX - RELEASE_PROB_MIN) * releasing_factor**RELEASE_EXPONENT

    if density <= link.k_critical:
        diffusion_flow = _diffusion_outflow(link, time_step, tau)
        if diffusion_flow > 0:
            flow = int(np.floor(min(
                PLATOON_WEIGHT * diffusion_flow + (1 - PLATOON_WEIGHT) * flow,
                flow,
            )))
        else:  # no inflow, so no diffusion flow: avoid releasing the maximum flow
            flow = np.random.binomial(n=int(np.floor(flow)), p=releasing_prob)
    else:
        flow = np.random.binomial(n=int(np.floor(flow)), p=releasing_prob)

    if flow < 0:
        raise ValueError(f"Negative sending flow detected, {flow} at {time_step}")
    return flow


def _apply_activity_stay(link, flow: float) -> float:
    """Pedestrians who stay on the link for an activity, each with activity_probability."""
    if link.activity_probability > 0 and flow > 1:
        flow -= np.random.binomial(n=int(np.floor(flow)), p=link.activity_probability)
    return flow


def _smooth(flow: float, previous_flow: float, cap: float) -> float:
    """Blend a flow with the previous step's flow, never exceeding ``cap``."""
    return min(
        np.floor(SMOOTHING_WEIGHT * flow + SMOOTHING_PREVIOUS_WEIGHT * previous_flow),
        cap,
    )
