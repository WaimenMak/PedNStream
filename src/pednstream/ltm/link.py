"""Representation for links in a transportation network.

Class structure:
    BaseLink          identity, endpoints, flow bookkeeping (inflow/outflow arrays)
    ├── VirtualLink   entry/exit point of origin/destination nodes
    └── Link          physical link: geometry, fundamental diagram, gates, travel time
        ├── Regular   bidirectional link, interacts with its reverse link
        └── Separator controllable link, the corridor width is split with its reverse link

The flow model (sending/receiving flows) lives in link_flow.py.
"""

import numpy as np
from pednstream.ltm import link_flow
from pednstream.utils.functions import BiDirectionalFd, cal_link_flow_kv
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Time-step convention: methods take the current step t. Flows (sending/receiving) are
# computed from the state at t - 1 and stored at index t - 1; cumulative flows, density
# and speed are stored at index t. Index 0 holds the initial state.


@dataclass
class LinkConfig:
    """Configuration dataclass for Link parameters."""

    link_id: tuple[int, int] | str  # (start_node, end_node), e.g. (1, 2); virtual links: "virtual_in_<node>" / "virtual_out_<node>"
    start_node: int | None  # node id; None for the outside end of a virtual link
    end_node: int | None
    unit_time: float
    length: float
    width: float
    free_flow_speed: float
    k_critical: float
    k_jam: float
    is_controller: bool = False
    activity_probability: float = 0.0
    gamma: float = 2e-3 # Defaults to diffusion coefficient
    bi_factor: float = 1
    fd_type: str = "yperman"
    speed_noise_std: int = 0
    front_gate_width: float | None = None
    back_gate_width: float | None = None

    def __post_init__(self) -> None:
        """Applies conditional defaults."""
        if self.front_gate_width is None:
            self.front_gate_width = self.width    

        if self.back_gate_width is None:
            self.back_gate_width = self.width

        # Makes sure this values are never none after instance creation
        if self.back_gate_width is None or self.front_gate_width is None:
            raise ValueError("Gate widths must be provided or default to link width. " \
            "Please provide valid values for front_gate_width and back_gate_width.")
        
        if self.k_jam - self.k_critical <= 0:
            raise ValueError("k_jam must be greater than k_critical " \
            "to avoid division by zero")
        

def virtual_link_config(node_id: int, direction: str, unit_time: float = 1) -> LinkConfig:
    """Build the LinkConfig of a virtual (origin/destination) link attached to a node.

    Virtual links only carry flows in and out of the network, so their physical
    parameters are neutral placeholders that are never used for density or speed.

    Args:
        node_id (int): id of the node the virtual link is attached to.
        direction (str): "in" (flow entering the network at the node) or "out".
        unit_time (float): unit time of the simulation.
    """
    if direction not in ("in", "out"):
        raise ValueError("Direction must be either 'in' or 'out'.")
    return LinkConfig(
        link_id=f"virtual_{direction}_{node_id}",
        start_node=None if direction == "in" else node_id,
        end_node=node_id if direction == "in" else None,
        unit_time=unit_time,
        length=1.0,
        width=1.0,
        free_flow_speed=1.0,
        k_critical=1.0,
        k_jam=2.0,
    )


class LinkCreator(ABC):
    """Abstract base class for creating different types of links."""

    @abstractmethod
    def create_link(self, config: LinkConfig) -> Link:
        """Creates a network link."""
        pass

class SeparatorCreator(LinkCreator):
    """Creator for the Separator link type."""

    def create_link(self, config) -> Link:
        """Creates a Separator link."""
        return Separator(config)
    

class RegularCreator(LinkCreator):
    """Creator for the Regular link type."""

    def create_link(self, config: LinkConfig) -> Link:
        """Creates a Regular link."""
        return Regular(config)

class VirtualLinkCreator(LinkCreator):
    """Creator for the Virtual link type."""

    def create_link(self, config, **kwargs) -> VirtualLink:
        """Creates a Virtual link."""
        return VirtualLink(config)  


class BaseLink(ABC):
    """Common part of all links: identity, endpoints and flow bookkeeping."""

    def __init__(self, config: LinkConfig) -> None:
        """Initializes a link with its configuration."""
        self.config = config
        self.link_id = config.link_id
        # Endpoints are node ids (None for the outside end of a virtual link). Links do
        # not hold Node objects; the Network wires links to nodes.
        self.start_node = config.start_node
        self.end_node = config.end_node
        self.unit_time = config.unit_time
        # Time-series state is allocated per simulation run, see allocate_state().
        self.simulation_steps: int | None = None

    @property
    def is_allocated(self) -> bool:
        """Whether the time-series state has been allocated."""
        return self.simulation_steps is not None

    def allocate_state(self, simulation_steps: int) -> None:
        """Allocate (or reset) the time-series state for a simulation run.

        Arrays have length ``simulation_steps + 1`` so that they can be indexed
        by time step, with index 0 holding the initial state. Calling this again
        discards previous results, so one link can be reused across runs.

        Args:
            simulation_steps (int): number of simulation steps of the run.
        """
        if simulation_steps < 1:
            raise ValueError("simulation_steps must be a positive integer.")
        self.simulation_steps = simulation_steps
        size = simulation_steps + 1
        self.inflow = np.zeros(size)
        self.outflow = np.zeros(size)
        self.cumulative_inflow = np.zeros(size)
        self.cumulative_outflow = np.zeros(size)
        self.sending_flow = -1 * np.ones(size)  # -1: not computed yet
        self.receiving_flow = -1 * np.ones(size)
        return None

    def update_cum_outflow(self, q_j: float, time_step: int) -> None:
        """Update commulative outflow for given time step.

        Args:
            q_j (float): flow for outflow at time step
            time_step (int): time step

        Returns:
            None
        """
        PREVIOUS_STEP = time_step - 1
        self.outflow[time_step] = q_j
        self.cumulative_outflow[time_step] = (
            self.cumulative_outflow[PREVIOUS_STEP] + q_j
        )
        return None

    def update_cum_inflow(self, q_i: float, time_step: int) -> None:
        """Update commulative inflow for given time step.

        Args:
            q_i (float): flow for inflow at time step
            time_step (int): time step

        Returns:
            None
        """
        PREVIOUS_STEP = time_step - 1
        self.inflow[time_step] = q_i
        self.cumulative_inflow[time_step] = self.cumulative_inflow[PREVIOUS_STEP] + q_i
        return None


class Link(BaseLink):
    """Physical link: geometry, fundamental diagram, gates and travel time.

    Subclasses define the area and how the link interacts with its reverse link through
    the hooks get_density, _fd_speed, _reverse_occupancy, _reverse_sending_flow and
    _record_gates.
    """

    def __init__(self, config: LinkConfig) -> None:
        """Initializes a Link with specific configurations."""
        super().__init__(config)
        self._is_controller = config.is_controller
        self.gamma = config.gamma
        self._reverse_link: Link | None = None
        self.activity_probability = config.activity_probability
        # Physical attributes
        self.length = config.length
        self._width = config.width  # width of link
        self.free_flow_speed = config.free_flow_speed
        self.capacity = self.free_flow_speed * config.k_critical
        self.k_jam = config.k_jam
        self.k_critical = config.k_critical
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.max_travel_time = (
            self.length / 0.05
        )  # Jam threshold, equivalent to speed of 0.01 m/s
        self._front_gate_width = config.front_gate_width
        self._back_gate_width = config.back_gate_width
        self.speed_density_fd = BiDirectionalFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            bi_factor=config.bi_factor,
            model_type=config.fd_type,
            noise_std=config.speed_noise_std)
        self.free_flow_travel_time = min(
            self.length / self.free_flow_speed, self.max_travel_time
        )
        self._travel_time_running_sum = self.free_flow_travel_time
        self.free_flow_tau = round(self.free_flow_travel_time / self.unit_time)

    def allocate_state(self, simulation_steps: int) -> None:
        """Allocate (or reset) flow, density, speed, travel time and gate time series."""
        super().allocate_state(simulation_steps)
        size = simulation_steps + 1
        self.num_pedestrians = np.zeros(size, dtype=np.float32)
        self.density = np.zeros(size, dtype=np.float32)
        self.speed = np.zeros(size, dtype=np.float32)
        self.link_flow = np.zeros(size, dtype=np.float32)
        self.travel_time = np.zeros(size, dtype=np.float32)
        self.travel_time[0] = self.free_flow_travel_time
        self.avg_travel_time = np.full(size, self.free_flow_travel_time, dtype=np.float32)
        self._travel_time_running_sum = self.free_flow_travel_time
        self.back_gate_width_data = self._back_gate_width * np.ones(size)
        self.front_gate_width_data = self._front_gate_width * np.ones(size)
        return None

    @property
    def reverse_link(self) -> Link | None:
        """Gets the reverse link."""
        return self._reverse_link

    @reverse_link.setter
    def reverse_link(self, link: Link | None) -> None:
        """Sets the reverse link.

        Args:
            link (Link | None): the reverse link to set. If None, unsets the reverse link.
        """
        self._reverse_link = link

        if link == self:
            raise ValueError("A link cannot be its own reverse link.")

    @property
    def is_controller(self) -> bool:
        """If link is a controller."""
        return self._is_controller

    @is_controller.setter
    def is_controller(self, value: bool) -> None:
        """Sets the value for is_controller."""
        self._is_controller = value

    @property
    def avg_travel_time_window(self) -> int:
        """Average travel time window."""
        # Using moving average for efficiency
        return round(100 / self.unit_time)

    @property
    def width(self):
        """Width of the link."""
        return self._width

    @property
    def front_gate_width(self):
        """Width of the front gate."""
        return self._front_gate_width

    @front_gate_width.setter
    def front_gate_width(self, value: float):
        self._front_gate_width = value

    @property
    def back_gate_width(self):
        """Width of the back gate."""
        return self._back_gate_width

    @back_gate_width.setter
    def back_gate_width(self, value: float):
        self._back_gate_width = value

    @property
    @abstractmethod
    def area(self) -> float:
        """Area of the link."""
        pass

    def update_density_flow(self, time_step: int) -> None:
        """Updates the number of pedestrians and the density of the link.

        Args:
            time_step (int): current time step

        Returns:
            None
        """
        PREVIOUS_STEP = time_step - 1
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[PREVIOUS_STEP] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area
        return None

    def calculate_sending_flow(self, time_step: int) -> float:
        """Calculate the sending flow of the link, stored at ``time_step - 1``.

        Args:
            time_step (int): Current time step

        Returns:
            float: sending flow
        """
        return link_flow.sending_flow(self, time_step)

    def calculate_receiving_flow(self, time_step: int, with_reverse: bool = False) -> float:
        """Calculate the receiving flow of the link, stored at ``time_step - 1``.

        With ``with_reverse``, the reverse link's sending flow at ``time_step - 1`` is
        subtracted, so it must be computed first (the node computes the sending flows of
        its incoming links, which include these reverse links, before the receiving flows).

        Args:
            time_step (int): current time step
            with_reverse (bool): whether to subtract the reverse link's sending flow. Default is False.

        Returns:
            float: receiving flow
        """
        PREVIOUS_STEP = time_step - 1
        # Order matters: the occupancy may draw a random number.
        reverse_occupancy = self._reverse_occupancy(PREVIOUS_STEP)
        reverse_sending_flow = self._reverse_sending_flow(PREVIOUS_STEP) if with_reverse else 0
        self.receiving_flow[PREVIOUS_STEP] = link_flow.receiving_flow(
            self,
            time_step,
            reverse_occupancy=reverse_occupancy,
            reverse_sending_flow=reverse_sending_flow,
            with_reverse=with_reverse,
        )
        return self.receiving_flow[PREVIOUS_STEP]

    def update_speeds(self, time_step: int) -> None:
        """Update speed, travel time and link flow from the density at ``time_step``.

        Args:
            time_step (int): current time step.

        Returns:
            None
        """
        speed = self._fd_speed(time_step)
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time
        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )
        self._update_avg_travel_time(time_step)
        self._record_gates(time_step)
        return None

    def _update_avg_travel_time(self, time_step: int) -> None:
        """Moving average of the travel time over avg_travel_time_window steps."""
        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )

    @abstractmethod
    def get_density(self, time_step: int) -> float:
        """Density used by the flow model at a time step."""
        pass

    @abstractmethod
    def _fd_speed(self, time_step: int) -> float:
        """Speed from the fundamental diagram at a time step."""
        pass

    @abstractmethod
    def _reverse_occupancy(self, previous_step: int) -> float:
        """Pedestrians of the reverse link occupying space in this link."""
        pass

    @abstractmethod
    def _reverse_sending_flow(self, previous_step: int) -> float:
        """Sending flow of the reverse link competing for this link's capacity."""
        pass

    @abstractmethod
    def _record_gates(self, time_step: int) -> None:
        """Record gate (and separator) widths at a time step, for outputs and visualization."""
        pass


class Separator(Link):
    """Represents a separator link in a transportation network.

    The corridor width is split between the link and its reverse link; the separator
    width is the control variable.
    """

    def __init__(self, config: LinkConfig) -> None:
        """Initializes a Separator link with specific configurations."""
        super().__init__(config)

        # Original width must be kept for downstream calculations.
        separator_width = self._width / 2
        # Modify values specific to separator
        self._separator_width = separator_width
        self._front_gate_width = separator_width
        self._back_gate_width = separator_width
        self.is_controller = True  # A separator shall alway be a controller.

    def allocate_state(self, simulation_steps: int) -> None:
        """Allocate (or reset) time-series state, including separator width history."""
        super().allocate_state(simulation_steps)
        # FIXME: Is the link collecting simulations results? Then maybe should be move to a component responsible for the results collection.
        self.separator_width_data = self._separator_width * np.ones(simulation_steps + 1)
        return None

    @property
    def area(self) -> float:
        """Area of the separator link."""
        return self.length * self.separator_width

    @property
    def separator_width(self):
        """Width of the separator."""
        return self._separator_width

    @separator_width.setter
    def separator_width(self, value: float):
        """Set the separator width; the reverse link gets the rest of the corridor width."""
        self._separator_width = value
        self._front_gate_width = value
        self._back_gate_width = value

        if self.reverse_link:
            self.reverse_link._separator_width = self._width - value
            self.reverse_link._front_gate_width = self._width - value
            self.reverse_link._back_gate_width = self._width - value

    def get_density(self, time_step: int) -> float:
        """Density of this lane only; the separator isolates it from the reverse link."""
        return self.density[time_step]

    def _fd_speed(self, time_step: int) -> float:
        """Speed from the fundamental diagram, without opposing flow."""
        return self.speed_density_fd(self.density[time_step], 0)

    def _reverse_occupancy(self, previous_step: int) -> float:
        """The reverse lane is separated, so it takes no space here."""
        return 0

    def _reverse_sending_flow(self, previous_step: int) -> float:
        """The reverse lane is separated, so it does not compete for capacity."""
        return 0

    def _record_gates(self, time_step: int) -> None:
        self.separator_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the separator width
        self.back_gate_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the back gate width


class Regular(Link):
    """Represents a regular (bidirectional) link in a transportation network."""

    @property
    def area(self) -> float:
        """Area of the link."""
        return self.length * self.width

    def get_density(self, time_step: int) -> float:
        """Density of both directions together.

        Args:
            time_step (int): time step

        Returns:
            float: density value
        """
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def _fd_speed(self, time_step: int) -> float:
        """Speed from the bidirectional fundamental diagram (own and opposing density)."""
        k_opp = 0
        if self.reverse_link:
            k_opp = self.reverse_link.density[time_step]
        return self.speed_density_fd(self.density[time_step], k_opp)

    def _reverse_occupancy(self, previous_step: int) -> float:
        """Random share of the reverse link's pedestrians occupying space in this link."""
        reverse_peds = self.reverse_link.num_pedestrians[previous_step]
        return np.random.binomial(n=reverse_peds, p=link_flow.REVERSE_OCCUPANCY_PROB)

    def _reverse_sending_flow(self, previous_step: int) -> float:
        """Sending flow of the reverse link, which competes for this link's capacity.

        Raises:
            Warning: if it is negative, i.e. not computed yet for this step.
        """
        reverse_sending_flow = self.reverse_link.sending_flow[previous_step]
        if reverse_sending_flow < 0:
            raise Warning(
                f"Negative reverse sending flow detected at time step {previous_step + 1} "
                f"for link {self.link_id}: {reverse_sending_flow}"
            )
        return reverse_sending_flow

    def _record_gates(self, time_step: int) -> None:
        self.back_gate_width_data[time_step] = self.back_gate_width
        self.front_gate_width_data[time_step] = self.front_gate_width


class VirtualLink(BaseLink):
    """Virtual link of an origin/destination node: where pedestrians enter or leave the network.

    It only carries flows; it has no geometry or speed. The demand profile lives on the
    node (see Network.set_demand).
    """

    def __init__(self, config: LinkConfig) -> None:
        """Initializes a Virtual link with specific configurations."""
        super().__init__(config)
        self._direction: str = ""

    @property
    def direction(self) -> str:
        """Get the direction of the virtual link."""
        return self._direction

    @direction.setter
    def direction(self, value: str) -> None:
        """Set the direction of the virtual link.

        Args:
            value (str): The direction to set. Should be either "in" or "out".

        Raises:
            ValueError: If the provided value is not "in" or "out".
        """
        if value not in ["in", "out"]:
            raise ValueError("Direction must be either 'in' or 'out'.")
        self._direction = value
