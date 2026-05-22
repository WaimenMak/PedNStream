"""Representation for links in a transportation network"""

import numpy as np
from pednstream.utils.functions import BiDirectionalFd, cal_link_flow_kv
from typing import Any
from numpy import ndarray
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LinkConfig:

    link_id: int 
    start_node: int  
    end_node : int  
    simulation_steps : int 
    unit_time: int
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
    speed_noise_std: float = 0
    front_gate_width: float | None = None
    back_gate_width: float | None = None
    reverse_link: int | None = None

    def __post_init__(self) -> None:
        """Applies conditional defaults"""

        if self.front_gate_width is None:
            self.front_gate_width = self.width    

        if self.back_gate_width is None:
            self.back_gate_width = self.width

        # Makes sure this values are never none after instance creation
        if self.back_gate_width is None or self.front_gate_width is None:
            raise ValueError


class LinkCreator(ABC):

    @abstractmethod
    def create_link(self, config: LinkConfig) -> Link:
        """Creates a network link"""
        pass



class SeparatorCreator(LinkCreator):
    """Creator for the Separator link type"""

    def create_link(self, config) -> Link:
        return Separator(config)
    

class RegularCreator(LinkCreator):
    """Creator for the Regular link type"""

    def create_link(self, config: LinkConfig) -> Link:
        return Regular(config)
    

class Link(ABC):
    """Interface for different Link types"""

    def __init__(self, config: LinkConfig) -> None:
        self.config = self.config

        self.link_id = config.link_id
        self.start_node = config.start_node
        self.end_node = config.end_node
        self.simulation_steps = config.simulation_steps
        self._is_controller: config.is_controller
        # Dynamic attributes
        self.inflow = np.zeros(self.simulation_steps + 1)  # adjust index
        self.outflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_inflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_outflow = np.zeros(self.simulation_steps + 1)
        self.sending_flow = -1 * np.ones(self.simulation_steps + 1)
        self.receiving_flow = -1 * np.ones(self.simulation_steps + 1)
        self.num_pedestrians = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.density = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.speed = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.link_flow = np.zeros(self.simulation_steps + 1, dtype=np.float32)
        self.gamma = config.gamma  
        self.reverse_link = config.reverse_link
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
        self.back_gate_width_data = self._back_gate_width * np.ones(
            self.simulation_steps + 1
        )
        self.front_gate_width_data = self._front_gate_width * np.ones(
            self.simulation_steps + 1
        )
        self.speed_density_fd = BiDirectionalFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            bi_factor=config.bi_factor
            model_type=config.fd_type,
            noise_std=config.speed_noise_std)
        self._exponent = (
            0.8  # private attribute for the releasing factor exponent, default is 1
        )
        self._travel_time_running_sum = self.travel_time[0]
        self.unit_time = config.unit_time
        self.free_flow_tau = round(self.travel_time[0] / self.unit_time)

    @property
    def is_controller(self) -> bool:
        """If link is a controller"""
        return self._is_controller
    
    @is_controller.setter
    def is_controller(self, value: bool) -> None:
        """Sets the value for is_controller"""
        self._is_controller = value

    @property
    def avg_travel_time_window(self) -> int:
        """Average travel time window"""
        # Using moving average for efficiency
        return round(100 / self.unit_time)

    @property
    def avg_travel_time(self) -> ndarray:
        """Average travel time"""
        avg = np.ones(self.simulation_steps + 1, dtype=np.float32) * self.travel_time[0]
        return avg

    @property
    def travel_time(self) -> ndarray:
        """Travel time."""
        _time = np.zeros(self.simulation_steps + 1, dtype=np.float32)

        _time[0] = min(self.length / self.free_flow_speed, self.max_travel_time)
        return _time

    @property
    def width(self):
        """Width of the link."""
        return self._width

    @property
    def front_gate_width(self):
        """Width of the front gate."""
        return self._back_gate_width

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
    def area(self):
        """Area of the link."""
        return self.length * self.width

    @staticmethod # TODO: why do we need this method, we already have dict.get, which does the same
    def _set_conditional_param(source: dict, param: str, default) -> Any:
        """Extracts the value of a parameter from a source.

        Args:
            source (dict): a flat keyword:value set of parameters
            param: name of the paramter to extract
            default: value to default if a parameter's value is None or parameter doesn't exist in source.

        Returns:
            (Any): parameter's value
        """
        if not isinstance(param, str):
            raise ValueError("Parameter name must be a string")

        value = source.get(param)
        if value is None:
            value = default
        return value

    
    @abstractmethod
    def operations(self)-> str:
        pass

    def set_conditional_param(self, source: dict, param: str, default):

        return None




class Separator(Link):
    
    def operations(self):
        return "a separtor link"
    

class Regular(Link):

    def operations(self):
        return "a regular link"




class Link1:
    """Represents a generic link in a transportation network """

    def __init__(
        self, link_id, start_node, end_node, simulation_steps, unit_time, **kwargs
    ):
        """Initialize a link with keyword parameters.

        Args:
            link_id (int): Id of the link
            start_node (): starting node
            kwargs (dict): parameters for the link, including:
                - length: Length of the link
                - width: Width of the link
                - back_gate_width: Width of the back gate
                - front_gate_width: Width of the front gate
                - free_flow_speed: Free flow speed
                - k_critical: Critical density
                - k_jam: Jam density
                - unit_time: Time step size
                - simulation_steps: Number of simulation steps
                - gamma: Optional, default 2e-2
                - exponent: for the releasing factor, default 1

        """
        self.link_id = link_id
        self.start_node = start_node
        self.end_node = end_node
        self.simulation_steps = simulation_steps
        self._is_controller: bool = False 
        # Dynamic attributes
        self.inflow = np.zeros(self.simulation_steps + 1)  # adjust index
        self.outflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_inflow = np.zeros(self.simulation_steps + 1)
        self.cumulative_outflow = np.zeros(self.simulation_steps + 1)
        self.sending_flow = -1 * np.ones(self.simulation_steps + 1)
        self.receiving_flow = -1 * np.ones(self.simulation_steps + 1)
        self.num_pedestrians = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.density = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.speed = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.link_flow = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.gamma = kwargs.get("gamma", 2e-3)  # Defaults to diffusion coefficient
        self.reverse_link = None
        self.activity_probability = kwargs.get("activity_probability", 0.0)
        # Physical attributes
        self.length = kwargs["length"]
        self._width = kwargs["width"]  # width of link
        self.free_flow_speed = kwargs["free_flow_speed"]
        self.capacity = self.free_flow_speed * kwargs["k_critical"]
        self.k_jam = kwargs["k_jam"]
        self.k_critical = kwargs["k_critical"]
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.max_travel_time = (
            self.length / 0.05
        )  # Jam threshold, equivalent to speed of 0.01 m/s
        #
        self._front_gate_width = self._set_conditional_param(
            kwargs, "front_gate_width", self.width
        )
        self._back_gate_width = self._set_conditional_param(
            kwargs, "back_gate_width", self.width
        )
        self.back_gate_width_data = self._back_gate_width * np.ones(
            simulation_steps + 1
        )
        self.front_gate_width_data = self._front_gate_width * np.ones(
            simulation_steps + 1
        )
        self.speed_density_fd = BiDirectionalFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            bi_factor=kwargs.get("bi_factor", 1),
            model_type=kwargs.get("fd_type", "yperman"),
            noise_std=kwargs.get("speed_noise_std", 0),
        )
        self._exponent = (
            0.8  # private attribute for the releasing factor exponent, default is 1
        )
        self._travel_time_running_sum = self.travel_time[0]
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time[0] / self.unit_time)

    @property
    def is_controller(self) -> bool:
        """If link is a controller"""
        return self._is_controller
    
    @is_controller.setter
    def is_controller(self, value: bool) -> None:
        """Sets the value for is_controller"""
        self._is_controller = value

    @property
    def avg_travel_time_window(self) -> int:
        """Average travel time window"""
        # Using moving average for efficiency
        return round(100 / self.unit_time)

    @property
    def avg_travel_time(self) -> ndarray:
        """Average travel time"""
        avg = np.ones(self.simulation_steps + 1, dtype=np.float32) * self.travel_time[0]
        return avg

    @property
    def travel_time(self) -> ndarray:
        """Travel time."""
        _time = np.zeros(self.simulation_steps + 1, dtype=np.float32)

        _time[0] = min(self.length / self.free_flow_speed, self.max_travel_time)
        return _time

    @property
    def width(self):
        """Width of the link."""
        return self._width

    @property
    def front_gate_width(self):
        """Width of the front gate."""
        return self._back_gate_width

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
    def area(self):
        """Area of the link."""
        return self.length * self.width

    def _set_conditional_param(self, source: dict, param: str, default) -> Any:
        """Extracts the value of a parameter from a source.

        Args:
            source (dict): a flat keyword:value set of parameters
            param: name of the paramter to extract
            default: value to default if a parameter's value is None or parameter doesn't exist in source.

        Returns:
            (Any): parameter's value
        """
        if not isinstance(param, str):
            raise ValueError("Parameter name must be a string")

        value = source.get(param)
        if value is None:
            value = default
        return value

    def update_cum_outflow(self, q_j: float, time_step: int) -> None:
        """Update commulative outflow for current time step."""
        self.outflow[time_step] = q_j
        self.cumulative_outflow[time_step] = (
            self.cumulative_outflow[time_step - 1] + q_j
        )
        return None

    def update_cum_inflow(self, q_i: float, time_step: int) -> None:
        """Update commulative inflow for current time step."""
        self.inflow[time_step] = q_i
        self.cumulative_inflow[time_step] = self.cumulative_inflow[time_step - 1] + q_i
        return None

    def update_link_density_flow(self, time_step: int) -> None:
        """Updates the link density flow.
        Args:
            time_step (int): time step

        Returns:
            None
        """
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area
        return None

    def update_speeds(self, time_step: int) -> None:
        """Update the speeds of the link based on the density.
        Args:
            time_step (int): current time step + 1, is the future time step

        Returns:
            None
        """
        # TODO: improve descriptions in docstrings
        k_self = self.density[time_step]
        k_opp = 0
        if self.reverse_link:
            k_opp = self.reverse_link.density[time_step]

        speed = self.speed_density_fd(
            k_self, k_opp
        )  # speed for actual link speed derived from FD, speed_eff for the sending flow calculation
        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time
        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )
        # update the back_gate_width_data
        self.back_gate_width_data[time_step] = self.back_gate_width
        self.front_gate_width_data[time_step] = self.front_gate_width
        return None

    def compute_density(self, time_step: int) -> float:
        """Computes the density of the link for the current time step.

        Args:
            time_step (int): time step

        Returns:
            float: density value
        """
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def compute_outflow(self, time_step: int, tau: int) -> int:
        """Computes outflow with diffusion behavior for the current time step.

        Args:
            time_step (int): time step
            tau (int):

        Returns:
            int: outflow value
        """
        travel_time = self.avg_travel_time[
            time_step
        ]  # use average travel time to calculate

        F = 1 / (1 + self.gamma * travel_time)

        idx = min(max(0, time_step + 1 - tau), len(self.inflow) - 1)
        idx1 = max(0, idx - 1)
        idx2 = max(0, idx - 2)
        idx3 = max(0, idx - 3)
        sending_flow = (
            F * self.inflow[idx]
            + F * (1 - F) * self.inflow[idx1]
            + F * (1 - F) ** 2 * self.inflow[idx2]
            + F * (1 - F) ** 3 * self.inflow[idx3]
        )

        return max(np.ceil(sending_flow), 0)

    def calculate_sending_flow(self, time_step: int) -> float:
        """Calculate the sending flow of the link at a given time step.

        Args:
            time_step (int): Current time step (t - 1)

        Returns:
            float: sending flow
        """
        # get the total density
        density = self.compute_density(time_step)

        tau = round(
            self.avg_travel_time[time_step] / self.unit_time
        )  # use average travel time to calculate tau

        """ for the initial stage """
        # if time_step - tau < 0:
        if time_step <= self.free_flow_tau:
            self.sending_flow[time_step] = 0
            return self.sending_flow[time_step]

        else:
            # if time_step - tau + 1 < 0: # congestion stage
            """ for the normal stage or the congestion stage """

            idx = min(max(0, time_step + 1 - tau), len(self.cumulative_inflow) - 1)

            # congestion_factor = np.clip(self.density[time_step] / self.k_jam, 0, 1)
            congestion_factor = np.clip(
                (self.density[time_step] - self.k_critical)
                / (self.k_jam - self.k_critical),
                0,
                1,
            )  # this one is theoratically more realistic for unidirectional condition

            boundary_congestion = self.num_pedestrians[time_step]
            boundary_freeflow = max(
                0, self.cumulative_inflow[idx] - self.cumulative_outflow[time_step]
            )

            sending_flow_boundary = (
                congestion_factor * boundary_congestion
                + (1 - congestion_factor) * boundary_freeflow
            )

            sending_flow_max = (
                self.front_gate_width
                * self.k_critical
                * self.free_flow_speed
                * self.unit_time
            )
            sending_flow = min(sending_flow_boundary, sending_flow_max)
            # TODO: fix the flow release logic: if sending flow >0, then use diffusion flow

        """ The purpose is to mitigate the maximum sending flow to avoid unrealistic high flow (Added)"""
        # TODO: split this into more manageble and testable pieces.
        original_sending_flow = sending_flow
        if sending_flow > 0:
            releasing_factor = np.clip(density / self.k_jam, 0, 1)
            releasing_prob = 0.7 + (0.85 - 0.7) * releasing_factor**self._exponent

            # free flow stage
            if density <= self.k_critical:
                diffusion_flow = self.compute_density(time_step, tau)
                #     # If diffusion flow is active, it represents the arrival of a platoon.
                if diffusion_flow > 0:
                    weight = 0.8
                    sending_flow = int(
                        np.floor(
                            min(
                                weight * diffusion_flow + (1 - weight) * sending_flow,
                                sending_flow,
                            )
                        )
                    )

                else:  # Prevent maximum flow when the inflow is 0, therefore duffusion flow is 0
                    num_passing_peds = int(np.floor(sending_flow))
                    num_leave = np.random.binomial(
                        n=num_passing_peds, p=releasing_prob
                    )  # 70% of the people will leave
                    sending_flow = num_leave
            else:
                num_passing_peds = int(np.floor(sending_flow))
                num_leave = np.random.binomial(
                    n=num_passing_peds, p=releasing_prob
                )  # 70% of the people will leave
                sending_flow = num_leave
            if sending_flow < 0:
                raise ValueError(
                    f"Negative sending flow detected, {sending_flow} at {time_step}"
                )

        """ Stochastic model for pedestrians performing activities (Added)"""
        if self.activity_probability > 0 and sending_flow > 1:
            # Number of pedestrians who could potentially leave
            potential_leavers = int(np.floor(sending_flow))
            # Each pedestrian has an independent probability of staying for an "activity".
            # We use a binomial distribution to find out how many stay.
            num_staying = np.random.binomial(
                n=potential_leavers, p=self.activity_probability
            )
            # Reduce the sending flow by the number of people who decided to stay
            sending_flow -= num_staying

        """ Smooth the sending flow to avoid unrealistic high flow (Added) """
        sending_flow = max(0, sending_flow)
        sending_flow = min(
            np.floor(0.8 * sending_flow + 0.2 * self.sending_flow[time_step - 1]),
            original_sending_flow,
        )
        if sending_flow < 0:
            raise ValueError(
                "Negative sending flow detected, sending flow more than original flow"
            )
        self.sending_flow[time_step] = sending_flow
        return self.sending_flow[time_step]

    def calculate_receiving_flow(self, time_step: int) -> float:
        """Calculate the receiving flow of the link at a given time step.

        Args:
            time_step (int): Current time step - 1

        Returns:
            float: recieving flow
        """

        # TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length / (self.shockwave_speed * self.unit_time))
        reverse_peds = self.reverse_link.num_pedestrians[time_step]
        reverse_peds_rand = np.random.binomial(n=reverse_peds, p=0.9)

        if time_step + 1 - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area - reverse_peds_rand
        else:
            receiving_flow_boundary = max(
                0,
                self.cumulative_outflow[time_step + 1 - tau_shockwave]
                + self.k_jam * self.area
                - reverse_peds_rand
                - self.cumulative_inflow[time_step],
            )

        receiving_flow_max = (
            self.back_gate_width
            * self.k_critical
            * self.free_flow_speed
            * self.unit_time
        )
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        """smooth the receiving flow (Optional when we use the mitigation method)"""
        if self.receiving_flow[time_step - 1] >= 0:
            receiving_flow = min(
                np.floor(
                    receiving_flow * 0.8 + self.receiving_flow[time_step - 1] * 0.2
                ),
                receiving_flow,
            )
        return receiving_flow

    def calculate_receiving_flow_with_reverse(
        self, time_step: int, reverse_sending_flow: float
    ) -> float:
        """Calculate receiving flow considering reverse link interaction.

        Args:
            time_step (int): time step
            reverse_sending_flow (float): reverse sending flow

        Returns:
            float: recieving flow for reverse nodes
        """
        forward_receiving_flow = self.calculate_receiving_flow(time_step)

        receiving_flow = forward_receiving_flow - reverse_sending_flow
        return max(receiving_flow, 0)


@DeprecationWarning
class OldLink:
    """Physical link with full traffic dynamics."""

    def __init__(
        self, link_id, start_node, end_node, simulation_steps, unit_time, **kwargs
    ):
        """Initialize a Link with parameters from kwargs.

        :param link_id: ID of the link
        :param start_node: Starting node
        :param end_node: Ending node
        :param kwargs: Keyword arguments including:
            - length: Length of the link
            - width: Width of the link
            - back_gate_width: Width of the back gate
            - front_gate_width: Width of the front gate
            - free_flow_speed: Free flow speed
            - k_critical: Critical density
            - k_jam: Jam density
            - unit_time: Time step size
            - simulation_steps: Number of simulation steps
            - gamma: Optional, default 2e-2
            - exponent: for the releasing factor, default 1
        """
        super().__init__(link_id, start_node, end_node, simulation_steps)

        # Physical attributes
        self.length = kwargs["length"]
        self._width = kwargs["width"]  # width of the link
        if kwargs.get("back_gate_width", None) is not None:
            self._back_gate_width = kwargs["back_gate_width"]
        else:
            self._back_gate_width = (
                self.width,
            )  # width of the gate, for gate control in the tail
        if kwargs.get("front_gate_width", None) is not None:
            self._front_gate_width = kwargs["front_gate_width"]
        else:
            self._front_gate_width = (
                self.width
            )  # width of the gate, for gate control in the head
        self.back_gate_width_data = self._back_gate_width * np.ones(
            simulation_steps + 1
        )
        self.front_gate_width_data = self._front_gate_width * np.ones(
            simulation_steps + 1
        )
        self.free_flow_speed = kwargs["free_flow_speed"]
        self.capacity = self.free_flow_speed * kwargs["k_critical"]
        self.k_jam = kwargs["k_jam"]
        self.k_critical = kwargs["k_critical"]
        self.shockwave_speed = self.capacity / (self.k_jam - self.k_critical)
        self.current_speed = self.free_flow_speed
        self.max_travel_time = (
            self.length / 0.05
        )  # Jam threshold, equivalent to speed of 0.01 m/s

        self.speed_density_fd = BiDirectionalFd(
            v_f=self.free_flow_speed,
            k_critical=self.k_critical,
            k_jam=self.k_jam,
            bi_factor=kwargs.get("bi_factor", 1),
            model_type=kwargs.get("fd_type", "yperman"),
            noise_std=kwargs.get("speed_noise_std", 0),
        )

        self.exponent = (
            0.8  # private attribute for the releasing factor exponent, default is 1
        )

        self.travel_time = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.travel_time[0] = min(
            self.length / self.free_flow_speed, self.max_travel_time
        )
        self._travel_time_running_sum = self.travel_time[0]
        self.unit_time = unit_time
        self.free_flow_tau = round(self.travel_time[0] / self.unit_time)

        # For efficient moving average calculation
        self.avg_travel_time_window = round(100 / self.unit_time)
        self.avg_travel_time = (
            np.ones(simulation_steps + 1, dtype=np.float32) * self.travel_time[0]
        )

        # Additional dynamic attributes
        self.num_pedestrians = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.density = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.speed = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.link_flow = np.zeros(simulation_steps + 1, dtype=np.float32)
        self.gamma = kwargs.get(
            "gamma", 2e-3
        )  # Default value if not provided, diffusion coefficient
        self.reverse_link = None
        self.activity_probability = kwargs.get("activity_probability", 0.0)

    @property
    def width(self):
        return self._width

    @property
    def front_gate_width(self):
        return self._front_gate_width

    @front_gate_width.setter
    def front_gate_width(self, value: float):
        self._front_gate_width = value

    @property
    def back_gate_width(self):
        return self._back_gate_width

    @back_gate_width.setter
    def back_gate_width(self, value: float):
        self._back_gate_width = value

    @property
    def area(self):
        """Calculates the area of the link dynamically based on its width and length."""
        return self.length * self.width

    def update_link_density_flow(self, time_step: int):
        num_peds = self.inflow[time_step] - self.outflow[time_step]
        self.num_pedestrians[time_step] = self.num_pedestrians[time_step - 1] + num_peds
        self.density[time_step] = self.num_pedestrians[time_step] / self.area

    def update_speeds(self, time_step: int):
        """
        Update the speed of the link based on the density
        :param time_step: current time step + 1, is the future time step
        :return:
        """
        k_self = self.density[time_step]
        k_opp = 0
        if self.reverse_link:
            k_opp = self.reverse_link.density[time_step]

        speed = self.speed_density_fd(
            k_self, k_opp
        )  # speed for actual link speed derived from FD, speed_eff for the sending flow calculation
        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time
        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )
        # update the back_gate_width_data
        self.back_gate_width_data[time_step] = self.back_gate_width
        self.front_gate_width_data[time_step] = self.front_gate_width

    def get_density(self, time_step: int):
        """Get the density of the link."""
        reverse_num_peds = 0
        if self.reverse_link is not None:
            reverse_num_peds = self.reverse_link.num_pedestrians[time_step]
        return (self.num_pedestrians[time_step] + reverse_num_peds) / self.area

    def get_outflow(self, time_step: int, tau: int) -> int:
        """Get outflow with diffusion behavior."""
        travel_time = self.avg_travel_time[
            time_step
        ]  # use average travel time to calculate

        F = 1 / (1 + self.gamma * travel_time)

        idx = min(max(0, time_step + 1 - tau), len(self.inflow) - 1)
        idx1 = max(0, idx - 1)
        idx2 = max(0, idx - 2)
        idx3 = max(0, idx - 3)
        sending_flow = (
            F * self.inflow[idx]
            + F * (1 - F) * self.inflow[idx1]
            + F * (1 - F) ** 2 * self.inflow[idx2]
            + F * (1 - F) ** 3 * self.inflow[idx3]
        )

        return max(np.ceil(sending_flow), 0)

    def cal_sending_flow(self, time_step: int) -> float:
        """Calculate the sending flow of the link at a given time step.

        :param time_step: Current time step (t - 1)
        """
        # get the total density
        density = self.get_density(time_step)

        tau = round(
            self.avg_travel_time[time_step] / self.unit_time
        )  # use average travel time to calculate tau

        """ for the initial stage """
        # if time_step - tau < 0:
        if time_step <= self.free_flow_tau:
            self.sending_flow[time_step] = 0
            return self.sending_flow[time_step]

        else:
            # if time_step - tau + 1 < 0: # congestion stage
            """ for the normal stage or the congestion stage """

            idx = min(max(0, time_step + 1 - tau), len(self.cumulative_inflow) - 1)

            # congestion_factor = np.clip(self.density[time_step] / self.k_jam, 0, 1)
            congestion_factor = np.clip(
                (self.density[time_step] - self.k_critical)
                / (self.k_jam - self.k_critical),
                0,
                1,
            )  # this one is theoratically more realistic for unidirectional condition

            boundary_congestion = self.num_pedestrians[time_step]
            boundary_freeflow = max(
                0, self.cumulative_inflow[idx] - self.cumulative_outflow[time_step]
            )

            sending_flow_boundary = (
                congestion_factor * boundary_congestion
                + (1 - congestion_factor) * boundary_freeflow
            )

            sending_flow_max = (
                self.front_gate_width
                * self.k_critical
                * self.free_flow_speed
                * self.unit_time
            )
            sending_flow = min(sending_flow_boundary, sending_flow_max)
            # TODO: fix the flow release logic: if sending flow >0, then use diffusion flow

        """ The purpose is to mitigate the maximum sending flow to avoid unrealistic high flow (Added)"""
        original_sending_flow = sending_flow
        if sending_flow > 0:
            releasing_factor = np.clip(density / self.k_jam, 0, 1)
            releasing_prob = 0.7 + (0.85 - 0.7) * releasing_factor**self.exponent

            # free flow stage
            if density <= self.k_critical:
                diffusion_flow = self.get_outflow(time_step, tau)
                #     # If diffusion flow is active, it represents the arrival of a platoon.
                if diffusion_flow > 0:
                    weight = 0.8
                    sending_flow = int(
                        np.floor(
                            min(
                                weight * diffusion_flow + (1 - weight) * sending_flow,
                                sending_flow,
                            )
                        )
                    )

                else:  # Prevent maximum flow when the inflow is 0, therefore duffusion flow is 0
                    num_passing_peds = int(np.floor(sending_flow))
                    num_leave = np.random.binomial(
                        n=num_passing_peds, p=releasing_prob
                    )  # 70% of the people will leave
                    sending_flow = num_leave
            else:
                num_passing_peds = int(np.floor(sending_flow))
                num_leave = np.random.binomial(
                    n=num_passing_peds, p=releasing_prob
                )  # 70% of the people will leave
                sending_flow = num_leave
            if sending_flow < 0:
                raise ValueError(
                    f"Negative sending flow detected, {sending_flow} at {time_step}"
                )

        """ Stochastic model for pedestrians performing activities (Added)"""
        if self.activity_probability > 0 and sending_flow > 1:
            # Number of pedestrians who could potentially leave
            potential_leavers = int(np.floor(sending_flow))
            # Each pedestrian has an independent probability of staying for an "activity".
            # We use a binomial distribution to find out how many stay.
            num_staying = np.random.binomial(
                n=potential_leavers, p=self.activity_probability
            )
            # Reduce the sending flow by the number of people who decided to stay
            sending_flow -= num_staying

        """ Smooth the sending flow to avoid unrealistic high flow (Added) """
        sending_flow = max(0, sending_flow)
        sending_flow = min(
            np.floor(0.8 * sending_flow + 0.2 * self.sending_flow[time_step - 1]),
            original_sending_flow,
        )
        if sending_flow < 0:
            raise ValueError(
                "Negative sending flow detected, sending flow more than original flow"
            )
        self.sending_flow[time_step] = sending_flow
        return self.sending_flow[time_step]

    def cal_receiving_flow(self, time_step: int) -> float:
        """
        Calculate the receiving flow of the link at a given time step
        :param time_step: Current time step - 1
        """

        # TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length / (self.shockwave_speed * self.unit_time))
        reverse_peds = self.reverse_link.num_pedestrians[time_step]
        reverse_peds_rand = np.random.binomial(n=reverse_peds, p=0.9)

        if time_step + 1 - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area - reverse_peds_rand
        else:
            receiving_flow_boundary = max(
                0,
                self.cumulative_outflow[time_step + 1 - tau_shockwave]
                + self.k_jam * self.area
                - reverse_peds_rand
                - self.cumulative_inflow[time_step],
            )

        receiving_flow_max = (
            self.back_gate_width
            * self.k_critical
            * self.free_flow_speed
            * self.unit_time
        )
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        """smooth the receiving flow (Optional when we use the mitigation method)"""
        if self.receiving_flow[time_step - 1] >= 0:
            receiving_flow = min(
                np.floor(
                    receiving_flow * 0.8 + self.receiving_flow[time_step - 1] * 0.2
                ),
                receiving_flow,
            )

        return receiving_flow

    def cal_receiving_flow_with_reverse(
        self, time_step: int, reverse_sending_flow: float
    ) -> float:
        """Calculate receiving flow considering reverse link interaction"""
        forward_receiving_flow = self.cal_receiving_flow(time_step)

        receiving_flow = forward_receiving_flow - reverse_sending_flow
        return max(receiving_flow, 0)


class Separator2(Link):
    """Separator: control object in the network, it adjust the width of the bidirection link"""

    def __init__(
        self, link_id, start_node, end_node, simulation_steps, unit_time, **kwargs
    ):
        super().__init__(
            link_id, start_node, end_node, simulation_steps, unit_time, **kwargs
        )
        self._separator_width = self._width / 2
        self._front_gate_width = self._width / 2
        self._back_gate_width = self._width / 2
        self.separator_width_data = self._width / 2 * np.ones(simulation_steps + 1)

    def get_density(self, time_step: int):
        return self.density[time_step]

    def update_speeds(self, time_step: int):
        """
        Update the speed of the link based on the density
        :param time_step: current time step + 1, is the future time step
        :return:
        """
        # TODO: continue here
        k_self = self.density[time_step]
        speed = self.speed_density_fd(k_self, 0)
        # Update travel time and speed
        self.speed[time_step] = speed
        self.travel_time[time_step] = (
            self.length / speed if speed > 0 else self.max_travel_time
        )  # avoid infinite travel time

        self.link_flow[time_step] = cal_link_flow_kv(
            self.density[time_step], self.speed[time_step]
        )

        self._travel_time_running_sum += self.travel_time[time_step]
        if time_step >= self.avg_travel_time_window:
            self._travel_time_running_sum -= self.travel_time[
                time_step - self.avg_travel_time_window
            ]
            self.avg_travel_time[time_step] = (
                self._travel_time_running_sum / self.avg_travel_time_window
            )

        self.separator_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the separator width
        self.back_gate_width_data[time_step] = (
            self._separator_width
        )  # for visualization of the back gate width

    @property
    def area(self):
        return self.length * self._separator_width

    @property
    def separator_width(self):
        return self._separator_width

    @separator_width.setter
    def separator_width(self, value):
        """
        Sets the width of this link and dynamically adjusts the reverse link's width
        to maintain a constant total corridor width.
        value: float, the width of the separator, has minimum and maximum value
        """
        self._separator_width = value
        # Update gate widths to match separator width
        self._front_gate_width = value
        self._back_gate_width = value

        if self.reverse_link:
            self.reverse_link._separator_width = self._width - value
            # Also update the reverse link's gate widths
            self.reverse_link._front_gate_width = self._width - value
            self.reverse_link._back_gate_width = self._width - value

    def cal_receiving_flow(self, time_step: int) -> float:
        """
        Calculate the receiving flow of the link at a given time step
        :param time_step: Current time step - 1
        """

        # TODO: is using length the correct way to calculate receiving flow?
        tau_shockwave = round(self.length / (self.shockwave_speed * self.unit_time))

        if time_step + 1 - tau_shockwave < 0:
            receiving_flow_boundary = self.k_jam * self.area

        else:
            receiving_flow_boundary = (
                self.cumulative_outflow[time_step + 1 - tau_shockwave]
                + self.k_jam * self.area
                - self.cumulative_inflow[time_step]
            )

        receiving_flow_max = (
            self.back_gate_width
            * self.k_critical
            * self.free_flow_speed
            * self.unit_time
        )
        receiving_flow = min(receiving_flow_boundary, receiving_flow_max)
        if receiving_flow < 0:
            print(f"Negative receiving flow detected, {receiving_flow} at {time_step}")
        receiving_flow = max(receiving_flow, 0)

        """smooth the receiving flow (Optional when we use the mitigation method)"""
        if self.receiving_flow[time_step - 1] >= 0:
            receiving_flow = min(
                np.floor(
                    receiving_flow * 0.8 + self.receiving_flow[time_step - 1] * 0.2
                ),
                receiving_flow,
            )

        return receiving_flow

    def cal_receiving_flow_with_reverse(
        self, time_step: int, reverse_sending_flow: float
    ) -> float:
        """Calculate receiving flow for separator (no reverse link interaction)"""
        forward_receiving_flow = self.cal_receiving_flow(time_step)
        return max(forward_receiving_flow, 0)


    